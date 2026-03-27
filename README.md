# ANE Inference Engine

Speculative decoding on Apple Silicon using the **Neural Engine (ANE)** for draft generation and **Metal GPU** for verification — running truly in parallel on independent hardware.

## Result

**1.14x average speedup** over autoregressive baseline on M4 16GB:

| Config | tok/s | Speedup | Acceptance |
|--------|-------|---------|------------|
| Qwen2.5-14B Q4 baseline (GPU only) | 10.5 | 1.00x | — |
| **+ 0.5B draft on ANE (INT8, pipelined)** | **12.0** | **1.14x** | **76%** |

Measured across 5 diverse prompts, 80 tokens each. Fully reproducible.

## Why This Matters

Every existing inference engine on Apple Silicon uses **one compute unit at a time**. But Apple Silicon has three — CPU, GPU, and ANE — each with independent memory bandwidth:

```
         ┌─────────┐
         │ Unified  │
         │ Memory   │
         └────┬─────┘
              │
    ┌─────────┼─────────┐
    │         │         │
  ┌─┴──┐  ┌──┴──┐  ┌──┴──┐
  │ CPU │  │ GPU │  │ ANE │
  └────┘  └─────┘  └─────┘
   8 GB/s  55 GB/s  22 GB/s   ← independent DCS channels
```

We proved that running ANE and GPU simultaneously causes **only 7.5% GPU degradation**. This enables true parallel speculative decoding.

## Architecture

```
Cycle K:
  ┌──────────────────────────────────────────────────────────┐
  │ ANE (CoreML, INT8)              GPU (llama.cpp, Metal)   │
  │                                                          │
  │ Generate 1 draft candidate      Verify [bonus + cand]    │
  │ for cycle K+1                   from cycle K             │
  │ (~14ms)                         (~100ms)                 │
  │                                                          │
  │        ← running in parallel on independent HW →         │
  └──────────────────────────────────────────────────────────┘
                            │
                   Accept/reject + bonus token
                   ~1.76 tokens per cycle (76% acceptance)
```

### Key Optimizations

1. **Merged bonus eval** — Instead of a separate GPU call for the bonus token (~100ms), prepend it to the next cycle's batch. Saves one full GPU forward pass per cycle.

2. **ANE/GPU pipelined overlap** — While GPU verifies cycle K, ANE speculatively generates candidates for cycle K+1 (assuming all accepted). 70%+ prefetch hit rate.

3. **INT8 quantization** — Draft model quantized to INT8 on CoreML. ANE latency: 26ms → 18ms (1.4x faster).

4. **Sequence length reduction** — Draft uses seq_len=64 instead of 128. CoreML has no KV cache, so shorter = faster. 18ms → 14ms.

Combined effect: **0.61x → 1.14x** (from sync to fully optimized pipelined).

## Quick Start

### Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- ~10GB disk for models

### Setup

```bash
git clone https://github.com/caiovicentino/ane-inference-engine
cd ane-inference-engine

# Install dependencies
pip install coremltools torch numpy transformers safetensors huggingface_hub

# Install llama.cpp with Metal support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# Run tests (uses mock backends, no models needed)
python -m pytest tests/ -v
```

### Download Models

```python
from huggingface_hub import snapshot_download, hf_hub_download

# Draft model (0.5B, for ANE)
snapshot_download("Qwen/Qwen2.5-0.5B", local_dir="models/Qwen2.5-0.5B")

# Main model (14B Q4, for GPU)
hf_hub_download("Qwen/Qwen2.5-14B-Instruct-GGUF",
    "qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
    local_dir="models/Qwen2.5-14B-GGUF")
# ... download remaining parts
```

### Run Benchmark

```bash
# Convert draft to CoreML + INT8
python tools/convert_draft.py --model models/Qwen2.5-0.5B --seq-len 64 --int8

# Run speculative decoding benchmark
python benchmarks/end_to_end.py \
    --draft models/draft_0.5b_seq64_int8.mlpackage \
    --main models/Qwen2.5-14B-GGUF/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
    --n-candidates 1 --pipelined
```

## Project Structure

```
ane-inference-engine/
├── draft/              # Draft model for ANE
│   ├── model.py        #   Qwen2-compatible transformer (PyTorch)
│   ├── tokenizer.py    #   Tokenizer wrapper
│   └── convert.py      #   PyTorch → CoreML conversion
├── engine/             # Speculative decoding engine
│   ├── coordinator.py  #   Synchronous spec-dec loop
│   ├── pipelined.py    #   Merged bonus + ANE/GPU overlap (best)
│   ├── threaded.py     #   Background prefetch variant
│   ├── speculative.py  #   Verify + accept/reject logic
│   ├── sampler.py      #   Greedy / temperature sampling
│   ├── kv_cache.py     #   KV cache position tracking
│   └── adaptive.py     #   Dynamic draft count tuning
├── gpu/                # GPU backend
│   ├── backend.py      #   Abstract GPUBackend interface
│   ├── llama_bridge.py #   llama.cpp Metal implementation
│   └── mock_backend.py #   Mock for testing
├── benchmarks/         # Benchmark scripts
│   ├── end_to_end.py   #   Full pipeline benchmark
│   ├── bandwidth.py    #   ANE/GPU bandwidth measurement
│   └── compare.py      #   Speedup comparison
├── tools/              # Training notebooks
│   ├── train_draft_online_v2_colab.ipynb  # Best distillation approach
│   └── ...
├── tests/              # Test suite (123 tests)
├── DESIGN.md           # Original design document
├── FINDINGS.md         # Detailed results and analysis
└── requirements.txt
```

## Findings Summary

See [FINDINGS.md](FINDINGS.md) for the full analysis. Key takeaways:

- **The architecture works**: ANE and GPU run in parallel with 7.5% degradation
- **The bottleneck is CoreML**: No KV cache support forces full context recomputation per draft token (14ms minimum for 0.5B)
- **Pipelining is critical**: Merged bonus eval + prefetch overlap turned 0.61x into 1.14x
- **Larger main models benefit more**: 1.5B (0.83x) → 7B (0.44x) → 14B (1.14x). The trend favors 32B+ models
- **Draft quality matters**: The 0.5B pretrained model (76% acceptance) massively outperforms distilled small models (10% acceptance)

## Comprehensive Benchmark Results

| Main Model | Draft | Method | tok/s | vs Baseline |
|---|---|---|---|---|
| 1.5B Q4 | 0.5B ANE | Sync N=4 | 7.8 | 0.83x |
| 7B Q4 | 0.5B ANE | Sync N=4 | 7.2 | 0.43x |
| 14B Q4 | 0.5B ANE | Sync N=2 | 7.2 | 0.71x |
| 14B Q4 | 0.5B ANE | **Pipelined N=1 INT8** | **12.0** | **1.14x** |
| 14B Q4 | 80M distilled | Sync N=4 | 4.4 | 0.48x |
| 14B Q4 | 43M distilled | Sync N=4 | 3.9 | 0.42x |

## Known Limitations

1. **CoreML has no KV cache** — Each draft token requires full forward pass over the context. A direct ANE interface would eliminate this.
2. **Marginal speedup** — 1.14x is real but modest. The approach scales better with larger (slower) main models.
3. **Fixed sequence length** — CoreML requires fixed input shapes. Dynamic shapes would improve efficiency.
4. **Single-token optimal** — N=1 candidates works best because the ANE latency per forward pass is high relative to GPU speed.

## Future Work

- **Direct ANE access** (bypass CoreML) for KV cache support
- **Multi-token prediction heads** on the draft model
- **Larger main models** (32B+) where the speedup compounds
- **Apple Silicon M4 Pro/Max/Ultra** with faster ANE

## Based On

- [ane-speculative-decoding](https://github.com/caiovicentino/ane-speculative-decoding): Proof that ANE+GPU have independent bandwidth
- [maderix/ANE](https://github.com/maderix/ANE): ANE reverse engineering reference
- Leviathan et al. 2023: *Fast Inference from Transformers via Speculative Decoding*
- Chen et al. 2023: *Accelerating Large Language Model Decoding with Speculative Sampling*

## License

MIT
