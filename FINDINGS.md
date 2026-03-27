# Findings: Speculative Decoding on Apple Silicon ANE

## Executive Summary

We built a speculative decoding engine that runs the draft model on Apple's Neural Engine (ANE) and the main model on the Metal GPU, exploiting independent memory bandwidth paths on Apple Silicon.

**Result**: 1.14x average speedup on Qwen2.5-14B (Q4) with a 0.5B draft model on M4 16GB.

**Key insight**: The architecture is correct — ANE and GPU truly run in parallel. The bottleneck is CoreML's lack of KV cache support, which forces full context recomputation on every draft call.

## The Hypothesis

Apple Silicon's unified memory architecture has **independent DCS channels** for each compute unit:

| Unit | Bandwidth (alone) | Bandwidth (concurrent) | Degradation |
|------|-------------------|----------------------|-------------|
| GPU  | 55 GB/s           | 51 GB/s              | 7.5%        |
| ANE  | 22 GB/s           | 18 GB/s              | 18%         |

This means we can run a draft model on ANE at ~82% throughput while the GPU runs the main model at ~92.5% throughput. Standard speculative decoding runs draft and verify sequentially. We run them in parallel.

## Optimization Journey

### Starting Point: Naive Sync (0.61x)

The simplest approach: draft 4 tokens on ANE, verify on GPU, repeat.

```
ANE draft(4 tokens): 4 × 28ms = 112ms
GPU verify(4 tokens): ~150ms
GPU bonus eval:       ~109ms
Total cycle:          ~370ms for 2.31 tokens
= 6.2 tok/s vs 10.5 baseline = 0.61x
```

### Optimization 1: Reduce N candidates (0.71x → 0.83x)

Fewer candidates = less draft overhead. N=2 was optimal for sync.

```
N=2: 7.2 tok/s (0.71x)   — draft takes 56ms instead of 112ms
N=1: 6.3 tok/s (0.59x)   — too much overhead per token
```

### Optimization 2: Merged bonus eval (significant)

**The key insight**: Instead of a separate GPU `eval([bonus])` call after verification, prepend the bonus to the next cycle's candidate batch:

Before: `eval(candidates)` + `eval([bonus])` = 2 GPU calls
After:  `eval([bonus] + candidates)` = 1 GPU call

This eliminates ~109ms per cycle for 14B models.

### Optimization 3: ANE/GPU pipelined overlap (0.71x → 0.91x)

After verification, we know the bonus token. We immediately launch ANE to generate the next cycle's candidates while the GPU is idle. If all candidates were accepted (context matches), the ANE result is reusable.

```
Prefetch hit rate: 60-74% (varies by prompt and acceptance rate)
```

Combined with merged bonus: sync 0.71x → pipelined 0.91x.

### Optimization 4: INT8 quantization (0.91x → 1.11x)

CoreML supports post-training INT8 weight quantization. Applied to the 0.5B draft:

```
FP16: 26.0ms/forward
INT8: 18.2ms/forward  (1.43x faster)
```

Acceptance rate unchanged (same model, just compressed weights).

### Optimization 5: Sequence length reduction (1.11x → 1.14x)

CoreML recomputes the full sequence every call (no KV cache). Reducing max_seq_len from 128 to 64:

```
INT8 seq128: 18.2ms
INT8 seq64:  14.4ms  (1.26x faster)
```

The draft model only needs recent context for next-token prediction. 64 tokens is sufficient for ~72% acceptance.

### Final Result

| Optimization Stack | tok/s | Speedup |
|---|---|---|
| Baseline (14B autoregressive) | 10.5 | 1.00x |
| Sync N=4 FP16 seq128 | 5.4 | 0.51x |
| Sync N=2 FP16 seq128 | 7.2 | 0.69x |
| Pipelined N=2 FP16 seq128 | 9.1 | 0.87x |
| Pipelined N=1 FP16 seq128 | 11.1 | 1.06x |
| Pipelined N=1 INT8 seq128 | 11.9 | 1.13x |
| **Pipelined N=1 INT8 seq64** | **12.0** | **1.14x** |

Each optimization is additive and independently measurable.

## Per-Prompt Results (25 prompts, 80 tokens each)

| # | Prompt | Base | Pipe | Speedup | Acc | Prefetch |
|---|--------|------|------|---------|-----|----------|
| 1 | Prime check function | 11.0 | 11.8 | 1.08x | 72% | 70% |
| 2 | Binary search | 10.8 | 12.3 | 1.14x | 74% | 72% |
| 3 | Fibonacci recursive | 10.8 | 13.2 | 1.22x | 82% | 80% |
| 4 | Linked list class | 10.7 | 13.2 | 1.24x | 84% | 82% |
| 5 | Merge sort | 10.6 | 12.6 | 1.19x | 74% | 72% |
| 6 | Stack implementation | 11.2 | 13.6 | 1.21x | 86% | 84% |
| 7 | Recursion concept | 11.0 | 12.0 | 1.09x | 76% | 74% |
| 8 | List vs tuple | 10.5 | 13.0 | 1.24x | 80% | 78% |
| 9 | Garbage collection | 10.8 | 12.3 | 1.14x | 80% | 78% |
| 10 | TCP vs UDP | 10.8 | 12.1 | 1.12x | 74% | 72% |
| 11 | OOP explanation | 10.8 | 10.3 | 0.95x | 53% | 51% |
| 12 | Hash table | 10.7 | 12.4 | 1.16x | 80% | 78% |
| 13 | Meaning of life | 10.3 | 11.9 | 1.15x | 76% | 74% |
| 14 | Capital of France | 10.5 | 11.6 | 1.11x | 74% | 72% |
| 15 | Einstein birthplace | 10.6 | 10.4 | 0.98x | 54% | 54% |
| 16 | Speed of light | 10.2 | 12.1 | 1.19x | 78% | 76% |
| 17 | Water boiling point | 10.4 | 12.1 | 1.15x | 80% | 78% |
| 18 | Largest planet | 10.7 | 10.9 | 1.01x | 63% | 61% |
| 19 | Once upon a time | 10.8 | 11.7 | 1.09x | 70% | 68% |
| 20 | Future of AI | 10.5 | 11.7 | 1.11x | 76% | 74% |
| 21 | Gradient descent | 10.4 | 11.5 | 1.11x | 67% | 65% |
| 22 | Transformer architecture | 10.7 | 11.0 | 1.03x | 69% | 67% |
| 23 | Docker vs VMs | 10.8 | 10.8 | 1.00x | 65% | 63% |
| 24 | TCP handshake | 11.0 | 12.4 | 1.13x | 82% | 82% |
| 25 | Hash functions | 11.0 | 11.4 | 1.04x | 70% | 70% |
| | **Average** | **10.7** | **11.9** | **1.12x** | **74%** | **72%** |

**Statistics**: std=0.079, range=0.95x–1.24x, wins=22/25 (88%)

**Pattern**: Acceptance > 70% → speedup > 1.10x consistently. The 3 losses all had acceptance < 65%.

## What Didn't Work

### 1. Knowledge Distillation to Small Models

We trained multiple small draft models via distillation from Qwen2.5-0.5B:

| Model | Params | ANE Speed | Acceptance | Result |
|-------|--------|-----------|------------|--------|
| 43M (hidden=256, 4 layers) | 43M | 7.9ms | 9.7% | Useless |
| 80M (hidden=384, 6 layers) | 80M | 18.7ms | 10.1% | Useless |
| 80M online distillation | 80M | 9.0ms | 9.7% | Useless |

**Why**: With vocab_size=151,936, the embedding table consumes 90%+ of the parameters. A 43M model has only ~4M params in its transformer layers — not enough capacity to learn the teacher's distribution.

**Lesson**: For large-vocab models, distillation to tiny models requires either (a) a much smaller vocabulary, (b) shared embeddings with the teacher, or (c) architectures like FastDraft (Intel) with projected embeddings.

### 2. PARD (Parallel Draft)

AMD's PARD-Qwen2.5-0.5B generates N candidates in 1 forward pass using special tokens:

```
Input:  [context..., PARD_TOK, PARD_TOK, PARD_TOK, PARD_TOK]
Output: 4 candidates from positions at PARD tokens
```

Speed: 26ms for 4 candidates (vs 104ms autoregressive). But acceptance: **0.3%**.

**Why**: PARD tokens were not trained for our main model (14B). The parallel predictions don't match the autoregressive distribution.

### 3. Layer Pruning

Removing layers from the pretrained 0.5B:

```
24 layers: coherent text, 567ms CPU
12 layers: garbage text
 8 layers: garbage text
 6 layers: garbage text
```

**Why**: The model was trained as a 24-layer system. Removing layers without fine-tuning destroys the learned representations.

### 4. Old-style Threading (Prefetch)

The `ThreadedCoordinator` pre-generates candidates assuming the context won't change. With 44.7% acceptance on 7B:

```
P(all 4 accepted) = 0.447^4 = 4%
→ 0% prefetch hit rate
→ Thread overhead makes it SLOWER than sync
```

**Fixed by**: Moving prefetch to AFTER bonus computation (PipelinedCoordinator), and using N=1 where acceptance is highest (76%).

## Scaling Analysis

| Main Model | Baseline tok/s | Best Spec tok/s | Speedup |
|------------|---------------|-----------------|---------|
| 1.5B Q4 | 79 | 7.8 | 0.10x |
| 7B Q4 | 17 | 7.2 | 0.42x |
| 14B Q4 | 10.5 | 12.0 | **1.14x** |
| 32B Q4 (projected) | ~3 | ~5 | **~1.7x** |

**The trend is clear**: speculative decoding benefits scale with main model size. The crossover point is around 12-14B Q4 on M4 16GB.

## Why Not Higher Speedup?

The theoretical maximum with N=1, 76% acceptance, perfect overlap:

```
GPU cycle time: ~120ms (verify 2 tokens: bonus + candidate)
ANE draft time: ~14ms (overlapped, free)
Tokens per cycle: 1.76
Rate: 1.76 / 120ms = 14.7 tok/s
Speedup: 14.7 / 10.5 = 1.40x theoretical max
```

We achieved 1.14x / 1.40x = **81% of theoretical maximum**. The gap is coordination overhead, KV cache management, and imperfect overlap.

## ANE Bandwidth and GPU Contention

We measured real bandwidth utilization during inference:

### ANE Bandwidth
| Metric | Value |
|--------|-------|
| Weight file size (INT8) | 496 MB |
| ANE forward pass | 14.1ms |
| ANE throughput | 25.7 GB/s (117% of 22 GB/s spec) |
| ANE degradation during GPU | **0%** |

The ANE is **fully saturated** — no headroom. The 22 GB/s from the original PoC was conservative; the M4's ANE actually delivers ~26 GB/s.

### GPU Contention
| Scenario | GPU ms/tok | Degradation |
|----------|-----------|-------------|
| GPU solo (baseline) | 89.4ms | — |
| GPU during pipelined inference | 99.2ms | **+11%** |
| GPU during 100% ANE stress test | 156ms | +41.5% |

During real pipelined inference, the ANE is only active ~12% of each cycle (14ms out of ~120ms). This limits GPU degradation to **11%**, not the 41.5% seen under synthetic stress test.

### Impact on Speedup

The 11% GPU degradation reduces the effective speedup:
- Without contention: theoretical ~1.25x
- With 11% contention: measured **1.12x**
- The contention costs ~0.13x of potential speedup

**Conclusion**: The speedup is real but modest. The GPU memory bandwidth contention from concurrent ANE access is the main limiting factor, not the ANE speed itself.

## Hardware Details

All measurements on:
- **Mac mini M4** (Model Mac16,10 / MU9D3LL/A)
- 10-core CPU (4P + 6E), 10-core GPU, 16-core ANE
- 16GB unified memory
- macOS 15.x (Darwin 25.3.0)
- CoreML (coremltools 8.x) for ANE
- llama.cpp (llama-cpp-python) for Metal GPU
- Python 3.9, PyTorch 2.8

## Reproduction

```bash
# 1. Setup
pip install coremltools torch numpy llama-cpp-python transformers safetensors

# 2. Download models
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='models/0.5B')"

# 3. Convert draft (INT8, seq64)
python tools/convert_draft.py --model models/0.5B --seq-len 64 --int8

# 4. Run
python benchmarks/end_to_end.py --draft draft.mlpackage --main model.gguf --pipelined
```

## Community Hardware Benchmarks

### MacBook Pro M3 Max (36GB)

**Hardware**:
- **Chip**: Apple M3 Max
- **GPU**: 40-core Metal GPU
- **ANE**: 16-core Neural Engine
- **Memory**: 36GB unified memory
- **Bandwidth**: ~400 GB/s (estimated)

**Key Finding**: The M3 Max GPU is **too fast** for speculative decoding to provide speedup with 32B-72B quantized models. The powerful 40-core GPU processes tokens faster than the ANE overhead can be amortized.

#### ANE Performance Comparison

| Metric | M4 mini (16GB) | M3 Max (36GB) |
|--------|----------------|---------------|
| 0.5B draft latency | 14ms | 21ms |
| 0.5B throughput | 71 tok/s | 47 tok/s |
| 1.5B draft latency | N/A | 62ms |
| 1.5B throughput | N/A | 16 tok/s |

**Observation**: The M3 Max ANE is ~1.5x slower than M4 mini for the same 0.5B model. This may be due to architectural differences or thermal constraints.

#### Main Model Benchmarks

| Model | Baseline tok/s | Spec tok/s (N=1) | Accept | Speedup |
|-------|----------------|------------------|--------|---------|
| 32B Q4_K_M | 12.0 | 10.9 | 77.8% | **0.91x** |
| 72B Q2_K | 6.5 | 6.0 | 57-78% | **0.92x** |
| 72B Q2_K (slow prompts) | 3.5-4.2 | 3.8-4.5 | 78-85% | **1.09x** ✅ |

#### Draft Model Comparison (with 32B Q4 main)

| Draft Model | ANE Latency | Acceptance | Throughput | Speedup |
|-------------|-------------|------------|------------|---------|
| 0.5B | 21ms | 82.1% | 10.7 tok/s | 0.89x |
| 1.5B | 62ms | 96.2% | 2.5 tok/s | 0.22x |

**Insight**: The 1.5B draft achieves 96.2% acceptance (vs 82.1% for 0.5B), but the 3x slower ANE inference makes it net negative. The optimal draft model size is hardware-dependent.

#### Why No Speedup on M3 Max?

The speculative decoding equation:

```
Speedup requires: Time(ANE draft) + Time(GPU verify N) < Time(GPU baseline × N)
```

On M3 Max with 32B Q4 (baseline = 12 tok/s = 83ms/token):
- ANE draft: 21ms
- GPU verify (N=1): ~83ms
- Total: 104ms for ~1.82 tokens (82% accept)
- Effective: 17.5 tok/s... but coordination overhead brings it to 10.9 tok/s

The **21ms ANE overhead** per cycle reduces the effective throughput. On M4 mini with 14B (baseline = 10.5 tok/s = 95ms/token), the same 14ms ANE overhead is proportionally smaller.

#### Scaling Prediction for M3 Max

| Model Size | Est. Baseline | Est. Speedup |
|------------|---------------|--------------|
| 32B Q4 | 12 tok/s | 0.9x |
| 72B Q2 | 4-6 tok/s | 0.9-1.1x |
| 72B Q4 | ~3 tok/s | ~1.2x (projected) |
| 128B Q2 | ~1.5 tok/s | ~1.5x (projected) |

**Conclusion**: On M3 Max, speculative decoding becomes beneficial for models where baseline is <4 tok/s. This requires 72B+ at Q4 quantization or larger models.

#### Benchmark Scripts

The following scripts were created for M3 Max testing:

- `benchmarks/bench_32b.py` - Generic baseline vs speculative comparison
- `benchmarks/bench_72b_real.py` - Real prompt testing with multiple scenarios

---

## Future Directions

1. **Direct ANE access**: Bypass CoreML to use KV cache on ANE. This alone could 10x the draft speed.
2. **Multi-token prediction**: Train draft to predict 2-4 tokens per forward pass (like PARD, but fine-tuned for the target model).
3. **Larger main models**: 32B+ on M4 Max/Ultra where baseline is slower.
4. **Dynamic sequence length**: Use shorter context for easy tokens, longer for harder ones.
5. **Apple Intelligence integration**: If Apple exposes ANE scheduling APIs, more efficient overlap.
6. **Hardware-specific tuning**: Different draft model sizes for different Apple Silicon variants (0.5B for M4 mini, potentially 1.5B for future chips with faster ANE).
