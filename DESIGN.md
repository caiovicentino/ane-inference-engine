# ANE Inference Engine — Design Document

**A heterogeneous LLM inference engine for Apple Silicon that runs the draft model on ANE and the main model on GPU simultaneously.**

Target: **3.3x speedup** over single-GPU inference (27 → 85 tok/s on M4).

## The Problem

No existing inference engine exploits Apple Silicon's heterogeneous compute:

| Engine | Draft | Main | Parallelism | Result |
|--------|-------|------|-------------|--------|
| llama.cpp speculative | CPU or GPU | GPU | **Sequential** | Slower (0.4x) |
| MLX | — | GPU | Single engine | No speculative |
| CoreML | ANE | — | Single engine | No LLM support |
| **This project** | **ANE (CoreML)** | **GPU (Metal)** | **Parallel pipeline** | **3.3x target** |

## Proven Data Points

From our research (measured on M4 16GB):

```
ANE draft (46M, CoreML):   1,199 tok/s alone, 743 tok/s during GPU load
GPU main (4B Q4, Metal):   27.5 tok/s alone, 25.4 tok/s during ANE load
GPU degradation with ANE:  7.5% (independent bandwidth paths confirmed)
ANE DCS channels:          Separate from GFX DCS in Apple Memory Controller
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Coordinator (CPU Thread)                    │
│                                                                │
│  ┌─────────────┐    Token Queue    ┌──────────────────┐       │
│  │ ANE Pipeline │ ──────────────→  │  GPU Pipeline     │       │
│  │  (Draft)     │  [t1,t2,t3,t4]  │  (Verify+Accept)  │       │
│  │              │ ←────────────── │                    │       │
│  │ CoreML       │   Accept/Reject │  Metal Compute     │       │
│  │ 743 tok/s    │                 │  25 tok/s verify   │       │
│  └─────────────┘                  └──────────────────┘       │
│         │                                  │                   │
│         ▼                                  ▼                   │
│  ┌─────────────┐                  ┌──────────────────┐       │
│  │ ANE DCS     │                  │ GFX DCS          │       │
│  │ (bandwidth) │                  │ (bandwidth)      │       │
│  └──────┬──────┘                  └────────┬─────────┘       │
│         │        Independent paths         │                   │
│         └──────────────┬───────────────────┘                   │
│                        ▼                                       │
│                 ┌────────────┐                                 │
│                 │    DRAM    │                                 │
│                 │  120 GB/s  │                                 │
│                 └────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### Stage 1: Draft Generation (ANE)

The draft model runs continuously on the ANE via CoreML, generating candidate tokens.

**Input**: Last accepted token ID
**Output**: N candidate token IDs + logits
**Target**: 4 candidates in <5ms

```
While not stopped:
    candidates = draft_model.predict(last_token)  // CoreML → ANE
    candidate_queue.push(candidates)              // Non-blocking
```

### Stage 2: Batch Verification (GPU)

The main model verifies all candidates in one forward pass on the GPU via Metal.

**Input**: N candidate token IDs from the queue
**Output**: Verified logits for each position
**Target**: Verify 4 candidates in ~37ms

```
While not stopped:
    candidates = candidate_queue.pop()            // Block until ready
    logits = main_model.forward_batch(candidates) // Metal → GPU
    accepted = verify_and_accept(logits, candidates)
    update_kv_cache(accepted)
    notify_draft(last_accepted_token)
```

### Stage 3: Token Sampling & Coordination (CPU)

Manages the accept/reject logic, KV cache, and output stream.

```
For each verification result:
    n_accepted = count_matching_prefix(draft_logits, main_logits)
    output_tokens(accepted_tokens)
    rollback_kv_cache(n_candidates - n_accepted)
    signal_draft(new_starting_token)
```

## Components to Build

### Phase 1: Draft Model on ANE (Week 1-2)

**Goal**: Run a real LLM draft model on the ANE via CoreML.

1. **Model conversion**: Build a pure-PyTorch Qwen-compatible transformer that traces cleanly
   - Use the architecture from our working 46M PoC (1199 tok/s)
   - Scale to match Qwen2.5-0.5B or Qwen3.5-0.8B dimensions
   - Load real weights from safetensors/GGUF into the pure-PyTorch model
   - Convert via coremltools to .mlpackage

2. **Tokenizer**: Wrap the Qwen tokenizer for use outside HuggingFace
   - SentencePiece or tiktoken (Qwen uses a custom BPE)
   - Shared between draft and main model

3. **Benchmarks**: Measure real ANE inference speed with real weights
   - Target: >200 tok/s for 0.5-0.8B model
   - Measure: accuracy of predictions (acceptance rate proxy)

**Key file**: `draft/model.py`, `draft/convert.py`, `draft/tokenizer.py`

### Phase 2: GPU Main Model Backend (Week 2-3)

**Goal**: Load and run the main model on GPU with batch verification support.

Options (pick one):
- **A) Wrap llama.cpp as a library** — call `llama_decode()` from our coordinator
- **B) Use MLX** — native Python, good Metal support, but single-threaded
- **C) Build custom Metal backend** — maximum control, most work

Recommended: **Option A** (llama.cpp as library)
- llama.cpp already has the fastest Metal kernels
- `libllama.so` can be loaded via ctypes/cffi
- We control when `llama_decode()` is called (batch verification)
- KV cache management is already implemented

**Key file**: `gpu/backend.py`, `gpu/llama_bridge.py`

### Phase 3: Pipeline Coordinator (Week 3-4)

**Goal**: Run draft and main in parallel, manage the speculative loop.

1. **Thread pool**:
   - ANE thread: runs CoreML predictions in a loop
   - GPU thread: runs llama.cpp verification batches
   - Main thread: coordinates, samples, outputs

2. **Queues**:
   - `draft_queue`: ANE → GPU (candidate tokens)
   - `accept_queue`: GPU → ANE (last accepted token)
   - `output_queue`: GPU → User (final tokens)

3. **Speculative loop**:
   ```
   ANE generates N candidates
        ↓ (non-blocking push)
   GPU receives candidates
        ↓
   GPU runs forward pass on all N+1 positions (batch)
        ↓
   Compare draft logits vs main logits
        ↓
   Accept longest matching prefix
        ↓
   Output accepted tokens
        ↓
   Signal ANE with new starting token
        ↓
   Repeat
   ```

4. **KV cache sync**:
   - Main model: extend KV cache for accepted tokens, rollback rejected
   - Draft model: reset to last accepted position
   - This is the hardest part — draft (CoreML) and main (llama.cpp) have different KV cache formats

**Key file**: `engine/coordinator.py`, `engine/speculative.py`, `engine/kv_cache.py`

### Phase 4: Optimization (Week 4-5)

1. **Tune draft count**: Find optimal N (candidates per cycle)
   - More candidates = higher throughput if acceptance is high
   - More candidates = more wasted compute if acceptance is low
   - Adaptive: start with N=4, increase if acceptance >80%, decrease if <40%

2. **Async pipeline**: Overlap draft generation with verification
   - While GPU verifies batch K, ANE generates batch K+1
   - Double-buffering for candidates

3. **KV cache optimization**:
   - Draft model without KV cache (single-token mode, always recompute)
   - Main model with full KV cache (only extends/rollbacks)

4. **Memory management**:
   - Draft model: ~500MB (CoreML package)
   - Main model: ~2.5GB (GGUF via llama.cpp)
   - KV cache: shared DRAM, independent access paths
   - Total: ~3.5GB of 16GB — fits comfortably

## File Structure

```
ane-inference-engine/
├── README.md
├── DESIGN.md                  # This document
├── requirements.txt
├── Makefile
│
├── draft/                     # ANE draft model
│   ├── model.py               # Pure PyTorch transformer (traceable)
│   ├── convert.py             # PyTorch → CoreML conversion
│   ├── tokenizer.py           # Shared tokenizer
│   └── benchmark.py           # ANE speed measurement
│
├── gpu/                       # GPU main model
│   ├── backend.py             # Abstract backend interface
│   ├── llama_bridge.py        # llama.cpp C library bridge
│   └── benchmark.py           # GPU speed measurement
│
├── engine/                    # Pipeline coordinator
│   ├── coordinator.py         # Main loop, thread management
│   ├── speculative.py         # Speculative decoding logic
│   ├── kv_cache.py            # KV cache management
│   ├── sampler.py             # Token sampling
│   └── pipeline.py            # Pipeline stages
│
├── benchmarks/                # Performance testing
│   ├── bandwidth.py           # IOReport bandwidth measurement
│   ├── end_to_end.py          # Full pipeline benchmark
│   └── compare.py             # Compare with llama.cpp baseline
│
├── tests/
│   ├── test_draft.py
│   ├── test_gpu.py
│   ├── test_pipeline.py
│   └── test_speculative.py
│
└── tools/
    ├── download_model.py      # Download and prepare models
    └── profile.py             # IOReport profiling during inference
```

## Success Criteria

| Metric | Target | How to measure |
|--------|--------|---------------|
| End-to-end tok/s | >50 tok/s | `benchmarks/end_to_end.py` |
| Speedup vs baseline | >1.8x | Compare with `llama-bench` baseline |
| Draft acceptance rate | >50% | Count accepted/total during generation |
| GPU degradation | <10% | IOReport bandwidth during pipeline |
| Memory usage | <6 GB total | `vm_stat` during inference |
| First token latency | <100ms | Time from prompt to first output token |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| CoreML conversion fails for real model | Blocks Phase 1 | Build pure-PyTorch model, load weights manually |
| Acceptance rate too low (<30%) | Reduces speedup to <1.5x | Use larger draft (0.8B), tune temperature |
| KV cache sync overhead | Eats the speedup | Stateless draft (no KV cache, recompute each time) |
| llama.cpp API changes | Breaks GPU backend | Pin to specific version, abstract behind interface |
| ANE scheduling latency | Adds per-token overhead | Batch multiple tokens, amortize scheduling |

## Non-Goals (for v1)

- Training the draft model (use existing Qwen3.5-0.8B)
- Supporting models >8B (memory limited on 16GB)
- Multi-user serving
- Streaming to network clients
- Supporting non-Apple hardware

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Draft model conversion | Qwen-compatible model on ANE via CoreML |
| 2 | Draft benchmarking | Measured tok/s, accuracy assessment |
| 3 | GPU backend | llama.cpp bridge with batch verification |
| 4 | Pipeline v1 | End-to-end speculative decoding working |
| 5 | Optimization | Tuning, profiling, reaching >50 tok/s |
| 6 | Documentation | Paper, benchmarks, release |

## References

1. Our speculative decoding research: `RESEARCH.md`
2. Our IOReport bandwidth measurements: independent ANE DCS paths
3. Leviathan et al. 2023: "Fast Inference from Transformers via Speculative Decoding"
4. maderix/ANE: ANE reverse engineering and direct access
5. llama.cpp Metal backend: optimized GPU kernels for quantized inference
