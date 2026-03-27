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

## Per-Prompt Results

| Prompt | Baseline | Pipelined | Speedup | Acceptance | Prefetch |
|--------|----------|-----------|---------|------------|----------|
| Prime function | 10.7 | 12.3 | 1.16x | 72% | 70% |
| Recursion concept | 10.8 | 11.8 | 1.09x | 76% | 74% |
| Meaning of life | 10.4 | 12.1 | 1.16x | 76% | 74% |
| Binary search | 10.3 | 11.7 | 1.14x | 74% | 72% |
| List vs tuple | 10.5 | 12.2 | 1.16x | 80% | 78% |
| **Average** | **10.5** | **12.0** | **1.14x** | **76%** | **74%** |

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

## Hardware Details

All measurements on:
- MacBook Pro M4 (10-core CPU, 10-core GPU, 16-core ANE)
- 16GB unified memory
- macOS 15.x
- CoreML (coremltools 8.x) for ANE
- llama.cpp (llama-cpp-python) for Metal GPU

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

## Future Directions

1. **Direct ANE access**: Bypass CoreML to use KV cache on ANE. This alone could 10x the draft speed.
2. **Multi-token prediction**: Train draft to predict 2-4 tokens per forward pass (like PARD, but fine-tuned for the target model).
3. **Larger main models**: 32B+ on M4 Max/Ultra where baseline is slower.
4. **Dynamic sequence length**: Use shorter context for easy tokens, longer for harder ones.
5. **Apple Intelligence integration**: If Apple exposes ANE scheduling APIs, more efficient overlap.
