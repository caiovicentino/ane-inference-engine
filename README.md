# ANE Inference Engine

A heterogeneous LLM inference engine for Apple Silicon that runs draft and main models on different compute engines simultaneously.

**Target: 3.3x speedup** (27 → 85 tok/s for 4B models on M4)

## Why

Every existing inference engine on Apple Silicon uses ONE compute engine at a time. But Apple Silicon has THREE: CPU, GPU, and ANE — each with independent memory bandwidth paths.

We proved that running ANE and GPU simultaneously causes only 7.5% GPU degradation. This means speculative decoding with the draft on ANE and the main on GPU can run in parallel — unlike llama.cpp's sequential approach.

## How

```
ANE (CoreML):  Draft model generates 4 candidate tokens    →  5ms
GPU (Metal):   Main model verifies all 4 in one batch pass → 37ms
Coordinator:   Accept matching tokens, output, repeat       →  3.8 tokens per 42ms cycle
```

See [DESIGN.md](DESIGN.md) for the full architecture.

## Status

Phase 1: Draft model conversion — **starting**

## Based On

- [ane-speculative-decoding](https://github.com/caiovicentino/ane-speculative-decoding): Proof that ANE+GPU have independent bandwidth
- [maderix/ANE](https://github.com/maderix/ANE): ANE reverse engineering reference
- Leviathan et al. 2023: Speculative decoding theory

## License

MIT
