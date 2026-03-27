#!/usr/bin/env python3
"""Benchmark GPU main model backend — prompt processing, generation, batch verify."""

import argparse
import time

import numpy as np


def benchmark_llama(
    model_path: str,
    n_ctx: int = 2048,
    prompt_len: int = 128,
    gen_tokens: int = 100,
    batch_sizes: tuple = (1, 4, 8),
) -> dict:
    """Benchmark the real llama.cpp backend."""
    from gpu.backend import GPUBackendConfig
    from gpu.llama_bridge import LlamaCppBackend

    config = GPUBackendConfig(
        model_path=model_path, n_ctx=n_ctx,
        n_gpu_layers=-1, verbose=False,
    )
    backend = LlamaCppBackend()
    backend.load(config)

    results = {}
    prompt = list(range(1, prompt_len + 1))

    # -- prompt processing ---------------------------------------------------
    t0 = time.perf_counter()
    backend.eval(prompt)
    dt = time.perf_counter() - t0
    pp_tps = prompt_len / dt
    results["prompt"] = {"ms": dt * 1000, "tok_per_sec": pp_tps}
    print(f"Prompt ({prompt_len} tok): {dt*1000:.1f} ms  ({pp_tps:.1f} tok/s)")

    # -- single-token generation ---------------------------------------------
    times = []
    for i in range(gen_tokens):
        t0 = time.perf_counter()
        backend.eval([prompt_len + i + 1])
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    gen_tps = 1.0 / t.mean()
    results["generation"] = {"mean_ms": t.mean() * 1000, "tok_per_sec": gen_tps}
    print(f"Generation: {t.mean()*1000:.2f} ms/tok  ({gen_tps:.1f} tok/s)")

    # -- batch verification --------------------------------------------------
    backend.reset()
    backend.eval(prompt)
    for bs in batch_sizes:
        batch = list(range(2000, 2000 + bs))
        t0 = time.perf_counter()
        backend.eval(batch)
        dt = time.perf_counter() - t0
        results[f"batch_{bs}"] = {"ms": dt * 1000}
        print(f"Batch verify ({bs} tok): {dt*1000:.2f} ms")
        backend.kv_truncate(prompt_len)

    backend.close()
    return results


def benchmark_mock(n_runs: int = 1000) -> dict:
    """Quick sanity benchmark of the mock backend."""
    from gpu.mock_backend import MockGPUBackend

    backend = MockGPUBackend(vocab_size=151936, n_ctx=2048)
    backend.load()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        backend.eval([42])
        times.append(time.perf_counter() - t0)
        backend.reset()

    ms = np.mean(times) * 1000
    print(f"Mock backend: {ms:.4f} ms/eval ({n_runs} runs)")
    backend.close()
    return {"mock_ms": ms}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark GPU backend")
    p.add_argument("--model", type=str, help="GGUF model path")
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-tokens", type=int, default=100)
    p.add_argument("--mock", action="store_true", help="Benchmark mock only")
    args = p.parse_args()

    if args.mock:
        benchmark_mock()
    elif args.model:
        benchmark_llama(args.model, args.n_ctx, args.prompt_len, args.gen_tokens)
    else:
        print("Usage: --model <path.gguf> or --mock")
