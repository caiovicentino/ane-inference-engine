#!/usr/bin/env python3
"""
Compare speculative decoding vs single-model baseline.

With mock backends this measures the algorithmic overhead.
With real models (GPU + ANE) it measures the actual speedup.
"""

import argparse
import time

import numpy as np

from engine.coordinator import Coordinator, CoordinatorConfig
from engine.sampler import greedy
from gpu.mock_backend import MockGPUBackend


def baseline_generate(backend: MockGPUBackend, prompt: list[int], max_tokens: int):
    """
    Single-model autoregressive generation (no speculation).

    Each cycle evaluates exactly 1 token — the simplest baseline.
    """
    backend.reset()
    backend.eval(prompt)
    output = []
    for _ in range(max_tokens):
        logits = backend.eval([0])  # placeholder token; mock ignores value semantics
        tok = int(np.argmax(logits[0]))
        output.append(tok)
    return output


def speculative_generate(
    draft_fn, backend, prompt, max_tokens, n_candidates=4,
):
    """Speculative decoding generation."""
    backend.reset()
    config = CoordinatorConfig(n_candidates=n_candidates)
    coord = Coordinator(draft_fn, backend, greedy, config)
    return coord.generate(prompt, max_tokens), coord.stats


def compare(
    vocab_size: int = 1000,
    prompt_len: int = 32,
    max_tokens: int = 200,
    n_candidates: int = 4,
    n_runs: int = 5,
):
    """Run both approaches and compare."""
    backend = MockGPUBackend(vocab_size=vocab_size, n_ctx=4096)
    backend.load()
    prompt = list(range(1, prompt_len + 1))

    def draft_fn(ctx, n):
        rng = np.random.RandomState(len(ctx))
        return [int(rng.randint(0, vocab_size)) for _ in range(n)]

    # --- Baseline ---
    base_times = []
    for _ in range(n_runs):
        backend.reset()
        t0 = time.perf_counter()
        baseline_generate(backend, prompt, max_tokens)
        base_times.append(time.perf_counter() - t0)
    base_tps = max_tokens / np.mean(base_times)

    # --- Speculative ---
    spec_times, spec_tokens = [], []
    last_stats = {}
    for _ in range(n_runs):
        backend.reset()
        t0 = time.perf_counter()
        toks, stats = speculative_generate(
            draft_fn, backend, prompt, max_tokens, n_candidates,
        )
        spec_times.append(time.perf_counter() - t0)
        spec_tokens.append(len(toks))
        last_stats = stats
    spec_tps = np.mean(spec_tokens) / np.mean(spec_times)

    # --- Report ---
    speedup = spec_tps / base_tps
    print(f"\n{'='*55}")
    print(f" Baseline vs Speculative Decoding  (mock backend)")
    print(f"{'='*55}")
    print(f"  Baseline       : {base_tps:8.1f} tok/s  ({max_tokens} tokens)")
    print(f"  Speculative    : {spec_tps:8.1f} tok/s  (N={n_candidates})")
    print(f"  Speedup        : {speedup:.2f}x")
    print(f"  Acceptance rate: {last_stats.get('acceptance_rate', 0):.1%}")
    print(f"  Tokens/cycle   : {np.mean(spec_tokens)/max(1,last_stats.get('cycles',1)):.2f}")
    print(f"  Target         : >1.8x  (design goal)")
    print(f"{'='*55}")

    return {"baseline_tps": base_tps, "speculative_tps": spec_tps, "speedup": speedup}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare baseline vs speculative")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()
    compare(max_tokens=args.max_tokens, n_candidates=args.n_candidates, n_runs=args.runs)
