#!/usr/bin/env python3
"""
End-to-end benchmark for the speculative decoding pipeline.

Measures tok/s, acceptance rate, latency, and per-cycle timing.
Works with mock backends (no GPU required) and real models.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np

from engine.coordinator import Coordinator, CoordinatorConfig
from engine.threaded import ThreadedCoordinator
from engine.adaptive import AdaptiveConfig
from engine.sampler import greedy
from gpu.mock_backend import MockGPUBackend


@dataclass
class BenchResult:
    total_tokens: int
    wall_time_s: float
    tok_per_sec: float
    acceptance_rate: float
    cycles: int
    tokens_per_cycle: float
    adaptive_n: int
    prefetch_hit_rate: float


def _mock_draft_fn(vocab_size: int, seed: int = 0):
    """Deterministic draft function for benchmarking."""
    def fn(context, n):
        rng = np.random.RandomState(len(context) + seed)
        return [int(rng.randint(0, vocab_size)) for _ in range(n)]
    return fn


def benchmark_sync(
    vocab_size: int = 1000,
    n_ctx: int = 4096,
    prompt_len: int = 32,
    max_tokens: int = 200,
    n_candidates: int = 4,
    n_runs: int = 5,
) -> BenchResult:
    """Benchmark the synchronous coordinator with mock backend."""
    backend = MockGPUBackend(vocab_size=vocab_size, n_ctx=n_ctx)
    backend.load()
    draft_fn = _mock_draft_fn(vocab_size)
    config = CoordinatorConfig(n_candidates=n_candidates)
    prompt = list(range(1, prompt_len + 1))

    times, tok_counts = [], []
    last_stats = {}

    for _ in range(n_runs):
        coord = Coordinator(draft_fn, backend, greedy, config)
        t0 = time.perf_counter()
        tokens = coord.generate(prompt, max_tokens=max_tokens)
        dt = time.perf_counter() - t0
        times.append(dt)
        tok_counts.append(len(tokens))
        last_stats = coord.stats
        backend.reset()

    avg_time = np.mean(times)
    avg_tok = np.mean(tok_counts)

    return BenchResult(
        total_tokens=int(avg_tok),
        wall_time_s=avg_time,
        tok_per_sec=avg_tok / avg_time,
        acceptance_rate=last_stats.get("acceptance_rate", 0),
        cycles=last_stats.get("cycles", 0),
        tokens_per_cycle=avg_tok / max(1, last_stats.get("cycles", 1)),
        adaptive_n=n_candidates,
        prefetch_hit_rate=0.0,
    )


def benchmark_threaded(
    vocab_size: int = 1000,
    n_ctx: int = 4096,
    prompt_len: int = 32,
    max_tokens: int = 200,
    n_candidates: int = 4,
    adaptive: bool = True,
    n_runs: int = 5,
) -> BenchResult:
    """Benchmark the threaded coordinator with adaptive draft count."""
    backend = MockGPUBackend(vocab_size=vocab_size, n_ctx=n_ctx)
    backend.load()
    draft_fn = _mock_draft_fn(vocab_size)
    config = CoordinatorConfig(n_candidates=n_candidates)
    adapt_cfg = AdaptiveConfig(initial_n=n_candidates) if adaptive else None
    prompt = list(range(1, prompt_len + 1))

    times, tok_counts = [], []
    last_stats = {}

    for _ in range(n_runs):
        tc = ThreadedCoordinator(draft_fn, backend, greedy, config, adapt_cfg)
        tc.start()
        t0 = time.perf_counter()
        tokens = tc.generate(prompt, max_tokens=max_tokens)
        dt = time.perf_counter() - t0
        tc.stop()
        times.append(dt)
        tok_counts.append(len(tokens))
        last_stats = tc.stats
        backend.reset()

    avg_time = np.mean(times)
    avg_tok = np.mean(tok_counts)

    return BenchResult(
        total_tokens=int(avg_tok),
        wall_time_s=avg_time,
        tok_per_sec=avg_tok / avg_time,
        acceptance_rate=last_stats.get("acceptance_rate", 0),
        cycles=last_stats.get("cycles", 0),
        tokens_per_cycle=avg_tok / max(1, last_stats.get("cycles", 1)),
        adaptive_n=last_stats.get("adaptive_n", n_candidates),
        prefetch_hit_rate=last_stats.get("prefetch_hit_rate", 0),
    )


def _print_result(label: str, r: BenchResult):
    print(f"\n{'='*50}")
    print(f" {label}")
    print(f"{'='*50}")
    print(f"  Tokens generated : {r.total_tokens}")
    print(f"  Wall time        : {r.wall_time_s*1000:.1f} ms")
    print(f"  Throughput       : {r.tok_per_sec:.1f} tok/s")
    print(f"  Acceptance rate  : {r.acceptance_rate:.1%}")
    print(f"  Cycles           : {r.cycles}")
    print(f"  Tokens/cycle     : {r.tokens_per_cycle:.2f}")
    print(f"  Adaptive N       : {r.adaptive_n}")
    print(f"  Prefetch hit rate: {r.prefetch_hit_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end speculative benchmark")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=1000)
    args = parser.parse_args()

    print("Benchmarking speculative decoding pipeline (mock backends)")
    print(f"  max_tokens={args.max_tokens}, n_candidates={args.n_candidates}, "
          f"runs={args.runs}")

    r_sync = benchmark_sync(
        vocab_size=args.vocab_size, max_tokens=args.max_tokens,
        n_candidates=args.n_candidates, n_runs=args.runs,
    )
    _print_result("Synchronous Coordinator", r_sync)

    r_threaded = benchmark_threaded(
        vocab_size=args.vocab_size, max_tokens=args.max_tokens,
        n_candidates=args.n_candidates, n_runs=args.runs,
    )
    _print_result("Threaded Coordinator (adaptive)", r_threaded)

    # Comparison
    print(f"\n{'='*50}")
    print(f" Threaded vs Sync speedup: {r_threaded.tok_per_sec/r_sync.tok_per_sec:.2f}x")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
