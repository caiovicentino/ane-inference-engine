#!/usr/bin/env python3
"""
Benchmark: Qwen2.5-32B with speculative decoding on ANE + Metal GPU.

Compares:
- Baseline: Single model (32B Q4 on GPU)
- Speculative: 0.5B draft on ANE + 32B verifier on GPU

Requirements:
- models/draft_0.5b_seq64.mlpackage (CoreML draft model)
- models/Qwen2.5-32B-GGUF/qwen2.5-32b-instruct-q4_k_m-*.gguf (split GGUF)
"""

import argparse
import time
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.pipeline import make_coreml_draft_fn
from engine.pipelined import PipelinedCoordinator
from engine.coordinator import CoordinatorConfig
from engine.sampler import greedy
from gpu.llama_bridge import LlamaCppBackend
from gpu.backend import GPUBackendConfig


@dataclass
class BenchmarkResult:
    mode: str
    total_tokens: int
    wall_time_s: float
    tok_per_sec: float
    cycles: int = 0
    acceptance_rate: float = 0.0
    prefetch_hit_rate: float = 0.0


def find_gguf_files(model_dir: str) -> str:
    """Find the first part of the split GGUF files."""
    model_path = Path(model_dir)
    # Look for split GGUF pattern
    parts = sorted(model_path.glob("*-00001-of-*.gguf"))
    if parts:
        return str(parts[0])
    # Look for single GGUF
    ggufs = sorted(model_path.glob("*.gguf"))
    if ggufs:
        return str(ggufs[0])
    raise FileNotFoundError(f"No GGUF files found in {model_dir}")


def baseline_generate(
    backend: LlamaCppBackend,
    prompt: list[int],
    max_tokens: int,
) -> list[int]:
    """
    Single-model autoregressive generation (no speculation).
    This is the baseline: GPU-only, one token at a time.
    """
    backend.reset()
    logits = backend.eval(prompt)
    output = []

    for _ in range(max_tokens):
        # Sample from last token's logits
        last_logits = logits[-1] if len(logits.shape) > 1 else logits
        next_token = int(np.argmax(last_logits))
        output.append(next_token)

        # Generate next token
        logits = backend.eval([next_token])

    return output


def speculative_generate(
    draft_fn,
    backend: LlamaCppBackend,
    prompt: list[int],
    max_tokens: int,
    n_candidates: int = 1,
) -> tuple[list[int], dict]:
    """
    Speculative decoding: ANE draft + GPU verify.
    """
    backend.reset()
    config = CoordinatorConfig(n_candidates=n_candidates, max_seq_len=4096)
    coord = PipelinedCoordinator(draft_fn, backend, greedy, config)
    tokens = coord.generate(prompt, max_tokens=max_tokens)
    return tokens, coord.stats


def benchmark_baseline(
    model_path: str,
    prompt: list[int],
    max_tokens: int,
    n_runs: int,
    n_ctx: int = 4096,
) -> BenchmarkResult:
    """Benchmark baseline (GPU-only) generation."""
    print(f"\n[Baseline] Loading 32B model from {model_path}...")

    config = GPUBackendConfig(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,  # All layers on GPU
        n_batch=512,
        verbose=False,
    )

    backend = LlamaCppBackend()
    backend.load(config)

    print(f"[Baseline] Warming up...")
    baseline_generate(backend, prompt, min(10, max_tokens))
    backend.reset()

    print(f"[Baseline] Running {n_runs} iterations...")
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        tokens = baseline_generate(backend, prompt, max_tokens)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  Run {i+1}: {len(tokens)} tokens in {dt:.2f}s ({len(tokens)/dt:.1f} tok/s)")
        backend.reset()

    backend.close()

    avg_time = np.mean(times)
    return BenchmarkResult(
        mode="Baseline (GPU-only)",
        total_tokens=max_tokens,
        wall_time_s=avg_time,
        tok_per_sec=max_tokens / avg_time,
    )


def benchmark_speculative(
    model_path: str,
    draft_path: str,
    prompt: list[int],
    max_tokens: int,
    n_candidates: int,
    n_runs: int,
    n_ctx: int = 4096,
    seq_len: int = 64,
) -> BenchmarkResult:
    """Benchmark speculative decoding (ANE + GPU)."""
    print(f"\n[Speculative] Loading 32B model from {model_path}...")

    config = GPUBackendConfig(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False,
    )

    backend = LlamaCppBackend()
    backend.load(config)

    print(f"[Speculative] Loading CoreML draft model from {draft_path}...")
    draft_fn = make_coreml_draft_fn(draft_path, max_seq_len=seq_len, pad_token_id=0)

    print(f"[Speculative] Warming up...")
    speculative_generate(draft_fn, backend, prompt, min(10, max_tokens), n_candidates)
    backend.reset()

    print(f"[Speculative] Running {n_runs} iterations with N={n_candidates}...")
    times = []
    token_counts = []
    last_stats = {}

    for i in range(n_runs):
        t0 = time.perf_counter()
        tokens, stats = speculative_generate(draft_fn, backend, prompt, max_tokens, n_candidates)
        dt = time.perf_counter() - t0
        times.append(dt)
        token_counts.append(len(tokens))
        last_stats = stats
        print(f"  Run {i+1}: {len(tokens)} tokens in {dt:.2f}s ({len(tokens)/dt:.1f} tok/s), "
              f"accept={stats.get('acceptance_rate', 0):.1%}")
        backend.reset()

    backend.close()

    avg_time = np.mean(times)
    avg_tokens = np.mean(token_counts)

    return BenchmarkResult(
        mode=f"Speculative (N={n_candidates})",
        total_tokens=int(avg_tokens),
        wall_time_s=avg_time,
        tok_per_sec=avg_tokens / avg_time,
        cycles=last_stats.get("cycles", 0),
        acceptance_rate=last_stats.get("acceptance_rate", 0),
        prefetch_hit_rate=last_stats.get("prefetch_hit_rate", 0),
    )


def print_results(baseline: BenchmarkResult, speculative: BenchmarkResult):
    """Print comparison results."""
    speedup = speculative.tok_per_sec / baseline.tok_per_sec

    print("\n" + "=" * 60)
    print(" Qwen2.5-32B: Baseline vs Speculative Decoding")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Speculative':>15}")
    print("-" * 60)
    print(f"{'Throughput (tok/s)':<25} {baseline.tok_per_sec:>15.2f} {speculative.tok_per_sec:>15.2f}")
    print(f"{'Wall time (s)':<25} {baseline.wall_time_s:>15.2f} {speculative.wall_time_s:>15.2f}")
    print(f"{'Cycles':<25} {'N/A':>15} {speculative.cycles:>15}")
    print(f"{'Acceptance rate':<25} {'N/A':>15} {speculative.acceptance_rate:>14.1%}")
    print(f"{'Prefetch hit rate':<25} {'N/A':>15} {speculative.prefetch_hit_rate:>14.1%}")
    print("-" * 60)
    print(f"\n{'SPEEDUP':>40}: {speedup:.2f}x")

    if speedup > 1:
        print(f"{'STATUS':>40}: FASTER than baseline!")
    else:
        print(f"{'STATUS':>40}: Slower (overhead exceeds gains)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark 32B model with speculative decoding")
    parser.add_argument("--model", type=str, default="models/Qwen2.5-32B-GGUF",
                       help="Path to GGUF model directory")
    parser.add_argument("--draft", type=str, default="models/draft_0.5b_seq64.mlpackage",
                       help="Path to CoreML draft model")
    parser.add_argument("--max-tokens", type=int, default=80,
                       help="Number of tokens to generate")
    parser.add_argument("--n-candidates", type=int, default=1,
                       help="Number of draft candidates per cycle")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of benchmark iterations")
    parser.add_argument("--n-ctx", type=int, default=4096,
                       help="Context window size")
    parser.add_argument("--seq-len", type=int, default=64,
                       help="Draft model sequence length")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline benchmark")
    args = parser.parse_args()

    # Find GGUF file
    gguf_path = find_gguf_files(args.model)
    print(f"Using GGUF: {gguf_path}")
    print(f"Using draft: {args.draft}")

    # Test prompt (same as in FINDINGS.md)
    prompt = list(range(1, 33))  # Simple token sequence

    print(f"\nBenchmark configuration:")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  N candidates: {args.n_candidates}")
    print(f"  Runs: {args.runs}")
    print(f"  Context: {args.n_ctx}")

    # Run benchmarks
    baseline_result = None
    if not args.skip_baseline:
        baseline_result = benchmark_baseline(
            gguf_path, prompt, args.max_tokens, args.runs, args.n_ctx
        )

    speculative_result = benchmark_speculative(
        gguf_path, args.draft, prompt, args.max_tokens,
        args.n_candidates, args.runs, args.n_ctx, args.seq_len
    )

    # Print results
    if baseline_result:
        print_results(baseline_result, speculative_result)
    else:
        print(f"\n[Speculative Only] {speculative_result.tok_per_sec:.2f} tok/s")


if __name__ == "__main__":
    main()
