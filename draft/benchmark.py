#!/usr/bin/env python3
"""Benchmark draft model — CoreML (ANE) and PyTorch (CPU) modes."""

import argparse
import time

import numpy as np


def benchmark_coreml(
    model_path: str = "draft_model.mlpackage",
    seq_len: int = 512,
    n_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """Benchmark a CoreML model and report tok/s."""
    import coremltools as ct

    print(f"Loading CoreML model: {model_path}")
    model = ct.models.MLModel(model_path)

    input_ids = np.zeros((1, seq_len), dtype=np.int32)

    print(f"Warmup ({warmup} runs) ...")
    for _ in range(warmup):
        model.predict({"input_ids": input_ids})

    print(f"Benchmarking ({n_runs} runs) ...")
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict({"input_ids": input_ids})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    ms_mean = times.mean() * 1000
    ms_std = times.std() * 1000
    ms_p50 = float(np.percentile(times, 50)) * 1000
    ms_p99 = float(np.percentile(times, 99)) * 1000
    tok_s = 1.0 / times.mean()

    print(f"\n=== CoreML results (seq_len={seq_len}) ===")
    print(f"  Mean : {ms_mean:7.2f} ms  (± {ms_std:.2f})")
    print(f"  P50  : {ms_p50:7.2f} ms")
    print(f"  P99  : {ms_p99:7.2f} ms")
    print(f"  Tok/s: {tok_s:7.1f}  (target: >200)")

    return dict(mean_ms=ms_mean, std_ms=ms_std, p50_ms=ms_p50,
                p99_ms=ms_p99, tok_per_sec=tok_s)


def benchmark_pytorch(
    model_path: str = None,
    seq_len: int = 64,
    n_runs: int = 50,
    warmup: int = 5,
) -> dict:
    """Benchmark a PyTorch model on CPU for comparison."""
    import torch
    from draft.model import DraftModel, DraftModelConfig, TINY_TEST_CONFIG

    if model_path:
        model = DraftModel.from_pretrained(model_path, max_seq_len=seq_len)
    else:
        print("Using TINY_TEST_CONFIG (random weights) for quick PyTorch baseline.")
        config = DraftModelConfig(**TINY_TEST_CONFIG.__dict__)
        config.max_seq_len = seq_len
        model = DraftModel(config)

    model.eval()
    print(f"Parameters: {model.count_parameters():,}")
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))

    print(f"Warmup ({warmup}) ...")
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)

    print(f"Benchmarking ({n_runs}) ...")
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(input_ids)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    ms_mean = times.mean() * 1000
    tok_s = 1.0 / times.mean()

    print(f"\n=== PyTorch CPU results (seq_len={seq_len}) ===")
    print(f"  Mean : {ms_mean:7.2f} ms")
    print(f"  Tok/s: {tok_s:7.1f}")

    return dict(mean_ms=ms_mean, tok_per_sec=tok_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark draft model")
    parser.add_argument("--model-path", type=str, default="draft_model.mlpackage")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--pytorch", action="store_true",
                        help="Run PyTorch CPU benchmark instead")
    parser.add_argument("--pytorch-model", type=str, default=None,
                        help="HF model path for PyTorch mode")
    args = parser.parse_args()

    if args.pytorch:
        benchmark_pytorch(args.pytorch_model, args.seq_len, args.runs)
    else:
        benchmark_coreml(args.model_path, args.seq_len, args.runs)
