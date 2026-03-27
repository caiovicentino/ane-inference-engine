#!/usr/bin/env python3
"""
Benchmark: Qwen2.5-72B with speculative decoding using REAL prompts.

Uses actual text prompts instead of arbitrary token IDs for better acceptance rates.
"""

import argparse
import time
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.pipeline import make_coreml_draft_fn
from engine.pipelined import PipelinedCoordinator
from engine.coordinator import CoordinatorConfig
from engine.sampler import greedy
from gpu.llama_bridge import LlamaCppBackend
from gpu.backend import GPUBackendConfig
from draft.tokenizer import QwenTokenizer


# Test prompts from FINDINGS.md (real-world scenarios)
TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. This is a simple test of",
    "In machine learning, the process of training a neural network involves",
    "Python is a programming language that is known for its",
    "The capital of France is Paris, and it is famous for",
    "Once upon a time in a land far away, there lived a",
]


def find_gguf_files(model_dir: str) -> str:
    model_path = Path(model_dir)
    parts = sorted(model_path.glob("*-00001-of-*.gguf"))
    if parts:
        return str(parts[0])
    ggufs = sorted(model_path.glob("*.gguf"))
    if ggufs:
        return str(ggufs[0])
    raise FileNotFoundError(f"No GGUF files found in {model_dir}")


def baseline_generate(backend, prompt_ids, max_tokens, tokenizer=None):
    """Single-model autoregressive generation."""
    backend.reset()
    logits = backend.eval(prompt_ids)
    output = []

    for _ in range(max_tokens):
        last_logits = logits[-1] if len(logits.shape) > 1 else logits
        next_token = int(np.argmax(last_logits))
        output.append(next_token)

        # Check EOS
        if tokenizer and next_token == tokenizer.eos_token_id:
            break

        logits = backend.eval([next_token])

    return output


def speculative_generate(draft_fn, backend, prompt_ids, max_tokens, n_candidates, eos_token_id=None):
    """Speculative decoding generation."""
    backend.reset()
    config = CoordinatorConfig(
        n_candidates=n_candidates,
        max_seq_len=2048,
        eos_token_id=eos_token_id,
    )
    coord = PipelinedCoordinator(draft_fn, backend, greedy, config)
    tokens = coord.generate(prompt_ids, max_tokens=max_tokens)
    return tokens, coord.stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark 72B with real prompts")
    parser.add_argument("--model", type=str, default="models/Qwen2.5-72B-GGUF")
    parser.add_argument("--draft", type=str, default="models/draft_0.5b_seq64.mlpackage")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--n-candidates", type=int, default=1)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    gguf_path = find_gguf_files(args.model)
    print(f"Using GGUF: {gguf_path}")
    print(f"Using draft: {args.draft}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = QwenTokenizer("Qwen/Qwen2.5-0.5B")

    # Configure backend
    gpu_config = GPUBackendConfig(
        model_path=gguf_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False,
    )

    # Load draft function
    print("Loading CoreML draft model...")
    draft_fn = make_coreml_draft_fn(args.draft, max_seq_len=args.seq_len, pad_token_id=0)

    print(f"\nRunning {len(TEST_PROMPTS)} prompts with max {args.max_tokens} tokens each...")
    print(f"N candidates: {args.n_candidates}")
    print("-" * 60)

    baseline_times = []
    speculative_times = []
    acceptance_rates = []

    for i, prompt in enumerate(TEST_PROMPTS):
        prompt_ids = tokenizer.encode(prompt)
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt[:50]}...\"")
        print(f"    Tokens: {len(prompt_ids)}")

        # Baseline
        backend = LlamaCppBackend()
        backend.load(gpu_config)

        t0 = time.perf_counter()
        base_output = baseline_generate(backend, prompt_ids, args.max_tokens, tokenizer)
        base_time = time.perf_counter() - t0
        baseline_times.append(base_time)
        base_tps = len(base_output) / base_time

        backend.close()

        # Speculative
        backend = LlamaCppBackend()
        backend.load(gpu_config)

        t0 = time.perf_counter()
        spec_output, stats = speculative_generate(
            draft_fn, backend, prompt_ids, args.max_tokens,
            args.n_candidates, tokenizer.eos_token_id
        )
        spec_time = time.perf_counter() - t0
        speculative_times.append(spec_time)
        spec_tps = len(spec_output) / spec_time
        acceptance = stats.get('acceptance_rate', 0)
        acceptance_rates.append(acceptance)

        backend.close()

        speedup = spec_tps / base_tps
        print(f"    Baseline: {len(base_output)} tok in {base_time:.2f}s ({base_tps:.1f} tok/s)")
        print(f"    Speculative: {len(spec_output)} tok in {spec_time:.2f}s ({spec_tps:.1f} tok/s)")
        print(f"    Acceptance: {acceptance:.1%} | Speedup: {speedup:.2f}x")

    # Summary
    avg_base_tps = sum(args.max_tokens / t for t in baseline_times) / len(baseline_times)
    avg_spec_tps = sum(args.max_tokens / t for t in speculative_times) / len(speculative_times)
    avg_acceptance = np.mean(acceptance_rates)
    overall_speedup = avg_spec_tps / avg_base_tps

    print("\n" + "=" * 60)
    print(" SUMMARY: Qwen2.5-72B Q2_K - Real Prompts")
    print("=" * 60)
    print(f" Avg Baseline:     {avg_base_tps:.2f} tok/s")
    print(f" Avg Speculative:  {avg_spec_tps:.2f} tok/s")
    print(f" Avg Acceptance:   {avg_acceptance:.1%}")
    print(f" Overall Speedup:  {overall_speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
