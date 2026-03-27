#!/usr/bin/env python3
"""
Knowledge distillation: train a small (~43M) draft model from Qwen2.5-0.5B.

Run on Google Colab (free T4 GPU):
    1. Upload this file + draft/model.py to Colab
    2. !pip install torch transformers datasets safetensors
    3. !python train_draft.py
    4. Download the saved weights → use locally with CoreML

Or run locally:
    python tools/train_draft.py --device mps --steps 5000
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Small draft model config (~43M params)
# ---------------------------------------------------------------------------

SMALL_DRAFT_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_seq_len": 128,
    "tie_word_embeddings": True,
}


def make_student(device="cpu"):
    """Create the small student draft model."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from draft.model import DraftModel, DraftModelConfig

    config = DraftModelConfig(**SMALL_DRAFT_CONFIG)
    model = DraftModel(config).to(device)
    print(f"Student: {model.count_parameters():,} params")
    return model


def make_teacher(device="cpu"):
    """Load Qwen2.5-0.5B as the teacher."""
    from transformers import AutoModelForCausalLM

    print("Loading teacher (Qwen2.5-0.5B)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        teacher = teacher.to(device)
    teacher.eval()
    n = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {n:,} params")
    return teacher


def make_dataloader(seq_len=128, batch_size=8, max_samples=50000):
    """Load a text dataset and tokenize it."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading dataset + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Tokenize and chunk into fixed-length sequences
    all_ids = []
    for text in ds["text"]:
        if text.strip():
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)

    # Chunk into seq_len blocks
    n_chunks = min(len(all_ids) // seq_len, max_samples)
    chunks = [
        torch.tensor(all_ids[i * seq_len : (i + 1) * seq_len], dtype=torch.long)
        for i in range(n_chunks)
    ]
    print(f"Dataset: {n_chunks} chunks of {seq_len} tokens")

    return DataLoader(chunks, batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combined KL-divergence (soft targets) + cross-entropy (hard targets).

    Args:
        student_logits: (B, L, V)
        teacher_logits: (B, L, V)
        labels: (B, L) token IDs for next-token prediction
        temperature: softens distributions (higher = softer)
        alpha: weight for distillation vs CE loss (1.0 = pure distillation)
    """
    V = student_logits.size(-1)

    # Soft loss: KL divergence on temperature-scaled logits
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

    # Hard loss: standard cross-entropy on next-token prediction
    ce = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * kl + (1 - alpha) * ce


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    device: str = "cuda",
    steps: int = 5000,
    batch_size: int = 8,
    lr: float = 3e-4,
    temperature: float = 2.0,
    alpha: float = 0.7,
    log_every: int = 100,
    save_dir: str = "draft_small_weights",
    seq_len: int = 128,
):
    student = make_student(device)
    teacher = make_teacher(device)
    dataloader = make_dataloader(seq_len=seq_len, batch_size=batch_size)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup + cosine decay
    warmup_steps = min(500, steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    student.train()
    data_iter = iter(dataloader)
    total_loss = 0
    t0 = time.time()

    print(f"\nTraining: {steps} steps, batch={batch_size}, lr={lr}, temp={temperature}")
    print(f"Device: {device}")
    print("-" * 60)

    for step in range(1, steps + 1):
        # Get batch (cycle through dataset)
        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids = next(data_iter).to(device)

        # Labels: shifted input for next-token prediction
        labels = input_ids[:, 1:].contiguous()
        input_for_loss = input_ids[:, :-1].contiguous()

        # Pad student input to max_seq_len
        B, L = input_for_loss.shape
        if L < seq_len:
            pad = torch.zeros(B, seq_len - L, dtype=torch.long, device=device)
            student_input = torch.cat([input_for_loss, pad], dim=1)
        else:
            student_input = input_for_loss[:, :seq_len]
            labels = labels[:, :seq_len]
            L = seq_len

        # Student forward
        student_logits = student(student_input)[:, :L, :]

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(input_for_loss[:, :L])
            teacher_logits = teacher_out.logits

        # Align vocab sizes (student may have fewer tokens)
        min_v = min(student_logits.size(-1), teacher_logits.size(-1))
        s_logits = student_logits[:, :, :min_v]
        t_logits = teacher_logits[:, :, :min_v]

        loss = distillation_loss(s_logits, t_logits, labels, temperature, alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % log_every == 0:
            avg = total_loss / log_every
            elapsed = time.time() - t0
            tps = step * batch_size * L / elapsed
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d} | loss={avg:.4f} | lr={cur_lr:.2e} | "
                  f"{tps:.0f} tok/s | {elapsed:.0f}s")
            total_loss = 0

    # Save weights
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "draft_small.safetensors")
    from safetensors.torch import save_file
    state = {k: v.cpu() for k, v in student.state_dict().items()
             if not k.startswith(("rope_", "causal_"))}
    save_file(state, save_path)

    # Save config
    import json
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(SMALL_DRAFT_CONFIG, f, indent=2)

    print(f"\nSaved to {save_dir}/")
    print(f"  draft_small.safetensors ({os.path.getsize(save_path)/1e6:.1f} MB)")
    print(f"  config.json")
    return student


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train small draft model via distillation")
    p.add_argument("--device", default="cuda", help="cuda, mps, or cpu")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--save-dir", default="draft_small_weights")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    train(
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        save_dir=args.save_dir,
    )
