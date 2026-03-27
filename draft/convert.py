#!/usr/bin/env python3
"""Convert draft model from PyTorch to CoreML (.mlpackage) for ANE execution."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def convert(
    model_path: str = None,
    output_path: str = "draft_model.mlpackage",
    seq_len: int = 512,
    verify: bool = True,
):
    """
    Trace a DraftModel and convert to CoreML format.

    Args:
        model_path: HuggingFace model directory (None → random weights).
        output_path: Where to save the .mlpackage.
        seq_len: Fixed sequence length compiled into the model.
        verify: Compare PyTorch vs CoreML outputs after conversion.
    """
    import coremltools as ct
    from draft.model import DraftModel, DraftModelConfig, QWEN2_5_0_5B_CONFIG

    # --- load / create model ------------------------------------------------
    if model_path:
        print(f"Loading model from {model_path} ...")
        model = DraftModel.from_pretrained(model_path, max_seq_len=seq_len)
    else:
        print("No model path — using random weights (Qwen2.5-0.5B config).")
        config = DraftModelConfig(
            **{k: v for k, v in QWEN2_5_0_5B_CONFIG.__dict__.items()}
        )
        config.max_seq_len = seq_len
        model = DraftModel(config)

    model.eval()
    print(f"Parameters: {model.count_parameters():,}")

    # --- trace --------------------------------------------------------------
    example = torch.randint(0, model.config.vocab_size, (1, seq_len))

    print("Tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    # --- convert ------------------------------------------------------------
    print("Converting to CoreML ...")
    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    print(f"Saving → {output_path}")
    ml_model.save(output_path)

    # --- verify -------------------------------------------------------------
    if verify:
        print("Verifying (PyTorch vs CoreML) ...")
        with torch.no_grad():
            pt_out = model(example).numpy()

        cml_out = ml_model.predict(
            {"input_ids": example.numpy().astype(np.int32)}
        )["logits"]

        max_diff = float(np.max(np.abs(pt_out - cml_out)))
        print(f"  Max abs diff: {max_diff:.6f}")
        if max_diff < 0.5:
            print("  OK — within fp16 tolerance.")
        else:
            print("  WARNING — large divergence detected.")

    print("Done.")
    return ml_model


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert draft model → CoreML")
    parser.add_argument("--model-path", type=str, default=None,
                        help="HuggingFace model directory")
    parser.add_argument("--output", type=str, default="draft_model.mlpackage")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    convert(
        model_path=args.model_path,
        output_path=args.output,
        seq_len=args.seq_len,
        verify=not args.no_verify,
    )
