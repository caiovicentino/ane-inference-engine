"""
High-level inference pipeline: text in → text out.

Ties together the draft model, main model, tokenizer, and coordinator.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from engine.coordinator import Coordinator, CoordinatorConfig
from engine.sampler import greedy
from gpu.backend import GPUBackend


@dataclass
class PipelineConfig:
    n_candidates: int = 4
    max_new_tokens: int = 256
    max_seq_len: int = 512       # draft model context window
    temperature: float = 0.0     # 0 = greedy
    top_k: int = 0
    top_p: float = 1.0
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Draft function factory
# ---------------------------------------------------------------------------

def make_draft_fn(
    model: torch.nn.Module,
    max_seq_len: int,
    pad_token_id: int = 0,
) -> Callable[[list[int], int], list[int]]:
    """
    Wrap a DraftModel into the ``draft_fn(context, n) → list[int]`` callable
    expected by the Coordinator.

    The draft model is stateless (no KV cache): each call builds a padded
    input, runs a full forward pass, and reads the logit at the last real
    position.
    """
    model.eval()

    def draft_fn(context: list[int], n_candidates: int) -> list[int]:
        current = list(context)
        candidates: list[int] = []
        for _ in range(n_candidates):
            # Truncate if context exceeds model window
            seq = current[-max_seq_len:]
            # Right-pad to fixed length
            padded = seq + [pad_token_id] * (max_seq_len - len(seq))
            ids = torch.tensor([padded], dtype=torch.long)
            with torch.no_grad():
                logits = model(ids)                     # (1, max_seq_len, V)
            pos = len(seq) - 1
            next_token = int(torch.argmax(logits[0, pos]).item())
            candidates.append(next_token)
            current.append(next_token)
        return candidates

    return draft_fn


def make_coreml_draft_fn(
    mlpackage_path: str,
    max_seq_len: int,
    pad_token_id: int = 0,
) -> Callable[[list[int], int], list[int]]:
    """
    Wrap a CoreML .mlpackage into the ``draft_fn(context, n) → list[int]``
    callable expected by the Coordinator.

    Runs on ANE via CoreML — significantly faster than PyTorch CPU.
    """
    import coremltools as ct

    model = ct.models.MLModel(
        mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE,
    )

    def draft_fn(context: list[int], n_candidates: int) -> list[int]:
        current = list(context)
        candidates: list[int] = []
        for _ in range(n_candidates):
            seq = current[-max_seq_len:]
            padded = seq + [pad_token_id] * (max_seq_len - len(seq))
            input_ids = np.array([padded], dtype=np.int32)
            out = model.predict({"input_ids": input_ids})
            logits = out["logits"]
            pos = len(seq) - 1
            next_token = int(np.argmax(logits[0, pos]))
            candidates.append(next_token)
            current.append(next_token)
        return candidates

    return draft_fn


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    End-to-end inference: encode prompt → speculative decode → decode text.

    Example
    -------
    >>> pipe = InferencePipeline(draft_model, gpu_backend, tokenizer, config)
    >>> print(pipe("Once upon a time"))
    """

    def __init__(
        self,
        draft_model: torch.nn.Module,
        main_backend: GPUBackend,
        tokenizer,
        config: PipelineConfig = None,
    ):
        self.config = config or PipelineConfig()
        self.tokenizer = tokenizer

        draft_fn = make_draft_fn(
            draft_model,
            max_seq_len=self.config.max_seq_len,
            pad_token_id=self.config.pad_token_id,
        )

        coord_config = CoordinatorConfig(
            n_candidates=self.config.n_candidates,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_seq_len=self.config.max_seq_len,
        )

        self.coordinator = Coordinator(
            draft_fn=draft_fn,
            main_backend=main_backend,
            sample_fn=greedy,         # Phase 4 will plug in temperature sampling
            config=coord_config,
        )

    def __call__(self, prompt: str, max_new_tokens: int = None) -> str:
        max_new = max_new_tokens or self.config.max_new_tokens
        prompt_ids = self.tokenizer.encode(prompt)
        output_ids = self.coordinator.generate(prompt_ids, max_tokens=max_new)
        return self.tokenizer.decode(output_ids)

    @property
    def stats(self) -> dict:
        return self.coordinator.stats
