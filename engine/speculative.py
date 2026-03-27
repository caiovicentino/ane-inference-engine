"""
Pure speculative decoding verification logic.

No model dependencies — operates on logits arrays only.
"""

from typing import Callable

import numpy as np

from engine.sampler import greedy


def verify_candidates(
    candidates: list[int],
    verify_logits: np.ndarray,
    bonus_logits: np.ndarray,
    sample_fn: Callable[[np.ndarray], int] = greedy,
) -> tuple[list[int], int]:
    """
    Compare draft candidates against the main model's logits.

    Args:
        candidates:    N draft token IDs.
        verify_logits: (N, vocab_size) — main-model logits that predict each
                       candidate position.  ``verify_logits[i]`` is the
                       distribution the main model assigns to position i.
        bonus_logits:  (vocab_size,) — logits for the position *after* the
                       last candidate (used when all candidates are accepted).
        sample_fn:     logits → token_id  (default: greedy argmax).

    Returns:
        (accepted, bonus):
            accepted — prefix of *candidates* that match the main model.
            bonus    — the main model's own pick at the first rejection
                       point, or sampled from *bonus_logits* if all match.
    """
    for i, cand in enumerate(candidates):
        main_pick = sample_fn(verify_logits[i])
        if cand != main_pick:
            # First mismatch → return accepted prefix + main's choice
            return list(candidates[:i]), main_pick

    # All candidates accepted → bonus from the extra position
    return list(candidates), sample_fn(bonus_logits)


def acceptance_rate(n_accepted: int, n_candidates: int) -> float:
    """Fraction of candidates accepted (0.0–1.0)."""
    if n_candidates == 0:
        return 0.0
    return n_accepted / n_candidates
