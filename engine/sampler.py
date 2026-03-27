"""Token sampling strategies: greedy, top-k, top-p, temperature."""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / e.sum()


def greedy(logits: np.ndarray) -> int:
    """Return the token with the highest logit."""
    return int(np.argmax(logits))


def sample(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    rng: np.random.Generator = None,
) -> int:
    """
    Sample a token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: (vocab_size,) raw logits.
        temperature: 0 = greedy; higher = more random.
        top_k: Keep only top-k tokens (0 = disabled).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        rng: Numpy random generator (for reproducibility).

    Returns:
        Sampled token ID.
    """
    if temperature == 0 or top_k == 1:
        return greedy(logits)

    rng = rng or np.random.default_rng()
    logits = np.array(logits, dtype=np.float64) / temperature

    # --- top-k filtering ---
    if top_k > 0 and top_k < len(logits):
        top_idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_idx] = logits[top_idx]
        logits = mask

    # --- top-p (nucleus) filtering ---
    if top_p < 1.0:
        sorted_idx = np.argsort(logits)[::-1]
        sorted_probs = softmax(logits[sorted_idx])
        cumsum = np.cumsum(sorted_probs)
        cutoff = int(np.searchsorted(cumsum, top_p)) + 1
        keep = sorted_idx[:cutoff]
        mask = np.full_like(logits, -np.inf)
        mask[keep] = logits[keep]
        logits = mask

    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))
