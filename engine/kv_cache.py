"""
KV cache position tracking for the main (verifier) model.

Wraps GPUBackend cache operations and collects statistics that
the coordinator uses for adaptive tuning.
"""

from gpu.backend import GPUBackend
import numpy as np


class KVCacheManager:
    """
    Tracks the accepted prefix length independently of the raw KV position.

    Invariant
    ---------
    After every coordinator cycle:
        ``backend.kv_pos == accepted_length``
    The manager enforces this via :meth:`rollback`.
    """

    def __init__(self, backend: GPUBackend):
        self.backend = backend
        self._accepted_len = 0
        self._stats = {
            "extends": 0,
            "rollbacks": 0,
            "tokens_rolled_back": 0,
            "tokens_accepted": 0,
        }

    # -- operations ----------------------------------------------------------

    def eval(self, tokens: list[int]) -> np.ndarray:
        """Evaluate tokens through the backend (extends raw KV cache)."""
        logits = self.backend.eval(tokens)
        self._stats["extends"] += 1
        return logits

    def accept(self, n_tokens: int) -> None:
        """Mark *n_tokens* as permanently accepted."""
        self._accepted_len += n_tokens
        self._stats["tokens_accepted"] += n_tokens

    def rollback(self) -> None:
        """Truncate raw KV cache back to the accepted prefix."""
        if self.backend.kv_pos > self._accepted_len:
            rolled = self.backend.kv_pos - self._accepted_len
            self.backend.kv_truncate(self._accepted_len)
            self._stats["rollbacks"] += 1
            self._stats["tokens_rolled_back"] += rolled

    def reset(self) -> None:
        """Full reset — wipe KV cache and counters."""
        self.backend.reset()
        self._accepted_len = 0

    # -- read-only -----------------------------------------------------------

    @property
    def accepted_length(self) -> int:
        return self._accepted_len

    @property
    def raw_kv_pos(self) -> int:
        return self.backend.kv_pos

    @property
    def stats(self) -> dict:
        return dict(self._stats)
