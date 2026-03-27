"""
Deterministic mock GPU backend for testing the speculative pipeline.

Generates reproducible logits from a seeded RNG so tests can predict
and verify outputs without loading a real model.
"""

import numpy as np
from gpu.backend import GPUBackend, GPUBackendConfig


class MockGPUBackend(GPUBackend):
    """
    Mock that satisfies the GPUBackend contract.

    Logits are deterministic: ``seed = token_id * 1000 + kv_position``.
    This means the *same* token at the *same* position always produces
    the *same* logit vector, making test assertions reliable.
    """

    def __init__(self, vocab_size: int = 1000, n_ctx: int = 2048):
        self._vocab_size_val = vocab_size
        self._n_ctx_val = n_ctx
        self._kv_pos = 0
        self._loaded = False
        self._eval_count = 0

    def load(self, config: GPUBackendConfig = None) -> None:
        if config:
            self._n_ctx_val = config.n_ctx
        self._loaded = True
        self._kv_pos = 0
        self._eval_count = 0

    def eval(self, tokens: list[int]) -> np.ndarray:
        if not tokens:
            return np.empty((0, self._vocab_size_val), dtype=np.float32)

        n = len(tokens)
        logits = np.zeros((n, self._vocab_size_val), dtype=np.float32)
        for i, tok in enumerate(tokens):
            seed = tok * 1000 + self._kv_pos + i
            rng = np.random.RandomState(seed)
            logits[i] = rng.randn(self._vocab_size_val).astype(np.float32)

        self._kv_pos += n
        self._eval_count += 1
        return logits

    @property
    def kv_pos(self) -> int:
        return self._kv_pos

    def kv_truncate(self, pos: int) -> None:
        self._kv_pos = min(pos, self._kv_pos)

    def reset(self) -> None:
        self._kv_pos = 0
        self._eval_count = 0

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val

    @property
    def n_ctx(self) -> int:
        return self._n_ctx_val

    def close(self) -> None:
        self._loaded = False
        self._kv_pos = 0
