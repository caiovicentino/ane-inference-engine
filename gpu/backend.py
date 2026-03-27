"""Abstract interface for the GPU main model backend."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class GPUBackendConfig:
    """Configuration for loading the GPU verifier model."""
    model_path: str = ""
    n_ctx: int = 2048          # context window
    n_gpu_layers: int = -1     # -1 = offload all layers to GPU
    n_batch: int = 512         # max tokens per eval call
    verbose: bool = False


class GPUBackend(ABC):
    """
    Abstract GPU backend for the main (verifier) model.

    Contract
    --------
    * ``eval(tokens)`` processes tokens left-to-right, extends KV cache,
      and returns logits of shape ``(len(tokens), vocab_size)``.
    * ``kv_pos`` always equals the total number of tokens evaluated
      minus any that were truncated.
    * ``kv_truncate(pos)`` discards KV entries at positions ``≥ pos``.
    * ``reset()`` clears everything (equivalent to ``kv_truncate(0)``).
    """

    # -- core operations -----------------------------------------------------

    @abstractmethod
    def load(self, config: GPUBackendConfig) -> None:
        """Load model and allocate KV cache."""

    @abstractmethod
    def eval(self, tokens: list[int]) -> np.ndarray:
        """
        Run a forward pass on *tokens*, appending to the KV cache.

        Returns
        -------
        np.ndarray, shape ``(len(tokens), vocab_size)``, dtype float32.
        """

    # -- KV cache ------------------------------------------------------------

    @property
    @abstractmethod
    def kv_pos(self) -> int:
        """Number of tokens currently held in the KV cache."""

    @abstractmethod
    def kv_truncate(self, pos: int) -> None:
        """Remove KV entries from *pos* onward (keep ``[0, pos)``)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all state (KV cache, internal counters)."""

    # -- metadata ------------------------------------------------------------

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def n_ctx(self) -> int: ...

    # -- lifecycle -----------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
