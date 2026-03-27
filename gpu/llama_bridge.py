"""
GPU backend using llama.cpp via llama-cpp-python.

Install
-------
pip install llama-cpp-python                              # CPU only
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python  # Metal GPU
"""

import numpy as np
from gpu.backend import GPUBackend, GPUBackendConfig


class LlamaCppBackend(GPUBackend):
    """
    Wraps llama-cpp-python for Metal GPU inference with KV cache control.

    KV truncation uses the fast ``n_tokens`` setter on the underlying Llama
    object.  In causal models the stale KV entries beyond the truncation
    point are never read (attention is left-to-right), so simply rewinding
    the position counter is safe and essentially free (~0.002 ms).
    """

    def __init__(self):
        self._llm = None
        self._kv_pos = 0
        self._vocab_size_val = 0
        self._n_ctx_val = 0

    # -- loading -------------------------------------------------------------

    def load(self, config: GPUBackendConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required.\n"
                "  pip install llama-cpp-python\n"
                "For Metal GPU:\n"
                "  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python"
            )

        self._llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_batch=config.n_batch,
            logits_all=True,
            verbose=config.verbose,
        )
        self._vocab_size_val = self._llm.n_vocab()
        self._n_ctx_val = config.n_ctx
        self._kv_pos = 0

    # -- eval ----------------------------------------------------------------

    def eval(self, tokens: list[int]) -> np.ndarray:
        if self._llm is None:
            raise RuntimeError("Model not loaded — call load() first.")
        if not tokens:
            return np.empty((0, self._vocab_size_val), dtype=np.float32)

        self._llm.eval(tokens)

        n = len(tokens)
        logits = np.zeros((n, self._vocab_size_val), dtype=np.float32)
        for i in range(n):
            logits[i] = np.array(
                self._llm.scores[self._kv_pos + i], dtype=np.float32,
            )

        self._kv_pos += n
        return logits

    # -- KV cache ------------------------------------------------------------

    @property
    def kv_pos(self) -> int:
        return self._kv_pos

    def kv_truncate(self, pos: int) -> None:
        if pos >= self._kv_pos or self._llm is None:
            return

        # Fast truncate: just rewind the position counter.
        # Stale KV entries beyond `pos` are never read by causal attention.
        self._llm.n_tokens = pos
        self._kv_pos = pos

    def reset(self) -> None:
        if self._llm is not None:
            self._llm.reset()
        self._kv_pos = 0

    # -- metadata ------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._vocab_size_val

    @property
    def n_ctx(self) -> int:
        return self._n_ctx_val

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._llm is not None:
            self._llm.close()
        self._llm = None
        self._kv_pos = 0
