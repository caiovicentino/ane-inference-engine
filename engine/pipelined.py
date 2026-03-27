"""
Pipelined speculative decoding coordinator.

Two key optimizations over the synchronous coordinator:

1. **Merged bonus eval**: Instead of a separate GPU call to eval the bonus
   token, we prepend it to the next cycle's candidate batch. This saves
   one full GPU forward pass per cycle (~100ms for 14B).

2. **ANE/GPU overlap**: While GPU verifies cycle K, ANE speculatively
   generates candidates for cycle K+1 (assuming all K candidates were
   accepted). If the assumption holds, we skip the ANE wait entirely.
   If not, we regenerate — same cost as synchronous.

On Apple Silicon, ANE and GPU are separate hardware units that run
truly in parallel (both CoreML and llama.cpp release the GIL).
"""

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Optional

import numpy as np

from engine.kv_cache import KVCacheManager
from engine.sampler import greedy
from engine.speculative import verify_candidates, acceptance_rate
from gpu.backend import GPUBackend
from engine.coordinator import CoordinatorConfig


class PipelinedCoordinator:
    """
    Speculative decoding with merged bonus + ANE/GPU overlap.

    Usage::

        pc = PipelinedCoordinator(draft_fn, backend, greedy, config)
        tokens = pc.generate(prompt, max_tokens=200)
    """

    def __init__(
        self,
        draft_fn: Callable[[list[int], int], list[int]],
        main_backend: GPUBackend,
        sample_fn: Callable[[np.ndarray], int] = greedy,
        config: CoordinatorConfig = None,
    ):
        self.draft_fn = draft_fn
        self.kv = KVCacheManager(main_backend)
        self.sample_fn = sample_fn
        self.config = config or CoordinatorConfig()

        self._context: list[int] = []
        self._last_main_logits: Optional[np.ndarray] = None

        # Pipeline state
        self._pending_bonus: Optional[int] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Stats
        self._total_accepted = 0
        self._total_candidates = 0
        self._total_cycles = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0

    def process_prompt(self, tokens: list[int]) -> None:
        """Feed the initial prompt through the main model."""
        self.kv.reset()
        logits = self.kv.eval(tokens)
        self.kv.accept(len(tokens))
        self._context = list(tokens)
        self._last_main_logits = logits[-1]
        self._pending_bonus = None

    def step(self, prefetched_candidates: Optional[list[int]] = None) -> tuple[list[int], Optional[Future]]:
        """
        One speculative-decoding cycle with merged bonus + overlap.

        Returns (new_tokens, ane_future) where ane_future is a Future
        for the next cycle's candidates (may be None or stale).
        """
        n = self.config.n_candidates

        # Use prefetched candidates if available, else generate sync
        if prefetched_candidates is not None:
            candidates = prefetched_candidates
            self._prefetch_hits += 1
        else:
            self._prefetch_misses += 1
            candidates = self.draft_fn(self._context, n)

        # --- GPU eval: merged bonus + candidates ---
        if self._pending_bonus is not None:
            # Merge bonus from previous cycle into this batch
            batch = [self._pending_bonus] + candidates
            all_logits = self.kv.eval(batch)

            # Accept the bonus token (always valid)
            self.kv.accept(1)

            # logits[0] = prediction after bonus → verifies candidates[0]
            # logits[i] = prediction after candidates[i-1] → verifies candidates[i]
            # logits[N] = prediction after candidates[-1] → bonus sampling
            verify_logits = all_logits[:n]
            bonus_logits = all_logits[n]
        else:
            # First cycle: no pending bonus
            all_logits = self.kv.eval(candidates)

            verify_logits = np.empty((n, all_logits.shape[1]), dtype=np.float32)
            verify_logits[0] = self._last_main_logits
            if n > 1:
                verify_logits[1:] = all_logits[:n - 1]
            bonus_logits = all_logits[n - 1]

        # --- Verify ---
        accepted, bonus = verify_candidates(
            candidates, verify_logits, bonus_logits, self.sample_fn,
        )
        n_accepted = len(accepted)

        # --- KV cache management ---
        self.kv.accept(n_accepted)
        self.kv.rollback()

        # --- Update state ---
        self._context.extend(accepted)
        self._context.append(bonus)
        self._pending_bonus = bonus

        # Stats
        self._total_accepted += n_accepted
        self._total_candidates += n
        self._total_cycles += 1

        # --- Start ANE for next cycle (optimistic: assume all accepted) ---
        # Include the bonus token so context is correct
        all_accepted = (n_accepted == n)
        if all_accepted:
            optimistic_context = list(self._context)
            ane_future = self._executor.submit(self.draft_fn, optimistic_context, n)
        else:
            ane_future = None

        return accepted + [bonus], ane_future

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 100,
    ) -> list[int]:
        """Generate tokens using the pipelined coordinator."""
        self.process_prompt(prompt_tokens)
        output: list[int] = []
        ane_future: Optional[Future] = None

        while len(output) < max_tokens:
            # Get prefetched candidates if available
            prefetched = None
            if ane_future is not None:
                try:
                    prefetched = ane_future.result(timeout=0.5)
                except Exception:
                    prefetched = None

            new_tokens, ane_future = self.step(prefetched)

            for tok in new_tokens:
                output.append(tok)
                if (
                    self.config.eos_token_id is not None
                    and tok == self.config.eos_token_id
                ):
                    if ane_future:
                        ane_future.cancel()
                    return output
                if len(output) >= max_tokens:
                    if ane_future:
                        ane_future.cancel()
                    return output

        return output

    @property
    def stats(self) -> dict:
        return {
            "cycles": self._total_cycles,
            "total_candidates": self._total_candidates,
            "total_accepted": self._total_accepted,
            "acceptance_rate": acceptance_rate(
                self._total_accepted, self._total_candidates
            ),
            "prefetch_hits": self._prefetch_hits,
            "prefetch_misses": self._prefetch_misses,
            "prefetch_hit_rate": (
                self._prefetch_hits / max(1, self._prefetch_hits + self._prefetch_misses)
            ),
            "context_length": len(self._context),
            "kv_cache": self.kv.stats,
        }
