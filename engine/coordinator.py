"""
Speculative decoding coordinator.

Orchestrates the draft → verify → accept loop:

1. Draft model (ANE) generates N candidate tokens.
2. Main model (GPU) evaluates all N in one batch.
3. Verification accepts a prefix + produces a bonus token.
4. KV cache is rolled back for rejected candidates.
5. Bonus token is evaluated to prepare the next cycle.

This is the synchronous (single-threaded) version.
Phase 4 adds parallel ANE/GPU threads.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from engine.kv_cache import KVCacheManager
from engine.sampler import greedy
from engine.speculative import verify_candidates, acceptance_rate
from gpu.backend import GPUBackend


@dataclass
class CoordinatorConfig:
    n_candidates: int = 4           # draft tokens per cycle
    eos_token_id: Optional[int] = None
    max_seq_len: int = 2048         # hard context limit


class Coordinator:
    """
    Synchronous speculative-decoding loop.

    Parameters
    ----------
    draft_fn : callable(context: list[int], n: int) -> list[int]
        Given the accepted context and a count, return *n* candidate token IDs.
    main_backend : GPUBackend
        The verifier model.
    sample_fn : callable(logits) -> int
        Sampling strategy (default: greedy).
    config : CoordinatorConfig
        Tuning knobs.
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

        # running stats
        self._total_accepted = 0
        self._total_candidates = 0
        self._total_cycles = 0

    # -- public API ----------------------------------------------------------

    def process_prompt(self, tokens: list[int]) -> None:
        """Feed the initial prompt through the main model."""
        self.kv.reset()
        main_logits = self.kv.eval(tokens)
        self.kv.accept(len(tokens))
        self._context = list(tokens)
        self._last_main_logits = main_logits[-1]

    def step(self) -> list[int]:
        """
        Run one speculative-decoding cycle.

        Returns
        -------
        list[int]
            Newly generated tokens (accepted candidates + bonus).
        """
        n = self.config.n_candidates

        # 1. Draft generates candidates
        candidates = self.draft_fn(self._context, n)

        # 2. Main model evaluates candidates
        main_logits = self.kv.eval(candidates)

        # 3. Build verification arrays
        #    verify_logits[0] = last_main_logits  (verifies candidates[0])
        #    verify_logits[i] = main_logits[i-1]  (verifies candidates[i])
        #    bonus_logits     = main_logits[N-1]   (bonus position)
        N = len(candidates)
        verify_logits = np.empty((N, main_logits.shape[1]), dtype=np.float32)
        verify_logits[0] = self._last_main_logits
        if N > 1:
            verify_logits[1:] = main_logits[: N - 1]
        bonus_logits = main_logits[N - 1]

        # 4. Verify
        accepted, bonus = verify_candidates(
            candidates, verify_logits, bonus_logits, self.sample_fn,
        )
        n_accepted = len(accepted)

        # 5. KV cache: accept the prefix, rollback the rest
        self.kv.accept(n_accepted)
        self.kv.rollback()

        # 6. Eval bonus token → gives us last_main_logits for next cycle
        bonus_logits_result = self.kv.eval([bonus])
        self.kv.accept(1)
        self._last_main_logits = bonus_logits_result[0]

        # 7. Update context
        self._context.extend(accepted)
        self._context.append(bonus)

        # 8. Stats
        self._total_accepted += n_accepted
        self._total_candidates += N
        self._total_cycles += 1

        return accepted + [bonus]

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 100,
    ) -> list[int]:
        """
        Generate up to *max_tokens* via speculative decoding.

        Stops early on EOS if ``config.eos_token_id`` is set.
        """
        self.process_prompt(prompt_tokens)
        output: list[int] = []

        while len(output) < max_tokens:
            new = self.step()
            for tok in new:
                output.append(tok)
                if (
                    self.config.eos_token_id is not None
                    and tok == self.config.eos_token_id
                ):
                    return output
                if len(output) >= max_tokens:
                    return output

        return output

    # -- diagnostics ---------------------------------------------------------

    @property
    def avg_acceptance_rate(self) -> float:
        return acceptance_rate(self._total_accepted, self._total_candidates)

    @property
    def stats(self) -> dict:
        return {
            "cycles": self._total_cycles,
            "total_candidates": self._total_candidates,
            "total_accepted": self._total_accepted,
            "acceptance_rate": self.avg_acceptance_rate,
            "context_length": len(self._context),
            "kv_cache": self.kv.stats,
        }
