"""
Threaded speculative decoding coordinator.

Overlaps ANE draft generation with GPU verification via a prefetch thread:

    ANE thread (background):
        Continuously generates the *next* candidate batch using the
        latest accepted context.

    Main thread (caller):
        Pulls prefetched candidates → GPU verify → accept/reject.

A version stamp on each prefetched batch lets the main thread detect
stale candidates (context changed since they were generated) and
fall back to synchronous generation (~5 ms penalty on ANE).
"""

import threading
from queue import Queue, Empty, Full
from typing import Callable, Optional

import numpy as np

from engine.adaptive import AdaptiveDraftCount, AdaptiveConfig
from engine.coordinator import CoordinatorConfig
from engine.kv_cache import KVCacheManager
from engine.sampler import greedy
from engine.speculative import verify_candidates, acceptance_rate
from gpu.backend import GPUBackend


class ThreadedCoordinator:
    """
    Speculative decoding with background ANE prefetching.

    Usage::

        tc = ThreadedCoordinator(draft_fn, backend)
        tc.start()
        tokens = tc.generate(prompt, max_tokens=200)
        tc.stop()
    """

    def __init__(
        self,
        draft_fn: Callable[[list[int], int], list[int]],
        main_backend: GPUBackend,
        sample_fn: Callable[[np.ndarray], int] = greedy,
        config: CoordinatorConfig = None,
        adaptive_config: AdaptiveConfig = None,
    ):
        self.draft_fn = draft_fn
        self.kv = KVCacheManager(main_backend)
        self.sample_fn = sample_fn
        self.config = config or CoordinatorConfig()

        self._adaptive = AdaptiveDraftCount(adaptive_config) if adaptive_config else None

        self._context: list[int] = []
        self._last_main_logits: Optional[np.ndarray] = None
        self._context_version: int = 0

        # prefetch plumbing
        self._prefetch_queue: Queue = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # stats
        self._total_accepted = 0
        self._total_candidates = 0
        self._total_cycles = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the background ANE prefetch thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._ane_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the prefetch thread."""
        self._stop.set()
        # unblock the thread if it's waiting on put()
        try:
            self._prefetch_queue.get_nowait()
        except Empty:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- ANE background loop -------------------------------------------------

    def _ane_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ctx = list(self._context)
                n = self._adaptive.n if self._adaptive else self.config.n_candidates
                ver = self._context_version

            if not ctx:
                # No prompt yet — spin-wait briefly
                self._stop.wait(0.01)
                continue

            candidates = self.draft_fn(ctx, n)

            # Drop stale prefetch if present
            try:
                self._prefetch_queue.get_nowait()
            except Empty:
                pass

            try:
                self._prefetch_queue.put((candidates, ver), timeout=0.5)
            except Full:
                pass

    # -- main-thread API -----------------------------------------------------

    def process_prompt(self, tokens: list[int]) -> None:
        """Feed the initial prompt through the main model."""
        self.kv.reset()
        logits = self.kv.eval(tokens)
        self.kv.accept(len(tokens))

        with self._lock:
            self._context = list(tokens)
            self._last_main_logits = logits[-1]
            self._context_version += 1

    def step(self) -> list[int]:
        """One speculative-decoding cycle (called from main thread)."""
        n = self._adaptive.n if self._adaptive else self.config.n_candidates

        # Try prefetched candidates
        candidates = None
        try:
            prefetched, ver = self._prefetch_queue.get_nowait()
            if ver == self._context_version and len(prefetched) == n:
                candidates = prefetched
                self._prefetch_hits += 1
        except Empty:
            pass

        if candidates is None:
            self._prefetch_misses += 1
            candidates = self.draft_fn(self._context, n)

        # --- GPU verify (main thread) ---
        main_logits = self.kv.eval(candidates)
        N = len(candidates)

        verify_logits = np.empty((N, main_logits.shape[1]), dtype=np.float32)
        verify_logits[0] = self._last_main_logits
        if N > 1:
            verify_logits[1:] = main_logits[: N - 1]
        bonus_logits = main_logits[N - 1]

        accepted, bonus = verify_candidates(
            candidates, verify_logits, bonus_logits, self.sample_fn,
        )
        n_accepted = len(accepted)

        # KV cache
        self.kv.accept(n_accepted)
        self.kv.rollback()
        bonus_result = self.kv.eval([bonus])
        self.kv.accept(1)

        # Context update (under lock so ANE sees it)
        with self._lock:
            self._last_main_logits = bonus_result[0]
            self._context.extend(accepted)
            self._context.append(bonus)
            self._context_version += 1

        # Adaptive tuning
        if self._adaptive:
            self._adaptive.update(n_accepted, N)

        # Stats
        self._total_accepted += n_accepted
        self._total_candidates += N
        self._total_cycles += 1

        return accepted + [bonus]

    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 100,
    ) -> list[int]:
        """Generate tokens using the threaded pipeline."""
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
    def stats(self) -> dict:
        return {
            "cycles": self._total_cycles,
            "total_candidates": self._total_candidates,
            "total_accepted": self._total_accepted,
            "acceptance_rate": acceptance_rate(
                self._total_accepted, self._total_candidates
            ),
            "adaptive_n": self._adaptive.n if self._adaptive else self.config.n_candidates,
            "prefetch_hits": self._prefetch_hits,
            "prefetch_misses": self._prefetch_misses,
            "prefetch_hit_rate": (
                self._prefetch_hits / max(1, self._prefetch_hits + self._prefetch_misses)
            ),
            "context_length": len(self._context),
            "kv_cache": self.kv.stats,
        }
