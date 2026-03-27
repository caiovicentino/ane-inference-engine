"""
Adaptive draft candidate count.

Tracks acceptance rate over a sliding window and adjusts N:
  - acceptance > up_threshold   →  N += 1  (draft is good, be greedier)
  - acceptance < down_threshold →  N -= 1  (draft is bad, waste less GPU)

This is the core of the "tune draft count" optimisation from the design.
"""

from collections import deque
from dataclasses import dataclass


@dataclass
class AdaptiveConfig:
    initial_n: int = 4
    min_n: int = 1
    max_n: int = 8
    up_threshold: float = 0.80     # increase N when acceptance above this
    down_threshold: float = 0.40   # decrease N when acceptance below this
    window: int = 5                # cycles in the sliding window


class AdaptiveDraftCount:
    """Dynamically adjusts the number of draft candidates per cycle."""

    def __init__(self, config: AdaptiveConfig = None):
        self.cfg = config or AdaptiveConfig()
        self._n = self.cfg.initial_n
        self._history: deque[tuple[int, int]] = deque(maxlen=self.cfg.window)

    # -- public API ----------------------------------------------------------

    @property
    def n(self) -> int:
        """Current number of candidates to request from the draft model."""
        return self._n

    def update(self, n_accepted: int, n_candidates: int) -> int:
        """
        Record one cycle's result and return the (possibly adjusted) N.

        Args:
            n_accepted:   how many candidates the main model accepted.
            n_candidates: how many were proposed (may differ from self.n
                          if the caller used a different count).

        Returns:
            The new value of N (same object as ``self.n``).
        """
        self._history.append((n_accepted, n_candidates))

        rate = self.windowed_acceptance_rate
        if rate > self.cfg.up_threshold and self._n < self.cfg.max_n:
            self._n += 1
        elif rate < self.cfg.down_threshold and self._n > self.cfg.min_n:
            self._n -= 1

        return self._n

    def reset(self) -> None:
        self._n = self.cfg.initial_n
        self._history.clear()

    # -- diagnostics ---------------------------------------------------------

    @property
    def windowed_acceptance_rate(self) -> float:
        if not self._history:
            return 0.0
        total_a = sum(a for a, _ in self._history)
        total_c = sum(c for _, c in self._history)
        return total_a / max(1, total_c)

    @property
    def history_len(self) -> int:
        return len(self._history)
