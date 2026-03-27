"""Tests for speculative decoding logic, sampler, and KV cache manager."""

import pytest
import numpy as np

from engine.sampler import greedy, sample, softmax
from engine.speculative import verify_candidates, acceptance_rate
from engine.kv_cache import KVCacheManager
from gpu.mock_backend import MockGPUBackend


# ============================================================================
# Sampler
# ============================================================================

class TestGreedy:
    def test_argmax(self):
        logits = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        assert greedy(logits) == 3

    def test_negative_logits(self):
        logits = np.array([-10.0, -5.0, -1.0, -20.0])
        assert greedy(logits) == 2

    def test_single_element(self):
        assert greedy(np.array([42.0])) == 0


class TestSoftmax:
    def test_sums_to_one(self):
        p = softmax(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(p.sum(), 1.0)

    def test_uniform(self):
        p = softmax(np.zeros(5))
        np.testing.assert_allclose(p, np.full(5, 0.2), atol=1e-7)

    def test_large_values(self):
        p = softmax(np.array([1000.0, 1001.0, 1000.0]))
        assert np.isfinite(p).all()
        assert np.isclose(p.sum(), 1.0)


class TestSample:
    def test_greedy_mode(self):
        logits = np.array([0.0, 0.0, 10.0, 0.0])
        assert sample(logits, temperature=0) == 2

    def test_top_k_1(self):
        logits = np.array([1.0, 5.0, 2.0])
        assert sample(logits, top_k=1) == 1

    def test_top_k_limits_choices(self):
        logits = np.zeros(100)
        logits[42] = 100.0
        logits[7] = 99.0
        rng = np.random.default_rng(0)
        results = {sample(logits, temperature=1.0, top_k=2, rng=rng) for _ in range(50)}
        assert results <= {42, 7}

    def test_top_p(self):
        logits = np.array([10.0, 0.0, 0.0, 0.0])
        # With very low top_p, should always pick the dominant token
        rng = np.random.default_rng(0)
        results = [sample(logits, temperature=1.0, top_p=0.5, rng=rng) for _ in range(20)]
        assert all(r == 0 for r in results)

    def test_reproducible_with_rng(self):
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        r1 = sample(logits, temperature=1.0, rng=np.random.default_rng(123))
        r2 = sample(logits, temperature=1.0, rng=np.random.default_rng(123))
        assert r1 == r2


# ============================================================================
# Verification
# ============================================================================

def _make_logits(vocab: int, hot_idx: int, hot_val: float = 10.0) -> np.ndarray:
    """Create a logit vector where ``hot_idx`` is the greedy winner."""
    v = np.zeros(vocab, dtype=np.float32)
    v[hot_idx] = hot_val
    return v


class TestVerifyCandidates:
    V = 64  # small vocab for tests

    def test_all_accepted(self):
        cands = [10, 20, 30]
        verify = np.stack([_make_logits(self.V, 10),
                           _make_logits(self.V, 20),
                           _make_logits(self.V, 30)])
        bonus = _make_logits(self.V, 50)

        accepted, bonus_tok = verify_candidates(cands, verify, bonus)
        assert accepted == [10, 20, 30]
        assert bonus_tok == 50

    def test_first_rejected(self):
        cands = [10, 20, 30]
        verify = np.stack([_make_logits(self.V, 63),   # mismatch!
                           _make_logits(self.V, 20),
                           _make_logits(self.V, 30)])
        bonus = _make_logits(self.V, 50)

        accepted, bonus_tok = verify_candidates(cands, verify, bonus)
        assert accepted == []
        assert bonus_tok == 63  # main model's actual pick

    def test_partial_accept(self):
        cands = [10, 20, 30, 40]
        verify = np.stack([_make_logits(self.V, 10),    # match
                           _make_logits(self.V, 20),    # match
                           _make_logits(self.V, 55),    # mismatch
                           _make_logits(self.V, 40)])
        bonus = _make_logits(self.V, 50)

        accepted, bonus_tok = verify_candidates(cands, verify, bonus)
        assert accepted == [10, 20]
        assert bonus_tok == 55

    def test_single_candidate_accepted(self):
        cands = [7]
        verify = np.stack([_make_logits(self.V, 7)])
        bonus = _make_logits(self.V, 42)

        accepted, bonus_tok = verify_candidates(cands, verify, bonus)
        assert accepted == [7]
        assert bonus_tok == 42

    def test_single_candidate_rejected(self):
        cands = [7]
        verify = np.stack([_make_logits(self.V, 9)])
        bonus = _make_logits(self.V, 42)

        accepted, bonus_tok = verify_candidates(cands, verify, bonus)
        assert accepted == []
        assert bonus_tok == 9

    def test_last_rejected(self):
        cands = [10, 20, 30]
        verify = np.stack([_make_logits(self.V, 10),
                           _make_logits(self.V, 20),
                           _make_logits(self.V, 63)])   # mismatch at last
        bonus = _make_logits(self.V, 50)

        accepted, bonus_tok = verify_candidates(cands, verify, bonus)
        assert accepted == [10, 20]
        assert bonus_tok == 63


class TestAcceptanceRate:
    def test_all(self):
        assert acceptance_rate(4, 4) == 1.0

    def test_none(self):
        assert acceptance_rate(0, 4) == 0.0

    def test_half(self):
        assert acceptance_rate(2, 4) == 0.5

    def test_empty(self):
        assert acceptance_rate(0, 0) == 0.0


# ============================================================================
# KV Cache Manager
# ============================================================================

class TestKVCacheManager:
    @pytest.fixture
    def kv(self):
        b = MockGPUBackend(vocab_size=100, n_ctx=256)
        b.load()
        return KVCacheManager(b)

    def test_initial_state(self, kv):
        assert kv.accepted_length == 0
        assert kv.raw_kv_pos == 0

    def test_eval_advances_raw_pos(self, kv):
        kv.eval([1, 2, 3])
        assert kv.raw_kv_pos == 3
        assert kv.accepted_length == 0  # not yet accepted

    def test_accept(self, kv):
        kv.eval([1, 2, 3])
        kv.accept(2)
        assert kv.accepted_length == 2
        assert kv.raw_kv_pos == 3  # still 3 in raw cache

    def test_rollback(self, kv):
        kv.eval([1, 2, 3, 4, 5])
        kv.accept(3)
        kv.rollback()  # truncate raw to accepted (3)
        assert kv.raw_kv_pos == 3
        assert kv.accepted_length == 3

    def test_rollback_noop_if_aligned(self, kv):
        kv.eval([1, 2, 3])
        kv.accept(3)
        kv.rollback()  # raw == accepted, nothing to do
        assert kv.raw_kv_pos == 3

    def test_stats_tracking(self, kv):
        kv.eval([1, 2, 3, 4])
        kv.accept(2)
        kv.rollback()
        s = kv.stats
        assert s["extends"] == 1
        assert s["rollbacks"] == 1
        assert s["tokens_rolled_back"] == 2
        assert s["tokens_accepted"] == 2

    def test_reset(self, kv):
        kv.eval([1, 2, 3])
        kv.accept(3)
        kv.reset()
        assert kv.accepted_length == 0
        assert kv.raw_kv_pos == 0

    def test_multiple_cycles(self, kv):
        # Cycle 1
        kv.eval([1, 2, 3])      # prompt
        kv.accept(3)

        kv.eval([10, 20, 30])   # candidates
        kv.accept(2)            # accept 2
        kv.rollback()           # rollback 1
        assert kv.raw_kv_pos == 5
        assert kv.accepted_length == 5

        # Cycle 2
        kv.eval([40, 50])       # new candidates
        kv.accept(1)
        kv.rollback()
        assert kv.raw_kv_pos == 6
        assert kv.accepted_length == 6
