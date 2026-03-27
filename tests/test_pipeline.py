"""Tests for the coordinator and inference pipeline (Phase 3)."""

import pytest
import numpy as np

from engine.coordinator import Coordinator, CoordinatorConfig
from engine.sampler import greedy
from gpu.mock_backend import MockGPUBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(vocab=100, n_ctx=512):
    b = MockGPUBackend(vocab_size=vocab, n_ctx=n_ctx)
    b.load()
    return b


def _random_draft_fn(seed=0):
    """Draft function that returns deterministic pseudo-random candidates."""
    def fn(context, n):
        rng = np.random.RandomState(len(context) + seed)
        return [int(rng.randint(0, 100)) for _ in range(n)]
    return fn


def _constant_draft_fn(token=42):
    """Draft function that always returns the same token."""
    def fn(context, n):
        return [token] * n
    return fn


# ---------------------------------------------------------------------------
# Coordinator — process_prompt
# ---------------------------------------------------------------------------

class TestProcessPrompt:
    def test_sets_context(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b)
        c.process_prompt([1, 2, 3])
        assert c._context == [1, 2, 3]

    def test_advances_kv_cache(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b)
        c.process_prompt([1, 2, 3, 4, 5])
        assert c.kv.accepted_length == 5
        assert c.kv.raw_kv_pos == 5

    def test_stores_last_logits(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b)
        c.process_prompt([1, 2, 3])
        assert c._last_main_logits is not None
        assert c._last_main_logits.shape == (100,)


# ---------------------------------------------------------------------------
# Coordinator — step
# ---------------------------------------------------------------------------

class TestStep:
    def test_returns_tokens(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        c.process_prompt([1, 2, 3])
        tokens = c.step()
        assert isinstance(tokens, list)
        assert len(tokens) >= 1  # at least the bonus token
        assert all(isinstance(t, int) for t in tokens)

    def test_context_grows(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        c.process_prompt([1, 2, 3])
        before = len(c._context)
        tokens = c.step()
        assert len(c._context) == before + len(tokens)

    def test_kv_cache_consistent(self):
        """After step, kv_pos == accepted_length == len(context)."""
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        c.process_prompt([1, 2, 3])
        c.step()
        assert c.kv.accepted_length == c.kv.raw_kv_pos
        assert c.kv.accepted_length == len(c._context)

    def test_multiple_steps(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        c.process_prompt([1, 2, 3])
        total = 0
        for _ in range(10):
            tokens = c.step()
            total += len(tokens)
            # Invariant must hold after every step
            assert c.kv.accepted_length == len(c._context)
            assert c.kv.raw_kv_pos == c.kv.accepted_length
        assert total >= 10  # at least 1 token per step (bonus)

    def test_stats_updated(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        c.process_prompt([1, 2, 3])
        c.step()
        s = c.stats
        assert s["cycles"] == 1
        assert s["total_candidates"] == 4
        assert 0 <= s["acceptance_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Coordinator — generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_respects_max_tokens(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        tokens = c.generate([1, 2, 3], max_tokens=20)
        assert len(tokens) <= 20

    def test_at_least_one_token(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        tokens = c.generate([1, 2, 3], max_tokens=1)
        assert len(tokens) >= 1

    def test_eos_stops_early(self):
        b = _make_backend()
        # Use a draft that always returns token 99 (our EOS)
        eos = 99
        config = CoordinatorConfig(n_candidates=4, eos_token_id=eos)

        def eos_draft(context, n):
            return [eos] * n

        c = Coordinator(eos_draft, b, config=config)
        tokens = c.generate([1, 2, 3], max_tokens=1000)
        # Should stop well before 1000
        assert len(tokens) < 1000
        assert tokens[-1] == eos

    def test_long_generation(self):
        b = _make_backend(n_ctx=4096)
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=4))
        tokens = c.generate([1, 2, 3], max_tokens=200)
        assert len(tokens) <= 200
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# Coordinator — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_prompt_token(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=2))
        tokens = c.generate([42], max_tokens=10)
        assert len(tokens) >= 1

    def test_n_candidates_1(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=1))
        c.process_prompt([1, 2, 3])
        tokens = c.step()
        assert len(tokens) >= 1
        assert c.kv.accepted_length == len(c._context)

    def test_n_candidates_8(self):
        b = _make_backend()
        c = Coordinator(_random_draft_fn(), b, config=CoordinatorConfig(n_candidates=8))
        c.process_prompt([1, 2, 3])
        tokens = c.step()
        assert len(tokens) >= 1
        assert c.kv.accepted_length == len(c._context)


# ---------------------------------------------------------------------------
# Full pipeline smoke test (with mock draft model)
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    def test_draft_fn_factory(self):
        """make_draft_fn produces a working callable."""
        import torch
        from draft.model import DraftModel, TINY_TEST_CONFIG
        from engine.pipeline import make_draft_fn

        torch.manual_seed(0)
        model = DraftModel(TINY_TEST_CONFIG)
        fn = make_draft_fn(model, max_seq_len=TINY_TEST_CONFIG.max_seq_len)
        cands = fn([1, 2, 3], 4)
        assert len(cands) == 4
        assert all(isinstance(c, int) for c in cands)
        assert all(0 <= c < TINY_TEST_CONFIG.vocab_size for c in cands)

    def test_end_to_end_with_draft_model(self):
        """Full loop: DraftModel + MockGPUBackend + Coordinator."""
        import torch
        from draft.model import DraftModel, TINY_TEST_CONFIG
        from engine.pipeline import make_draft_fn

        torch.manual_seed(0)
        model = DraftModel(TINY_TEST_CONFIG)
        backend = MockGPUBackend(vocab_size=TINY_TEST_CONFIG.vocab_size, n_ctx=256)
        backend.load()

        draft_fn = make_draft_fn(model, max_seq_len=TINY_TEST_CONFIG.max_seq_len)
        config = CoordinatorConfig(n_candidates=4)
        coord = Coordinator(draft_fn, backend, greedy, config)

        tokens = coord.generate([1, 2, 3], max_tokens=20)
        assert 1 <= len(tokens) <= 20
        assert coord.kv.accepted_length == len(coord._context)

        s = coord.stats
        assert s["cycles"] >= 1
        assert s["context_length"] > 3  # prompt + generated
