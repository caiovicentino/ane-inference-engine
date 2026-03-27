"""Tests for the GPU backend (Phase 2)."""

import pytest
import numpy as np

from gpu.backend import GPUBackend, GPUBackendConfig
from gpu.mock_backend import MockGPUBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    b = MockGPUBackend(vocab_size=100, n_ctx=256)
    b.load()
    yield b
    b.close()


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

class TestInterface:
    def test_isinstance(self, backend):
        assert isinstance(backend, GPUBackend)

    def test_initial_state(self, backend):
        assert backend.kv_pos == 0
        assert backend.vocab_size == 100
        assert backend.n_ctx == 256


# ---------------------------------------------------------------------------
# eval()
# ---------------------------------------------------------------------------

class TestEval:
    def test_single_token_shape(self, backend):
        logits = backend.eval([42])
        assert logits.shape == (1, 100)
        assert logits.dtype == np.float32

    def test_multi_token_shape(self, backend):
        logits = backend.eval([1, 2, 3, 4])
        assert logits.shape == (4, 100)

    def test_empty(self, backend):
        logits = backend.eval([])
        assert logits.shape == (0, 100)

    def test_kv_pos_advances(self, backend):
        backend.eval([1, 2, 3])
        assert backend.kv_pos == 3
        backend.eval([4, 5])
        assert backend.kv_pos == 5

    def test_deterministic(self, backend):
        l1 = backend.eval([10, 20])
        backend.reset()
        l2 = backend.eval([10, 20])
        np.testing.assert_array_equal(l1, l2)

    def test_different_tokens(self, backend):
        l1 = backend.eval([10])
        backend.reset()
        l2 = backend.eval([99])
        assert not np.allclose(l1, l2)

    def test_position_matters(self, backend):
        """Same token at different KV positions → different logits."""
        backend.eval([1])          # token 1 at pos 0
        l_pos1 = backend.eval([5])  # token 5 at pos 1
        backend.reset()
        l_pos0 = backend.eval([5])  # token 5 at pos 0
        assert not np.allclose(l_pos0, l_pos1)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class TestKVCache:
    def test_truncate(self, backend):
        backend.eval([1, 2, 3, 4, 5])
        backend.kv_truncate(3)
        assert backend.kv_pos == 3

    def test_truncate_noop_beyond(self, backend):
        backend.eval([1, 2])
        backend.kv_truncate(10)
        assert backend.kv_pos == 2

    def test_truncate_to_zero(self, backend):
        backend.eval([1, 2, 3])
        backend.kv_truncate(0)
        assert backend.kv_pos == 0

    def test_reset(self, backend):
        backend.eval([1, 2, 3])
        backend.reset()
        assert backend.kv_pos == 0

    def test_eval_after_truncate(self, backend):
        """Can continue evaluating after a truncate."""
        backend.eval([1, 2, 3, 4])
        backend.kv_truncate(2)
        logits = backend.eval([5, 6])
        assert logits.shape == (2, 100)
        assert backend.kv_pos == 4

    def test_eval_after_reset(self, backend):
        backend.eval([1, 2])
        backend.reset()
        logits = backend.eval([3])
        assert logits.shape == (1, 100)
        assert backend.kv_pos == 1


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_with_statement(self):
        with MockGPUBackend(vocab_size=50) as b:
            b.load()
            logits = b.eval([1])
            assert logits.shape == (1, 50)


# ---------------------------------------------------------------------------
# Speculative decoding simulation
# ---------------------------------------------------------------------------

class TestSpeculativeFlow:
    def test_prompt_then_candidates(self, backend):
        # Prompt
        prompt_logits = backend.eval([1, 2, 3, 4, 5])
        assert prompt_logits.shape == (5, 100)
        assert backend.kv_pos == 5

        # Draft candidates
        cand_logits = backend.eval([10, 20, 30, 40])
        assert cand_logits.shape == (4, 100)
        assert backend.kv_pos == 9

        # Accept first 2, reject rest
        backend.kv_truncate(5 + 2)
        assert backend.kv_pos == 7

        # Next cycle
        new_logits = backend.eval([50, 60, 70])
        assert new_logits.shape == (3, 100)
        assert backend.kv_pos == 10

    def test_full_reject(self, backend):
        backend.eval([1, 2, 3])         # prompt
        backend.eval([10, 20])           # candidates
        backend.kv_truncate(3)           # all rejected
        assert backend.kv_pos == 3

    def test_full_accept(self, backend):
        backend.eval([1, 2, 3])          # prompt
        backend.eval([10, 20, 30])       # candidates
        # All accepted → no truncation
        assert backend.kv_pos == 6

    def test_many_cycles(self, backend):
        """Simulate 10 speculative cycles."""
        backend.eval(list(range(20)))    # prompt of 20 tokens
        prompt_end = 20

        accepted_total = 0
        for cycle in range(10):
            n_cands = 4
            cands = list(range(100 + cycle * 10, 100 + cycle * 10 + n_cands))
            backend.eval(cands)

            # Accept 2 out of 4 every cycle
            n_accept = 2
            backend.kv_truncate(prompt_end + accepted_total + n_accept)
            accepted_total += n_accept

        assert backend.kv_pos == prompt_end + 20  # 10 cycles × 2 accepted

    def test_logits_argmax_is_stable(self, backend):
        """Argmax of logits should be deterministic across runs."""
        backend.eval([1, 2, 3])
        l1 = backend.eval([42])
        tok1 = int(np.argmax(l1[0]))

        backend.reset()
        backend.eval([1, 2, 3])
        l2 = backend.eval([42])
        tok2 = int(np.argmax(l2[0]))

        assert tok1 == tok2
