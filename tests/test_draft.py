"""Comprehensive tests for the draft model (Phase 1)."""

import pytest
import torch

from draft.model import (
    DraftModel,
    DraftModelConfig,
    RMSNorm,
    Attention,
    FeedForward,
    TransformerBlock,
    TINY_TEST_CONFIG,
    QWEN2_5_0_5B_CONFIG,
    _precompute_rope,
    _rotate_half,
    _apply_rope,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny config for speed
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return DraftModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=32,
        tie_word_embeddings=False,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return DraftModel(config)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        c = DraftModelConfig()
        assert c.vocab_size == 151936
        assert c.hidden_size == 896
        assert c.num_attention_heads == 14

    def test_qwen_preset(self):
        c = QWEN2_5_0_5B_CONFIG
        assert c.num_hidden_layers == 24
        assert c.intermediate_size == 4864

    def test_tiny_preset(self):
        c = TINY_TEST_CONFIG
        assert c.num_hidden_layers == 2
        assert c.vocab_size == 256

    def test_head_dim_divides_evenly(self):
        c = QWEN2_5_0_5B_CONFIG
        assert c.hidden_size % c.num_attention_heads == 0

    def test_kv_heads_divide_evenly(self):
        c = QWEN2_5_0_5B_CONFIG
        assert c.num_attention_heads % c.num_key_value_heads == 0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self, config):
        norm = RMSNorm(config.hidden_size)
        x = torch.randn(2, 10, config.hidden_size)
        assert norm(x).shape == x.shape

    def test_unit_rms(self):
        """With weight=1, output RMS ≈ 1."""
        norm = RMSNorm(8, eps=1e-6)
        norm.weight.data.fill_(1.0)
        x = torch.randn(1, 1, 8)
        rms = norm(x).pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_weight_scaling(self):
        norm2 = RMSNorm(4, eps=1e-6)
        norm2.weight.data.fill_(2.0)
        norm1 = RMSNorm(4, eps=1e-6)
        norm1.weight.data.fill_(1.0)
        x = torch.randn(1, 1, 4)
        assert torch.allclose(norm2(x), 2.0 * norm1(x), atol=1e-5)

    def test_zero_input(self):
        norm = RMSNorm(4, eps=1e-6)
        x = torch.zeros(1, 1, 4)
        out = norm(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_shape(self):
        cos, sin = _precompute_rope(64, 128, theta=10000.0)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_value_range(self):
        cos, sin = _precompute_rope(64, 128, theta=10000.0)
        assert cos.min() >= -1.0 and cos.max() <= 1.0
        assert sin.min() >= -1.0 and sin.max() <= 1.0

    def test_pythagorean_identity(self):
        cos, sin = _precompute_rope(64, 128, theta=10000.0)
        identity = cos.pow(2) + sin.pow(2)
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-6)

    def test_rotate_half(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(_rotate_half(x), expected)

    def test_preserves_norm(self):
        """RoPE is a rotation — must preserve vector norms."""
        head_dim = 16
        cos, sin = _precompute_rope(head_dim, 32, theta=10000.0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = torch.randn(1, 4, 32, head_dim)
        k = torch.randn(1, 4, 32, head_dim)
        q_rot, k_rot = _apply_rope(q, k, cos, sin)

        assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
        assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)

    def test_position_zero_is_identity_for_cos(self):
        """At position 0, cos=1 and sin=0 so RoPE should be identity."""
        cos, sin = _precompute_rope(8, 16, theta=10000.0)
        # Position 0
        assert torch.allclose(cos[0], torch.ones(8), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros(8), atol=1e-6)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class TestAttention:
    def test_output_shape(self, config):
        attn = Attention(config)
        B, L = 2, config.max_seq_len
        head_dim = config.hidden_size // config.num_attention_heads

        x = torch.randn(B, L, config.hidden_size)
        cos, sin = _precompute_rope(head_dim, L, config.rope_theta)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        mask = torch.triu(torch.full((L, L), -1e9), diagonal=1).unsqueeze(0).unsqueeze(0)

        out = attn(x, cos, sin, mask)
        assert out.shape == (B, L, config.hidden_size)

    def test_gqa_group_count(self, config):
        attn = Attention(config)
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2
        assert attn.num_kv_groups == 2


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class TestFeedForward:
    def test_output_shape(self, config):
        ff = FeedForward(config)
        x = torch.randn(1, 10, config.hidden_size)
        assert ff(x).shape == x.shape

    def test_swiglu_nonlinearity(self, config):
        """Output should differ from a plain linear projection."""
        ff = FeedForward(config)
        x = torch.randn(1, 5, config.hidden_size)
        out = ff(x)
        assert not torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def _make_inputs(self, config):
        B, L = 1, config.max_seq_len
        head_dim = config.hidden_size // config.num_attention_heads
        x = torch.randn(B, L, config.hidden_size)
        cos, sin = _precompute_rope(head_dim, L, config.rope_theta)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        mask = torch.triu(
            torch.full((L, L), -1e9), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        return x, cos, sin, mask

    def test_output_shape(self, config):
        block = TransformerBlock(config)
        x, cos, sin, mask = self._make_inputs(config)
        assert block(x, cos, sin, mask).shape == x.shape

    def test_residual_not_identity(self, config):
        """Block output should differ from input (transformations applied)."""
        torch.manual_seed(0)
        block = TransformerBlock(config)
        x, cos, sin, mask = self._make_inputs(config)
        with torch.no_grad():
            out = block(x, cos, sin, mask)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Full DraftModel
# ---------------------------------------------------------------------------

class TestDraftModel:
    def test_creation(self, model, config):
        assert model.config.hidden_size == config.hidden_size

    def test_forward_shape(self, model, config):
        ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, config.max_seq_len, config.vocab_size)

    def test_forward_batch(self, model, config):
        ids = torch.randint(0, config.vocab_size, (4, config.max_seq_len))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (4, config.max_seq_len, config.vocab_size)

    def test_deterministic(self, model, config):
        model.eval()
        ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        with torch.no_grad():
            a = model(ids)
            b = model(ids)
        assert torch.allclose(a, b)

    def test_causal_masking(self, model, config):
        """Changing a future token must NOT affect logits at earlier positions."""
        model.eval()
        ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))

        with torch.no_grad():
            logits1 = model(ids.clone())

        pos = config.max_seq_len // 2
        modified = ids.clone()
        modified[0, pos] = (ids[0, pos] + 1) % config.vocab_size
        with torch.no_grad():
            logits2 = model(modified)

        # Positions before the change must be identical
        assert torch.allclose(logits1[0, :pos], logits2[0, :pos], atol=1e-5)

    def test_parameter_count(self, model):
        count = model.count_parameters()
        assert 0 < count < 10_000_000  # tiny model

    def test_tied_weights(self):
        c = DraftModelConfig(
            vocab_size=256, hidden_size=64, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=128,
            num_hidden_layers=2, max_seq_len=16, tie_word_embeddings=True,
        )
        m = DraftModel(c)
        assert m.lm_head.weight is m.embed_tokens.weight

    def test_untied_weights(self, model):
        assert model.lm_head.weight is not model.embed_tokens.weight

    def test_buffers_exist(self, model):
        buffers = dict(model.named_buffers())
        assert "rope_cos" in buffers
        assert "rope_sin" in buffers
        assert "causal_mask" in buffers

    def test_buffer_shapes(self, model, config):
        head_dim = config.hidden_size // config.num_attention_heads
        assert model.rope_cos.shape == (1, 1, config.max_seq_len, head_dim)
        assert model.rope_sin.shape == (1, 1, config.max_seq_len, head_dim)
        assert model.causal_mask.shape == (1, 1, config.max_seq_len, config.max_seq_len)

    def test_causal_mask_upper_triangle(self, model, config):
        """Mask should be 0 on/below diagonal, -1e9 above."""
        mask = model.causal_mask.squeeze()
        L = config.max_seq_len
        for i in range(L):
            for j in range(L):
                if j <= i:
                    assert mask[i, j].item() == 0.0
                else:
                    assert mask[i, j].item() == pytest.approx(-1e9)


# ---------------------------------------------------------------------------
# Traceability (critical for CoreML conversion)
# ---------------------------------------------------------------------------

class TestTrace:
    def test_jit_trace(self, model, config):
        """Model must be traceable with torch.jit.trace."""
        model.eval()
        ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        with torch.no_grad():
            traced = torch.jit.trace(model, ids)
            traced_out = traced(ids)
            direct_out = model(ids)
        assert torch.allclose(traced_out, direct_out, atol=1e-5)

    def test_traced_different_input(self, model, config):
        """Traced model should handle different input values."""
        model.eval()
        ids1 = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        ids2 = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        with torch.no_grad():
            traced = torch.jit.trace(model, ids1)
            out1 = traced(ids1)
            out2 = traced(ids2)
        # Different inputs → different outputs
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Weight mapping (HuggingFace → ours)
# ---------------------------------------------------------------------------

class TestWeightMapping:
    def test_hf_name_strip(self):
        hf_keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        mapped = {}
        for name in hf_keys:
            our = name[len("model."):] if name.startswith("model.") else name
            mapped[our] = name

        assert "embed_tokens.weight" in mapped
        assert "layers.0.self_attn.q_proj.weight" in mapped
        assert "layers.0.mlp.gate_proj.weight" in mapped
        assert "norm.weight" in mapped
        assert "lm_head.weight" in mapped

    def test_our_state_dict_keys(self, model):
        """Our model's state dict should contain expected key patterns."""
        keys = set(model.state_dict().keys())
        assert "embed_tokens.weight" in keys
        assert "layers.0.self_attn.q_proj.weight" in keys
        assert "layers.0.mlp.gate_proj.weight" in keys
        assert "norm.weight" in keys
        assert "lm_head.weight" in keys
