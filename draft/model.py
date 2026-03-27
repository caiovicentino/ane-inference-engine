"""
Pure PyTorch transformer for Qwen2.5-compatible draft model.

Designed for coremltools tracing → CoreML → ANE execution.
All operations are static (no dynamic shapes or control flow in forward).
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DraftModelConfig:
    """Configuration matching Qwen2.5 architecture."""
    vocab_size: int = 151936
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_seq_len: int = 512
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, path: str) -> "DraftModelConfig":
        """Load from a HuggingFace config.json."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            vocab_size=data.get("vocab_size", 151936),
            hidden_size=data.get("hidden_size", 896),
            num_attention_heads=data.get("num_attention_heads", 14),
            num_key_value_heads=data.get("num_key_value_heads", 2),
            intermediate_size=data.get("intermediate_size", 4864),
            num_hidden_layers=data.get("num_hidden_layers", 24),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=data.get("rope_theta", 1000000.0),
            max_seq_len=data.get("max_position_embeddings", 512),
            tie_word_embeddings=data.get("tie_word_embeddings", True),
        )


# Predefined configs ---------------------------------------------------------

QWEN2_5_0_5B_CONFIG = DraftModelConfig(
    vocab_size=151936,
    hidden_size=896,
    num_attention_heads=14,
    num_key_value_heads=2,
    intermediate_size=4864,
    num_hidden_layers=24,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    max_seq_len=512,
    tie_word_embeddings=True,
)

TINY_TEST_CONFIG = DraftModelConfig(
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


# ---------------------------------------------------------------------------
# RoPE helpers (matches HuggingFace Qwen2 implementation)
# ---------------------------------------------------------------------------

def _precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    """Precompute cosine/sine tables for Rotary Position Embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)                   # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims — standard HF convention."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and RoPE."""

    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)

        # GQA: expand KV heads via reshape (trace-friendly, no repeat_interleave)
        if self.num_kv_groups > 1:
            k = (
                k.unsqueeze(2)
                .expand(B, self.num_kv_heads, self.num_kv_groups, L, self.head_dim)
                .reshape(B, self.num_heads, L, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(B, self.num_kv_heads, self.num_kv_groups, L, self.head_dim)
                .reshape(B, self.num_heads, L, self.head_dim)
            )

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (Qwen2 MLP)."""

    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer layer: attention → FFN with pre-norm residuals."""

    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DraftModel(nn.Module):
    """
    Pure PyTorch Qwen2-compatible transformer for ANE execution.

    - Fixed sequence length (max_seq_len) for CoreML tracing
    - All buffers (RoPE, causal mask) precomputed
    - No dynamic control flow in forward()
    """

    def __init__(self, config: DraftModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE — shape (1, 1, max_seq_len, head_dim)
        head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = _precompute_rope(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos.unsqueeze(0).unsqueeze(0))
        self.register_buffer("rope_sin", sin.unsqueeze(0).unsqueeze(0))

        # Precompute causal mask — shape (1, 1, max_seq_len, max_seq_len)
        mask = torch.triu(
            torch.full((config.max_seq_len, config.max_seq_len), -1e9),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, max_seq_len) right-padded token IDs.
        Returns:
            logits: (batch, max_seq_len, vocab_size)
        """
        h = self.embed_tokens(input_ids)

        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin, self.causal_mask)

        h = self.norm(h)
        return self.lm_head(h)

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, model_path: str, max_seq_len: int = 512) -> "DraftModel":
        """
        Load from a HuggingFace model directory.

        Expects model_path/config.json and model_path/*.safetensors.
        """
        model_path = Path(model_path)
        config = DraftModelConfig.from_json(str(model_path / "config.json"))
        config.max_seq_len = max_seq_len
        model = cls(config)
        _load_hf_weights(model, model_path)
        return model


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_hf_weights(model: DraftModel, model_path: Path) -> None:
    """Load weights from HuggingFace safetensors, mapping 'model.*' → '*'."""
    from safetensors.torch import load_file

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    state_dict = {}
    for f in safetensors_files:
        state_dict.update(load_file(str(f)))

    # Strip the "model." prefix that HuggingFace uses
    mapped = {}
    for hf_name, tensor in state_dict.items():
        name = hf_name
        if name.startswith("model."):
            name = name[len("model."):]
        mapped[name] = tensor

    missing, unexpected = model.load_state_dict(mapped, strict=False)

    # Tied lm_head.weight is expected to be "missing"
    if model.config.tie_word_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]

    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
