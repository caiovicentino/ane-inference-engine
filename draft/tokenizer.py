"""Shared tokenizer wrapper for Qwen models."""

from typing import Optional


class QwenTokenizer:
    """Thin wrapper around HuggingFace Qwen tokenizer."""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-0.5B"):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True,
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        pid = self.tokenizer.pad_token_id
        return pid if pid is not None else self.tokenizer.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def pad(self, token_ids: list[int], max_length: int) -> list[int]:
        """Right-pad token IDs to max_length, or truncate if longer."""
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        return token_ids + [self.pad_token_id] * (max_length - len(token_ids))
