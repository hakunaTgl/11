"""Tokenizer helpers for the CLI agent."""
from transformers import AutoTokenizer


def load_tokenizer(model_path: str, trust_remote_code: bool = False) -> AutoTokenizer:
    """Load a tokenizer for the provided model path."""

    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
