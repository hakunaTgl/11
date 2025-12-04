"""Tokenizer helpers for the CLI agent."""
from transformers import AutoTokenizer


def load_tokenizer(model_path: str, trust_remote_code: bool = False) -> AutoTokenizer:
    """
    Load a tokenizer for the provided model path.

    Args:
        model_path (str): Path or identifier of the pretrained model.
        trust_remote_code (bool, optional): Whether to trust custom code from the model repository. Defaults to False.

    Returns:
        AutoTokenizer: The loaded tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
