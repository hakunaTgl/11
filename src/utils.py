"""Utility helpers for loading language models."""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    """Pick the best available device.

    Args:
        prefer_gpu: When True, prefer CUDA or MPS if available.

    Returns:
        A torch.device pointing to an available accelerator or CPU.
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[str] = "auto",
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
) -> AutoModelForCausalLM:
    """Load a CausalLM model and place it on the target device.

    Args:
        model_path: Local directory or Hugging Face model id.
        device: Optional torch.device. If not provided and ``device_map`` is not set,
            the model is moved to the best available device.
        dtype: Torch dtype name (e.g. "float16") or ``"auto"`` to let transformers
            pick an appropriate precision.
        trust_remote_code: Whether to allow custom model code from model hub.
        device_map: Pass ``"auto"`` to let transformers shard the model across devices.

    Returns:
        An instance of ``AutoModelForCausalLM`` ready for generation.
    """

    torch_dtype = dtype
    if dtype and dtype != "auto":
        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype: {dtype}. Expected one of: float16, bfloat16, float32, etc.")
        torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )

    if device_map is None:
        target_device = device or resolve_device()
        model.to(target_device)

    return model
