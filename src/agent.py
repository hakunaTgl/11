"""Command-line chat interface for a local transformers model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Allow running as script from src directory or as module
_src_dir = Path(__file__).parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from .tokenizer import load_tokenizer
from .utils import load_model, resolve_device


def build_prompt(
    *,
    system_prompt: Optional[str],
    history: List[dict],
    user_input: str,
    include_history: bool,
) -> str:
    """Format a lightweight dialogue prompt."""

    lines = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    if include_history:
        for turn in history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
    lines.append(f"User: {user_input}\nAssistant:")
    return "\n".join(lines)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: Optional[torch.device],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    # Only move to device if device is specified (not using device_map)
    if device is not None:
        inputs = inputs.to(device)
    do_sample = temperature > 0
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text


def interactive_chat(args, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: Optional[torch.device]):
    history: List[dict] = []
    print("Welcome to the local AI agent. Type /exit or /quit to leave.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not user_input:
            continue

        prompt = build_prompt(
            system_prompt=args.system,
            history=history,
            user_input=user_input,
            include_history=not args.no_history,
        )
        response = generate_response(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        history.append({"user": user_input, "assistant": response})
        print(f"Agent: {response}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local uncensored AI agent.")
    parser.add_argument("--model-path", required=True, help="Path or Hugging Face model id to load.")
    parser.add_argument("--prompt", help="Single prompt to run non-interactively.")
    parser.add_argument("--system", default="You are a helpful AI assistant.", help="System prompt injected at the start.")
    parser.add_argument("--device", default=None, help="Force a device (e.g. cpu, cuda, mps).")
    parser.add_argument("--dtype", default="auto", help="Torch dtype name (float16, bfloat16, auto).")
    parser.add_argument("--device-map", default=None, help="Pass 'auto' to let transformers shard the model across devices.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling value.")
    parser.add_argument("--no-history", action="store_true", help="Disable dialogue history in the prompt.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code when loading the model/tokenizer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # When device_map is used, let transformers handle device placement
    if args.device_map:
        target_device = None
    else:
        target_device = torch.device(args.device) if args.device else resolve_device()
    tokenizer = load_tokenizer(args.model_path, trust_remote_code=args.trust_remote_code)
    model = load_model(
        args.model_path,
        device=target_device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
    )

    if args.prompt:
        prompt = build_prompt(
            system_prompt=args.system,
            history=[],
            user_input=args.prompt,
            include_history=False,
        )
        response = generate_response(
            model,
            tokenizer,
            prompt,
            target_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        print(response)
    else:
        interactive_chat(args, model, tokenizer, target_device)


if __name__ == "__main__":
    main()
