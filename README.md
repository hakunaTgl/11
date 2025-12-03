# Uncensored AI Agent

A lightweight command-line agent for experimenting with local transformer models. The scripts keep the footprint minimal while providing interactive chat, prompt history, and tunable generation settings.

## Features
- CLI chat loop with `/exit` and `/quit` shortcuts
- Optional single-prompt invocation for scripting
- System prompt injection and history-aware context building
- Configurable generation controls (temperature, top-p, top-k, max tokens)
- Device selection with optional automatic sharding via `--device-map auto`

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install torch transformers
   ```

2. Download or reference a model (local directory or Hugging Face ID). Example using a GGUF-converted model is not supported directly by transformers; prefer a standard transformers checkpoint such as a LLaMA or Mistral variant.

## Usage
Run an interactive session:
```bash
python src/agent.py --model-path <model-id-or-path>
```

Run a single prompt non-interactively:
```bash
python src/agent.py --model-path <model-id-or-path> --prompt "Write a haiku about the ocean" --max-new-tokens 128
```

Key flags:
- `--system` – custom system prompt
- `--device` – force a device (`cpu`, `cuda`, `mps`)
- `--dtype` – precision hint (`float16`, `bfloat16`, `auto`)
- `--device-map auto` – let transformers shard across available hardware
- `--no-history` – disable conversation history in prompts
- `--trust-remote-code` – allow custom model/tokenizer code if required

## Notes
- Large models may require `--device-map auto` or quantized checkpoints to fit on your hardware.
- Ensure `pad_token_id` and `eos_token_id` are set on your tokenizer if the model defines custom tokens.
