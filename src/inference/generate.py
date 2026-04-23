from __future__ import annotations

import argparse

from src.data.tokenizer import BPETokenizer
from src.model.generate import generate_text
from src.model.transformer import build_model
from src.train.checkpoint import load_checkpoint
from src.utils.config import load_full_config
from src.utils.contracts import GenerationConfig
from src.utils.runtime import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Explain sparse routing.")
    args = parser.parse_args()

    config = load_full_config(args.config)
    device = resolve_device(config["runtime"].device)
    model = build_model(config["model"]).to(device)
    tokenizer = BPETokenizer.load(config["data"].tokenizer_path)
    load_checkpoint(args.checkpoint, model, optimizer=None, device=device, strict=False)
    text = generate_text(model, tokenizer, args.prompt, GenerationConfig(), device)
    print(text)


if __name__ == "__main__":
    main()
