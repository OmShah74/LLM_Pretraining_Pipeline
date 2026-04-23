from __future__ import annotations

import argparse

from src.train.common import build_pretrain_dataloaders, load_tokenizer, setup_run, train_language_model
from src.utils.config import load_full_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretraining.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_full_config(args.config)
    device, model = setup_run(args.config, config["runtime"], config["model"], config["train"])
    tokenizer = load_tokenizer(config["data"])
    train_loader, eval_loader = build_pretrain_dataloaders(config["data"], tokenizer, config["train"])
    train_language_model(
        stage="pretrain",
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
        runtime_config=config["runtime"],
        train_config=config["train"],
        tokenizer_path=config["data"].tokenizer_path,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
