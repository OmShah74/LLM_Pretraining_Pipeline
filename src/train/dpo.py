from __future__ import annotations

import argparse

from pathlib import Path

from src.train.checkpoint import load_checkpoint, load_latest_metadata
from src.train.common import build_dpo_dataloaders, load_tokenizer, setup_run, train_dpo
from src.utils.config import load_full_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPO.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_full_config(args.config)
    device, model = setup_run(args.config, config["runtime"], config["model"], config["train"])
    if config["train"].resume_from is None:
        latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", "sft")
        if latest is None:
            latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", "pretrain")
        if latest is not None:
            load_checkpoint(latest.path, model, optimizer=None, device=device, strict=False)
    tokenizer = load_tokenizer(config["data"])
    train_loader, eval_loader = build_dpo_dataloaders(config["data"], tokenizer, config["train"])
    train_dpo(
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
