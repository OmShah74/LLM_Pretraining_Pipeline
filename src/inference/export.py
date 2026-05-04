from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.train.checkpoint import load_latest_metadata
from src.utils.config import load_full_config
from src.utils.io import ensure_dir, write_json
from src.utils.logging import PipelineLogger


def run_export(config_path: str, stage: str = "dpo") -> dict[str, str]:
    config = load_full_config(config_path)
    logger = PipelineLogger(config["runtime"].log_dir, "export")
    export_dir = ensure_dir(Path(config["runtime"].artifact_dir) / "release")
    logger.event("export", "starting export", export_stage=stage, export_dir=str(export_dir))
    latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", stage)
    if latest is None:
        latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", "pretrain")
    if latest is None:
        raise FileNotFoundError("No checkpoint metadata found for export.")
    logger.event("export", "using checkpoint", export_stage=stage, checkpoint=latest.path, source_stage=latest.stage)
    checkpoint_target = export_dir / Path(latest.path).name
    shutil.copyfile(latest.path, checkpoint_target)
    tokenizer_target = export_dir / Path(config["data"].tokenizer_path).name
    shutil.copyfile(config["data"].tokenizer_path, tokenizer_target)
    model_card_path = export_dir / "model_card.json"
    payload = {
        "checkpoint": str(checkpoint_target),
        "tokenizer": str(tokenizer_target),
        "stage": latest.stage,
        "profile": config["runtime"].profile_name,
        "notes": "Prototype export for the MoE LLM pipeline.",
    }
    write_json(model_card_path, payload)
    logger.event("export", "complete", export_stage=latest.stage, model_card=str(model_card_path), checkpoint=str(checkpoint_target))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="dpo")
    args = parser.parse_args()
    run_export(args.config, args.stage)


if __name__ == "__main__":
    main()
