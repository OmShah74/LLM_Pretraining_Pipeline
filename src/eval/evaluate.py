from __future__ import annotations

import argparse
from pathlib import Path

from src.model.generate import generate_text
from src.train.checkpoint import load_checkpoint, load_latest_metadata
from src.train.common import build_pretrain_dataloaders, evaluate_language_model, load_tokenizer, setup_run
from src.utils.config import load_full_config
from src.utils.contracts import EvalReport, GenerationConfig
from src.utils.io import write_json


def run_eval(config_path: str, stage: str = "pretrain") -> EvalReport:
    config = load_full_config(config_path)
    device, model = setup_run(config_path, config["runtime"], config["model"], config["train"])
    tokenizer = load_tokenizer(config["data"])
    _, eval_loader = build_pretrain_dataloaders(config["data"], tokenizer, config["train"])
    latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", stage)
    if latest is not None:
        load_checkpoint(latest.path, model, optimizer=None, device=device, strict=False)
    metrics = evaluate_language_model(model, eval_loader, device)
    sample = generate_text(model, tokenizer, "Explain mixture of experts", GenerationConfig(), device)
    report = EvalReport(
        path=str(Path(config["runtime"].artifact_dir) / f"{stage}_eval_report.json"),
        stage=stage,
        metrics=metrics,
        samples=[{"prompt": "Explain mixture of experts", "output": sample}],
    )
    write_json(report.path, report.to_dict())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="pretrain")
    args = parser.parse_args()
    run_eval(args.config, args.stage)


if __name__ == "__main__":
    main()
