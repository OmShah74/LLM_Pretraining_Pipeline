from __future__ import annotations

from pathlib import Path

import torch

from src.utils.contracts import CheckpointMetadata
from src.utils.io import ensure_dir, read_json, write_json


def save_checkpoint(
    output_dir: str | Path,
    stage: str,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_path: str,
    tokenizer_path: str,
    metrics: dict[str, float],
) -> CheckpointMetadata:
    output_path = ensure_dir(output_dir)
    ckpt_path = output_path / f"{stage}_step_{step}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "stage": stage,
        },
        ckpt_path,
    )
    metadata = CheckpointMetadata(
        path=str(ckpt_path),
        stage=stage,
        step=step,
        epoch=epoch,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        metrics=metrics,
    )
    write_json(output_path / f"{stage}_latest.json", metadata.to_dict())
    return metadata


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, int | str]:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {"step": int(payload.get("step", 0)), "epoch": int(payload.get("epoch", 0)), "stage": str(payload.get("stage", ""))}


def load_latest_metadata(directory: str | Path, stage: str) -> CheckpointMetadata | None:
    metadata_path = Path(directory) / f"{stage}_latest.json"
    if not metadata_path.exists():
        return None
    payload = read_json(metadata_path)
    return CheckpointMetadata(**payload)
