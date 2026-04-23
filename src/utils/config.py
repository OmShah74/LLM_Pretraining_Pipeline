from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from src.utils.contracts import AttentionConfig, DataConfig, ModelConfig, MoEConfig, RuntimeConfig, TrainConfig

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    values: dict[str, Any] = {}
    for field_def in fields(cls):
        field_value = payload.get(field_def.name)
        if hasattr(field_def.type, "__dataclass_fields__") and isinstance(field_value, dict):
            values[field_def.name] = dataclass_from_dict(field_def.type, field_value)
        else:
            values[field_def.name] = field_value if field_value is not None else field_def.default
    return cls(**values)


def load_full_config(path: str | Path) -> dict[str, Any]:
    payload = load_yaml(path)
    return {
        "runtime": dataclass_from_dict(RuntimeConfig, payload.get("runtime", {})),
        "data": dataclass_from_dict(DataConfig, payload.get("data", {})),
        "model": ModelConfig(
            **{
                **payload.get("model", {}),
                "attn": dataclass_from_dict(AttentionConfig, payload.get("model", {}).get("attn", {})),
                "moe": dataclass_from_dict(MoEConfig, payload.get("model", {}).get("moe", {})),
            }
        ),
        "train": dataclass_from_dict(TrainConfig, payload.get("train", {})),
    }


def save_effective_config(path: str | Path, config: dict[str, Any]) -> None:
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if is_dataclass(value):
            normalized[key] = {field.name: getattr(value, field.name) for field in fields(value)}
            if key == "model":
                normalized[key]["attn"] = {
                    field.name: getattr(value.attn, field.name) for field in fields(value.attn)
                }
                normalized[key]["moe"] = {
                    field.name: getattr(value.moe, field.name) for field in fields(value.moe)
                }
        else:
            normalized[key] = value
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(normalized, handle, sort_keys=False)
