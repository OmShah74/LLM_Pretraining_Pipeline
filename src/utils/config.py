from __future__ import annotations

from dataclasses import fields, is_dataclass
import os
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
    runtime = dataclass_from_dict(RuntimeConfig, payload.get("runtime", {}))
    data = dataclass_from_dict(DataConfig, payload.get("data", {}))
    runtime = apply_runtime_env_overrides(runtime)
    data = apply_data_env_overrides(data, runtime)
    return {
        "runtime": runtime,
        "data": data,
        "model": ModelConfig(
            **{
                **payload.get("model", {}),
                "attn": dataclass_from_dict(AttentionConfig, payload.get("model", {}).get("attn", {})),
                "moe": dataclass_from_dict(MoEConfig, payload.get("model", {}).get("moe", {})),
            }
        ),
        "train": dataclass_from_dict(TrainConfig, payload.get("train", {})),
    }


def apply_runtime_env_overrides(runtime: RuntimeConfig) -> RuntimeConfig:
    runtime.project_root = os.environ.get("LLM_PROJECT_ROOT", runtime.project_root)
    runtime.artifact_dir = os.environ.get("LLM_ARTIFACT_DIR", runtime.artifact_dir)
    return runtime


def apply_data_env_overrides(data: DataConfig, runtime: RuntimeConfig) -> DataConfig:
    raw_dir = os.environ.get("LLM_RAW_DIR")
    processed_dir = os.environ.get("LLM_PROCESSED_DIR")
    tokenizer_path = os.environ.get("LLM_TOKENIZER_PATH")
    streaming_cache_dir = os.environ.get("LLM_STREAMING_CACHE_DIR")

    if raw_dir:
        data.raw_dir = raw_dir
    if processed_dir:
        data.processed_dir = processed_dir
    if tokenizer_path:
        data.tokenizer_path = tokenizer_path
    if streaming_cache_dir:
        data.streaming_cache_dir = streaming_cache_dir

    hf_home = os.environ.get("HF_HOME")
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if not streaming_cache_dir:
        if hf_datasets_cache:
            data.streaming_cache_dir = hf_datasets_cache
        elif hf_home:
            data.streaming_cache_dir = str(Path(hf_home) / "datasets")
    return data


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
