from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError

from src.utils.contracts import DataConfig


def _map_record(role: str, record: dict[str, Any], source: dict[str, Any]) -> dict[str, str] | None:
    if role == "pretrain":
        field = source.get("text_field", "text")
        text = record.get(field)
        if not isinstance(text, str):
            return None
        return {"text": text, "source": source["name"]}
    if role == "sft":
        prompt = record.get(source.get("prompt_field", "prompt"))
        response = record.get(source.get("response_field", "response"))
        if not isinstance(prompt, str) or not isinstance(response, str):
            return None
        return {"prompt": prompt, "response": response, "source": source["name"]}
    if role == "dpo":
        prompt = record.get(source.get("prompt_field", "prompt"))
        chosen = record.get(source.get("chosen_field", "chosen"))
        rejected = record.get(source.get("rejected_field", "rejected"))
        if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
            return None
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source["name"]}
    if role == "grpo":
        prompt = record.get(source.get("prompt_field", "prompt"))
        if not isinstance(prompt, str):
            return None
        mapped = {"prompt": prompt, "source": source["name"]}
        if isinstance(record.get(source.get("reference_field", "reference")), str):
            mapped["reference"] = record[source.get("reference_field", "reference")]
        return mapped
    return None


def iter_stream_source(source: dict[str, Any], config: DataConfig) -> Iterable[dict[str, str]]:
    try:
        dataset = load_dataset(
            path=source["path"],
            name=source.get("name_config"),
            split=source.get("split", "train"),
            streaming=True,
            trust_remote_code=bool(source.get("trust_remote_code", False)),
            cache_dir=config.streaming_cache_dir,
        )
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            f"Streaming source '{source['name']}' could not be loaded. "
            f"path={source['path']} name_config={source.get('name_config')} split={source.get('split', 'train')}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Streaming source '{source['name']}' failed during dataset initialization. "
            f"path={source['path']} name_config={source.get('name_config')} split={source.get('split', 'train')} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if source.get("shuffle", False):
        dataset = dataset.shuffle(seed=int(source.get("seed", 42)), buffer_size=int(source.get("shuffle_buffer", 10_000)))
    take = int(source.get("take", 0))
    count = 0
    role = str(source["role"])
    for record in dataset:
        mapped = _map_record(role, record, source)
        if mapped is None:
            continue
        yield mapped
        count += 1
        if take and count >= take:
            break


def get_stream_sources_for_role(config: DataConfig, role: str) -> list[dict[str, Any]]:
    return [source for source in config.stream_sources if source.get("role") == role and source.get("enabled", True)]


def validate_stream_source(source: dict[str, Any], config: DataConfig) -> dict[str, Any]:
    iterator = iter_stream_source(source, config)
    first = next(iterator, None)
    if first is None:
        raise RuntimeError(
            f"Streaming source '{source['name']}' loaded but produced no usable rows. "
            f"path={source['path']} name_config={source.get('name_config')} split={source.get('split', 'train')}"
        )
    return {
        "name": source["name"],
        "role": source["role"],
        "path": source["path"],
        "name_config": source.get("name_config"),
        "split": source.get("split", "train"),
        "sample_keys": sorted(first.keys()),
        "sample_preview": {key: str(value)[:120] for key, value in first.items()},
    }
