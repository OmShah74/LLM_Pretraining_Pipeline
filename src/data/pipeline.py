from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.data.quality import ApproxDeduper, quality_score
from src.data.streaming import get_stream_sources_for_role, iter_stream_source, validate_stream_source
from src.data.tokenizer import BPETokenizer
from src.utils.contracts import DataConfig, DatasetManifest, TokenizerArtifact
from src.utils.io import append_jsonl, ensure_dir, read_json, read_jsonl, write_json, write_jsonl


ROLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "pretrain": ("text",),
    "sft": ("prompt", "response"),
    "dpo": ("prompt", "chosen", "rejected"),
    "grpo": ("prompt",),
}


def _normalize_text(text: str, config: DataConfig) -> str:
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    if config.strip_html:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    if config.lowercase:
        text = text.lower()
    return text.strip()


def _local_sources(config: DataConfig) -> list[tuple[str, str, Path, bool]]:
    raw_dir = ensure_dir(config.raw_dir)
    return [
        ("pretrain_raw", "pretrain", raw_dir / config.pretrain_filename, True),
        ("sft_raw", "sft", raw_dir / config.sft_filename, True),
        ("dpo_raw", "dpo", raw_dir / config.dpo_filename, True),
        ("grpo_raw", "grpo", raw_dir / config.grpo_filename, False),
    ]


def _validate_local_rows(role: str, rows: list[dict[str, object]], source_path: Path) -> None:
    required_fields = ROLE_SCHEMAS[role]
    if not rows:
        raise ValueError(f"{source_path} contains no rows for role '{role}'.")
    for idx, row in enumerate(rows[:10]):
        missing = [field for field in required_fields if not isinstance(row.get(field), str) or not str(row.get(field)).strip()]
        if missing:
            raise ValueError(f"{source_path} row {idx} is missing required fields for role '{role}': {missing}")


def _prepare_local_manifests(config: DataConfig) -> list[DatasetManifest]:
    manifests: list[DatasetManifest] = []
    missing_required: list[str] = []
    for name, role, source_path, required in _local_sources(config):
        if not source_path.exists():
            if required:
                missing_required.append(str(source_path))
            continue
        rows = read_jsonl(source_path)
        _validate_local_rows(role, rows, source_path)
        manifests.append(DatasetManifest(name=name, role=role, path=str(source_path), num_records=len(rows), metadata={"synthetic": False}))
    if missing_required:
        raise FileNotFoundError("Real dataset files are required before training. Expected JSONL files at:\n" + "\n".join(missing_required))
    if not manifests:
        raise FileNotFoundError("No dataset files found in raw_dir. Populate the real dataset JSONL files first.")
    return manifests


def _clean_record(role: str, row: dict[str, str], config: DataConfig) -> dict[str, str] | None:
    cleaned: dict[str, str] = {}
    for field in ROLE_SCHEMAS[role]:
        value = _normalize_text(str(row.get(field, "")), config)
        if not value:
            return None
        cleaned[field] = value
    if role == "pretrain":
        text = cleaned["text"]
        if len(text) < config.min_chars or len(text) > config.max_chars:
            return None
        if quality_score(text, config) < config.quality_threshold:
            return None
    if role in {"sft", "dpo", "grpo"}:
        joined = " ".join(cleaned.values())
        if quality_score(joined, config) < (config.quality_threshold * 0.75):
            return None
    if isinstance(row.get("source"), str):
        cleaned["source"] = _normalize_text(row["source"], config)
    if isinstance(row.get("reference"), str):
        cleaned["reference"] = _normalize_text(row["reference"], config)
    return cleaned


def _stream_and_clean_role(role: str, config: DataConfig) -> DatasetManifest | None:
    sources = get_stream_sources_for_role(config, role)
    if not sources:
        return None
    processed_dir = ensure_dir(config.processed_dir)
    output_path = processed_dir / f"{role}_stream_cleaned.jsonl"
    partial_output_path = processed_dir / f"{role}_stream_cleaned.partial.jsonl"
    progress_path = processed_dir / f"{role}_stream_progress.json"
    if partial_output_path.exists():
        partial_output_path.unlink()
    if output_path.exists():
        output_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    deduper = ApproxDeduper(bits=config.simhash_bits, bands=config.simhash_bands)
    rows_written = 0
    max_records = int(config.max_records_per_role.get(role, 0))
    pending_rows: list[dict[str, str]] = []
    total_seen = 0
    total_cleaned = 0
    total_filtered = 0
    total_duplicates = 0
    progress_every = max(int(config.progress_every_rows), 1)
    checkpoint_every = max(int(config.checkpoint_every_rows), 1)
    flush_every = max(int(config.flush_every_rows), 1)
    source_stats: list[dict[str, int | str]] = []

    def flush_pending() -> None:
        nonlocal pending_rows
        if pending_rows:
            append_jsonl(partial_output_path, pending_rows)
            pending_rows = []

    def write_progress(current_source: str, source_seen: int, source_written: int, source_filtered: int, source_duplicates: int) -> None:
        write_json(
            progress_path,
            {
                "role": role,
                "current_source": current_source,
                "rows_seen": total_seen,
                "rows_cleaned": total_cleaned,
                "rows_written": rows_written,
                "rows_filtered": total_filtered,
                "rows_duplicates": total_duplicates,
                "partial_output_path": str(partial_output_path),
                "completed_output_path": str(output_path),
                "source_seen": source_seen,
                "source_written": source_written,
                "source_filtered": source_filtered,
                "source_duplicates": source_duplicates,
            },
        )

    for source in sources:
        print(f"[data.prepare] streaming role={role} source={source['name']} path={source['path']} split={source.get('split', 'train')}")
        source_seen = 0
        source_written = 0
        source_filtered = 0
        source_duplicates = 0
        for row in iter_stream_source(source, config):
            total_seen += 1
            source_seen += 1
            cleaned = _clean_record(role, row, config)
            if cleaned is None:
                total_filtered += 1
                source_filtered += 1
                if total_seen % progress_every == 0:
                    print(
                        f"[data.prepare] progress role={role} source={source['name']} "
                        f"seen={total_seen} written={rows_written} filtered={total_filtered} duplicates={total_duplicates}"
                    )
                continue
            total_cleaned += 1
            dedup_text = " ".join(cleaned[field] for field in ROLE_SCHEMAS[role])
            if config.dedup and deduper.seen(dedup_text):
                total_duplicates += 1
                source_duplicates += 1
                if total_seen % progress_every == 0:
                    print(
                        f"[data.prepare] progress role={role} source={source['name']} "
                        f"seen={total_seen} written={rows_written} filtered={total_filtered} duplicates={total_duplicates}"
                    )
                continue
            pending_rows.append(cleaned)
            rows_written += 1
            source_written += 1
            if len(pending_rows) >= flush_every:
                flush_pending()
            if rows_written % checkpoint_every == 0:
                flush_pending()
                write_progress(source["name"], source_seen, source_written, source_filtered, source_duplicates)
                print(
                    f"[data.prepare] checkpoint role={role} source={source['name']} "
                    f"written={rows_written} partial_output={partial_output_path}"
                )
            elif total_seen % progress_every == 0:
                print(
                    f"[data.prepare] progress role={role} source={source['name']} "
                    f"seen={total_seen} written={rows_written} filtered={total_filtered} duplicates={total_duplicates}"
                )
            if max_records and rows_written >= max_records:
                break
        source_stats.append(
            {
                "name": str(source["name"]),
                "seen": source_seen,
                "written": source_written,
                "filtered": source_filtered,
                "duplicates": source_duplicates,
            }
        )
        flush_pending()
        write_progress(source["name"], source_seen, source_written, source_filtered, source_duplicates)
        print(
            f"[data.prepare] source-complete role={role} source={source['name']} "
            f"seen={source_seen} written={source_written} filtered={source_filtered} duplicates={source_duplicates}"
        )
        if max_records and rows_written >= max_records:
            break

    flush_pending()
    if rows_written == 0:
        return None
    partial_output_path.replace(output_path)
    manifest = DatasetManifest(
        name=f"{role}_stream_cleaned",
        role=role,
        path=str(output_path),
        num_records=rows_written,
        metadata={"streaming": True, "sources": [source["name"] for source in sources], "source_stats": source_stats},
    )
    write_json(processed_dir / f"{manifest.name}_manifest.json", manifest.to_dict())
    if progress_path.exists():
        progress_path.unlink()
    print(f"[data.prepare] completed role={role} rows={rows_written} output={output_path}")
    return manifest


def validate_stream_sources(config: DataConfig, roles: list[str] | None = None) -> dict[str, object]:
    requested_roles = set(roles or [source.get("role") for source in config.stream_sources if source.get("enabled", True)])
    reports: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for source in config.stream_sources:
        if not source.get("enabled", True):
            continue
        role = str(source.get("role"))
        if role not in requested_roles:
            continue
        try:
            report = validate_stream_source(source, config)
            reports.append(report)
        except Exception as exc:
            errors.append(
                {
                    "name": str(source.get("name")),
                    "role": role,
                    "path": str(source.get("path")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not reports and not errors:
        raise ValueError("No enabled streaming sources matched the requested roles.")
    summary = {"validated_sources": reports, "errors": errors}
    if errors:
        joined = "\n".join(
            f"- source={item['name']} role={item['role']} path={item['path']} error={item['error']}"
            for item in errors
        )
        raise RuntimeError(f"Some streaming sources failed validation:\n{joined}")
    return summary


def prepare_raw_sources(config: DataConfig) -> list[DatasetManifest]:
    if config.use_streaming_sources and config.stream_sources:
        manifests: list[DatasetManifest] = []
        for role in ("pretrain", "sft", "dpo", "grpo"):
            manifest = _stream_and_clean_role(role, config)
            if manifest is not None:
                manifests.append(manifest)
        required_roles = {"pretrain", "sft", "dpo"}
        found_roles = {manifest.role for manifest in manifests}
        missing_roles = sorted(required_roles - found_roles)
        if missing_roles:
            raise FileNotFoundError(f"Streaming extraction did not yield required roles: {missing_roles}")
        return manifests
    return _prepare_local_manifests(config)


def clean_corpus(input_manifest: DatasetManifest, config: DataConfig) -> DatasetManifest:
    if input_manifest.metadata.get("streaming", False):
        return input_manifest

    processed_dir = ensure_dir(config.processed_dir)
    rows = read_jsonl(input_manifest.path)
    cleaned: list[dict[str, str]] = []
    deduper = ApproxDeduper(bits=config.simhash_bits, bands=config.simhash_bands)
    for row in rows:
        cleaned_row = _clean_record(input_manifest.role, row, config)
        if cleaned_row is None:
            continue
        dedup_text = " ".join(cleaned_row[field] for field in ROLE_SCHEMAS[input_manifest.role])
        if config.dedup and deduper.seen(dedup_text):
            continue
        cleaned.append(cleaned_row)

    if not cleaned:
        raise ValueError(f"No valid rows remained after cleaning {input_manifest.path}.")
    output_path = processed_dir / f"{input_manifest.name}_cleaned.jsonl"
    write_jsonl(output_path, cleaned)
    manifest = DatasetManifest(
        name=f"{input_manifest.name}_cleaned",
        role=input_manifest.role,
        path=str(output_path),
        num_records=len(cleaned),
        metadata={"input": input_manifest.path},
    )
    write_json(processed_dir / f"{manifest.name}_manifest.json", manifest.to_dict())
    return manifest


def train_or_load_tokenizer(config: DataConfig, manifests: list[DatasetManifest]) -> TokenizerArtifact:
    tokenizer_path = Path(config.tokenizer_path)
    tokenizer: BPETokenizer | None = None
    if tokenizer_path.exists():
        try:
            tokenizer = BPETokenizer.load(tokenizer_path)
        except Exception:
            tokenizer_path.unlink()
    if tokenizer is None:
        texts: list[str] = []
        for manifest in manifests:
            rows = read_jsonl(manifest.path)
            if manifest.role == "pretrain":
                texts.extend(row["text"] for row in rows)
            elif manifest.role == "sft":
                texts.extend(f"Instruction: {row['prompt']} Response: {row['response']}" for row in rows)
            elif manifest.role == "dpo":
                for row in rows:
                    texts.extend(
                        [
                            f"Instruction: {row['prompt']} Response: {row['chosen']}",
                            f"Instruction: {row['prompt']} Response: {row['rejected']}",
                        ]
                    )
            elif manifest.role == "grpo":
                texts.extend(f"Instruction: {row['prompt']} Response:" for row in rows)
        tokenizer = BPETokenizer.train(
            texts,
            max_vocab_size=config.max_vocab_size,
            min_frequency=config.min_frequency,
            lowercase=config.lowercase,
        )
        tokenizer.save(tokenizer_path)
    artifact = TokenizerArtifact(
        path=str(tokenizer_path),
        vocab_size=len(tokenizer.token_to_id),
        special_tokens={
            "pad": tokenizer.token_to_id[tokenizer.pad_token],
            "bos": tokenizer.token_to_id[tokenizer.bos_token],
            "eos": tokenizer.token_to_id[tokenizer.eos_token],
            "unk": tokenizer.token_to_id[tokenizer.unk_token],
        },
    )
    write_json(tokenizer_path.parent / "manifest.json", artifact.to_dict())
    return artifact


def tokenize_and_pack(manifest: DatasetManifest, tokenizer_path: str, config: DataConfig) -> DatasetManifest:
    tokenizer = BPETokenizer.load(tokenizer_path)
    rows = read_jsonl(manifest.path)
    processed_dir = ensure_dir(config.processed_dir)
    output_path = processed_dir / f"{manifest.name}_packed.json"
    sequences: list[list[int]] = []
    if manifest.role == "pretrain":
        sequences = [tokenizer.encode(row["text"], add_bos=True, add_eos=True) for row in rows]
    elif manifest.role == "sft":
        sequences = [
            tokenizer.encode(f"Instruction: {row['prompt']} Response: {row['response']}", add_bos=True, add_eos=True)
            for row in rows
        ]
    elif manifest.role == "dpo":
        for row in rows:
            sequences.append(tokenizer.encode(f"Instruction: {row['prompt']} Response: {row['chosen']}", add_bos=True, add_eos=True))
            sequences.append(tokenizer.encode(f"Instruction: {row['prompt']} Response: {row['rejected']}", add_bos=True, add_eos=True))
    elif manifest.role == "grpo":
        sequences = [tokenizer.encode(f"Instruction: {row['prompt']} Response:", add_bos=True, add_eos=False) for row in rows]
    write_json(output_path, {"sequences": sequences, "tokenizer_path": tokenizer_path, "role": manifest.role})
    packed_manifest = DatasetManifest(
        name=f"{manifest.name}_packed",
        role=manifest.role,
        path=str(output_path),
        num_records=len(sequences),
        metadata={"tokenizer_path": tokenizer_path},
    )
    write_json(processed_dir / f"{packed_manifest.name}_manifest.json", packed_manifest.to_dict())
    return packed_manifest


def build_mixture(manifests: list[DatasetManifest], output_path: str) -> DatasetManifest:
    rows = [manifest.to_dict() for manifest in manifests]
    write_json(output_path, {"manifests": rows})
    return DatasetManifest("mixture_manifest", "meta", output_path, len(rows), {})


def run_data_prep(config: DataConfig) -> dict[str, object]:
    raw_manifests = prepare_raw_sources(config)
    cleaned = [clean_corpus(manifest, config) for manifest in raw_manifests]
    tokenizer = train_or_load_tokenizer(config, cleaned)
    packed = [tokenize_and_pack(manifest, tokenizer.path, config) for manifest in cleaned]
    mixture = build_mixture(cleaned, str(Path(config.processed_dir) / "mixture_manifest.json"))
    return {"raw": raw_manifests, "cleaned": cleaned, "tokenizer": tokenizer, "packed": packed, "mixture": mixture}


def load_packed_sequences(path: str) -> list[list[int]]:
    return read_json(path)["sequences"]
