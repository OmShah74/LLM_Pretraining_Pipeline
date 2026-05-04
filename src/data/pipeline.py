from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.data.quality import ApproxDeduper, quality_score
from src.data.streaming import get_stream_sources_for_role, iter_stream_source, validate_stream_source
from src.data.tokenizer import BPETokenizer
from src.utils.contracts import DataConfig, DatasetManifest, TokenizerArtifact
from src.utils.io import append_jsonl, count_jsonl, ensure_dir, iter_jsonl, read_json, write_json, write_jsonl
from src.utils.logging import PipelineLogger


ROLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "pretrain": ("text",),
    "sft": ("prompt", "response"),
    "dpo": ("prompt", "chosen", "rejected"),
    "grpo": ("prompt",),
}


def _make_data_logger(config: DataConfig, run_name: str = "data_prep") -> PipelineLogger:
    log_dir = Path(config.log_dir)
    if config.log_dir == "artifacts/logs" and Path(config.processed_dir) != Path("artifacts/data/processed"):
        log_dir = Path(config.processed_dir).parent / "logs"
    return PipelineLogger(log_dir, run_name)


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
        sample_rows = []
        for row in iter_jsonl(source_path):
            sample_rows.append(row)
            if len(sample_rows) >= 10:
                break
        _validate_local_rows(role, sample_rows, source_path)
        manifests.append(DatasetManifest(name=name, role=role, path=str(source_path), num_records=count_jsonl(source_path), metadata={"synthetic": False}))
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


def _download_role_sources(role: str, config: DataConfig, logger: PipelineLogger) -> DatasetManifest | None:
    sources = get_stream_sources_for_role(config, role)
    if not sources:
        return None
    raw_dir = ensure_dir(config.raw_dir)
    output_path = raw_dir / getattr(config, f"{role}_filename")
    partial_output_path = raw_dir / f"{role}.download.partial.jsonl"
    progress_path = raw_dir / f"{role}_download_progress.json"
    if output_path.exists():
        rows = count_jsonl(output_path)
        logger.event("data.download", "reusing existing raw dataset", role=role, rows=rows, output=str(output_path))
        return DatasetManifest(
            name=f"{role}_raw",
            role=role,
            path=str(output_path),
            num_records=rows,
            metadata={"downloaded": True, "reused": True, "sources": [source["name"] for source in sources]},
        )
    if partial_output_path.exists():
        partial_output_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    rows_written = 0
    max_records = int(config.max_records_per_role.get(role, 0))
    pending_rows: list[dict[str, str]] = []
    total_seen = 0
    progress_every = max(int(config.progress_every_rows), 1)
    checkpoint_every = max(int(config.checkpoint_every_rows), 1)
    flush_every = max(int(config.flush_every_rows), 1)
    source_stats: list[dict[str, int | str]] = []

    def flush_pending() -> None:
        nonlocal pending_rows
        if pending_rows:
            append_jsonl(partial_output_path, pending_rows)
            pending_rows = []

    def write_progress(current_source: str, source_seen: int, source_written: int) -> None:
        write_json(
            progress_path,
            {
                "role": role,
                "stage": "download",
                "current_source": current_source,
                "rows_seen": total_seen,
                "rows_written": rows_written,
                "partial_output_path": str(partial_output_path),
                "completed_output_path": str(output_path),
                "source_seen": source_seen,
                "source_written": source_written,
            },
        )

    for source in sources:
        logger.event(
            "data.download",
            "starting source download/materialization",
            role=role,
            source=source["name"],
            path=source["path"],
            split=source.get("split", "train"),
        )
        source_seen = 0
        source_written = 0
        for row in iter_stream_source(source, config):
            total_seen += 1
            source_seen += 1
            pending_rows.append(row)
            rows_written += 1
            source_written += 1
            if len(pending_rows) >= flush_every:
                flush_pending()
            if rows_written % checkpoint_every == 0:
                flush_pending()
                write_progress(source["name"], source_seen, source_written)
                logger.event(
                    "data.download",
                    "checkpoint",
                    role=role,
                    source=source["name"],
                    seen=total_seen,
                    written=rows_written,
                    partial_output=str(partial_output_path),
                )
            elif total_seen % progress_every == 0:
                logger.event("data.download", "progress", role=role, source=source["name"], seen=total_seen, written=rows_written)
            if max_records and rows_written >= max_records:
                break
        source_stats.append(
            {
                "name": str(source["name"]),
                "seen": source_seen,
                "written": source_written,
            }
        )
        flush_pending()
        write_progress(source["name"], source_seen, source_written)
        logger.event(
            "data.download",
            "source complete",
            role=role,
            source=source["name"],
            seen=source_seen,
            written=source_written,
        )
        if max_records and rows_written >= max_records:
            break

    flush_pending()
    if rows_written == 0:
        return None
    partial_output_path.replace(output_path)
    manifest = DatasetManifest(
        name=f"{role}_raw",
        role=role,
        path=str(output_path),
        num_records=rows_written,
        metadata={
            "downloaded": True,
            "sources": [source["name"] for source in sources],
            "source_stats": source_stats,
            "hf_cache_dir": config.streaming_cache_dir,
        },
    )
    write_json(raw_dir / f"{manifest.name}_manifest.json", manifest.to_dict())
    if progress_path.exists():
        progress_path.unlink()
    logger.event("data.download", "role complete", role=role, rows=rows_written, output=str(output_path))
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
        raise ValueError("No enabled Hugging Face sources matched the requested roles.")
    summary = {"validated_sources": reports, "errors": errors}
    if errors:
        joined = "\n".join(
            f"- source={item['name']} role={item['role']} path={item['path']} error={item['error']}"
            for item in errors
        )
        raise RuntimeError(f"Some Hugging Face sources failed validation:\n{joined}")
    return summary


def prepare_raw_sources(config: DataConfig, logger: PipelineLogger | None = None) -> list[DatasetManifest]:
    logger = logger or _make_data_logger(config)
    if config.use_streaming_sources and config.stream_sources:
        manifests: list[DatasetManifest] = []
        for role in ("pretrain", "sft", "dpo", "grpo"):
            manifest = _download_role_sources(role, config, logger)
            if manifest is not None:
                manifests.append(manifest)
        required_roles = {"pretrain", "sft", "dpo"}
        found_roles = {manifest.role for manifest in manifests}
        missing_roles = sorted(required_roles - found_roles)
        if missing_roles:
            raise FileNotFoundError(f"Hugging Face source materialization did not yield required roles: {missing_roles}")
        return manifests
    return _prepare_local_manifests(config)


def clean_corpus(input_manifest: DatasetManifest, config: DataConfig, logger: PipelineLogger | None = None) -> DatasetManifest:
    logger = logger or _make_data_logger(config)
    processed_dir = ensure_dir(config.processed_dir)
    output_path = processed_dir / f"{input_manifest.name}_cleaned.jsonl"
    partial_output_path = processed_dir / f"{input_manifest.name}_cleaned.partial.jsonl"
    progress_path = processed_dir / f"{input_manifest.name}_clean_progress.json"
    if partial_output_path.exists():
        partial_output_path.unlink()
    if output_path.exists():
        output_path.unlink()
    if progress_path.exists():
        progress_path.unlink()
    deduper = ApproxDeduper(bits=config.simhash_bits, bands=config.simhash_bands)
    progress_every = max(int(config.progress_every_rows), 1)
    checkpoint_every = max(int(config.checkpoint_every_rows), 1)
    flush_every = max(int(config.flush_every_rows), 1)
    pending_rows: list[dict[str, str]] = []
    rows_seen = 0
    rows_cleaned = 0
    rows_filtered = 0
    rows_duplicates = 0

    def flush_pending() -> None:
        nonlocal pending_rows
        if pending_rows:
            append_jsonl(partial_output_path, pending_rows)
            pending_rows = []

    logger.event("data.clean", "starting clean", role=input_manifest.role, input=input_manifest.path, output=str(output_path))
    for row in iter_jsonl(input_manifest.path):
        rows_seen += 1
        cleaned_row = _clean_record(input_manifest.role, row, config)
        if cleaned_row is None:
            rows_filtered += 1
            if rows_seen % progress_every == 0:
                logger.event(
                    "data.clean",
                    "progress",
                    role=input_manifest.role,
                    seen=rows_seen,
                    cleaned=rows_cleaned,
                    filtered=rows_filtered,
                    duplicates=rows_duplicates,
                )
            continue
        dedup_text = " ".join(cleaned_row[field] for field in ROLE_SCHEMAS[input_manifest.role])
        if config.dedup and deduper.seen(dedup_text):
            rows_duplicates += 1
            if rows_seen % progress_every == 0:
                logger.event(
                    "data.clean",
                    "progress",
                    role=input_manifest.role,
                    seen=rows_seen,
                    cleaned=rows_cleaned,
                    filtered=rows_filtered,
                    duplicates=rows_duplicates,
                )
            continue
        pending_rows.append(cleaned_row)
        rows_cleaned += 1
        if len(pending_rows) >= flush_every:
            flush_pending()
        if rows_cleaned % checkpoint_every == 0:
            flush_pending()
            write_json(
                progress_path,
                {
                    "role": input_manifest.role,
                    "stage": "clean",
                    "rows_seen": rows_seen,
                    "rows_cleaned": rows_cleaned,
                    "rows_filtered": rows_filtered,
                    "rows_duplicates": rows_duplicates,
                    "partial_output_path": str(partial_output_path),
                    "completed_output_path": str(output_path),
                },
            )
            logger.event(
                "data.clean",
                "checkpoint",
                role=input_manifest.role,
                seen=rows_seen,
                cleaned=rows_cleaned,
                filtered=rows_filtered,
                duplicates=rows_duplicates,
                partial_output=str(partial_output_path),
            )
        elif rows_seen % progress_every == 0:
            logger.event(
                "data.clean",
                "progress",
                role=input_manifest.role,
                seen=rows_seen,
                cleaned=rows_cleaned,
                filtered=rows_filtered,
                duplicates=rows_duplicates,
            )

    flush_pending()
    if rows_cleaned == 0:
        raise ValueError(f"No valid rows remained after cleaning {input_manifest.path}.")
    partial_output_path.replace(output_path)
    manifest = DatasetManifest(
        name=f"{input_manifest.name}_cleaned",
        role=input_manifest.role,
        path=str(output_path),
        num_records=rows_cleaned,
        metadata={
            "input": input_manifest.path,
            "rows_seen": rows_seen,
            "rows_filtered": rows_filtered,
            "rows_duplicates": rows_duplicates,
        },
    )
    write_json(processed_dir / f"{manifest.name}_manifest.json", manifest.to_dict())
    if progress_path.exists():
        progress_path.unlink()
    logger.event(
        "data.clean",
        "complete",
        role=input_manifest.role,
        seen=rows_seen,
        cleaned=rows_cleaned,
        filtered=rows_filtered,
        duplicates=rows_duplicates,
        output=str(output_path),
    )
    return manifest


def train_or_load_tokenizer(config: DataConfig, manifests: list[DatasetManifest], logger: PipelineLogger | None = None) -> TokenizerArtifact:
    logger = logger or _make_data_logger(config)
    tokenizer_path = Path(config.tokenizer_path)
    tokenizer: BPETokenizer | None = None
    if tokenizer_path.exists():
        try:
            tokenizer = BPETokenizer.load(tokenizer_path)
            logger.event("data.tokenizer", "loaded existing tokenizer", path=str(tokenizer_path), vocab_size=len(tokenizer.token_to_id))
        except Exception:
            tokenizer_path.unlink()
    if tokenizer is None:
        def iter_texts():
            for manifest in manifests:
                logger.event("data.tokenizer", "streaming training texts", role=manifest.role, rows=manifest.num_records, path=manifest.path)
                for row in iter_jsonl(manifest.path):
                    if manifest.role == "pretrain":
                        yield row["text"]
                    elif manifest.role == "sft":
                        yield f"Instruction: {row['prompt']} Response: {row['response']}"
                    elif manifest.role == "dpo":
                        yield f"Instruction: {row['prompt']} Response: {row['chosen']}"
                        yield f"Instruction: {row['prompt']} Response: {row['rejected']}"
                    elif manifest.role == "grpo":
                        yield f"Instruction: {row['prompt']} Response:"
        tokenizer = BPETokenizer.train(
            iter_texts(),
            max_vocab_size=config.max_vocab_size,
            min_frequency=config.min_frequency,
            lowercase=config.lowercase,
        )
        tokenizer.save(tokenizer_path)
        logger.event("data.tokenizer", "trained tokenizer", path=str(tokenizer_path), vocab_size=len(tokenizer.token_to_id))
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


def tokenize_and_pack(
    manifest: DatasetManifest,
    tokenizer_path: str,
    config: DataConfig,
    logger: PipelineLogger | None = None,
) -> DatasetManifest:
    logger = logger or _make_data_logger(config)
    tokenizer = BPETokenizer.load(tokenizer_path)
    processed_dir = ensure_dir(config.processed_dir)
    output_path = processed_dir / f"{manifest.name}_packed.jsonl"
    partial_output_path = processed_dir / f"{manifest.name}_packed.partial.jsonl"
    if partial_output_path.exists():
        partial_output_path.unlink()
    if output_path.exists():
        output_path.unlink()
    pending_rows: list[dict[str, list[int]]] = []
    sequences_written = 0
    progress_every = max(int(config.progress_every_rows), 1)
    flush_every = max(int(config.flush_every_rows), 1)
    logger.event("data.pack", "starting tokenize/pack", role=manifest.role, rows=manifest.num_records, input=manifest.path, output=str(output_path))

    def append_tokens(tokens: list[int]) -> None:
        nonlocal sequences_written, pending_rows
        if len(tokens) < 2:
            return
        pending_rows.append({"tokens": tokens})
        sequences_written += 1
        if len(pending_rows) >= flush_every:
            append_jsonl(partial_output_path, pending_rows)
            pending_rows = []

    def flush_pending() -> None:
        nonlocal pending_rows
        if pending_rows:
            append_jsonl(partial_output_path, pending_rows)
            pending_rows = []

    if manifest.role == "pretrain":
        for idx, row in enumerate(iter_jsonl(manifest.path), start=1):
            append_tokens(tokenizer.encode(row["text"], add_bos=True, add_eos=True))
            if idx % progress_every == 0:
                logger.event("data.pack", "progress", role=manifest.role, rows=idx, sequences=sequences_written)
    elif manifest.role == "sft":
        for idx, row in enumerate(iter_jsonl(manifest.path), start=1):
            append_tokens(tokenizer.encode(f"Instruction: {row['prompt']} Response: {row['response']}", add_bos=True, add_eos=True))
            if idx % progress_every == 0:
                logger.event("data.pack", "progress", role=manifest.role, rows=idx, sequences=sequences_written)
    elif manifest.role == "dpo":
        for idx, row in enumerate(iter_jsonl(manifest.path), start=1):
            append_tokens(tokenizer.encode(f"Instruction: {row['prompt']} Response: {row['chosen']}", add_bos=True, add_eos=True))
            append_tokens(tokenizer.encode(f"Instruction: {row['prompt']} Response: {row['rejected']}", add_bos=True, add_eos=True))
            if idx % progress_every == 0:
                logger.event("data.pack", "progress", role=manifest.role, rows=idx, sequences=sequences_written)
    elif manifest.role == "grpo":
        for idx, row in enumerate(iter_jsonl(manifest.path), start=1):
            append_tokens(tokenizer.encode(f"Instruction: {row['prompt']} Response:", add_bos=True, add_eos=False))
            if idx % progress_every == 0:
                logger.event("data.pack", "progress", role=manifest.role, rows=idx, sequences=sequences_written)
    flush_pending()
    partial_output_path.replace(output_path)
    packed_manifest = DatasetManifest(
        name=f"{manifest.name}_packed",
        role=manifest.role,
        path=str(output_path),
        num_records=sequences_written,
        metadata={"tokenizer_path": tokenizer_path},
    )
    write_json(processed_dir / f"{packed_manifest.name}_manifest.json", packed_manifest.to_dict())
    logger.event("data.pack", "complete", role=manifest.role, sequences=sequences_written, output=str(output_path))
    return packed_manifest


def build_mixture(manifests: list[DatasetManifest], output_path: str, logger: PipelineLogger | None = None) -> DatasetManifest:
    rows = [manifest.to_dict() for manifest in manifests]
    write_json(output_path, {"manifests": rows})
    if logger is not None:
        logger.event("data.mixture", "wrote mixture manifest", manifests=len(rows), output=output_path)
    return DatasetManifest("mixture_manifest", "meta", output_path, len(rows), {})


def run_data_prep(config: DataConfig) -> dict[str, object]:
    logger = _make_data_logger(config)
    logger.event("data_prep", "starting data preparation", raw_dir=config.raw_dir, processed_dir=config.processed_dir)
    raw_manifests = prepare_raw_sources(config, logger)
    cleaned = [clean_corpus(manifest, config, logger) for manifest in raw_manifests]
    tokenizer = train_or_load_tokenizer(config, cleaned, logger)
    packed = [tokenize_and_pack(manifest, tokenizer.path, config, logger) for manifest in cleaned]
    mixture = build_mixture(cleaned, str(Path(config.processed_dir) / "mixture_manifest.json"), logger)
    logger.event("data_prep", "complete", raw=len(raw_manifests), cleaned=len(cleaned), packed=len(packed))
    return {"raw": raw_manifests, "cleaned": cleaned, "tokenizer": tokenizer, "packed": packed, "mixture": mixture}


def load_packed_sequences(path: str) -> list[list[int]]:
    path_obj = Path(path)
    if path_obj.suffix == ".jsonl":
        return [row["tokens"] for row in iter_jsonl(path_obj)]
    return read_json(path)["sequences"]
