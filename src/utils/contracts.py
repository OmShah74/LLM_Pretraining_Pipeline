from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AttentionConfig:
    num_heads: int = 4
    num_kv_heads: int = 2
    dropout: float = 0.0
    rope_base: int = 10000


@dataclass
class MoEConfig:
    enabled: bool = True
    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_weight: float = 0.01
    shared_expert: bool = True
    expert_hidden_mult: float = 4.0
    expert_interval: int = 2


@dataclass
class ModelConfig:
    vocab_size: int = 512
    max_seq_len: int = 128
    hidden_size: int = 192
    num_layers: int = 6
    mlp_hidden_size: int = 512
    attn: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_embeddings: bool = True


@dataclass
class RuntimeConfig:
    profile_name: str = "t4_prototype_spec"
    project_root: str = "."
    artifact_dir: str = "artifacts"
    seed: int = 42
    device: str = "auto"
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 4
    eval_batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    epochs: int = 1
    max_steps: int = 30
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    log_every_steps: int = 5
    eval_every_steps: int = 10
    save_every_steps: int = 10
    warmup_steps: int = 0
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.1
    grpo_group_size: int = 4
    grpo_beta: float = 0.02
    grpo_max_new_tokens: int = 48
    resume_from: str | None = None


@dataclass
class DataConfig:
    raw_dir: str = "artifacts/data/raw"
    processed_dir: str = "artifacts/data/processed"
    tokenizer_path: str = "artifacts/tokenizer/tokenizer.json"
    tokenizer_type: str = "byte_level_bpe"
    require_real_data: bool = True
    use_streaming_sources: bool = True
    streaming_cache_dir: str = "artifacts/cache/hf"
    stream_sources: list[dict[str, Any]] = field(default_factory=list)
    pretrain_filename: str = "pretrain.jsonl"
    sft_filename: str = "sft.jsonl"
    dpo_filename: str = "dpo.jsonl"
    grpo_filename: str = "grpo.jsonl"
    max_vocab_size: int = 512
    min_frequency: int = 1
    val_ratio: float = 0.2
    pretrain_seq_len: int = 128
    sft_seq_len: int = 128
    dpo_seq_len: int = 128
    min_chars: int = 20
    max_chars: int = 4000
    dedup: bool = True
    strip_html: bool = True
    lowercase: bool = False
    quality_threshold: float = 0.35
    min_alpha_fraction: float = 0.6
    max_line_repeat_ratio: float = 0.25
    simhash_bits: int = 64
    simhash_bands: int = 4
    progress_every_rows: int = 500
    checkpoint_every_rows: int = 2000
    flush_every_rows: int = 250
    max_records_per_role: dict[str, int] = field(default_factory=dict)
    dataset_plan: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: int = 20


@dataclass
class DatasetManifest:
    name: str
    role: str
    path: str
    num_records: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizerArtifact:
    path: str
    vocab_size: int
    special_tokens: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointMetadata:
    path: str
    stage: str
    step: int
    epoch: int
    config_path: str
    tokenizer_path: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    path: str
    stage: str
    metrics: dict[str, float]
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_project_path(runtime: RuntimeConfig, relative_path: str) -> Path:
    return Path(runtime.project_root).resolve() / relative_path
