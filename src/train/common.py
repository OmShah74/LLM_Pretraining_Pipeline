from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PackedSequenceDataset, PreferenceDataset, SFTDataset
from src.data.pipeline import load_packed_sequences
from src.data.tokenizer import BPETokenizer
from src.model.transformer import MoEDecoderLM, build_model
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import save_effective_config
from src.utils.contracts import DataConfig, ModelConfig, RuntimeConfig, TrainConfig
from src.utils.io import ensure_dir, read_jsonl, write_json
from src.utils.runtime import autocast_context, resolve_device, set_seed


def build_optimizer(model: torch.nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_epsilon,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, train_config: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(train_config.max_steps, 1)
    warmup_steps = min(train_config.warmup_steps, total_steps - 1) if total_steps > 1 else 0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(warmup_steps, 1))
        if train_config.lr_scheduler == "constant":
            return 1.0
        progress_num = max(current_step - warmup_steps, 0)
        progress_den = max(total_steps - warmup_steps, 1)
        progress = min(progress_num / progress_den, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(train_config.min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_tokenizer(data_config: DataConfig) -> BPETokenizer:
    return BPETokenizer.load(data_config.tokenizer_path)


def _resolve_processed_file(data_config: DataConfig, role: str, suffix: str) -> Path:
    processed_dir = Path(data_config.processed_dir)
    candidates = sorted(processed_dir.glob(f"*{role}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No processed file found for role='{role}' and suffix='{suffix}' in {processed_dir}")
    return candidates[0]


def build_pretrain_dataloaders(data_config: DataConfig, tokenizer: BPETokenizer, train_config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    packed = load_packed_sequences(str(_resolve_processed_file(data_config, "pretrain", "_packed.json")))
    split = max(1, int(len(packed) * (1.0 - data_config.val_ratio)))
    train_rows = packed[:split]
    val_rows = packed[split:] or packed[-1:]
    pad_id = tokenizer.token_to_id[tokenizer.pad_token]
    train_ds = PackedSequenceDataset(train_rows, data_config.pretrain_seq_len, pad_id)
    val_ds = PackedSequenceDataset(val_rows, data_config.pretrain_seq_len, pad_id)
    return (
        DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=train_config.eval_batch_size, shuffle=False),
    )


def build_sft_dataloaders(data_config: DataConfig, tokenizer: BPETokenizer, train_config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    rows = read_jsonl(_resolve_processed_file(data_config, "sft", "_cleaned.jsonl"))
    split = max(1, int(len(rows) * (1.0 - data_config.val_ratio)))
    train_ds = SFTDataset(rows[:split], tokenizer, data_config.sft_seq_len)
    val_ds = SFTDataset(rows[split:] or rows[-1:], tokenizer, data_config.sft_seq_len)
    return (
        DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=train_config.eval_batch_size, shuffle=False),
    )


def build_dpo_dataloaders(data_config: DataConfig, tokenizer: BPETokenizer, train_config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    rows = read_jsonl(_resolve_processed_file(data_config, "dpo", "_cleaned.jsonl"))
    split = max(1, int(len(rows) * (1.0 - data_config.val_ratio)))
    train_ds = PreferenceDataset(rows[:split], tokenizer, data_config.dpo_seq_len)
    val_ds = PreferenceDataset(rows[split:] or rows[-1:], tokenizer, data_config.dpo_seq_len)
    return (
        DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=train_config.eval_batch_size, shuffle=False),
    )


def setup_run(
    config_path: str,
    runtime_config: RuntimeConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> tuple[torch.device, MoEDecoderLM]:
    set_seed(runtime_config.seed)
    device = resolve_device(runtime_config.device)
    model = build_model(model_config).to(device)
    if runtime_config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    config_dir = ensure_dir(Path(runtime_config.artifact_dir) / "configs")
    save_effective_config(config_dir / f"{Path(config_path).stem}_effective.yaml", {
        "runtime": runtime_config,
        "model": model_config,
        "train": train_config,
    })
    return device, model


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_language_model(model: MoEDecoderLM, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    aux_losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            output = model(batch["input_ids"], batch.get("attention_mask"), batch["labels"])
            current_loss = output.loss if output.loss is not None else torch.tensor(0.0, device=device)
            losses.append(float(current_loss.detach().cpu().item()))
            aux_losses.append(float(output.aux_loss.detach().cpu().item()))
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20)), "aux_loss": sum(aux_losses) / max(len(aux_losses), 1)}


def train_language_model(
    *,
    stage: str,
    model: MoEDecoderLM,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    runtime_config: RuntimeConfig,
    train_config: TrainConfig,
    tokenizer_path: str,
    config_path: str,
) -> dict[str, float]:
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    start_step = 0
    start_epoch = 0
    if train_config.resume_from:
        state = load_checkpoint(train_config.resume_from, model, optimizer, device)
        start_step = int(state["step"])
        start_epoch = int(state["epoch"])

    scaler = torch.amp.GradScaler("cuda", enabled=runtime_config.use_mixed_precision and device.type == "cuda")
    global_step = start_step
    latest_metrics: dict[str, float] = {}
    checkpoint_dir = Path(runtime_config.artifact_dir) / "checkpoints"

    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            with autocast_context(device, runtime_config.use_mixed_precision):
                output = model(batch["input_ids"], batch.get("attention_mask"), batch["labels"])
                current_loss = output.loss if output.loss is not None else torch.tensor(0.0, device=device)
                total_loss = current_loss + output.aux_loss
                total_loss = total_loss / train_config.grad_accum_steps
            scaler.scale(total_loss).backward()
            if step % train_config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                latest_metrics = {
                    "train_loss": float(total_loss.detach().cpu().item()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                for key, value in output.metrics.items():
                    latest_metrics[key] = float(value)
                if global_step % train_config.eval_every_steps == 0:
                    latest_metrics.update({f"eval_{k}": v for k, v in evaluate_language_model(model, eval_loader, device).items()})
                if global_step % train_config.save_every_steps == 0:
                    save_checkpoint(checkpoint_dir, stage, global_step, epoch, model, optimizer, config_path, tokenizer_path, latest_metrics)
                if global_step >= train_config.max_steps:
                    break
        if global_step >= train_config.max_steps:
            break
    save_checkpoint(checkpoint_dir, stage, global_step, epoch, model, optimizer, config_path, tokenizer_path, latest_metrics)
    write_json(Path(runtime_config.artifact_dir) / f"{stage}_metrics.json", latest_metrics)
    return latest_metrics


def compute_sequence_logprob(model: MoEDecoderLM, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=input_ids, labels=None)
    log_probs = torch.log_softmax(output.logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
    mask = (labels > 0).float()
    return (gathered * mask).sum(dim=-1)


def train_dpo(
    *,
    model: MoEDecoderLM,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    runtime_config: RuntimeConfig,
    train_config: TrainConfig,
    tokenizer_path: str,
    config_path: str,
    stage: str = "dpo",
    beta: float = 0.1,
) -> dict[str, float]:
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    checkpoint_dir = Path(runtime_config.artifact_dir) / "checkpoints"
    global_step = 0
    latest_metrics: dict[str, float] = {}
    scaler = torch.amp.GradScaler("cuda", enabled=runtime_config.use_mixed_precision and device.type == "cuda")

    for epoch in range(train_config.epochs):
        model.train()
        for batch in train_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            with autocast_context(device, runtime_config.use_mixed_precision):
                chosen_scores = compute_sequence_logprob(model, chosen_input_ids, chosen_labels)
                rejected_scores = compute_sequence_logprob(model, rejected_input_ids, rejected_labels)
                preference_margin = chosen_scores - rejected_scores
                loss = -torch.log(torch.sigmoid(beta * preference_margin)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            latest_metrics = {
                "dpo_loss": float(loss.detach().cpu().item()),
                "preference_margin": float(preference_margin.mean().detach().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if global_step % train_config.save_every_steps == 0:
                save_checkpoint(checkpoint_dir, stage, global_step, epoch, model, optimizer, config_path, tokenizer_path, latest_metrics)
            if global_step >= train_config.max_steps:
                break
        if global_step >= train_config.max_steps:
            break

    val_margins: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            chosen_scores = compute_sequence_logprob(model, batch["chosen_input_ids"].to(device), batch["chosen_labels"].to(device))
            rejected_scores = compute_sequence_logprob(model, batch["rejected_input_ids"].to(device), batch["rejected_labels"].to(device))
            val_margins.append(float((chosen_scores - rejected_scores).mean().detach().cpu().item()))
    latest_metrics["eval_preference_margin"] = sum(val_margins) / max(len(val_margins), 1)
    save_checkpoint(checkpoint_dir, stage, global_step, epoch, model, optimizer, config_path, tokenizer_path, latest_metrics)
    write_json(Path(runtime_config.artifact_dir) / f"{stage}_metrics.json", latest_metrics)
    return latest_metrics
