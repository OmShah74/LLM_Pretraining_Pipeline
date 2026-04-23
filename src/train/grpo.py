from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from src.data.tokenizer import BPETokenizer
from src.model.transformer import MoEDecoderLM
from src.train.checkpoint import load_checkpoint, load_latest_metadata, save_checkpoint
from src.train.common import _resolve_processed_file, build_optimizer, build_scheduler, load_tokenizer, setup_run
from src.utils.config import load_full_config
from src.utils.io import read_jsonl, write_json
from src.utils.runtime import autocast_context


def _build_grpo_rows(config_path: str) -> tuple[dict[str, object], list[dict[str, str]]]:
    config = load_full_config(config_path)
    candidate_paths = []
    for role in ("grpo", "sft"):
        try:
            candidate_paths.append(_resolve_processed_file(config["data"], role, "_cleaned.jsonl"))
        except FileNotFoundError:
            continue
    for path in candidate_paths:
        if path.exists():
            return config, read_jsonl(path)
    raise FileNotFoundError("No GRPO-compatible dataset found. Expected grpo_raw_cleaned.jsonl or sft_raw_cleaned.jsonl.")


@torch.no_grad()
def _sample_group(
    model: MoEDecoderLM,
    tokenizer: BPETokenizer,
    prompt: str,
    group_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[tuple[list[int], str]]:
    model.eval()
    prompt_ids = tokenizer.encode(f"Instruction: {prompt} Response:", add_bos=True, add_eos=False)
    outputs: list[tuple[list[int], str]] = []
    eos_id = tokenizer.token_to_id[tokenizer.eos_token]
    for _ in range(group_size):
        generated = list(prompt_ids)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            result = model(input_ids=input_ids)
            logits = result.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if token_id == eos_id:
                break
        completion_ids = generated[len(prompt_ids):]
        outputs.append((completion_ids, tokenizer.decode(completion_ids)))
    return outputs


def _reward_fn(prompt: str, completion: str, reference: str | None) -> float:
    completion_words = set(completion.lower().split())
    if not completion_words:
        return -1.0
    reward = 0.0
    if reference:
        reference_words = set(reference.lower().split())
        overlap = len(completion_words & reference_words)
        reward += overlap / max(len(reference_words), 1)
    if len(completion_words) < 4:
        reward -= 0.2
    if "i do not know" in completion.lower():
        reward -= 0.1
    if "moe" in prompt.lower() and "expert" in completion.lower():
        reward += 0.1
    return reward


def _sequence_logprob(model: MoEDecoderLM, full_ids: list[int], prompt_len: int, device: torch.device) -> torch.Tensor:
    input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    label_tensor = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    output = model(input_ids=input_tensor, labels=None)
    log_probs = torch.log_softmax(output.logits, dim=-1)
    gathered = torch.gather(log_probs, -1, label_tensor.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(gathered)
    start_index = max(prompt_len - 1, 0)
    mask[:, start_index:] = 1.0
    return (gathered * mask).sum(dim=-1).mean()


def run_grpo_experimental(config_path: str) -> dict[str, float | str]:
    config, rows = _build_grpo_rows(config_path)
    device, model = setup_run(config_path, config["runtime"], config["model"], config["train"])
    tokenizer = load_tokenizer(config["data"])

    latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", "dpo")
    if latest is None:
        latest = load_latest_metadata(Path(config["runtime"].artifact_dir) / "checkpoints", "sft")
    if latest is not None:
        load_checkpoint(latest.path, model, optimizer=None, device=device, strict=False)

    reference_model = copy.deepcopy(model).to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    optimizer = build_optimizer(model, config["train"])
    scheduler = build_scheduler(optimizer, config["train"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["runtime"].use_mixed_precision and device.type == "cuda")

    group_size = config["train"].grpo_group_size
    max_new_tokens = config["train"].grpo_max_new_tokens
    beta = config["train"].grpo_beta
    global_step = 0
    latest_metrics: dict[str, float | str] = {}
    checkpoint_dir = Path(config["runtime"].artifact_dir) / "checkpoints"

    while global_step < config["train"].max_steps:
        for row in rows:
            prompt = row["prompt"]
            reference = row.get("response") or row.get("chosen")
            samples = _sample_group(model, tokenizer, prompt, group_size, max_new_tokens, device)
            rewards = torch.tensor(
                [_reward_fn(prompt, completion_text, reference) for _, completion_text in samples],
                dtype=torch.float32,
                device=device,
            )
            advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss = torch.tensor(0.0, device=device)
            prompt_ids = tokenizer.encode(f"Instruction: {prompt} Response:", add_bos=True, add_eos=False)
            for advantage, (completion_ids, _) in zip(advantages, samples):
                full_ids = prompt_ids + completion_ids
                if len(full_ids) < 2:
                    continue
                with autocast_context(device, config["runtime"].use_mixed_precision):
                    policy_logprob = _sequence_logprob(model, full_ids, len(prompt_ids), device)
                    reference_logprob = _sequence_logprob(reference_model, full_ids, len(prompt_ids), device)
                    kl_penalty = policy_logprob - reference_logprob
                    loss = loss + (-(advantage.detach() * policy_logprob) + beta * kl_penalty)
            loss = loss / max(len(samples), 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"].grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            latest_metrics = {
                "stage": "grpo",
                "grpo_loss": float(loss.detach().cpu().item()),
                "mean_reward": float(rewards.mean().detach().cpu().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if global_step % config["train"].save_every_steps == 0:
                save_checkpoint(
                    checkpoint_dir,
                    "grpo",
                    global_step,
                    0,
                    model,
                    optimizer,
                    config_path,
                    config["data"].tokenizer_path,
                    {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in latest_metrics.items() if k != "stage"},
                )
            if global_step >= config["train"].max_steps:
                break

    save_checkpoint(
        checkpoint_dir,
        "grpo",
        global_step,
        0,
        model,
        optimizer,
        config_path,
        config["data"].tokenizer_path,
        {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in latest_metrics.items() if k != "stage"},
    )
    write_json(Path(config["runtime"].artifact_dir) / "grpo_metrics.json", latest_metrics)
    return latest_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPO alignment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_grpo_experimental(args.config)


if __name__ == "__main__":
    main()
