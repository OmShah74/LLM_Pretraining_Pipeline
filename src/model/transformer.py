from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.model.layers import GroupedQueryAttention, RMSNorm, SwiGLU
from src.model.moe import Top2SharedExpertMoE
from src.utils.contracts import ModelConfig


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    aux_loss: torch.Tensor
    metrics: dict[str, float]


class DenseMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, inner_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.attn = GroupedQueryAttention(config.hidden_size, config.attn)
        use_moe = config.moe.enabled and (layer_idx % max(config.moe.expert_interval, 1) == 0)
        self.ffn = (
            Top2SharedExpertMoE(config.hidden_size, config.moe)
            if use_moe
            else DenseMLPBlock(config.hidden_size, config.mlp_hidden_size)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        attn_output = self.attn(self.attn_norm(x), attention_mask)
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.mlp_norm(x))
        aux_loss = getattr(self.ffn, "last_aux_loss", torch.tensor(0.0, device=x.device))
        metrics = getattr(self.ffn, "last_metrics", {})
        x = x + self.dropout(ffn_output)
        return x, aux_loss.to(x.device), metrics


class MoEDecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config, idx) for idx in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        x = self.embed_tokens(input_ids)
        aux_losses: list[torch.Tensor] = []
        metrics: dict[str, float] = {}
        for index, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x, aux_loss, block_metrics = checkpoint(
                    block,
                    x,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x, aux_loss, block_metrics = block(x, attention_mask)
            aux_losses.append(aux_loss)
            for key, value in block_metrics.items():
                metrics[f"layer_{index}_{key}"] = value
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        total_aux = torch.stack(aux_losses).sum() if aux_losses else torch.tensor(0.0, device=input_ids.device)
        return ModelOutput(logits=logits, loss=loss, aux_loss=total_aux, metrics=metrics)


def build_dense_block(config: ModelConfig) -> DenseMLPBlock:
    return DenseMLPBlock(config.hidden_size, config.mlp_hidden_size)


def build_moe_block(config: ModelConfig) -> Top2SharedExpertMoE:
    return Top2SharedExpertMoE(config.hidden_size, config.moe)


def build_model(config: ModelConfig) -> MoEDecoderLM:
    return MoEDecoderLM(config)
