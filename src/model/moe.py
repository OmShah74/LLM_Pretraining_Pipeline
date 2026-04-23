from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.model.layers import SwiGLU
from src.utils.contracts import MoEConfig


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, inner_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class Top2SharedExpertMoE(nn.Module):
    def __init__(self, hidden_size: int, config: MoEConfig):
        super().__init__()
        self.config = config
        inner_size = int(hidden_size * config.expert_hidden_mult)
        self.router = nn.Linear(hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(hidden_size, inner_size) for _ in range(config.num_experts)])
        self.shared_expert = ExpertMLP(hidden_size, inner_size) if config.shared_expert else None
        self.last_aux_loss = torch.tensor(0.0)
        self.last_metrics: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        flat_x = x.reshape(num_tokens, hidden_size)
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k = min(self.config.top_k, self.config.num_experts)
        topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        capacity = max(1, math.ceil((num_tokens * top_k / self.config.num_experts) * self.config.capacity_factor))
        routed_output = torch.zeros_like(flat_x)
        accepted_mask = torch.zeros(num_tokens, dtype=torch.bool, device=flat_x.device)
        tokens_per_expert = torch.zeros(self.config.num_experts, device=flat_x.device)
        probability_mass = router_probs.mean(dim=0)

        for expert_idx, expert in enumerate(self.experts):
            expert_assignment = topk_indices == expert_idx
            if not expert_assignment.any():
                continue
            token_positions, route_positions = torch.where(expert_assignment)
            expert_weights = topk_probs[token_positions, route_positions]
            if token_positions.numel() > capacity:
                selected = torch.topk(expert_weights, k=capacity).indices
                token_positions = token_positions[selected]
                expert_weights = expert_weights[selected]
            expert_output = expert(flat_x[token_positions])
            routed_output[token_positions] += expert_output * expert_weights.unsqueeze(-1)
            accepted_mask[token_positions] = True
            tokens_per_expert[expert_idx] = token_positions.numel()

        if self.shared_expert is not None:
            shared_output = self.shared_expert(flat_x)
            routed_output = routed_output + shared_output

        dropped_fraction = float((~accepted_mask).float().mean().detach().cpu().item())
        fraction_tokens = tokens_per_expert / max(float(num_tokens * top_k), 1.0)
        aux_loss = (self.config.num_experts * (probability_mass * fraction_tokens).sum()) * self.config.aux_loss_weight
        self.last_aux_loss = aux_loss
        self.last_metrics = {
            f"expert_{idx}_load": float(tokens_per_expert[idx].detach().cpu().item()) / max(capacity, 1)
            for idx in range(self.config.num_experts)
        }
        self.last_metrics["router_aux_loss"] = float(aux_loss.detach().cpu().item())
        self.last_metrics["dropped_fraction"] = dropped_fraction
        self.last_metrics["capacity"] = float(capacity)
        return routed_output.view(batch_size, seq_len, hidden_size)
