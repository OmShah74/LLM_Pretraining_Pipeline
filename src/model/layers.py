from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.contracts import AttentionConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: int) -> torch.Tensor:
    head_dim = x.size(-1)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    cos = torch.repeat_interleave(torch.cos(freqs), 2, dim=-1)[None, :, None, :]
    sin = torch.repeat_interleave(torch.sin(freqs), 2, dim=-1)[None, :, None, :]
    return (x * cos) + (_rotate_half(x) * sin)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inner_size)
        self.up_proj = nn.Linear(hidden_size, inner_size)
        self.down_proj = nn.Linear(inner_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size: int, config: AttentionConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.rope_base = config.rope_base

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        repeat_factor = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(repeat_factor, dim=2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = apply_rope(q, positions, self.rope_base)
        k = apply_rope(k, positions, self.rope_base)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~key_mask, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attended = torch.einsum("bhts,bshd->bthd", attn_probs, v).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attended)
