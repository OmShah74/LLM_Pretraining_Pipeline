import torch

from src.model.layers import apply_rope
from src.model.transformer import build_model
from src.utils.contracts import AttentionConfig, ModelConfig, MoEConfig


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        mlp_hidden_size=64,
        attn=AttentionConfig(num_heads=4, num_kv_heads=2, dropout=0.0, rope_base=10000),
        moe=MoEConfig(enabled=True, num_experts=4, top_k=2, shared_expert=True, expert_interval=1),
    )


def test_rope_shape() -> None:
    x = torch.randn(2, 8, 4, 8)
    positions = torch.arange(8)
    rotated = apply_rope(x, positions, 10000)
    assert rotated.shape == x.shape


def test_model_forward_and_metrics() -> None:
    model = build_model(_config())
    input_ids = torch.randint(0, 63, (2, 8))
    labels = torch.randint(0, 63, (2, 8))
    outputs = model(input_ids, labels=labels)
    assert outputs.logits.shape == (2, 8, 64)
    assert outputs.loss is not None
    assert outputs.aux_loss.dim() == 0
    assert any("expert" in key for key in outputs.metrics)
