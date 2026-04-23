from pathlib import Path

import torch

from src.model.transformer import build_model
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.utils.contracts import AttentionConfig, ModelConfig, MoEConfig


def test_checkpoint_save_load(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        hidden_size=16,
        num_layers=2,
        mlp_hidden_size=32,
        attn=AttentionConfig(num_heads=4, num_kv_heads=2, dropout=0.0, rope_base=10000),
        moe=MoEConfig(enabled=True, num_experts=2, top_k=2, shared_expert=True, expert_interval=1),
    )
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metadata = save_checkpoint(tmp_path, "unit", 1, 0, model, optimizer, "cfg.yaml", "tok.json", {"loss": 1.0})
    assert Path(metadata.path).exists()
    state = load_checkpoint(metadata.path, model, optimizer)
    assert state["step"] == 1
