from __future__ import annotations


def kaggle_commands(config_path: str = "configs/profiles/real_150m_plus_spec.yaml") -> list[str]:
    return [
        "!pip install -r requirements.txt",
        f"!python -m src.data.prepare --config {config_path}",
        f"!python -m src.train.pretrain --config {config_path}",
        f"!python -m src.train.sft --config {config_path}",
        f"!python -m src.train.dpo --config {config_path}",
    ]
