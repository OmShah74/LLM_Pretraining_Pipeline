from __future__ import annotations

from pathlib import Path


def colab_commands(repo_url: str, config_path: str = "configs/profiles/real_150m_plus_spec.yaml") -> list[str]:
    return [
        "from google.colab import drive",
        "drive.mount('/content/drive')",
        f"!git clone {repo_url}",
        "%cd LLM_Pretraining_Pipeline",
        "!pip install -r requirements.txt",
        f"!python -m src.data.prepare --config {config_path}",
        f"!python -m src.train.pretrain --config {config_path}",
        f"!python -m src.train.sft --config {config_path}",
        f"!python -m src.train.dpo --config {config_path}",
    ]
