from __future__ import annotations


def colab_commands(repo_url: str, config_path: str = "configs/profiles/real_150m_plus_spec.yaml") -> list[str]:
    return [
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=True)",
        "import os",
        "DRIVE_ROOT = '/content/drive/MyDrive/LLM_Pretraining_Pipeline'",
        "os.environ['LLM_PROJECT_ROOT'] = '/content/LLM_Pretraining_Pipeline'",
        "os.environ['LLM_ARTIFACT_DIR'] = f'{DRIVE_ROOT}/artifacts'",
        "os.environ['LLM_RAW_DIR'] = f'{DRIVE_ROOT}/artifacts/data/raw'",
        "os.environ['LLM_PROCESSED_DIR'] = f'{DRIVE_ROOT}/artifacts/data/processed'",
        "os.environ['LLM_TOKENIZER_PATH'] = f'{DRIVE_ROOT}/artifacts/tokenizer/tokenizer.json'",
        "os.environ['HF_HOME'] = f'{DRIVE_ROOT}/hf_cache'",
        "os.environ['HF_DATASETS_CACHE'] = f'{DRIVE_ROOT}/hf_cache/datasets'",
        "os.environ['TRANSFORMERS_CACHE'] = f'{DRIVE_ROOT}/hf_cache/transformers'",
        "os.environ['LLM_STREAMING_CACHE_DIR'] = os.environ['HF_DATASETS_CACHE']",
        "os.makedirs(DRIVE_ROOT, exist_ok=True)",
        f"!git clone {repo_url} /content/LLM_Pretraining_Pipeline",
        "%cd /content/LLM_Pretraining_Pipeline",
        "!pip install -r requirements.txt",
        f"!python -m src.data.prepare --config {config_path} --validate-only",
        f"!python -m src.data.prepare --config {config_path}",
        f"!python -m src.train.pretrain --config {config_path}",
        f"!python -m src.train.sft --config {config_path}",
        f"!python -m src.train.dpo --config {config_path}",
        f"!python -m src.train.grpo --config {config_path}",
    ]
