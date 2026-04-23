from pathlib import Path

from src.utils.config import load_full_config


def test_load_full_config() -> None:
    config = load_full_config(Path("configs/profiles/t4_prototype_spec.yaml"))
    assert config["runtime"].profile_name == "t4_prototype_spec"
    assert config["model"].moe.top_k == 2
