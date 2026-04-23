from pathlib import Path

from src.utils.io import write_jsonl
from src.data.pipeline import run_data_prep
from src.utils.contracts import DataConfig


def test_data_prep_pipeline(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_jsonl(raw_dir / "pretrain.jsonl", [{"text": "Real pretraining sample text for moe testing.", "source": "unit"}])
    write_jsonl(raw_dir / "sft.jsonl", [{"prompt": "Test prompt", "response": "Test response"}])
    write_jsonl(raw_dir / "dpo.jsonl", [{"prompt": "Test prompt", "chosen": "Better answer", "rejected": "Worse answer"}])
    config = DataConfig(
        raw_dir=str(raw_dir),
        processed_dir=str(tmp_path / "processed"),
        tokenizer_path=str(tmp_path / "tokenizer" / "tokenizer.json"),
        max_vocab_size=64,
        pretrain_seq_len=16,
        sft_seq_len=16,
        dpo_seq_len=16,
    )
    results = run_data_prep(config)
    assert Path(results["tokenizer"].path).exists()
    assert len(results["packed"]) == 3
