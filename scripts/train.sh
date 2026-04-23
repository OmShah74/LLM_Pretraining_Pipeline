#!/bin/bash
set -e
CONFIG=${1:-configs/profiles/real_150m_plus_spec.yaml}
python -m src.data.prepare --config "$CONFIG"
python -m src.train.pretrain --config "$CONFIG"
python -m src.train.sft --config "$CONFIG"
python -m src.train.dpo --config "$CONFIG"
python -m src.train.grpo --config "$CONFIG"
python -m src.eval.evaluate --config "$CONFIG" --stage dpo
python -m src.inference.export --config "$CONFIG" --stage dpo
