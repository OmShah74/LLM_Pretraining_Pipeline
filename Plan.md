# Fully Runnable MoE LLM Prototype

This repository implements a production-shaped, free-tier-friendly prototype for an end-to-end LLM pipeline with a decoder-only Transformer using RoPE, RMSNorm, SwiGLU, grouped-query attention, and Top-2 shared-expert hybrid MoE blocks.

## Architecture
- Decoder-only causal LM
- RMSNorm pre-norm stack
- RoPE positional encoding
- Grouped-query attention
- SwiGLU FFN
- Top-2 routed MoE MLP with shared expert path
- Dense fallback path for ablations

## Runnable Stages
- Data preparation: source registration, cleaning, deduplication, tokenizer training, packing, manifests
- Pretraining: causal LM training with checkpoints and resume support
- SFT: instruction tuning on prompt-response pairs
- DPO: preference optimization on chosen-rejected pairs
- Evaluation: perplexity and generation sample report
- Export: checkpoint, tokenizer, and model card packaging

## Runtime Profiles
- `configs/profiles/t4_prototype_spec.yaml`: scaled-down faithful prototype for free T4
- `configs/profiles/production_family_spec.yaml`: future production-family reference spec

## Main Commands
```bash
python -m src.data.prepare --config configs/profiles/t4_prototype_spec.yaml
python -m src.train.pretrain --config configs/profiles/t4_prototype_spec.yaml
python -m src.train.sft --config configs/profiles/t4_prototype_spec.yaml
python -m src.train.dpo --config configs/profiles/t4_prototype_spec.yaml
python -m src.eval.evaluate --config configs/profiles/t4_prototype_spec.yaml --stage dpo
python -m src.inference.export --config configs/profiles/t4_prototype_spec.yaml --stage dpo
```

## Notes
- The repo is offline-friendly for smoke validation and uses small synthetic datasets by default.
- GRPO is reserved as the next reinforcement phase and is scaffolded as an experimental extension point.
