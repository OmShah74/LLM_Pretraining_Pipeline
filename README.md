# MoE LLM Training Repo

This repo now includes a **streaming dataset backend** so you do not need to manually store raw JSONL files before preprocessing. The pipeline can stream from curated Hugging Face datasets, clean/filter/deduplicate them, materialize only the cleaned training-ready corpora, train a BPE tokenizer, and then continue into pretraining/SFT/DPO/GRPO.

## Google Drive on Colab
If Colab shows `MessageError: [dfs_ephemeral] Credentials propagation unsuccessful`, use the updated notebook flow:
- authenticate first with `google.colab.auth.authenticate_user()`
- then mount Drive with `force_remount=True`
- all caches and artifacts are redirected to Drive via environment variables

The notebook now sets:
- `LLM_ARTIFACT_DIR`
- `LLM_RAW_DIR`
- `LLM_PROCESSED_DIR`
- `LLM_TOKENIZER_PATH`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `TRANSFORMERS_CACHE`
- `LLM_STREAMING_CACHE_DIR`

This avoids depending on the small local Colab disk as much as possible.

## What Changed
- Streaming dataset backend added
- Heuristic quality filtering added
- Approximate near-duplicate removal added with SimHash banding
- Data prep no longer requires hand-created raw JSONL inputs when `stream_sources` are configured
- Cleaned/tokenized outputs are still materialized locally because tokenization and training need deterministic artifacts

## Architecture
- Decoder-only causal LM
- `RMSNorm`
- `RoPE`
- `Grouped Query Attention`
- `SwiGLU`
- Top-2 routed MoE with shared dense expert
- Capacity-aware routing and balancing loss
- Tied embeddings
- Gradient checkpointing support
- `AdamW` with proper no-decay groups
- Cosine LR scheduler with warmup

## Streaming Dataset Plan
Default streaming sources are defined in [configs/profiles/real_150m_plus_spec.yaml](</c:/Users/OM%20SHAH/Desktop/DeepLearning_Projects/LLM_Pretraining_Pipeline/configs/profiles/real_150m_plus_spec.yaml>):

- Pretraining:
  - `HuggingFaceFW/fineweb-edu` with `sample-10BT`
  - `DKYoon/SlimPajama-6B`
- SFT:
  - `ModelCloud/alpaca-data-cleaned`
  - `databricks/databricks-dolly-15k`
- DPO:
  - `mlabonne/ultrafeedback-binarized-preferences-cleaned`
- GRPO:
  - `openai/gsm8k`

## What Gets Stored
Not stored:
- full raw downloaded corpora

Stored:
- cleaned streamed subsets
- manifests
- tokenizer artifacts
- packed token sequences
- checkpoints and release artifacts

This is the practical compromise needed for resumable training.

## Cleaning / Filtering
Implemented now:
- UTF-8 cleanup
- HTML stripping
- URL stripping
- whitespace normalization
- optional lowercasing
- heuristic quality scoring
- exact deduplication
- approximate near-duplicate removal via SimHash
- per-role max record limits

Still not industrial-grade:
- no neural quality classifier yet
- no language ID model yet
- no safety classifier yet
- no large-scale datatrove/Spark pipeline yet

## Run Sequence
1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

2. Validate the streaming sources first
```bash
python -m src.data.prepare --config configs/profiles/real_150m_plus_spec.yaml --validate-only
```

3. Optional: validate only one role
```bash
python -m src.data.prepare --config configs/profiles/real_150m_plus_spec.yaml --validate-only --roles pretrain
```

4. Prepare streamed data, cleaned corpora, manifests, tokenizer, and packed sequences
```bash
python -m src.data.prepare --config configs/profiles/real_150m_plus_spec.yaml
```

5. Run pretraining
```bash
python -m src.train.pretrain --config configs/profiles/real_150m_plus_spec.yaml
```

6. Run SFT
```bash
python -m src.train.sft --config configs/profiles/real_150m_plus_spec.yaml
```

7. Run DPO
```bash
python -m src.train.dpo --config configs/profiles/real_150m_plus_spec.yaml
```

8. Run GRPO
```bash
python -m src.train.grpo --config configs/profiles/real_150m_plus_spec.yaml
```

9. Run evaluation
```bash
python -m src.eval.evaluate --config configs/profiles/real_150m_plus_spec.yaml --stage dpo
```

10. Export artifacts
```bash
python -m src.inference.export --config configs/profiles/real_150m_plus_spec.yaml --stage dpo
```

## How To Tell Whether Streaming Is Working
When a full prep run is working you should see log lines like:
- `[data.prepare] streaming role=pretrain source=fineweb_edu ...`
- `[data.prepare] completed role=pretrain rows=... output=...`

When source validation is working, it writes:
- `artifacts/data/processed/data_source_validation.json`

When full prep completes, you should see:
- `artifacts/data/processed/*_stream_cleaned.jsonl`
- `artifacts/data/processed/*_manifest.json`
- `artifacts/data/processed/data_prep_summary.json`

If a dataset source is bad, you should now get an explicit error naming:
- source name
- Hugging Face path
- optional config/subset
- split

## Practical Readiness
Ready now:
- streaming extraction
- cleaning and filtering
- approximate near-duplicate removal
- tokenizer training
- packed-sequence generation
- single-node pretraining/SFT/DPO/GRPO command paths

Still true:
- a free T4 is the hard throughput bottleneck
- a 150M+ run on multi-million-token data will still take many resumed sessions
- this is serious single-node infrastructure, not hyperscale distributed training infra
