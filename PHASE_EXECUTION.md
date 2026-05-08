# Phase Execution Guide (Alpha AGI)

This document tracks what to execute in order, how to run each phase, and what “done” means.

## Phase 0 — Environment & Baseline Validation

### Execute
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run smoke test:
```bash
PYTHONPATH=. python arch_a/tests.py
```

### Done criteria
- Dependencies install successfully.
- Smoke test passes end-to-end.

---

## Phase 1 — Config Foundation

### Execute
1. Validate architecture presets in `arch_a/config.py` (including `for_50m`).
2. Validate training hyperparameter dataclass in `arch_a/training/train_config.py`.

### Done criteria
- `for_50m` config exists and loads.
- `TrainConfig` exists with core knobs (batch size, LR, warmup, max steps, eval/save cadence, precision, compile).

---

## Phase 2 — Data Pipeline

### Execute
1. Prepare and clean raw manifest data:
```bash
python scripts/prepare_data.py --input <raw.jsonl> --output_dir <prepared_dir>
```
2. Tokenize and pack shards (train and val separately):
```bash
python scripts/tokenize_data.py --input <prepared_dir>/train.jsonl --output_prefix <out_dir>/train --tokenizer gpt2 --max_seq_len 1024
python scripts/tokenize_data.py --input <prepared_dir>/val.jsonl --output_prefix <out_dir>/val --tokenizer gpt2 --max_seq_len 1024
```
3. Load shards via `arch_a/training/data.py` in a training loop dry-run.

### Done criteria
- `data/manifest.schema.json` and `data/README.md` define required fields.
- Preprocessing emits deduplicated `train.jsonl` and `val.jsonl`.
- Tokenization emits `.bin` + `.idx` shards with correct chunk metadata.
- `StreamingShardLoader` can iterate batches without shape/type errors.

---

## Phase 3 — Trainer Integration (in progress)

### Execute
1. Create `train.py` entrypoint with optimizer/scheduler/grad-accum/checkpointing.
2. Connect `StreamingShardLoader` batches to model forward/backward.
3. Add periodic eval/save hooks.

### Done criteria
- 1k-step smoke train runs with finite loss.
- Checkpoint save/load resumes correctly.
- Basic logs show tokens/sec, loss trend, and no NaN/Inf divergence.

---

## Recommended gating before longer pretraining
- **Gate A:** 100-step dry run (sanity)
- **Gate B:** 1k-step smoke run (stability)
- **Gate C:** 10k-step pilot (learning signal + throughput)

Proceed to long runs only after all three gates pass.
