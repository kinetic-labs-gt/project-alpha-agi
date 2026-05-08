# Project-Alpha-AGI

Project Alpha AGI is an experimental architecture focused on reasoning quality, cost efficiency, and hardware portability.

## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run smoke tests
```bash
PYTHONPATH=. python arch_a/tests.py
```

If tests pass, you should see:
```
✓ All tests passed
```

## Docs
- `TESTING.md` — short test execution guide.
- `PHASE_EXECUTION.md` — end-to-end phase checklist (Phase 0–3+) and completion criteria.
- `data/README.md` + `data/manifest.schema.json` — data manifest specification.

## Current implementation modules
- Model and architecture core: `arch_a/model.py`, `arch_a/modules/*`
- Training utilities: `arch_a/training/*`
- Data preparation scripts: `scripts/prepare_data.py`, `scripts/tokenize_data.py`
