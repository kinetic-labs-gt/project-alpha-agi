# How to Run Tests

## Prerequisites
```bash
pip install -r requirements.txt
```

## Smoke test (recommended)
```bash
PYTHONPATH=. python arch_a/tests.py
```

## Why `PYTHONPATH=.`?
The test imports `arch_a` as a package. Setting `PYTHONPATH=.` ensures imports resolve from the repo root.

## Expected output
```text
✓ All tests passed
```
