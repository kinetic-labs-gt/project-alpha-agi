# Alpha-AGI (Arch-A) Comprehensive Documentation

Welcome to the Alpha-AGI (Arch-A) architecture. This repository contains a novel, highly efficient neural network architecture designed around **Non-Autoregressive Diffusion (NADD)** and **Adaptive Logic-Gated Recurrence (ALGR)**.

This document serves as a complete manual for understanding the codebase, the libraries used, and how to execute a full pre-training run from scratch.

---

## 1. Libraries and Their Roles

This repository strictly curates its dependencies to ensure high performance and hardware universalism.

*   **`torch` (PyTorch):** The core deep learning framework. It provides the tensor operations, autograd engine, and the `torch.compile` JIT compiler used for extreme acceleration.
*   **`numpy`:** Used heavily in the data pipeline (`StreamingShardLoader`). It allows us to memory-map (`np.memmap`) massive binary files directly from disk into RAM seamlessly, avoiding memory overflow during training.
*   **`transformers` (HuggingFace):** We use this strictly for the `AutoTokenizer` module during the data preparation phase to convert raw text into token IDs efficiently.
*   **`datasets` (HuggingFace):** An optional dependency useful if you intend to download massive datasets (like FineWeb or RedPajama) programmatically before feeding them into our data pipeline.
*   **`wandb` (Weights & Biases):** Integrated into `MetricsLogger` to provide real-time dashboard tracking of loss, perplexity, and hardware telemetry during long pre-training runs.
*   **`tqdm` & `pyyaml`:** Utility libraries for progress bars and reading the YAML configuration files, respectively.

---

## 2. Core Architecture Modules Explained

### `AlphaWindow` (`arch_a/modules/alpha_window.py`)
Traditional Transformers use global attention, which scales quadratically ($O(N^2)$) and requires massive KV-caches.
The `AlphaWindow` solves this by combining a fast, linear State Space Model (SSM) scan with a heavily restricted "local" causal attention window. This allows the model to process infinite sequence lengths in linear time ($O(N)$) without losing local reasoning accuracy.

### `ALGRController` (`arch_a/modules/algr.py`)
Standard models pass tokens through every layer exactly once. Arch-A uses **Adaptive Logic-Gated Recurrence (ALGR)**.
The controller dynamically loops a token through the *same* layer multiple times until a "halting gate" determines the token is "confident" enough to move on. Easy tokens exit immediately; complex reasoning tokens loop longer. The controller calculates an entropy penalty during training so the model learns to balance accuracy with speed.

### `NADDDecoder` (`arch_a/model.py`)
Instead of predicting tokens one-by-one from left-to-right (Autoregressive), the **Non-Autoregressive Diffusion (NADD)** decoder refines the entire sequence at once over a set number of "steps".

### `KernelDispatcher` (`arch_a/kernels/dispatcher.py`)
Ensures "Hardware Universalism". It detects your hardware at runtime. If you are on an Nvidia GPU, it compiles using Inductor. If you are on a Google TPU, it forces the `openxla` backend. If you are on an Apple Mac (`mps`), it safely bypasses compilation to prevent silent crashes.

### `StreamingShardLoader` (`arch_a/training/data.py`)
A custom dataloader designed for infinite scaling. Instead of loading JSON files into RAM, it reads raw `uint16` binary chunks directly from disk using `np.memmap`. It supports deterministic shuffling and saves its exact file/byte cursor during checkpoints so training can be perfectly paused and resumed.

---

## 3. How to Pre-Train Your AI Model

Follow these steps to train your own model (e.g., the 50M parameter preset).

### Step A: Data Preparation (Phase 2)
You need raw text data in `.jsonl` format.
1.  **Clean and Filter:** Run your raw text through the preparation script to remove duplicates, apply quality heuristics, and split into train/val sets.
    ```bash
    python scripts/prepare_data.py --input raw_data.jsonl --output_dir ./clean_data
    ```
2.  **Tokenize and Pack:** Convert the text into binary `.bin` shards. This script automatically packs sequences into perfectly sized blocks (e.g., 1024 tokens).
    ```bash
    python scripts/tokenize_data.py --input ./clean_data/train.jsonl --output_prefix data/train_shard --max_seq_len 1024
    ```

### Step B: Setting Hyperparameters
Hyperparameters are controlled via two dataclasses:
*   **`ArchAConfig` (`arch_a/config.py`):** Controls the size of the "Brain" (parameters).
    *   `vocab_size`: Size of your tokenizer (usually ~32k to 65k).
    *   `d_model`: The hidden dimension width (e.g., 512).
    *   `n_layers`: Number of neural blocks (e.g., 8).
    *   *Note: We have provided presets like `ArchAConfig.for_50m()` and `ArchAConfig.for_500m()` so you don't have to guess these numbers.*
*   **`TrainConfig` (`arch_a/training/train_config.py`):** Controls the training loop.
    *   `batch_size`: How many sequences to process at once.
    *   `grad_accum_steps`: Simulates a larger batch size by accumulating gradients over multiple steps before updating weights.
    *   `lr`: Learning rate (e.g., `3e-4`).

### Step C: Hardware Sanity Check (Phase 5)
Before committing to an 11-hour run, ensure your hardware is responding correctly:
```bash
PYTHONPATH=. python scripts/runtime_sanity.py
```
This tests forward and backward passes to ensure your GPU/TPU drivers and PyTorch bindings are fully functional.

### Step D: Launching the Training Loop (Phase 3)
Finally, start the trainer. The trainer handles mixed-precision (FP16/BF16) automatically for speed.
```bash
python train.py \
    --preset for_50m \
    --train_data "data/*.bin" \
    --save_dir "checkpoints" \
    --wandb
```

### Resuming a Run
If your computer crashes or you need to pause, simply pass the `--resume` flag pointing to the last saved checkpoint. The `CheckpointManager` and `StreamingShardLoader` will restore the exact epoch, file chunk, optimizer momentum, and RNG seeds.
```bash
python train.py --preset for_50m --train_data "data/*.bin" --resume "checkpoints/latest.pt"
```