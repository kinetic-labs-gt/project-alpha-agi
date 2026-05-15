import argparse
import os
import yaml
import torch
import glob
from torch.cuda.amp import autocast, GradScaler

from arch_a import ArchAConfig, ArchAForCausalLM
from arch_a.training.train_config import TrainConfig
from arch_a.training.checkpointing import CheckpointManager
from arch_a.training.logging import MetricsLogger
from arch_a.training.data import StreamingShardLoader

def load_yaml_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_evaluation(model, eval_loader, device, eval_steps=50):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        with autocast(enabled=(device.type == 'cuda')):
            for batch in eval_loader:
                if steps >= eval_steps:
                    break
                batch = batch.to(device)
                outputs = model(batch, labels=batch)
                total_loss += outputs.loss.item()
                steps += 1
    model.train()
    return total_loss / max(1, steps) if steps > 0 else float("inf")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Real Trainer Entrypoint")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--preset", type=str, default="for_50m", help="Config preset (e.g., for_50m, for_500m)")
    parser.add_argument("--train_data", type=str, required=True, help="Glob pattern for train shards (e.g. data/*.bin)")
    parser.add_argument("--val_data", type=str, help="Glob pattern for val shards")
    parser.add_argument("--resume", type=str, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    # 1. Setup configs
    if args.config and os.path.exists(args.config):
        raw_conf = load_yaml_config(args.config)
        arch_config = ArchAConfig.from_dict(raw_conf.get("arch", {}))
        train_config = TrainConfig(**raw_conf.get("training", {}))
    else:
        # Fallback to defaults or preset
        preset_method = getattr(ArchAConfig, args.preset, ArchAConfig.for_50m)
        arch_config = preset_method()
        train_config = TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Build Model
    model = ArchAForCausalLM(arch_config)
    model.to(device)

    if (
    train_config.compile
    and not args.no_compile
    and torch.cuda.is_available()
):
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
else:
    print("torch.compile DISABLED")

    # 3. Optimizers & Scalers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.max_steps,
        eta_min=train_config.lr * 0.1
    )

    scaler = GradScaler(enabled=(train_config.mixed_precision in ['fp16', 'bf16'] and device.type == 'cuda'))

    # 4. Managers
    ckpt_manager = CheckpointManager(save_dir=args.save_dir)
    logger = MetricsLogger(use_wandb=args.wandb)

    # 5. Dataloaders
    train_files = glob.glob(args.train_data)
    if not train_files:
        raise ValueError(f"No training shards found at {args.train_data}")

    train_loader = StreamingShardLoader(train_files, batch_size=train_config.batch_size, seed=train_config.seed)
    train_iter = iter(train_loader)

    val_loader = None
    val_iter = None
    if args.val_data:
        val_files = glob.glob(args.val_data)
        if val_files:
            val_loader = StreamingShardLoader(val_files, batch_size=train_config.batch_size, shuffle=False)
            val_iter = iter(val_loader)

    # 6. Resume logic
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        state = ckpt_manager.load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        global_step = state.get("global_step", 0)
        if "loader_state" in state and state["loader_state"] is not None:
            train_loader.load_state_dict(state["loader_state"])
            train_iter = iter(train_loader)

    # 7. Training Loop
    model.train()
    optimizer.zero_grad()

    print(f"Starting training loop. Device: {device}. Target steps: {train_config.max_steps}")

    while global_step < train_config.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            break

        batch = batch.to(device)
        batch_size, seq_len = batch.shape

        # Forward pass
        dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
        amp_dtype = dtype_map.get(train_config.mixed_precision, torch.float16)
        with autocast(enabled=(train_config.mixed_precision in ['fp16', 'bf16'] and device.type == 'cuda'), dtype=amp_dtype):
            outputs = model(batch, labels=batch)
            loss = outputs.loss / train_config.grad_accum_steps

        # NaN / Inf guard
        if logger.check_nan_inf(loss):
            print(f"Warning: NaN/Inf detected at step {global_step}. Skipping batch.")
            optimizer.zero_grad()
            continue

        # Backward pass
        scaler.scale(loss).backward()

        # Step handling
        if (global_step + 1) % train_config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            # Only step the scheduler if optimizer actually stepped
            # (i.e. scale was not decreased due to inf/nan gradients)
            if scaler.get_scale() >= scale_before:
                scheduler.step()

            optimizer.zero_grad()

            # Extract ALGR metadata if available
            algr_loops = 0.0
            algr_conf = 0.0
            if hasattr(outputs, 'algr_meta') and outputs.algr_meta.loops:
                algr_loops = sum(outputs.algr_meta.loops) / len(outputs.algr_meta.loops)
                algr_conf = sum(outputs.algr_meta.confidence) / max(1, len(outputs.algr_meta.confidence))

            # Log
            logger.log_step(
                step=global_step,
                loss=loss.item() * train_config.grad_accum_steps,
                lr=scheduler.get_last_lr()[0],
                grad_norm=grad_norm.item() if not torch.isnan(grad_norm) else 0.0,
                tokens_processed=batch_size * seq_len,
                algr_loops=algr_loops,
                algr_conf=algr_conf
            )

        # Evaluation & Checkpointing
        if (global_step + 1) % train_config.eval_every == 0 and val_iter is not None:
            val_loss = run_evaluation(model, val_iter, device)
            logger.log_eval(global_step, val_loss)

            if (global_step + 1) % train_config.save_every == 0:
                ckpt_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    config_dict=arch_config.to_dict(),
                    val_loss=val_loss,
                    scaler=scaler,
                    loader_state=train_loader.state_dict()
                )

        global_step += 1

    print("Training complete.")

if __name__ == "__main__":
    main()
