import time
import torch
import math
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class MetricsLogger:
    def __init__(self, use_wandb: bool = False, project_name: str = "arch_a", run_name: str = "run_01"):
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=project_name, name=run_name)

        self.step_start_time = time.time()
        self.nan_count = 0
        self.inf_count = 0

    def check_nan_inf(self, loss: torch.Tensor) -> bool:
        if torch.isnan(loss):
            self.nan_count += 1
            return True
        if torch.isinf(loss):
            self.inf_count += 1
            return True
        return False

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_processed: int,
        algr_loops: float = 0.0,
        algr_conf: float = 0.0,
    ):
        """Logs metrics for a single training step."""
        now = time.time()
        elapsed = now - self.step_start_time
        tokens_per_sec = tokens_processed / max(elapsed, 1e-6)

        # Perplexity estimation
        ppl = math.exp(min(loss, 20.0))

        # GPU Memory usage (if available)
        mem_mb = 0
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        metrics = {
            "train/loss": loss,
            "train/ppl": ppl,
            "train/lr": lr,
            "train/grad_norm": grad_norm,
            "perf/tokens_per_sec": tokens_per_sec,
            "algr/mean_loops": algr_loops,
            "algr/mean_confidence": algr_conf,
            "system/gpu_mem_mb": mem_mb,
            "system/nan_count": self.nan_count,
            "system/inf_count": self.inf_count
        }

        # Terminal logging
        if step % 10 == 0:
            self.logger.info(
                f"Step {step:05d} | Loss: {loss:.4f} | PPL: {ppl:.2f} | "
                f"Tok/s: {tokens_per_sec:.0f} | Loops: {algr_loops:.2f} | "
                f"Mem: {mem_mb:.0f}MB"
            )

        # WandB logging
        if self.use_wandb:
            wandb.log(metrics, step=step)

        self.step_start_time = time.time()

    def log_eval(self, step: int, val_loss: float):
        """Logs validation metrics."""
        val_ppl = math.exp(min(val_loss, 20.0))
        self.logger.info(f"=== EVAL Step {step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ===")

        if self.use_wandb:
            wandb.log({
                "val/loss": val_loss,
                "val/ppl": val_ppl
            }, step=step)
