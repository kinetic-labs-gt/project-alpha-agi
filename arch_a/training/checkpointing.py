import os
import glob
import torch
import numpy as np
import random
from typing import Dict, Any, Optional

class CheckpointManager:
    def __init__(self, save_dir: str, keep_top_k: int = 3):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        global_step: int,
        config_dict: Dict[str, Any],
        val_loss: float
    ):
        """Saves a checkpoint containing full state for exact resumption."""
        checkpoint_name = f"step_{global_step}_loss_{val_loss:.4f}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # Unwrap model if compiled or DDP
        model_to_save = model.module if hasattr(model, 'module') else model

        state = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "global_step": global_step,
            "config": config_dict,
            "val_loss": val_loss,
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "rng_state_numpy": np.random.get_state(),
            "rng_state_python": random.getstate()
        }

        torch.save(state, checkpoint_path)

        # Also maintain a symlink/copy to "latest.pt"
        latest_path = os.path.join(self.save_dir, "latest.pt")
        torch.save(state, latest_path)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keeps 'latest.pt' and the top-K checkpoints sorted by validation loss."""
        # Find all step_X_loss_Y.pt files
        files = glob.glob(os.path.join(self.save_dir, "step_*_loss_*.pt"))

        if len(files) <= self.keep_top_k:
            return

        # Parse losses from filenames
        def extract_loss(filepath):
            filename = os.path.basename(filepath)
            # step_100_loss_0.4500.pt -> "0.4500"
            try:
                loss_str = filename.split("_loss_")[1].replace(".pt", "")
                return float(loss_str)
            except Exception:
                return float("inf")

        # Sort files by loss (ascending)
        sorted_files = sorted(files, key=extract_loss)

        # Keep top K, delete the rest
        files_to_delete = sorted_files[self.keep_top_k:]
        for f in files_to_delete:
            try:
                os.remove(f)
            except OSError:
                pass

    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """Loads a full checkpoint state."""
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(state["model_state_dict"])

        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])

        if "rng_state_torch" in state:
            torch.set_rng_state(state["rng_state_torch"])
        if "rng_state_torch_cuda" in state and state["rng_state_torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["rng_state_torch_cuda"])
        if "rng_state_numpy" in state:
            np.random.set_state(state["rng_state_numpy"])
        if "rng_state_python" in state:
            random.setstate(state["rng_state_python"])

        return state
