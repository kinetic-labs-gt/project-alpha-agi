
from __future__ import annotations
import os
import torch

class KernelDispatcher:
    """
    Unified accelerator fabric abstraction.
    This does not magically compile to every backend, but it centralizes feature detection
    and allows the model to select safe kernels per platform.
    """
    def __init__(self):
        self.backend = self.detect_backend()

    def detect_backend(self):
        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return "rocm"
            return "cuda"
        try:
            import torch_xla.core.xla_model as xm
            return "xla"
        except ImportError:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @property
    def supports_amp(self):
        return self.backend in {"cuda", "mps", "rocm", "xla"}

    def maybe_compile(self, module):
        if not self.supports_amp:
            return module

        if self.backend == "mps":
            # MPS backend currently has poor compilation support and often hangs/fails.
            # Bypass torch.compile to retain safe execution.
            return module

        try:
            if self.backend == "xla":
                return torch.compile(module, backend="openxla")
            else:
                return torch.compile(module) # Defaults safely to inductor for CUDA/ROCm
        except Exception:
            return module

    def attention_fn(self):
        return torch.nn.functional.scaled_dot_product_attention if hasattr(torch.nn.functional, "scaled_dot_product_attention") else None
