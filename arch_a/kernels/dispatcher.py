
from __future__ import annotations
import os
import torch

class KernelDispatcher:
    """
    Unified accelerator fabric abstraction.
    This does not magically compile to every backend, but it centralizes feature detection
    and allows the model to select safe kernels per platform.
    """
    def __init__(self, verbose: bool = False):
        self.backend = self.detect_backend()
        self.compile_status = "uncompiled"
        self.attn_kernel = "sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "fallback"

        if verbose:
            self.print_capabilities()

    def print_capabilities(self):
        print(f"--- KernelDispatcher Capabilities ---")
        print(f"Backend:        {self.backend.upper()}")
        print(f"AMP Support:    {self.supports_amp}")
        print(f"Attention:      {self.attn_kernel}")
        print(f"-------------------------------------")

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
            self.compile_status = "bypassed (mps)"
            return module

        try:
            if self.backend == "xla":
                compiled = torch.compile(module, backend="openxla")
                self.compile_status = "compiled (openxla)"
                return compiled
            else:
                compiled = torch.compile(module) # Defaults safely to inductor for CUDA/ROCm
                self.compile_status = "compiled (inductor)"
                return compiled
        except Exception as e:
            self.compile_status = f"failed ({str(e)})"
            return module

    def attention_fn(self):
        return torch.nn.functional.scaled_dot_product_attention if hasattr(torch.nn.functional, "scaled_dot_product_attention") else None
