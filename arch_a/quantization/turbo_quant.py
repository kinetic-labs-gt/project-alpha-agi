from __future__ import annotations
import torch

class TurboQuantizer:
    """
    TurboQuant-inspired hardware-universal asymmetric group quantizer.
    Designed for high-speed OpenXLA/Triton compilation across CUDA/ROCm/TPU.
    """
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.qmin = 0
        self.qmax = (1 << bits) - 1

    def quantize(self, x: torch.Tensor):
        orig_shape = x.shape
        flat = x.flatten()
        n = flat.numel()

        # Pad to group_size
        pad_len = (-n) % self.group_size
        if pad_len:
            flat = torch.nn.functional.pad(flat, (0, pad_len))

        groups = flat.view(-1, self.group_size)

        # Asymmetric group-wise scaling
        vmax = groups.amax(dim=1, keepdim=True)
        vmin = groups.amin(dim=1, keepdim=True)

        # Ensure 0.0 is exactly representable to prevent zero-drift
        vmax = torch.max(vmax, torch.zeros_like(vmax))
        vmin = torch.min(vmin, torch.zeros_like(vmin))

        # Prevent division by zero
        scale = ((vmax - vmin) / self.qmax).clamp_min(1e-7)
        zero_point = (-vmin / scale).round().clamp(self.qmin, self.qmax)

        q = torch.clamp((groups / scale).round() + zero_point, self.qmin, self.qmax).to(torch.uint8)

        return q, scale.to(torch.float16), zero_point.to(torch.uint8), orig_shape

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, shape):
        import math
        qf = q.to(torch.float32)
        zf = zero_point.to(torch.float32)
        sf = scale.to(torch.float32)

        flat = (qf - zf) * sf
        flat = flat.flatten()[: math.prod(shape)]
        return flat.view(*shape)
