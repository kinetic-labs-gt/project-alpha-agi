
from __future__ import annotations
import torch

class ShadowResidualQuantizer:
    """
    1-bit residual correction for low-bit quantization.
    This is a utility; it does not alter the main forward graph unless invoked.
    """
    def __init__(self, bits: int = 4, block_size: int = 32):
        self.bits = bits
        self.block_size = block_size

    def quantize(self, x: torch.Tensor):
        orig_shape = x.shape
        flat = x.flatten()
        n = flat.numel()
        pad = (-n) % self.block_size
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blocks = flat.view(-1, self.block_size)
        scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        q = torch.clamp((blocks / scale) * (2 ** (self.bits - 1) - 1), -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1).round()
        # 1-bit shadow residual sign map
        residual = (blocks - (q / (2 ** (self.bits - 1) - 1)) * scale).sign().to(torch.int8)
        return q.to(torch.int8), residual, scale.to(torch.float16), orig_shape

    def dequantize(self, q: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor, orig_shape):
        import math
        qf = q.to(torch.float32)
        rf = residual.to(torch.float32)
        out = (qf / max(1, 2 ** (self.bits - 1) - 1)) * scale.to(torch.float32)
        out = out + 0.125 * rf * scale.to(torch.float32)
        flat = out.flatten()[: math.prod(orig_shape)]
        return flat.view(*orig_shape)
