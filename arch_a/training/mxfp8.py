
from __future__ import annotations
import math
import torch

def blockwise_mxfp8_quantize(x: torch.Tensor, block_size: int = 32, eps: float = 1e-6):
    """
    Simulated microscaling quantization.
    Returns:
        q: quantized tensor in float16 (for portability)
        scale: per-block scale factors
    """
    orig_shape = x.shape
    flat = x.flatten()
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    q = torch.clamp((blocks / scale) * 127.0, -127, 127).round().to(torch.int8)
    return q, scale.to(torch.float16), orig_shape

def blockwise_mxfp8_dequantize(q: torch.Tensor, scale: torch.Tensor, orig_shape, block_size: int = 32):
    qf = q.to(torch.float32)
    blocks = (qf / 127.0) * scale.to(torch.float32)
    flat = blocks.flatten()[: math.prod(orig_shape)]
    return flat.view(*orig_shape)
