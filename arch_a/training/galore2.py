
from __future__ import annotations
from typing import Iterable, Optional
import torch

def _low_rank_project(grad: torch.Tensor, rank: int):
    if grad.ndim != 2:
        return grad
    m, n = grad.shape
    k = max(1, min(rank, min(m, n)))
    if k >= min(m, n):
        return grad
    # Use a randomized low-rank approximation when available; fall back to SVD.
    grad_f = grad.float()
    try:
        q = torch.randn(n, k, device=grad.device, dtype=torch.float32)
        y = grad_f @ q
        q2, _ = torch.linalg.qr(y, mode='reduced')
        b = q2.transpose(0, 1) @ grad_f
        return (q2 @ b).to(grad.dtype)
    except Exception:
        u, s, vh = torch.linalg.svd(grad_f, full_matrices=False)
        return ((u[:, :k] * s[:k]) @ vh[:k, :]).to(grad.dtype)

@torch.no_grad()
def project_gradients_galore2(model: torch.nn.Module, rank: int = 8, min_numel: int = 1024):
    """
    Project large matrix gradients to a low-rank subspace.
    Intended as a memory-conscious training-time hook.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        if p.grad.ndim == 2 and p.numel() >= min_numel:
            projected = _low_rank_project(p.grad, rank)
            p.grad.data.copy_(projected)
