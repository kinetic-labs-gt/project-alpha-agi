from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import torch
from torch import nn
from .alpha_window import AlphaWindow, AlphaWindowState
from .normalization import RMSNorm

@dataclass
class ALGRMeta:
    loops: List[int]
    halt_prob: List[float]
    entropy: List[float]
    confidence: List[float]
    loss_penalty: Optional[torch.Tensor] = None

class ALGRBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        ssm_state_dim: int,
        alpha_window: int,
        d_ff: int,
        rope_theta: float,
        norm_eps: float,
        dropout: float,
        use_bias: bool,
        residual_fp32: bool,
        summary_dim: int,
    ):
        super().__init__()
        self.alpha_window = AlphaWindow(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            ssm_state_dim=ssm_state_dim,
            alpha_window=alpha_window,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            attn_dropout=dropout,
            use_bias=use_bias,
            residual_fp32=residual_fp32,
            summary_dim=summary_dim,
        )
        self.norm2 = RMSNorm(d_model, norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=use_bias),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=use_bias),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ssm_state: Optional[AlphaWindowState] = None, loop_idx: int = 0):
        residual = x
        x, new_state = self.alpha_window(x, ssm_state)
        x = residual + self.dropout(x)
        residual2 = x
        x = self.mlp(self.norm2(x))
        x = residual2 + self.dropout(x)
        return x, new_state

class ALGRController(nn.Module):
    """
    Adaptive Logic-Gated Recurrence controller.
    The controller loops each block until a halting gate crosses a threshold.
    """
    def __init__(
        self,
        layers: nn.ModuleList,
        d_model: int,
        max_loops: int = 3,
        confidence_threshold: float = 0.82,
        temperature: float = 1.0,
        device_map: Optional[Sequence[torch.device]] = None,
    ):
        super().__init__()
        self.layers = layers
        self.max_loops = max_loops
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.device_map = list(device_map) if device_map is not None else None
        self.halt_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def _move_optional_state(self, state, device):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(s.to(device, non_blocking=True) if s is not None else None for s in state)
        return state.to(device, non_blocking=True)

    def forward(self, layers, x, ssm_states, training: bool = True):
        if ssm_states is None:
            ssm_states = [None] * len(layers)
        meta_loops, meta_halt, meta_entropy, meta_conf = [], [], [], []
        penalty_tensors = []

        for i, layer in enumerate(layers):
            dev = self.device_map[i] if self.device_map is not None else x.device
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
            ssm_states[i] = self._move_optional_state(ssm_states[i], dev)

            loops = 0
            halted = False
            conf = 0.0
            entropy = 0.0
            halt_head_dev = next(self.halt_head.parameters()).device

            # Mask tracking which batch items have halted
            bsz = x.size(0)
            active_mask = torch.ones(bsz, dtype=torch.bool, device=x.device)

            curr_conf = torch.tensor(0.0, device=x.device)
            curr_ent = torch.tensor(0.0, device=x.device)

            while True:
                x_next, ssm_next = layer(x, ssm_states[i], loop_idx=loops)

                # Update only sequences that haven't halted yet
                x = torch.where(active_mask.unsqueeze(-1).unsqueeze(-1), x_next, x)
                ssm_h_next, ssm_summary_next = ssm_next

                if ssm_states[i] is None:
                    # If this is the very first loop pass and ssm_states[i] was initialized as None
                    ssm_h_curr = torch.zeros_like(ssm_h_next)
                    ssm_summary_curr = torch.zeros_like(ssm_summary_next)
                else:
                    ssm_h_curr, ssm_summary_curr = ssm_states[i]

                ssm_h = torch.where(active_mask.unsqueeze(-1), ssm_h_next, ssm_h_curr)
                ssm_summary = torch.where(active_mask.unsqueeze(-1), ssm_summary_next, ssm_summary_curr)
                ssm_states[i] = (ssm_h, ssm_summary)

                pooled = x.float().mean(dim=1)
                # Temporarily cast pooled vector to halt_head's device, avoiding parameter migration
                if pooled.device != halt_head_dev:
                    pooled_h = pooled.to(halt_head_dev)
                else:
                    pooled_h = pooled

                halt_logit = self.halt_head(pooled_h).squeeze(-1) / max(self.temperature, 1e-6)
                if halt_logit.device != x.device:
                    halt_logit = halt_logit.to(x.device)
                prob = torch.sigmoid(halt_logit)

                curr_conf = prob.mean()
                p = prob.clamp(1e-6, 1 - 1e-6)
                curr_ent = (-p * torch.log(p) - (1 - p) * torch.log(1 - p)).mean()

                # Collect differentiable probability for the penalty
                # We want to minimize the number of loops, which means we want to minimize
                # the probability of continuing. So we penalize (1.0 - prob.mean())
                penalty_tensors.append(1.0 - prob.mean())

                # Update active mask: items where prob >= threshold become inactive (False)
                active_mask = active_mask & (prob < self.confidence_threshold)

                loops += 1
                if not active_mask.any() or loops >= self.max_loops:
                    halted = True
                    break
            meta_loops.append(loops)
            meta_halt.append(float(halted))
            meta_entropy.append(curr_ent)
            meta_conf.append(curr_conf)

        if meta_entropy:
            # Transfer all metrics to CPU in a single batch to avoid multiple sync points.
            # We move them to a common device (x.device) before stacking to handle multi-GPU sharding.
            target_dev = x.device
            all_metrics = torch.stack([m.to(target_dev) for m in meta_entropy + meta_conf]).detach().cpu().tolist()
            half = len(all_metrics) // 2
            meta_entropy = all_metrics[:half]
            meta_conf = all_metrics[half:]

        if penalty_tensors:
            penalty_tensors = [p.to(x.device) for p in penalty_tensors]
            total_penalty = torch.stack(penalty_tensors).sum()
        else:
            total_penalty = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return x, ssm_states, ALGRMeta(meta_loops, meta_halt, meta_entropy, meta_conf, loss_penalty=total_penalty)
