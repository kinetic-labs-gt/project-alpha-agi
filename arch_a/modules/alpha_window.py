from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F
from .normalization import RMSNorm
from .rotary import RotaryEmbedding, apply_rope

AlphaWindowState = Tuple[torch.Tensor, torch.Tensor]  # (ssm_state, summary_state)

class AlphaWindow(nn.Module):
    """
    Hybrid SSM + attention block with explicit dtype discipline.
    The SSM is computed in fp32 for stability and cast back to the module dtype.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        ssm_state_dim: int,
        alpha_window: int,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
        use_bias: bool = False,
        residual_fp32: bool = True,
        summary_dim: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.ssm_state_dim = ssm_state_dim
        self.alpha_window = alpha_window
        self.residual_fp32 = residual_fp32
        self.summary_dim = summary_dim

        if d_model != n_heads * d_head:
            raise ValueError(f"d_model ({d_model}) must equal n_heads * d_head ({n_heads * d_head})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_rep = n_heads // n_kv_heads

        self.norm1 = RMSNorm(d_model, norm_eps)
        self.norm_ssm = RMSNorm(d_model, norm_eps)
        self.norm_attn = RMSNorm(d_model, norm_eps)

        self.in_proj = nn.Linear(d_model, ssm_state_dim, bias=use_bias)
        self.state_proj = nn.Linear(ssm_state_dim, d_model, bias=use_bias)
        self.decay = nn.Parameter(torch.zeros(ssm_state_dim))
        self.gate = nn.Linear(d_model, d_model, bias=use_bias)
        self.summary_proj = nn.Linear(d_model, summary_dim, bias=use_bias)
        self.summary_to_model = nn.Linear(summary_dim, d_model, bias=use_bias)

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=use_bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=use_bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=use_bias)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=use_bias)

        self.rotary = RotaryEmbedding(d_head, base=rope_theta, max_seq_len=alpha_window)
        self.attn_dropout = attn_dropout

    def _init_state(self, x: torch.Tensor) -> AlphaWindowState:
        bsz = x.size(0)
        device = x.device
        dtype = x.dtype
        ssm_state = torch.zeros(bsz, self.ssm_state_dim, device=device, dtype=dtype)
        summary_state = torch.zeros(bsz, self.summary_dim, device=device, dtype=dtype)
        return ssm_state, summary_state

    def _ssm_scan(self, x: torch.Tensor, ssm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stable sequential fp32 scan (resolving the NaN underflow/overflow bounds)
        x_fp32 = x.float()
        bsz, seq_len, _ = x_fp32.shape
        h = ssm_state.float()
        decay = torch.sigmoid(self.decay.float()).clamp(1e-4, 1 - 1e-4)

        in_w = self.in_proj.weight.float()
        in_b = self.in_proj.bias.float() if self.in_proj.bias is not None else None
        out_w = self.state_proj.weight.float()
        out_b = self.state_proj.bias.float() if self.state_proj.bias is not None else None

        u = torch.tanh(torch.nn.functional.linear(x_fp32, in_w, in_b))  # [B,T,S]

        h_seq = []
        for t in range(seq_len):
            h = h * decay + u[:, t, :]
            h_seq.append(h)

        h_seq_tensor = torch.stack(h_seq, dim=1) # [B,T,S]
        y = torch.nn.functional.linear(h_seq_tensor, out_w, out_b)
        return y, h

    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        local_len = min(seq_len, self.alpha_window)
        x_local = x[:, -local_len:]
        q = self.q_proj(self.norm_attn(x_local))
        k = self.k_proj(self.norm_attn(x_local))
        v = self.v_proj(self.norm_attn(x_local))

        q = q.view(bsz, local_len, self.n_heads, self.d_head).transpose(1, 2)  # B,H,T,D
        k = k.view(bsz, local_len, self.n_kv_heads, self.d_head).transpose(1, 2)  # B,K,T,D
        v = v.view(bsz, local_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        cos, sin = self.rotary.get_cos_sin(local_len, x.device, q.dtype)
        cos = cos.view(1, 1, local_len, self.d_head)
        sin = sin.view(1, 1, local_len, self.d_head)
        q, k = apply_rope(q, k, cos, sin)

        # causal mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.ones(local_len, local_len, device=x.device, dtype=torch.bool).triu(1)
        attn_scores = attn_scores.masked_fill(mask.view(1, 1, local_len, local_len), float("-inf"))
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)
        if self.attn_dropout and self.training:
            attn_probs = F.dropout(attn_probs, p=self.attn_dropout)
        out = torch.matmul(attn_probs, v)  # B,H,T,D
        out = out.transpose(1, 2).contiguous().view(bsz, local_len, self.n_heads * self.d_head)
        out = self.o_proj(out)
        if local_len < seq_len:
            pad = torch.zeros(bsz, seq_len - local_len, self.d_model, device=out.device, dtype=out.dtype)
            out = torch.cat([pad, out], dim=1)
        return out

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: Optional[AlphaWindowState] = None,
    ) -> Tuple[torch.Tensor, AlphaWindowState]:
        if ssm_state is None:
            ssm_state = self._init_state(x)
        ssm_h, summary_h = ssm_state

        orig_dtype = x.dtype
        x_norm = self.norm1(x)
        x_gate = torch.sigmoid(self.gate(x_norm))

        ssm_out, new_ssm_h = self._ssm_scan(self.norm_ssm(x_norm), ssm_h)
        attn_out = self._local_attention(x_norm)

        # Fractal summary update (EMA over batch summary)
        pooled = x_norm.float().mean(dim=1)
        summary_update = torch.tanh(self.summary_proj(pooled)).to(orig_dtype)
        summary_h = 0.97 * summary_h + 0.03 * summary_update
        summary_bias = self.summary_to_model(summary_h).unsqueeze(1)

        y = x_norm + x_gate * (ssm_out.to(orig_dtype) + attn_out + summary_bias)
        y = y.to(self.o_proj.weight.dtype)

        # keep residual branch dtype aligned
        if self.residual_fp32:
            y = y.float().to(orig_dtype)

        return y, (new_ssm_h.to(orig_dtype), summary_h.to(orig_dtype))
