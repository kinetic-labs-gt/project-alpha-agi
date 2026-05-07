from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import ArchAConfig
from .modules import ALGRBlock, ALGRController, NADDDecoder, RMSNorm
from .modules.alpha_window import AlphaWindowState
from .training.mxfp8 import blockwise_mxfp8_quantize, blockwise_mxfp8_dequantize

@dataclass
class ArchAOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    ssm_states: Optional[List[AlphaWindowState]] = None
    algr_meta: Optional[Any] = None
    aux_losses: Optional[Dict[str, torch.Tensor]] = None

class ArchAModel(nn.Module):
    def __init__(self, config: ArchAConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            ALGRBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                ssm_state_dim=config.ssm_state_dim,
                alpha_window=config.alpha_window,
                d_ff=config.d_ff,
                rope_theta=config.rope_theta,
                norm_eps=config.norm_eps,
                dropout=config.dropout,
                use_bias=config.use_bias,
                residual_fp32=config.use_fp32_residuals,
                summary_dim=config.residual_summary_dim,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.controller = ALGRController(
            self.layers,
            d_model=config.d_model,
            max_loops=config.algr_max_loops,
            confidence_threshold=config.algr_confidence_threshold,
            temperature=config.algr_temperature,
            device_map=None,
        )
        self.nadd_decoder = NADDDecoder(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            steps=config.nadd_steps,
            hidden_mult=config.nadd_hidden_mult,
            dropout=config.dropout,
            use_bias=config.use_bias,
            noise_scale=config.nadd_noise_scale,
        )
        self.device_map: Optional[List[torch.device]] = None
        if self.config.tie_embeddings:
            self.tie_weights()

    def shard_to_devices(self, devices: Sequence[str | torch.device]):
        devices = [torch.device(d) for d in devices]
        self.device_map = devices
        if len(devices) == 0:
            return self
        # simple contiguous partition
        per = math.ceil(len(self.layers) / len(devices))
        map_list = []
        for i, layer in enumerate(self.layers):
            dev = devices[min(i // per, len(devices) - 1)]
            layer.to(dev)
            map_list.append(dev)
        self.controller.device_map = map_list
        self.controller.halt_head.to(devices[0])
        self.token_embedding.to(devices[0])
        self.final_norm.to(devices[-1])
        self.nadd_decoder.to(devices[-1])
        # weight tying only when single device
        if self.config.tie_embeddings and len(devices) == 1:
            self.tie_weights()
        return self

    def tie_weights(self):
        self.nadd_decoder.out_proj.weight = self.token_embedding.weight
        return self

    def init_states(self, batch_size: int, device: Optional[torch.device] = None):
        states = []
        for i, layer in enumerate(self.layers):
            dev = device if device is not None else next(layer.parameters()).device
            ssm = torch.zeros(batch_size, self.config.ssm_state_dim, device=dev, dtype=next(layer.parameters()).dtype)
            summary = torch.zeros(batch_size, self.config.residual_summary_dim, device=dev, dtype=next(layer.parameters()).dtype)
            states.append((ssm, summary))
        return states

    def forward_backbone(self, input_ids: torch.Tensor, ssm_states=None, training: bool = True):
        emb_dev = self.token_embedding.weight.device
        if input_ids.device != emb_dev:
            input_ids = input_ids.to(emb_dev, non_blocking=True)
        x = self.token_embedding(input_ids)
        x = self.emb_dropout(x)
        x, ssm_states, algr_meta = self.controller(self.layers, x, ssm_states, training=training)
        x = self.final_norm(x)
        return x, ssm_states, algr_meta

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ssm_states=None,
        training_mode: str = "hybrid",
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        hidden, ssm_states, algr_meta = self.forward_backbone(input_ids, ssm_states=ssm_states, training=self.training)
        anchor = hidden.mean(dim=1)

        # AR head uses tied embedding weights when possible, otherwise the decoder projection weight.
        ar_weight = self.token_embedding.weight if (self.config.tie_embeddings and self.device_map is None) else self.nadd_decoder.out_proj.weight
        ar_logits = F.linear(hidden, ar_weight)

        refined_hidden, nadd_logits = self.nadd_decoder(hidden, anchor_state=anchor)
        if training_mode == "nadd":
            logits = nadd_logits
            hidden_out = refined_hidden
        elif training_mode == "ar":
            logits = ar_logits
            hidden_out = hidden
        else:
            logits = 0.6 * ar_logits + 0.4 * nadd_logits
            hidden_out = refined_hidden

        loss = None
        aux_losses = {}
        if labels is not None:
            if labels.device != logits.device:
                labels = labels.to(logits.device, non_blocking=True)
            # causal next-token loss
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            # lightly regularize halting entropy/loops
            # Use differentiable probabilities to allow training the halting gate
            loop_penalty = algr_meta.loss_penalty if algr_meta.loss_penalty is not None else torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            aux_losses["loop_penalty"] = loop_penalty
            loss = loss + 1e-4 * loop_penalty

        out = ArchAOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_out if output_hidden_states else None,
            ssm_states=ssm_states,
            algr_meta=algr_meta,
            aux_losses=aux_losses if aux_losses else None,
        )
        return out if return_dict else (out.logits, out.loss, out.hidden_states, out.ssm_states, out.algr_meta)

    @torch.no_grad()
    def generate_ar(self, input_ids: torch.Tensor, max_new_tokens: int = 32, temperature: float = 1.0):
        self.eval()
        seq = input_ids
        for _ in range(max_new_tokens):
            # AlphaWindow local attention does not currently cache K/V pairs for sequence generation.
            # We must pass the full sequence and compute from scratch, so ssm_states must be None
            # to avoid double-counting prefix recurrence state.
            out = self.forward(seq, ssm_states=None, training_mode="ar", return_dict=True)
            logits = out.logits[:, -1] / max(temperature, 1e-6)
            next_token = torch.multinomial(F.softmax(logits.float(), dim=-1), num_samples=1)
            next_token = next_token.to(seq.device)
            seq = torch.cat([seq, next_token], dim=1)
        return seq

    @torch.no_grad()
    def generate_nadd(self, input_ids: torch.Tensor):
        self.eval()
        hidden, _, _ = self.forward_backbone(input_ids, training=False)
        # NADDDecoder internally applies `steps` passes over its refiners.
        # Calling it repeatedly creates an unintended quadratic loop.
        refined, logits = self.nadd_decoder(hidden, anchor_state=hidden.mean(dim=1))
        return logits

class ArchAForCausalLM(ArchAModel):
    def __init__(self, config: ArchAConfig):
        super().__init__(config)
