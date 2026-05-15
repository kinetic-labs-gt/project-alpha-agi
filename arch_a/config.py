
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class ArchAConfig:
    vocab_size: int = 65536
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_head: int = 64
    d_ff: int = 3072
    ssm_state_dim: int = 128
    alpha_window: int = 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    norm_eps: float = 1e-5
    algr_max_loops: int = 3
    algr_confidence_threshold: float = 0.82
    algr_temperature: float = 1.0
    nadd_steps: int = 6
    nadd_hidden_mult: int = 2
    nadd_noise_scale: float = 0.15
    use_bias: bool = False
    use_checkpointing: bool = True
    tie_embeddings: bool = True
    use_fp32_residuals: bool = True
    use_structured_memory: bool = True
    device_map: Optional[Dict[int, str]] = None  # layer shard plan if provided
    quant_block_size: int = 32
    shadow_bits: int = 4
    polar_bits: int = 4
    scale_critical_lr: float = 1e-4
    scale_noncritical_lr: float = 3e-4
    galore_rank: int = 8
    spectron_power_iters: int = 1
    spectron_target_norm: float = 1.0
    residual_summary_dim: int = 128
    num_codebooks: int = 1
    model_name: str = "arch_a"

    @classmethod
    def for_debug(cls) -> "ArchAConfig":
        return cls(
            vocab_size=512, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2,
            d_head=16, d_ff=256, ssm_state_dim=32, alpha_window=32, max_seq_len=128,
            nadd_steps=2, algr_max_loops=1
        )

    @classmethod
    def for_2gpu_demo(cls) -> "ArchAConfig":
        return cls(
            vocab_size=1024, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
            d_head=16, d_ff=256, ssm_state_dim=32, alpha_window=32, max_seq_len=128,
            nadd_steps=2, algr_max_loops=1
        )

    @classmethod
    def for_20m(cls) -> "ArchAConfig":
        return cls(
            vocab_size=50257,          # GPT2 tokenizer vocab
            d_model=384,
            n_layers=6,
            n_heads=6,
            n_kv_heads=2,
            d_head=64,
            d_ff=1536,
            ssm_state_dim=64,
            alpha_window=128,
            max_seq_len=512,

            # Stability-focused settings
            dropout=0.1,
            attn_dropout=0.1,

            # Safer ALGR settings
            algr_max_loops=2,
            algr_confidence_threshold=0.88,
            algr_temperature=1.0,

            # Safer NADD settings
            nadd_steps=3,
            nadd_hidden_mult=2,
            nadd_noise_scale=0.10,

            # Memory/runtime safety
            use_checkpointing=True,
            use_fp32_residuals=True,
            use_structured_memory=False,

            # Optimizer helpers
            galore_rank=4,
            residual_summary_dim=64,

            model_name="arch_a_20m"
        )
    
    @classmethod
    def for_50m(cls) -> "ArchAConfig":
        return cls(
            vocab_size=32768, d_model=512, n_layers=8, n_heads=8, n_kv_heads=4,
            d_head=64, d_ff=2048, max_seq_len=1024
        )

    @classmethod
    def for_500m(cls) -> "ArchAConfig":
        return cls(
            vocab_size=32768, d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4,
            d_head=64, d_ff=4096, ssm_state_dim=128, alpha_window=256, max_seq_len=2048,
            nadd_steps=6, algr_max_loops=3
        )

    @classmethod
    def for_1b(cls) -> "ArchAConfig":
        return cls(
            vocab_size=65536, d_model=1536, n_layers=24, n_heads=24, n_kv_heads=6,
            d_head=64, d_ff=6144, ssm_state_dim=192, alpha_window=256, max_seq_len=4096,
            nadd_steps=8, algr_max_loops=3
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchAConfig":
        return cls(**d)
