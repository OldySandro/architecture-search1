from dataclasses import dataclass, field
from typing import Dict
from enum import Enum


# ── Attention Mechanisms ───────────────────────────────────────────────────────

class AttentionType(Enum):
    MHA    = "Multi-Head Attention (MHA)"
    MQA    = "Multi-Query Attention (MQA)"
    GQA    = "Grouped-Query Attention (GQA)"
    SLIDE  = "Sliding-Window Attention"
    LINEAR = "Linear Attention (Approx)"
    ALIBI  = "ALiBi Relative Pos"
    ROPE   = "RoPE Rotary Embed"
    HYBRID = "Hybrid Local+Global"


# ── FFN Variants ───────────────────────────────────────────────────────────────

class FFNType(Enum):
    DENSE    = "Dense FFN (SwiGLU)"
    MOE      = "Mixture-of-Experts"
    GATED    = "Gated Linear Unit"
    GEGLU    = "GeGLU Activation"
    MOE_TOPK = "MoE Top-K Sparse"


# ── Normalization ──────────────────────────────────────────────────────────────

class NormType(Enum):
    RMSNORM   = "RMSNorm"
    LAYERNORM = "LayerNorm"
    GROUPNORM = "GroupNorm"


# ── Positional Encoding ────────────────────────────────────────────────────────

class PosEncType(Enum):
    ROPE    = "RoPE"
    ALIBI   = "ALiBi"
    SINCOS  = "Sinusoidal"
    LEARNED = "Learned Absolute"
    NONE    = "None (ALiBi-style)"


# ── Optimizer Types ────────────────────────────────────────────────────────────

class OptimizerType(Enum):
    """Optimizer dengan footprint memori per parameter."""
    ADAM_FP32  = "Adam FP32 states (3× params)"
    ADAM_8BIT  = "Adam 8-bit / bitsandbytes (~1.5× params)"
    LION       = "Lion (2× params, momentum only)"
    ADAMW_BF16 = "AdamW BF16 states (3× params @ BF16)"
    ZERO1      = "ZeRO Stage-1 (optim sharded, N GPU)"
    ZERO2      = "ZeRO Stage-2 (optim+grad sharded)"
    ZERO3      = "ZeRO Stage-3 (full param+optim+grad)"


# ── Architecture Configuration ─────────────────────────────────────────────────

@dataclass
class ArchConfig:
    """Konfigurasi arsitektur pretraining lengkap."""
    arch_id:     str = ""
    arch_name:   str = ""
    arch_family: str = ""

    # ─ Core dims ──────────────────────────────────────────────────────────────
    vocab_size:  int   = 32000
    hidden_dim:  int   = 1024
    num_layers:  int   = 12
    seq_len:     int   = 2048
    batch_size:  int   = 1

    # ─ Attention ──────────────────────────────────────────────────────────────
    attn_type:          AttentionType = AttentionType.GQA
    num_heads:          int           = 8
    num_kv_heads:       int           = 2
    head_dim:           int           = 64
    window_size:        int           = 512
    global_attn_layers: int           = 4

    # ─ FFN ────────────────────────────────────────────────────────────────────
    ffn_type:               FFNType = FFNType.DENSE
    ffn_multiplier:         float   = 4.0
    num_experts:            int     = 8
    top_k_experts:          int     = 2
    expert_capacity_factor: float   = 1.25

    # ─ Optimizer ──────────────────────────────────────────────────────────────
    optimizer_type: OptimizerType = OptimizerType.ADAM_FP32

    # ─ Misc ───────────────────────────────────────────────────────────────────
    norm_type:                   NormType    = NormType.RMSNORM
    pos_enc:                     PosEncType  = PosEncType.ROPE
    tie_embeddings:              bool        = True
    use_flash_attn:              bool        = True
    use_gradient_checkpointing:  bool        = False
    use_mixed_precision:         bool        = True
    use_torch_compile:           bool        = True
    dropout:                     float       = 0.0

    # ─ Derived (dihitung oleh ArchitectureGenerator) ──────────────────────────
    param_count:             int   = 0
    vram_weights_gb:         float = 0.0
    vram_activations_gb:     float = 0.0
    vram_optimizer_gb:       float = 0.0
    vram_kv_cache_gb:        float = 0.0
    vram_fragmentation_gb:   float = 0.0
    vram_total_gb:           float = 0.0
    vram_usage_pct:          float = 0.0
    flops_per_token_fwd:     float = 0.0
    flops_per_token_bwd:     float = 0.0
    flops_attn_fwd:          float = 0.0
    flops_ffn_fwd:           float = 0.0
    arithmetic_intensity:    float = 0.0

    # ─ Profiling results ──────────────────────────────────────────────────────
    tokens_per_sec_estimate: float = 0.0
    mfu_estimate:            float = 0.0
    ms_per_step:             float = 0.0
    bottleneck:              str   = ""
    bottleneck_factors:      Dict  = field(default_factory=dict)
    fits_gpu:                bool  = True
    fitness_score:           float = 0.0
    compiler_speedup:        float = 1.0
    warp_divergence_pct:     float = 0.0
    sm_occupancy:            float = 0.0
