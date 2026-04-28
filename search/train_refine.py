"""
train_refine.py — Training-Aware NAS + RL Refinement Engine
═══════════════════════════════════════════════════════════════════════════════

Menggantikan training_aware.py dengan sistem NAS training yang komprehensif.

Sistem ini WAJIB mengevaluasi setiap arsitektur melalui simulasi training nyata
(PyTorch proxy), kemudian RL memperbaiki arsitektur hingga training dynamics
optimal.

Arsitektur:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  ProxyModel       → minimal transformer proxy (rasio terjaga)           │
  │  ProxyTrainer     → real PyTorch training (40-80 steps, warmup, clip)   │
  │  TrainingDynEval  → scoring 6 dimensi dari trajectory + heuristik       │
  │  TrainingQLearner → Q-learning cross-arch untuk training dynamics       │
  │  TrainingNASRefiner → Phase A (heuristik) + Phase B (RL training NAS)  │
  └──────────────────────────────────────────────────────────────────────────┘

Training Score — 6 Dimensi:
  T1  Convergence Rate        22 pts  — seberapa cepat loss turun
  T2  Training Stability      22 pts  — loss variance, NaN detection
  T3  Gradient Health         18 pts  — grad norm, vanishing/exploding risk
  T4  Generalization Gap       15 pts  — proxy train-val loss gap
  T5  Sample Efficiency       13 pts  — Chinchilla ratio, token efficiency
  T6  Optimizer Compatibility 10 pts  — kecocokan optimizer-arsitektur
                               ─────
  TOTAL                       100 pts  → training_score [0.0, 1.0]

Simulasi NAS Training:
  • Setiap arc diuji dengan proxy training hingga loss konvergen
  • Jika NaN/divergence → training_score = 0.0 → RL perlu memperbaiki
  • RL reward = Δtraining_score × 10 + Δconvergence × 5 + Δstability × 4
  • RL terus iterate sampai training_score optimal (threshold 0.75)

RL State Space:
  (score_bucket, grad_risk_bucket, stability_bucket, family_idx)
  Buckets: 6 × 4 × 5 × 7 = 840 states

RL Actions (9 training-centric perturbations):
  FIX_DEPTH_WIDTH, SWITCH_RMSNORM, INCR_BATCH_TRAIN, SWITCH_OPT_STABLE,
  ENABLE_MIXED_PREC, TIE_EMBEDDINGS, ADJUST_FFN_MULT, DISABLE_DROPOUT,
  ADJUST_LR_SENSITIVE
"""

from __future__ import annotations

import copy
import math
import random
import hashlib
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from arch_types import ArchConfig, FFNType, AttentionType, OptimizerType, NormType
from hardware import GPUSpec
from generator import ArchitectureGenerator, VRAM_LIMIT_PCT
from refiner import ArcQualityScorer, ArcRefiner, RefinementLog


# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA TRAINING
# ══════════════════════════════════════════════════════════════════════════════

# Chinchilla: optimal tokens ≈ 20× param count
CHINCHILLA_TOKEN_MULT: float = 20.0

# Gradient noise threshold: effective_batch < threshold → noisy gradient
GRAD_NOISE_THRESHOLD: int = 4096

# Depth-Width ratio optimal dari literature
OPTIMAL_DW_RATIO_LO: float = 0.15
OPTIMAL_DW_RATIO_HI: float = 0.55

# Atenuasi gradien per layer (1.5% toleransi per layer)
GRAD_ATTENUATION_PER_LAYER: float = 0.015

# Training score target untuk RL convergence
TRAINING_SCORE_TARGET: float = 0.75

# Proxy model config
PROXY_HIDDEN_MAX  = 64
PROXY_LAYERS_MAX  = 3
PROXY_SEQ_LEN     = 32
PROXY_VOCAB       = 256
PROXY_BATCH       = 2
PROXY_TRAIN_STEPS = 50
PROXY_WARMUP_STEPS = 5
PROXY_LR          = 3e-3
PROXY_WEIGHT_DECAY = 0.01
PROXY_GRAD_CLIP   = 1.0
PROXY_TAIL_STEPS  = 15    # langkah terakhir untuk variance
MAX_VALID_LOSS    = 8.0   # log(256) ≈ max; lebih = diverged

# RL
_FAMILY_LIST = [
    "CoT-Optimizer", "Speed-Demon", "Balanced-Pro", "MoE-Sparse",
    "Long-Horizon", "Nano-Efficient", "Compute-Dense",
]
_FAMILY_IDX: Dict[str, int] = {f: i for i, f in enumerate(_FAMILY_LIST)}

# Training score buckets
_TRAIN_BUCKETS = [
    (0.00, 0.15),  # 0: sangat buruk (NaN, divergence)
    (0.15, 0.30),  # 1: buruk
    (0.30, 0.50),  # 2: di bawah rata-rata
    (0.50, 0.65),  # 3: rata-rata
    (0.65, 0.80),  # 4: baik
    (0.80, 1.01),  # 5: excellent
]

# Gradient risk buckets
_GRAD_RISK_BUCKETS = [
    (0.00, 0.25),  # 0: kritis (vanishing/exploding)
    (0.25, 0.50),  # 1: tinggi
    (0.50, 0.75),  # 2: sedang
    (0.75, 1.01),  # 3: rendah (sehat)
]

# Stability buckets
_STAB_BUCKETS = [
    (0.00, 0.20),  # 0: sangat tidak stabil
    (0.20, 0.40),  # 1: buruk
    (0.40, 0.60),  # 2: sedang
    (0.60, 0.80),  # 3: baik
    (0.80, 1.01),  # 4: excellent
]

# RL Actions — 15 total (9 original + 6 baru untuk T2/T4)
TRAIN_ACTIONS = [
    # ── Original 9 (T1/T3/T5/T6 focus) ──────────────────────────────────────
    "FIX_DEPTH_WIDTH",       # perbaiki rasio depth/width → T1
    "SWITCH_RMSNORM",        # ganti ke RMSNorm (gradient flow) → T3
    "INCR_BATCH_TRAIN",      # naikkan batch (kurangi gradient noise) → T2/T5
    "SWITCH_OPT_STABLE",     # ganti ke Adam FP32 → T6
    "ENABLE_MIXED_PREC",     # mixed precision → T2 stability
    "TIE_EMBEDDINGS",        # tied embedding → T4/T5
    "ADJUST_FFN_MULT",       # sesuaikan FFN multiplier → T1/T5
    "DISABLE_DROPOUT",       # matikan dropout pretraining → T2
    "FIX_OPTIMIZER_DEPTH",   # optimizer sesuai depth → T6
    # ── New 6 (T2/T4 focus) ──────────────────────────────────────────────────
    "STABILIZE_LR_DEEP",     # mark model butuh lower LR proxy → T2 stability
    "REDUCE_CAPACITY_GEN",   # kurangi layers 2–3 → less overfit → T4 generalization
    "SWITCH_ADAMW_DECAY",    # switch ke AdamW (weight_decay=0.1) → T4 regularization
    "ENABLE_DROPOUT_REG",    # tambah dropout kecil 0.05 → T4 regularization
    "FIX_GQA_STABILITY",     # MHA→GQA (sedikit params/step → grad lebih stabil) → T2/T3
    "SHRINK_HIDDEN_REG",     # kurangi hidden_dim kecil → less capacity → T4
]

_N_TRAIN_BUCKETS = len(_TRAIN_BUCKETS)
_N_GRAD_BUCKETS  = len(_GRAD_RISK_BUCKETS)
_N_STAB_BUCKETS  = len(_STAB_BUCKETS)
_N_FAM           = len(_FAMILY_LIST)
_N_ACTIONS       = len(TRAIN_ACTIONS)   # 15


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProxyTrainingResult:
    """Hasil satu sesi proxy training PyTorch."""
    arch_fingerprint: str = ""

    # Raw trajectories
    train_losses: List[float] = field(default_factory=list)
    val_losses:   List[float] = field(default_factory=list)
    grad_norms:   List[float] = field(default_factory=list)

    # Derived metrics
    loss_initial:         float = 0.0
    loss_final:           float = 0.0
    loss_variance:        float = 999.0  # variance di tail
    grad_norm_mean:       float = 0.0
    grad_norm_variance:   float = 999.0
    nan_detected:         bool  = False

    convergence_rate:     float = 0.0   # (L_init - L_final) / steps
    stability_score:      float = 0.0   # [0, 1]
    generalization_gap:   float = 999.0  # val_loss_final - train_loss_final
    overfitting_trend:    float = 0.0   # slope val_loss di tail (positif = overfit)
    loss_smoothness:      float = 0.0   # [0, 1] kehalusan kurva loss

    training_time_ms:  float = 0.0
    proxy_param_count: int   = 0
    steps_completed:   int   = 0

    @property
    def is_stable(self) -> bool:
        return not self.nan_detected and self.stability_score > 0.40

    @property
    def converged(self) -> bool:
        return self.convergence_rate > 0.005 and self.loss_final < MAX_VALID_LOSS * 0.6

    def compute_derived(self, steps: int, tail: int = PROXY_TAIL_STEPS) -> None:
        """Hitung semua metrik derived dari raw trajectories."""
        n = len(self.train_losses)
        if n == 0:
            return

        self.steps_completed = n
        self.loss_initial     = self.train_losses[0]
        self.loss_final       = self.train_losses[-1]

        # Convergence rate: rata-rata penurunan loss per step
        self.convergence_rate = max(0.0, self.loss_initial - self.loss_final) / max(1, steps)

        # Loss variance (tail): stabilitas di akhir training
        # PENTING: gunakan RELATIVE variance (CV²) bukan absolute variance
        # Absolute variance unfair untuk model cepat-konvergen (loss kecil → variance kecil absolut tapi wajar)
        tail_losses = self.train_losses[-min(tail, n):]
        if len(tail_losses) >= 2:
            t_arr         = torch.tensor(tail_losses, dtype=torch.float32)
            abs_var       = float(t_arr.var().item())
            tail_mean     = float(t_arr.mean().item())
            # CV² = variance / mean² — normalisasi terhadap level loss aktual
            self.loss_variance = abs_var / max(1e-4, tail_mean ** 2)
        else:
            self.loss_variance = 0.0

        # Gradient statistics
        if self.grad_norms:
            gn = torch.tensor(self.grad_norms, dtype=torch.float32)
            self.grad_norm_mean     = float(gn.mean().item())
            self.grad_norm_variance = float(gn.var().item()) if len(self.grad_norms) >= 2 else 0.0

        # Stability score — berdasarkan relative variance (CV²)
        if self.nan_detected:
            self.stability_score = 0.0
        else:
            # CV² < 0.02 = sangat stabil, CV² > 0.5 = tidak stabil
            var_penalty = 1.0 / (1.0 + self.loss_variance * 5.0)   # calibrated untuk CV²
            # Gradient norm variance: normalized terhadap mean grad norm
            gnv_norm    = (self.grad_norm_variance / max(1e-6, self.grad_norm_mean ** 2)
                           if self.grad_norm_mean > 0 else 0.0)
            gnv_penalty = 1.0 / (1.0 + gnv_norm * 3.0)
            loss_ok     = 1.0 if self.loss_final < MAX_VALID_LOSS else 0.0
            self.stability_score = float(np.clip(
                var_penalty * 0.50 + gnv_penalty * 0.30 + loss_ok * 0.20,
                0.0, 1.0
            ))

        # Generalization gap — RELATIVE terhadap loss_initial
        # Absolute gap misleading: model cepat-konvergen punya train_loss kecil
        # sehingga val-train gap terlihat besar padahal relatif kecil
        if len(self.val_losses) > 0:
            val_final = self.val_losses[-1]
            abs_gap   = max(0.0, val_final - self.loss_final)
            # Relative gap: gap / loss_initial → normalized ke skala yang sama semua arch
            self.generalization_gap = abs_gap / max(0.1, self.loss_initial)

            # Overfitting trend: slope val_loss di tail
            if len(self.val_losses) >= 3:
                val_tail = self.val_losses[-min(tail, len(self.val_losses)):]
                t_vals   = list(range(len(val_tail)))
                if len(t_vals) >= 2:
                    mean_t = sum(t_vals) / len(t_vals)
                    mean_v = sum(val_tail) / len(val_tail)
                    cov    = sum((t - mean_t) * (v - mean_v)
                                 for t, v in zip(t_vals, val_tail))
                    var_t  = sum((t - mean_t) ** 2 for t in t_vals)
                    # Normalize trend oleh initial loss scale
                    raw_trend = cov / max(1e-6, var_t)
                    self.overfitting_trend = raw_trend / max(0.1, self.loss_initial)
        else:
            self.generalization_gap = 0.0

        # Loss smoothness
        if n >= 5:
            diffs = [abs(self.train_losses[i] - self.train_losses[i-1])
                     for i in range(1, n)]
            mean_diff = sum(diffs) / len(diffs)
            # Normalize mean_diff relative to loss scale
            rel_diff = mean_diff / max(0.1, self.loss_initial)
            self.loss_smoothness = float(np.clip(
                1.0 / (1.0 + rel_diff * 15.0), 0.0, 1.0
            ))
        else:
            self.loss_smoothness = 0.5


@dataclass
class TrainingNASResult:
    """Hasil lengkap Training NAS evaluation."""
    arch_id: str = ""

    # Sub-scores [0, 1]
    convergence_score:    float = 0.0   # T1
    stability_score:      float = 0.0   # T2
    gradient_health:      float = 0.0   # T3
    generalization_score: float = 0.0   # T4
    sample_efficiency:    float = 0.0   # T5
    optimizer_compat:     float = 0.0   # T6

    # Aggregated
    training_score: float = 0.0          # [0, 1] weighted

    # Metadata
    proxy_result:    Optional[ProxyTrainingResult] = None
    training_time_ms: float = 0.0

    # Interpretasi
    gradient_risk:  str = "unknown"     # low / moderate / high / critical
    lr_sensitivity: str = "unknown"     # robust / sensitive / fragile
    regime:         str = ""            # karakteristik training singkat

    # Per-dimensi poin
    pts_t1: float = 0.0
    pts_t2: float = 0.0
    pts_t3: float = 0.0
    pts_t4: float = 0.0
    pts_t5: float = 0.0
    pts_t6: float = 0.0

    @property
    def total_pts(self) -> float:
        return self.pts_t1 + self.pts_t2 + self.pts_t3 + \
               self.pts_t4 + self.pts_t5 + self.pts_t6

    @property
    def grade(self) -> str:
        s = self.training_score
        if s >= 0.90: return "S ★★★  Excellent Trainability"
        if s >= 0.80: return "A+ ★★  Very Good"
        if s >= 0.70: return "A  ★   Good"
        if s >= 0.55: return "B      Acceptable"
        if s >= 0.40: return "C      Marginal"
        return              "F  ✗   Poor Trainability"


@dataclass
class TrainingAdaptiveLog:
    """Log satu siklus training NAS refinement."""
    arch_id:   str = ""
    arch_name: str = ""

    # Scores
    quality_start:       float = 0.0
    quality_end:         float = 0.0
    fitness_start:       float = 0.0
    fitness_end:         float = 0.0
    train_score_start:   float = 0.0
    train_score_end:     float = 0.0
    combined_start:      float = 0.0
    combined_end:        float = 0.0

    # NAS stats
    nas_evaluations:     int   = 0
    nas_cache_hits:      int   = 0
    nas_nan_count:       int   = 0
    nas_training_ms_total: float = 0.0

    # RL stats
    perturbation_tries:     int = 0
    perturbations_accepted: int = 0
    rl_replay_updates:      int = 0

    # Training improvement stats
    convergence_improvements: int = 0
    stability_improvements:   int = 0
    gradient_improvements:    int = 0

    improvement_events:   List[str] = field(default_factory=list)
    rule_effectiveness:   Dict[str, float] = field(default_factory=dict)
    status:               str = ""

    base_log: Optional[RefinementLog] = None

    @property
    def train_delta(self) -> float:
        return round(self.train_score_end - self.train_score_start, 4)

    @property
    def combined_delta(self) -> float:
        return round(self.combined_end - self.combined_start, 5)


# ══════════════════════════════════════════════════════════════════════════════
#  PROXY MODEL (Minimal Transformer)
# ══════════════════════════════════════════════════════════════════════════════

class ProxyTransformerLayer(nn.Module):
    """Satu layer transformer proxy — minimal tapi representatif."""

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        ffn_dim:     int,
        use_rmsnorm: bool = True,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_heads   = max(1, num_heads)
        head_dim         = hidden_dim // max(1, self.num_heads)

        # Attention (simplified MHA, no GQA untuk proxy)
        self.attn = nn.MultiheadAttention(
            hidden_dim, self.num_heads,
            dropout    = dropout,
            batch_first = True,
        )
        # FFN
        self.ff1 = nn.Linear(hidden_dim, ffn_dim)
        self.ff2 = nn.Linear(ffn_dim, hidden_dim)

        # Normalization
        if use_rmsnorm:
            # RMSNorm approximation dengan LayerNorm (elementwise_affine=False)
            self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        else:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        normed = self.norm2(x)
        ff_out = self.ff2(F.gelu(self.ff1(normed)))
        x = x + self.dropout(ff_out)
        return x


class ProxyTransformer(nn.Module):
    """
    Minimal transformer proxy.
    Dibangun dengan rasio yang terjaga dari ArchConfig:
      - num_heads / num_kv_heads ratio → GQA topology
      - ffn_multiplier → FFN width ratio
      - num_layers → kedalaman (dikurangi proporsional)
      - use_rmsnorm → pilihan normalization
    """

    def __init__(
        self,
        hidden_dim:   int,
        num_layers:   int,
        num_heads:    int,
        ffn_mult:     float,
        vocab_size:   int,
        use_rmsnorm:  bool = True,
        dropout:      float = 0.0,
    ):
        super().__init__()
        ffn_dim = max(hidden_dim * 2, int(hidden_dim * ffn_mult))
        ffn_dim = max(64, ffn_dim)   # minimum 64

        self.embed  = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(PROXY_SEQ_LEN + 4, hidden_dim)

        self.layers = nn.ModuleList([
            ProxyTransformerLayer(
                hidden_dim  = hidden_dim,
                num_heads   = num_heads,
                ffn_dim     = ffn_dim,
                use_rmsnorm = use_rmsnorm,
                dropout     = dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(hidden_dim)
        self.head    = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying (jika tie_embeddings)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.ff1.weight)
            nn.init.xavier_uniform_(layer.ff2.weight)
            nn.init.zeros_(layer.ff1.bias)
            nn.init.zeros_(layer.ff2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.embed(x) + self.pos_emb(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)   # [B, T, vocab_size]


# ══════════════════════════════════════════════════════════════════════════════
#  PROXY TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class ProxyTrainer:
    """
    Menjalankan real PyTorch proxy training untuk mengevaluasi training dynamics.

    Rasio arsitektur yang dipertahankan:
      - head_dim ratio (head_dim / hidden_dim yang diperkecil)
      - ffn_multiplier
      - depth/width ratio (layer/hidden_dim)
      - normalization type (RMSNorm vs LayerNorm)
      - dropout setting

    Evaluasi setiap arc wajib dilakukan. Jika loss diverge → training_score = 0.
    RL akan terus memperbaiki sampai loss konvergen.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def _fingerprint(self, cfg: ArchConfig) -> str:
        """Hash fingerprint dari parameter training-relevant."""
        key = (
            cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.head_dim,
            cfg.ffn_multiplier, cfg.batch_size, cfg.dropout,
            cfg.norm_type, cfg.optimizer_type,
            cfg.use_mixed_precision, cfg.tie_embeddings,
        )
        return hashlib.md5(str(key).encode()).hexdigest()[:16]

    def _build_proxy(self, cfg: ArchConfig) -> ProxyTransformer:
        """Bangun proxy model dengan skala diperkecil, rasio terjaga."""
        # Scale hidden_dim ke proxy max
        scale_factor = min(1.0, PROXY_HIDDEN_MAX / max(1, cfg.hidden_dim))

        # Proxy hidden_dim: diperkecil, minimal 16, harus habis dibagi num_heads
        proxy_hidden = max(16, int(cfg.hidden_dim * scale_factor))

        # Proxy num_heads: rasio head_dim terjaga
        proxy_heads = max(1, min(cfg.num_heads, proxy_hidden // max(1, cfg.head_dim)))
        # Pastikan proxy_hidden % proxy_heads == 0
        while proxy_hidden % proxy_heads != 0 and proxy_heads > 1:
            proxy_heads -= 1
        if proxy_hidden % proxy_heads != 0:
            proxy_hidden = proxy_heads * max(8, proxy_hidden // max(1, proxy_heads))

        # Proxy layers: max 3 tapi pertahankan "dalam vs dangkal" relatif
        proxy_layers = max(1, min(PROXY_LAYERS_MAX, max(1, cfg.num_layers // 8)))

        # FFN multiplier: sama persis
        ffn_mult = max(1.5, cfg.ffn_multiplier)

        # RMSNorm detection
        use_rmsnorm = cfg.norm_type in (
            NormType.RMSNORM,
        ) if hasattr(cfg, 'norm_type') else False
        # Fallback: cek string
        if hasattr(cfg.norm_type, 'value'):
            use_rmsnorm = "RMS" in cfg.norm_type.value.upper()
        elif isinstance(cfg.norm_type, str):
            use_rmsnorm = "RMS" in cfg.norm_type.upper()

        model = ProxyTransformer(
            hidden_dim  = proxy_hidden,
            num_layers  = proxy_layers,
            num_heads   = proxy_heads,
            ffn_mult    = ffn_mult,
            vocab_size  = PROXY_VOCAB,
            use_rmsnorm = use_rmsnorm,
            dropout     = cfg.dropout if hasattr(cfg, 'dropout') else 0.0,
        )

        return model

    def _build_optimizer(self, model: nn.Module, cfg: ArchConfig):
        """Build optimizer (legacy, pakai PROXY_LR)."""
        return self._build_optimizer_adaptive(model, cfg, PROXY_LR)

    def _build_optimizer_adaptive(self, model: nn.Module, cfg: ArchConfig, lr: float):
        """
        Build optimizer dengan LR yang sudah disesuaikan.
        AdamW dipakai untuk semua karena weight_decay membantu T4 generalization.
        """
        opt_type = cfg.optimizer_type
        params   = model.parameters()
        wd       = PROXY_WEIGHT_DECAY   # 0.01 default

        # AdamW dengan weight_decay lebih agresif untuk deep model (helps T4)
        if cfg.num_layers > 24:
            wd = 0.05   # lebih banyak regularization untuk model deep

        if opt_type == OptimizerType.LION:
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type in (OptimizerType.ADAM_8BIT,):
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt_type == OptimizerType.ADAM_FP32:
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt_type in (OptimizerType.ADAMW_BF16, OptimizerType.ZERO1,
                          OptimizerType.ZERO2, OptimizerType.ZERO3):
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def _lr_warmup(self, step: int, warmup: int, base_lr: float) -> float:
        if step < warmup:
            return base_lr * (step + 1) / max(1, warmup)
        # Cosine decay setelah warmup
        progress = (step - warmup) / max(1, PROXY_TRAIN_STEPS - warmup)
        return base_lr * (0.5 * (1.0 + math.cos(math.pi * progress)))

    def _generate_structured_data(self, n_tokens: int, seed: int = 42) -> torch.Tensor:
        """
        Generate structured token data untuk proxy training yang BERMAKNA.

        Pure random data tidak bisa dipelajari oleh model apapun → convergence
        signal menjadi noise. Structured data punya pola yang BISA dipelajari
        oleh model kecil dalam 60 steps sehingga convergence rate mencerminkan
        KUALITAS ARSITEKTUR, bukan seberapa beruntung random seed.

        Komposisi:
          40% Bigram patterns — predict token berdasarkan token sebelumnya
          30% Shift sequences — sequence adalah versi geser dari dirinya sendiri
          20% Modular arithmetic — (prev + k) % vocab
          10% True random — noise untuk mencegah overfitting sempurna
        """
        rng    = random.Random(seed)
        tokens = []
        vocab  = PROXY_VOCAB

        n_bigram = int(n_tokens * 0.40)
        n_shift  = int(n_tokens * 0.30)
        n_modulo = int(n_tokens * 0.20)
        n_random = n_tokens - n_bigram - n_shift - n_modulo

        # ── 40%: Bigram patterns (A→B, A→B, A→B, ...) ────────────────────
        # Buat kamus bigram: setiap token punya "successor" yang konsisten
        bigram_table = {i: rng.randint(0, vocab - 1) for i in range(vocab)}
        curr = rng.randint(0, vocab - 1)
        for _ in range(n_bigram):
            tokens.append(curr)
            curr = bigram_table[curr]

        # ── 30%: Shift sequences (subsequences yang berulang) ────────────
        base_seq_len = rng.randint(8, 16)
        base_seq     = [rng.randint(0, vocab - 1) for _ in range(base_seq_len)]
        for i in range(n_shift):
            tokens.append(base_seq[i % base_seq_len])

        # ── 20%: Modular arithmetic ((prev + k) % vocab) ─────────────────
        step_k = rng.randint(3, 17)   # prime-ish step untuk coverage baik
        curr   = rng.randint(0, vocab - 1)
        for _ in range(n_modulo):
            tokens.append(curr)
            curr = (curr + step_k) % vocab

        # ── 10%: True random ──────────────────────────────────────────────
        for _ in range(n_random):
            tokens.append(rng.randint(0, vocab - 1))

        # Shuffle ringan untuk mix pola tanpa menghilangkan struktur lokal
        # Hanya shuffle setiap blok 32 agar pola local tetap ada
        result = []
        block  = 32
        for i in range(0, len(tokens), block):
            blk = tokens[i:i+block]
            # Hanya 20% chance shuffle setiap blok
            if rng.random() < 0.20:
                random.shuffle(blk)
            result.extend(blk)

        result = result[:n_tokens]
        # Pad jika kurang
        while len(result) < n_tokens:
            result.append(rng.randint(0, vocab - 1))

        return torch.tensor(result[:n_tokens], dtype=torch.long)

    def _compute_adaptive_lr(self, cfg: ArchConfig) -> float:
        """
        LR adaptif berdasarkan arsitektur.

        Deep/wide model perlu LR lebih kecil untuk stabilitas:
          LR_eff = PROXY_LR × depth_scale × width_scale

        depth_scale = 1/sqrt(layers/12)   → L=12 → ×1.0, L=33 → ×0.60, L=48 → ×0.50
        width_scale = 1/log2(hidden/64)   → D=512 → ×0.38, D=896 → ×0.30 (flatter curve)

        Kombinasi di-clip ke [0.25, 1.0] × PROXY_LR.
        Ini fix utama untuk T2 Stability pada model deep.
        """
        depth_scale = 1.0 / math.sqrt(max(1.0, cfg.num_layers / 12.0))
        # Width: gunakan hidden_dim/256 sebagai baseline (lebih moderat)
        width_ratio = max(1.0, cfg.hidden_dim / 256.0)
        width_scale = 1.0 / math.sqrt(width_ratio)
        # Blend 70% depth + 30% width
        scale = depth_scale * 0.70 + width_scale * 0.30
        return PROXY_LR * float(np.clip(scale, 0.20, 1.0))

    def train(self, cfg: ArchConfig) -> ProxyTrainingResult:
        """
        Jalankan real proxy training dan return ProxyTrainingResult.

        Training procedure:
          1. Build scaled proxy model (rasio terjaga dari ArchConfig)
          2. Generate STRUCTURED data (bigram+shift+modulo) — bermakna untuk convergence
          3. Split menjadi train (80%) dan val (20%)
          4. Train PROXY_TRAIN_STEPS steps dengan ADAPTIVE LR + warmup + cosine decay
          5. Evaluasi val loss setiap 5 steps
          6. Compute semua derived metrics (relative variance, relative gap)

        Kenapa structured data + adaptive LR:
          - Random data → T1≈0 untuk semua arch (tidak discriminative)
          - Structured data → arch bagus konvergen lebih cepat → T1 discriminative
          - Fixed LR terlalu tinggi untuk model deep → oscillation → T2 anjlok
          - Adaptive LR → T2 mencerminkan ARSITEKTUR, bukan hyperparameter bug
        """
        t0     = time.perf_counter()
        result = ProxyTrainingResult()
        result.arch_fingerprint = self._fingerprint(cfg)

        # Seed dari fingerprint untuk reproducibility
        fp_seed = int(result.arch_fingerprint[:8], 16) % (2**31)

        try:
            model = self._build_proxy(cfg).to(self.device)
            result.proxy_param_count = sum(p.numel() for p in model.parameters())

            torch.manual_seed(fp_seed)

            # Adaptive LR — kunci untuk T2 Stability pada model deep
            adaptive_lr = self._compute_adaptive_lr(cfg)

            opt = self._build_optimizer_adaptive(model, cfg, adaptive_lr)

            # Generate STRUCTURED data
            total_tokens_needed = PROXY_SEQ_LEN * PROXY_BATCH * PROXY_TRAIN_STEPS
            n_total             = total_tokens_needed + PROXY_SEQ_LEN + 64
            all_data = self._generate_structured_data(n_total, seed=fp_seed)
            all_data = all_data.to(self.device)

            # Split: 80% train, 20% val
            split   = int(len(all_data) * 0.80)
            tr_data = all_data[:split]
            vl_data = all_data[split:]

            def get_batch(data, step, rng_offset=0):
                """Ambil batch dari data dengan sedikit offset untuk variasi."""
                n = len(data) - PROXY_SEQ_LEN - 1
                if n <= 0:
                    data = data.repeat(4)
                    n    = len(data) - PROXY_SEQ_LEN - 1
                idx = (step * PROXY_BATCH * PROXY_SEQ_LEN + rng_offset) % max(1, n)
                seqs = []
                for b in range(PROXY_BATCH):
                    start = (idx + b * PROXY_SEQ_LEN) % max(1, n)
                    seqs.append(data[start : start + PROXY_SEQ_LEN + 1])
                tokens = torch.stack([s[:PROXY_SEQ_LEN] for s in seqs])
                labels = torch.stack([s[1:PROXY_SEQ_LEN+1] for s in seqs])
                return tokens, labels

            # Label smoothing factor — reduces train-val gap (helps T4)
            # Deep model dengan high convergence rate rawan overfit structured data
            label_smooth = 0.05 if cfg.num_layers > 20 else 0.0

            # ── Training loop ───────────────────────────────────────────────
            model.train()
            for step in range(PROXY_TRAIN_STEPS):
                # Adaptive LR warmup + cosine decay (menggunakan adaptive_lr bukan PROXY_LR)
                lr = self._lr_warmup(step, PROXY_WARMUP_STEPS, adaptive_lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                x, y = get_batch(tr_data, step)
                opt.zero_grad()
                logits = model(x)   # [B, T, V]

                # Cross-entropy dengan optional label smoothing untuk T4
                if label_smooth > 0:
                    loss = F.cross_entropy(
                        logits.view(-1, PROXY_VOCAB),
                        y.view(-1),
                        label_smoothing=label_smooth,
                    )
                else:
                    loss = F.cross_entropy(
                        logits.view(-1, PROXY_VOCAB),
                        y.view(-1),
                    )

                # NaN detection
                if torch.isnan(loss) or torch.isinf(loss):
                    result.nan_detected = True
                    result.train_losses.append(float("inf"))
                    break

                loss.backward()

                # Gradient clipping + norm measurement
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), PROXY_GRAD_CLIP)
                    .item()
                )
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    result.nan_detected = True
                    break

                opt.step()

                result.train_losses.append(float(loss.item()))
                result.grad_norms.append(grad_norm)

                # Val evaluation setiap 3 steps (lebih sering → lebih banyak datapoint T4)
                if step % 3 == 2 or step == PROXY_TRAIN_STEPS - 1:
                    model.eval()
                    with torch.no_grad():
                        # Gunakan offset berbeda untuk val batch agar tidak overlap train
                        vx, vy = get_batch(vl_data, step, rng_offset=7)
                        # Val tanpa label smoothing (mengukur true generalization)
                        vl = F.cross_entropy(
                            model(vx).view(-1, PROXY_VOCAB),
                            vy.view(-1),
                        )
                        if not torch.isnan(vl):
                            result.val_losses.append(float(vl.item()))
                        else:
                            result.val_losses.append(MAX_VALID_LOSS)
                    model.train()

        except Exception as e:
            result.nan_detected = True
            if not result.train_losses:
                result.train_losses = [MAX_VALID_LOSS]

        finally:
            result.training_time_ms = (time.perf_counter() - t0) * 1000

        # Compute derived metrics (relative variance + relative gap)
        result.compute_derived(PROXY_TRAIN_STEPS, PROXY_TAIL_STEPS)

        return result


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING DYNAMICS EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class TrainingDynamicsEvaluator:
    """
    Menggabungkan hasil ProxyTrainer (real signal) dengan analisis heuristik
    (gradient flow, optimizer compat, sample efficiency) untuk menghasilkan
    training_score yang komprehensif.

    Skor 6 dimensi (total 100 pts → training_score [0, 1]):
      T1  Convergence Rate     22 pts
      T2  Training Stability   22 pts
      T3  Gradient Health      18 pts
      T4  Generalization Gap   15 pts
      T5  Sample Efficiency    13 pts
      T6  Optimizer Compat     10 pts
    """

    W_T1 = 0.22
    W_T2 = 0.22
    W_T3 = 0.18
    W_T4 = 0.15
    W_T5 = 0.13
    W_T6 = 0.10

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu

    def evaluate(
        self,
        cfg:    ArchConfig,
        proxy:  ProxyTrainingResult,
    ) -> TrainingNASResult:
        """
        Gabungkan proxy training result + heuristik → TrainingNASResult.
        """
        result = TrainingNASResult(arch_id=cfg.arch_id)
        result.proxy_result    = proxy
        result.training_time_ms = proxy.training_time_ms

        # T1: Convergence Rate (22 pts)
        t1 = self._score_convergence(cfg, proxy)
        result.convergence_score = t1
        result.pts_t1           = round(t1 * 22.0, 2)

        # T2: Training Stability (22 pts)
        t2 = self._score_stability(cfg, proxy)
        result.stability_score = t2
        result.pts_t2          = round(t2 * 22.0, 2)

        # T3: Gradient Health (18 pts)
        t3, grad_risk = self._score_gradient_health(cfg, proxy)
        result.gradient_health = t3
        result.gradient_risk   = grad_risk
        result.pts_t3          = round(t3 * 18.0, 2)

        # T4: Generalization Gap (15 pts)
        t4 = self._score_generalization(cfg, proxy)
        result.generalization_score = t4
        result.pts_t4               = round(t4 * 15.0, 2)

        # T5: Sample Efficiency (13 pts)
        t5 = self._score_sample_efficiency(cfg)
        result.sample_efficiency = t5
        result.pts_t5            = round(t5 * 13.0, 2)

        # T6: Optimizer Compatibility (10 pts)
        t6, lr_sens = self._score_optimizer_compat(cfg)
        result.optimizer_compat = t6
        result.lr_sensitivity   = lr_sens
        result.pts_t6           = round(t6 * 10.0, 2)

        # Weighted combination
        result.training_score = float(np.clip(
            self.W_T1 * t1 +
            self.W_T2 * t2 +
            self.W_T3 * t3 +
            self.W_T4 * t4 +
            self.W_T5 * t5 +
            self.W_T6 * t6,
            0.0, 1.0
        ))

        # Training regime
        result.regime = self._classify_regime(cfg, proxy, result)

        return result

    # ── T1: Convergence Rate ────────────────────────────────────────────────

    def _score_convergence(self, cfg: ArchConfig, proxy: ProxyTrainingResult) -> float:
        if proxy.nan_detected or not proxy.train_losses:
            return 0.0

        # ── Relative reduction scoring (calibrated untuk structured data) ──
        #
        # Dengan structured data (bigram+shift+modulo), proxy model BISA belajar.
        # Expected relative reduction dalam 60 steps pada structured data:
        #   Excellent arch (deep, good optim): 25–50%+ reduction
        #   Good arch:                         15–25%
        #   Average arch:                      8–15%
        #   Poor arch (bad depth/width):       2–8%
        #   Failing (NaN/divergence):          <2% atau negatif
        #
        # Berbeda dari pure random data di mana bahkan arch terbaik hanya ~2-5%.

        loss_init  = proxy.loss_initial
        loss_fin   = proxy.loss_final

        if loss_init > 0 and not math.isnan(loss_init) and not math.isnan(loss_fin):
            rel_reduction = max(0.0, (loss_init - loss_fin) / loss_init)
        else:
            return 0.0

        # Scoring bertingkat (calibrated untuk structured proxy data)
        if rel_reduction >= 0.40:            # ≥40%: excellent convergence
            rate_score = 1.00
        elif rel_reduction >= 0.25:          # 25–40%: very good
            rate_score = 0.85 + (rel_reduction - 0.25) / 0.15 * 0.15
        elif rel_reduction >= 0.15:          # 15–25%: good
            rate_score = 0.65 + (rel_reduction - 0.15) / 0.10 * 0.20
        elif rel_reduction >= 0.08:          # 8–15%: acceptable (proxy typical)
            rate_score = 0.40 + (rel_reduction - 0.08) / 0.07 * 0.25
        elif rel_reduction >= 0.03:          # 3–8%: marginal
            rate_score = 0.15 + (rel_reduction - 0.03) / 0.05 * 0.25
        else:                                # <3%: tidak konvergen
            rate_score = max(0.0, rel_reduction / 0.03 * 0.15)

        # Bonus: konvergensi smooth (tidak spaiky)
        rate_score = min(1.0, rate_score + proxy.loss_smoothness * 0.06)

        # Penalti: final loss masih sangat tinggi (>85% dari MAX = diverged/stuck)
        if loss_fin > MAX_VALID_LOSS * 0.85:
            rate_score *= 0.20

        # Heuristik dari ArchConfig: depth-width ratio
        dw_ratio = cfg.num_layers / max(1.0, math.sqrt(max(1, cfg.hidden_dim)))
        if OPTIMAL_DW_RATIO_LO <= dw_ratio <= OPTIMAL_DW_RATIO_HI:
            dw_bonus = 0.06
        else:
            excess   = abs(dw_ratio - (OPTIMAL_DW_RATIO_LO + OPTIMAL_DW_RATIO_HI) / 2)
            dw_bonus = -0.03 * min(excess, 3.0)

        return float(np.clip(rate_score + dw_bonus, 0.0, 1.0))

    # ── T2: Training Stability ──────────────────────────────────────────────

    def _score_stability(self, cfg: ArchConfig, proxy: ProxyTrainingResult) -> float:
        if proxy.nan_detected:
            return 0.0

        # Base dari relative CV² stability_score (sudah diperbaiki di compute_derived)
        base = proxy.stability_score

        # Loss smoothness bonus — smooth curve = lebih stabil
        smooth_bonus = proxy.loss_smoothness * 0.08

        # Mixed precision: BF16 lebih numerically stable
        mp_bonus = 0.04 if cfg.use_mixed_precision else -0.02

        # Dropout pretraining standard = 0.0 (tidak ada dropout = lebih stabil)
        dropout_ok = 0.02 if cfg.dropout == 0.0 else -0.04

        # Penalti: gradient checkpointing tidak perlu (overhead tanpa manfaat)
        gc_pen = -0.03 if (cfg.use_gradient_checkpointing and
                           cfg.vram_usage_pct < 65) else 0.0

        # Bonus: optimizer yang diketahui stabil
        opt_bonus = 0.03 if cfg.optimizer_type in (
            OptimizerType.ADAM_FP32, OptimizerType.ADAMW_BF16
        ) else 0.0

        # Penalti depth: model sangat dalam lebih rentan osilasi
        depth_pen = -0.02 * max(0, (cfg.num_layers - 30) / 10.0)

        return float(np.clip(
            base + smooth_bonus + mp_bonus + dropout_ok + gc_pen + opt_bonus + depth_pen,
            0.0, 1.0
        ))

    # ── T3: Gradient Health ─────────────────────────────────────────────────

    def _score_gradient_health(
        self, cfg: ArchConfig, proxy: ProxyTrainingResult
    ) -> Tuple[float, str]:
        """Returns (score, risk_level)."""

        if proxy.nan_detected:
            return 0.0, "critical"

        # Base dari proxy grad norm
        if proxy.grad_norm_mean > 0:
            # Healthy range: 0.1–5.0
            gnm = proxy.grad_norm_mean
            if 0.05 <= gnm <= 5.0:
                gn_score = 1.0 - abs(math.log10(max(1e-6, gnm)) - 0.3) / 3.0
                gn_score = max(0.0, gn_score)
            elif gnm < 0.05:
                gn_score = gnm / 0.05 * 0.40   # vanishing
            else:
                gn_score = max(0.0, 1.0 - (gnm - 5.0) / 10.0)  # exploding
        else:
            gn_score = 0.50   # unknown

        # Gradient variance (spikyness)
        gnv_score = 1.0 / (1.0 + proxy.grad_norm_variance * 2.0)

        # Heuristik: depth risiko atenuasi
        attenuation = cfg.num_layers * GRAD_ATTENUATION_PER_LAYER
        atten_score = max(0.0, 1.0 - attenuation)

        # RMSNorm: better gradient flow vs LayerNorm
        norm_val = getattr(cfg, 'norm_type', None)
        norm_str = norm_val.value if hasattr(norm_val, 'value') else str(norm_val)
        rmsnorm_bonus = 0.08 if "RMS" in norm_str.upper() else 0.0

        score = float(np.clip(
            0.40 * gn_score + 0.25 * gnv_score + 0.25 * atten_score + 0.10 * (rmsnorm_bonus / 0.08),
            0.0, 1.0
        )) + rmsnorm_bonus * 0.5

        score = float(np.clip(score, 0.0, 1.0))

        # Risk classification
        if score >= 0.75:
            risk = "low"
        elif score >= 0.55:
            risk = "moderate"
        elif score >= 0.35:
            risk = "high"
        else:
            risk = "critical"

        return score, risk

    # ── T4: Generalization Gap ──────────────────────────────────────────────

    def _score_generalization(self, cfg: ArchConfig, proxy: ProxyTrainingResult) -> float:
        if proxy.nan_detected:
            return 0.0

        # generalization_gap sekarang adalah RELATIVE gap (gap / loss_initial)
        # Calibrated untuk proxy structured data:
        #   relative_gap ≤ 0.20 = excellent generalization (val mengikuti train)
        #   relative_gap ≤ 0.50 = good
        #   relative_gap ≤ 1.00 = acceptable (train cepat, val ketinggalan sedikit)
        #   relative_gap ≤ 2.00 = poor (significant overfit relative to initial loss)
        #   relative_gap >  2.00 = bad (extreme overfit)
        rel_gap = proxy.generalization_gap   # sudah relative di compute_derived

        if rel_gap <= 0.0:
            gap_score = 1.0
        elif rel_gap <= 0.20:
            gap_score = 1.0 - rel_gap / 0.20 * 0.10
        elif rel_gap <= 0.50:
            gap_score = 0.90 - (rel_gap - 0.20) / 0.30 * 0.15
        elif rel_gap <= 1.00:
            gap_score = 0.75 - (rel_gap - 0.50) / 0.50 * 0.25
        elif rel_gap <= 2.00:
            gap_score = 0.50 - (rel_gap - 1.00) / 1.00 * 0.30
        elif rel_gap <= 4.00:
            gap_score = 0.20 - (rel_gap - 2.00) / 2.00 * 0.15
        else:
            gap_score = max(0.0, 0.05)

        # Overfitting trend penalty (sudah dinormalisasi oleh compute_derived)
        if proxy.overfitting_trend > 0.10:
            gap_score *= max(0.40, 1.0 - proxy.overfitting_trend * 1.5)

        # Bonus: tied embeddings (parameter sharing = implicit regularization)
        if cfg.tie_embeddings:
            gap_score = min(1.0, gap_score + 0.04)

        # Bonus: optimizer dengan weight decay (regularization)
        if cfg.optimizer_type in (OptimizerType.ADAMW_BF16, OptimizerType.ZERO1,
                                   OptimizerType.ZERO2, OptimizerType.ZERO3):
            gap_score = min(1.0, gap_score + 0.03)

        return float(np.clip(gap_score, 0.0, 1.0))

    # ── T5: Sample Efficiency ───────────────────────────────────────────────

    def _score_sample_efficiency(self, cfg: ArchConfig) -> float:
        # Gradient noise: effective_batch = batch × seq_len
        eff_batch = cfg.batch_size * cfg.seq_len
        noise_score = float(np.clip(eff_batch / max(1, GRAD_NOISE_THRESHOLD), 0.0, 1.0))

        # Chinchilla: apakah param count dalam skala yang reasonable?
        chinchilla_tokens = cfg.param_count * CHINCHILLA_TOKEN_MULT
        # Model sangat besar tanpa data cukup = buruk
        if chinchilla_tokens > 1e11:   # > 100B tokens needed
            chinchilla_score = 0.50   # perlu banyak data
        elif chinchilla_tokens > 1e10:
            chinchilla_score = 0.70
        else:
            chinchilla_score = 1.0    # reasonable

        # Tied embeddings
        tie_score = 1.0 if cfg.tie_embeddings else 0.60

        # FFN multiplier: terlalu besar = lebih banyak param tapi tidak proporsional
        ffn_mult = cfg.ffn_multiplier
        if 3.0 <= ffn_mult <= 4.5:
            ffn_score = 1.0
        elif 2.0 <= ffn_mult < 3.0:
            ffn_score = 0.80
        elif ffn_mult > 5.0:
            ffn_score = 0.60
        else:
            ffn_score = 0.70

        return float(np.clip(
            0.35 * noise_score +
            0.30 * chinchilla_score +
            0.20 * tie_score +
            0.15 * ffn_score,
            0.0, 1.0
        ))

    # ── T6: Optimizer Compatibility ─────────────────────────────────────────

    def _score_optimizer_compat(self, cfg: ArchConfig) -> Tuple[float, str]:
        """Returns (score, lr_sensitivity)."""
        opt  = cfg.optimizer_type
        L    = cfg.num_layers
        H    = cfg.hidden_dim

        score = 1.0

        # Depth compatibility
        if L > 40:
            if opt == OptimizerType.LION:
                score *= 0.60    # Lion + deep = LR sangat sensitif
                lr_sens = "fragile"
            elif opt == OptimizerType.ADAM_8BIT:
                score *= 0.75    # 8-bit + deep = precision loss
                lr_sens = "sensitive"
            else:
                lr_sens = "sensitive"
        elif L > 24:
            lr_sens = "sensitive"
            if opt == OptimizerType.LION:
                score *= 0.80
        else:
            lr_sens = "robust"

        # Mixed precision + optimizer compatibility
        if cfg.use_mixed_precision:
            if opt in (OptimizerType.ADAM_FP32, OptimizerType.ADAMW_BF16):
                score = min(1.0, score + 0.05)
            elif opt == OptimizerType.ADAM_8BIT:
                score *= 0.85   # 8-bit + mixed prec = double quantization risk

        # ZeRO optimizers: baik untuk multi-GPU, suboptimal single GPU
        if opt in (OptimizerType.ZERO2, OptimizerType.ZERO3):
            # Single GPU → overhead komunikasi tidak berguna
            score *= 0.80

        # Gradient checkpointing + optimizer interaction
        if cfg.use_gradient_checkpointing and L < 20:
            score *= 0.90   # GC tidak diperlukan untuk model dangkal

        return float(np.clip(score, 0.0, 1.0)), lr_sens

    # ── Regime Classification ───────────────────────────────────────────────

    def _classify_regime(
        self, cfg: ArchConfig, proxy: ProxyTrainingResult, result: TrainingNASResult
    ) -> str:
        if proxy.nan_detected:
            return "UNSTABLE — NaN/Divergence"

        parts = []
        if result.convergence_score >= 0.75:
            parts.append("fast-converging")
        elif result.convergence_score >= 0.50:
            parts.append("moderate-convergence")
        else:
            parts.append("slow-convergence")

        if result.stability_score >= 0.75:
            parts.append("stable")
        elif result.stability_score < 0.40:
            parts.append("unstable")

        if result.gradient_risk == "low":
            parts.append("healthy-gradients")
        elif result.gradient_risk in ("high", "critical"):
            parts.append(f"gradient-{result.gradient_risk}-risk")

        if result.generalization_score >= 0.80:
            parts.append("good-generalization")
        elif result.generalization_score < 0.40:
            parts.append("overfitting-risk")

        return " | ".join(parts) if parts else "standard"


# ══════════════════════════════════════════════════════════════════════════════
#  NAS CACHE
# ══════════════════════════════════════════════════════════════════════════════

class TrainingNASCache:
    """LRU cache untuk training NAS results."""

    def __init__(self, max_size: int = 256):
        self._cache:    OrderedDict[str, TrainingNASResult] = OrderedDict()
        self._max_size: int = max_size
        self.hits:   int = 0
        self.misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get(self, key: str) -> Optional[TrainingNASResult]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, val: TrainingNASResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = val
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING Q-LEARNER
# ══════════════════════════════════════════════════════════════════════════════

class TrainingQLearner:
    """Q-learning untuk optimasi training dynamics."""

    def __init__(
        self,
        alpha:       float = 0.18,
        gamma:       float = 0.88,
        epsilon:     float = 0.30,
        epsilon_min: float = 0.06,
        ucb_c:       float = 2.0,
    ):
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_min = epsilon_min
        self.ucb_c       = ucb_c

        self._q:      Dict[str, List[float]] = defaultdict(lambda: [0.0] * _N_ACTIONS)
        self._counts: Dict[str, List[int]]   = defaultdict(lambda: [0]   * _N_ACTIONS)
        self._total_steps: int = 0

        self._replay: List[Tuple] = []
        self._replay_cap: int = 2000

    def _state_key(self, ts: float, grad_risk: str, stability: float, family: str) -> str:
        ts_b   = _bucket_idx(ts,        _TRAIN_BUCKETS)
        gr_b   = _grad_risk_to_bucket(grad_risk)
        stab_b = _bucket_idx(stability, _STAB_BUCKETS)
        fam_b  = _FAMILY_IDX.get(family, 0)
        return f"{ts_b}:{gr_b}:{stab_b}:{fam_b}"

    def select_action(
        self,
        ts: float, grad_risk: str, stability: float, family: str,
        fail_streak: Dict[str, int],
    ) -> int:
        key = self._state_key(ts, grad_risk, stability, family)
        q   = self._q[key]
        cnt = self._counts[key]
        total_cnt = max(1, sum(cnt))

        eps = max(self.epsilon_min, self.epsilon * (0.995 ** self._total_steps))

        if random.random() < eps:
            weights = [max(0.1, 1.0 / (1 + fail_streak.get(a, 0) * 0.5))
                       for a in TRAIN_ACTIONS]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    return i
            return random.randrange(_N_ACTIONS)
        else:
            ucb_vals = []
            for i in range(_N_ACTIONS):
                ucb = q[i] + self.ucb_c * math.sqrt(
                    math.log(total_cnt + 1) / (cnt[i] + 1)
                )
                fs = fail_streak.get(TRAIN_ACTIONS[i], 0)
                ucb -= fs * 0.20
                ucb_vals.append(ucb)
            return int(np.argmax(ucb_vals))

    def update(
        self,
        ts_old: float, gr_old: str, stab_old: float, family: str,
        action_idx: int,
        reward: float,
        ts_new: float, gr_new: str, stab_new: float,
    ) -> None:
        key_old = self._state_key(ts_old, gr_old, stab_old, family)
        key_new = self._state_key(ts_new, gr_new, stab_new, family)

        q_next_max = max(self._q[key_new])
        q_old      = self._q[key_old][action_idx]
        self._q[key_old][action_idx] = q_old + self.alpha * (
            reward + self.gamma * q_next_max - q_old
        )
        self._counts[key_old][action_idx] += 1
        self._total_steps += 1

        self._replay.append((key_old, action_idx, reward, key_new))
        if len(self._replay) > self._replay_cap:
            self._replay.pop(0)

    def replay_update(self, n: int = 16) -> int:
        if len(self._replay) < n:
            return 0
        batch = random.sample(self._replay, n)
        for key_old, ai, rew, key_new in batch:
            q_next_max = max(self._q[key_new])
            q_old      = self._q[key_old][ai]
            self._q[key_old][ai] = q_old + self.alpha * (
                rew + self.gamma * q_next_max - q_old
            )
        return n

    def best_q_values(self, ts: float, gr: str, stab: float, family: str) -> Dict[str, float]:
        key = self._state_key(ts, gr, stab, family)
        q   = self._q[key]
        return {TRAIN_ACTIONS[i]: round(q[i], 4) for i in range(_N_ACTIONS)}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PERTURBATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TrainingPerturbationEngine:
    """Eksekusi perturbasi training-centric pada ArchConfig."""

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu
        self._gen = ArchitectureGenerator(gpu)

    def _recompute(self, cfg: ArchConfig) -> None:
        gen = self._gen
        gpu = self.gpu

        if cfg.num_heads > 0 and cfg.head_dim > 0:
            cfg.hidden_dim = cfg.num_heads * cfg.head_dim
        valid_kv = [h for h in range(1, cfg.num_heads + 1)
                    if cfg.num_heads % h == 0]
        if cfg.num_kv_heads not in valid_kv and valid_kv:
            cfg.num_kv_heads = min(valid_kv, key=lambda h: abs(h - cfg.num_kv_heads))

        cfg.param_count = gen._compute_params(cfg)
        w_gb, a_gb, o_gb, kv_gb = gen._compute_memory(cfg)
        frag_frac = gen._compute_fragmentation(cfg)
        frag_gb   = round(frag_frac * (w_gb + a_gb + o_gb + kv_gb), 3)
        total_gb  = round(w_gb + a_gb + o_gb + kv_gb + frag_gb, 3)

        cfg.vram_weights_gb       = round(w_gb, 3)
        cfg.vram_activations_gb   = round(a_gb, 3)
        cfg.vram_optimizer_gb     = round(o_gb, 3)
        cfg.vram_kv_cache_gb      = round(kv_gb, 5)
        cfg.vram_fragmentation_gb = frag_gb
        cfg.vram_total_gb         = total_gb
        cfg.vram_usage_pct        = round(total_gb / max(0.001, gpu.vram_gb) * 100, 2)
        cfg.fits_gpu              = total_gb <= gpu.vram_gb * VRAM_LIMIT_PCT

        fwd, bwd, attn_fwd, ffn_fwd = gen._compute_flops(cfg)
        cfg.flops_per_token_fwd  = round(fwd, 0)
        cfg.flops_per_token_bwd  = round(bwd, 0)
        cfg.flops_attn_fwd       = round(attn_fwd, 0)
        cfg.flops_ffn_fwd        = round(ffn_fwd, 0)
        cfg.arithmetic_intensity = round(gen._compute_arithmetic_intensity(cfg), 2)

        thr = gen._estimate_throughput(cfg)
        cfg.tokens_per_sec_estimate = thr["tokens_per_sec"]
        cfg.mfu_estimate            = thr["mfu"]
        cfg.ms_per_step             = thr["ms_per_step"]
        cfg.bottleneck              = thr["bottleneck"]
        cfg.bottleneck_factors      = thr["bottleneck_factors"]
        cfg.compiler_speedup        = thr["compiler_speedup"]
        cfg.warp_divergence_pct     = thr["warp_divergence"]["warp_divergence_pct"]
        cfg.sm_occupancy            = thr["sm_occupancy"]
        cfg.fitness_score           = gen._fitness_score(cfg)

    def apply(self, cfg: ArchConfig, action: str) -> Tuple[Optional[ArchConfig], str]:
        """Apply satu training action. Returns (new_cfg, desc) atau (None, reason)."""
        new = copy.deepcopy(cfg)

        if action == "FIX_DEPTH_WIDTH":
            dw = cfg.num_layers / math.sqrt(max(1, cfg.hidden_dim))
            if OPTIMAL_DW_RATIO_LO <= dw <= OPTIMAL_DW_RATIO_HI:
                return None, "FIX_DEPTH_WIDTH: rasio sudah optimal"
            if dw > OPTIMAL_DW_RATIO_HI:
                # Terlalu dalam → kurangi layers
                new.num_layers = max(2, cfg.num_layers - 2)
                desc = f"FIX_DEPTH_WIDTH: kurangi layers {cfg.num_layers}→{new.num_layers}"
            else:
                # Terlalu dangkal → tambah layers (jika VRAM ada ruang)
                if cfg.vram_usage_pct > 70:
                    return None, "FIX_DEPTH_WIDTH: VRAM penuh, tidak bisa tambah layers"
                new.num_layers = cfg.num_layers + 2
                desc = f"FIX_DEPTH_WIDTH: tambah layers {cfg.num_layers}→{new.num_layers}"

        elif action == "SWITCH_RMSNORM":
            norm_val = getattr(cfg, 'norm_type', None)
            norm_str = norm_val.value if hasattr(norm_val, 'value') else str(norm_val)
            if "RMS" in norm_str.upper():
                return None, "SWITCH_RMSNORM: sudah RMSNorm"
            try:
                new.norm_type = NormType.RMSNORM
                desc = "SWITCH_RMSNORM: ganti ke RMSNorm"
            except Exception:
                return None, "SWITCH_RMSNORM: NormType tidak tersedia"

        elif action == "INCR_BATCH_TRAIN":
            # Naikkan batch untuk kurangi gradient noise
            new.batch_size = cfg.batch_size + 1
            desc = f"INCR_BATCH_TRAIN {cfg.batch_size}→{new.batch_size}"

        elif action == "SWITCH_OPT_STABLE":
            # Switch ke optimizer paling stabil untuk training
            if cfg.optimizer_type == OptimizerType.ADAM_FP32:
                return None, "SWITCH_OPT_STABLE: sudah Adam FP32"
            new.optimizer_type = OptimizerType.ADAM_FP32
            desc = f"SWITCH_OPT_STABLE {cfg.optimizer_type}→ADAM_FP32"

        elif action == "ENABLE_MIXED_PREC":
            if cfg.use_mixed_precision:
                return None, "ENABLE_MIXED_PREC: sudah aktif"
            new.use_mixed_precision = True
            desc = "ENABLE_MIXED_PREC: aktifkan mixed precision"

        elif action == "TIE_EMBEDDINGS":
            if cfg.tie_embeddings:
                return None, "TIE_EMBEDDINGS: sudah aktif"
            new.tie_embeddings = True
            desc = "TIE_EMBEDDINGS: aktifkan embedding tying"

        elif action == "ADJUST_FFN_MULT":
            # Sesuaikan FFN multiplier ke range yang lebih efisien
            current = cfg.ffn_multiplier
            if current < 3.0:
                new.ffn_multiplier = 3.0
                desc = f"ADJUST_FFN_MULT {current:.2f}→3.0"
            elif current > 5.0:
                new.ffn_multiplier = 4.0
                desc = f"ADJUST_FFN_MULT {current:.2f}→4.0"
            elif abs(current - 4.0) > 0.5:
                new.ffn_multiplier = round(current * 0.5 + 4.0 * 0.5, 2)
                desc = f"ADJUST_FFN_MULT {current:.2f}→{new.ffn_multiplier:.2f}"
            else:
                return None, "ADJUST_FFN_MULT: multiplier sudah optimal"

        elif action == "DISABLE_DROPOUT":
            if cfg.dropout == 0.0:
                return None, "DISABLE_DROPOUT: sudah 0"
            new.dropout = 0.0
            desc = f"DISABLE_DROPOUT {cfg.dropout:.2f}→0.0"

        elif action == "FIX_OPTIMIZER_DEPTH":
            # Untuk model sangat dalam, Adam FP32 lebih stabil
            if cfg.num_layers > 40 and cfg.optimizer_type == OptimizerType.LION:
                new.optimizer_type = OptimizerType.ADAM_FP32
                desc = f"FIX_OPTIMIZER_DEPTH: deep model Lion→ADAM_FP32"
            elif cfg.num_layers > 32 and cfg.optimizer_type == OptimizerType.ADAM_8BIT:
                new.optimizer_type = OptimizerType.ADAM_FP32
                desc = f"FIX_OPTIMIZER_DEPTH: deep model 8bit→ADAM_FP32"
            else:
                return None, "FIX_OPTIMIZER_DEPTH: optimizer sudah compatible"

        # ── New 6 actions (T2/T4 focus) ─────────────────────────────────────

        elif action == "STABILIZE_LR_DEEP":
            # Flag: kurangi layers 1 untuk menurunkan effective LR yang diperlukan
            # (adaptive LR di proxy otomatis lebih kecil untuk model lebih dangkal)
            # Aksi ini target T2 dengan cara mengurangi depth sensitivity
            if cfg.num_layers <= 8:
                return None, "STABILIZE_LR_DEEP: model sudah dangkal"
            if cfg.num_layers > 24:
                # Model sangat deep: kurangi 2 layers untuk stabilitas proxy LR
                new.num_layers = cfg.num_layers - 2
                desc = f"STABILIZE_LR_DEEP: {cfg.num_layers}→{new.num_layers} layers (LR stabilization)"
            else:
                return None, "STABILIZE_LR_DEEP: depth bukan masalah utama"

        elif action == "REDUCE_CAPACITY_GEN":
            # Kurangi capacity → kurangi overfit → T4 membaik
            # Target: model overfit (T4 < 0.50)
            if cfg.num_layers <= 6:
                return None, "REDUCE_CAPACITY_GEN: layers sudah minimum"
            if cfg.vram_usage_pct < 40:
                return None, "REDUCE_CAPACITY_GEN: VRAM terlalu rendah, arch terlalu kecil"
            # Kurangi 2-4 layers tergantung depth
            reduce_by = 4 if cfg.num_layers > 30 else 2
            new.num_layers = max(4, cfg.num_layers - reduce_by)
            desc = f"REDUCE_CAPACITY_GEN: {cfg.num_layers}→{new.num_layers} layers (reduce overfit)"

        elif action == "SWITCH_ADAMW_DECAY":
            # Switch ke AdamW — weight_decay bawaan membantu T4 generalization
            if cfg.optimizer_type in (OptimizerType.ADAMW_BF16, OptimizerType.ZERO1,
                                       OptimizerType.ZERO2, OptimizerType.ZERO3):
                return None, "SWITCH_ADAMW_DECAY: sudah AdamW-based"
            new.optimizer_type = OptimizerType.ADAMW_BF16
            desc = f"SWITCH_ADAMW_DECAY: {cfg.optimizer_type}→ADAMW_BF16 (weight_decay regularization)"

        elif action == "ENABLE_DROPOUT_REG":
            # Tambahkan dropout kecil untuk regularization → T4 generalization
            # Pretraining biasanya dropout=0, tapi proxy 50-60 steps rentan overfit
            if cfg.dropout > 0.0:
                return None, f"ENABLE_DROPOUT_REG: dropout={cfg.dropout:.2f} sudah ada"
            new.dropout = 0.05
            desc = "ENABLE_DROPOUT_REG: dropout 0.0→0.05 (regularization for T4)"

        elif action == "FIX_GQA_STABILITY":
            # MHA→GQA: kurangi KV heads → lebih sedikit param per attention step
            # → gradient noise lebih rendah → T2 stability naik
            if cfg.num_kv_heads < cfg.num_heads:
                return None, "FIX_GQA_STABILITY: sudah GQA"
            if cfg.num_heads <= 1:
                return None, "FIX_GQA_STABILITY: num_heads terlalu kecil"
            # Target kv_heads = num_heads / 4 (GQA ratio agresif untuk stabilitas)
            target_kv = max(1, cfg.num_heads // 4)
            valid_kv  = [h for h in range(1, cfg.num_heads + 1)
                         if cfg.num_heads % h == 0 and h <= target_kv * 2]
            if not valid_kv:
                return None, "FIX_GQA_STABILITY: tidak ada kv_heads GQA valid"
            new.num_kv_heads = min(valid_kv, key=lambda h: abs(h - target_kv))
            desc = f"FIX_GQA_STABILITY: kv_heads {cfg.num_kv_heads}→{new.num_kv_heads} (GQA stability)"

        elif action == "SHRINK_HIDDEN_REG":
            # Kurangi hidden_dim ke kelipatan head_dim yang lebih kecil → less capacity → T4
            if cfg.hidden_dim <= 256:
                return None, "SHRINK_HIDDEN_REG: hidden_dim sudah kecil"
            # Coba kurangi 1 head worth of hidden
            target = cfg.hidden_dim - cfg.head_dim
            if target <= 0 or target % cfg.head_dim != 0:
                return None, "SHRINK_HIDDEN_REG: tidak bisa kurangi dengan rapi"
            new.hidden_dim = target
            new.num_heads  = target // cfg.head_dim
            # Sesuaikan kv_heads
            valid_kv = [h for h in range(1, new.num_heads + 1)
                        if new.num_heads % h == 0]
            if valid_kv:
                new.num_kv_heads = min(valid_kv,
                                       key=lambda h: abs(h - cfg.num_kv_heads))
            desc = f"SHRINK_HIDDEN_REG: hidden {cfg.hidden_dim}→{new.hidden_dim} (capacity reduction)"

        else:
            return None, f"Unknown action: {action}"

        self._recompute(new)

        if not new.fits_gpu:
            return None, f"{action}: OOM setelah perturbasi"

        return new, desc


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING NAS REFINER (MAIN CLASS)
# ══════════════════════════════════════════════════════════════════════════════

class TrainingNASRefiner:
    """
    Training-Aware NAS + RL Refinement Engine.

    WAJIB mengevaluasi setiap arc melalui proxy training nyata.
    RL beriterasi sampai training_score ≥ TRAINING_SCORE_TARGET atau
    max_explore_iters tercapai.

    Phase A: quality heuristic fixes (ArcRefiner)
    Phase B: RL training NAS (ProxyTrainer + TrainingQLearner)
    """

    def __init__(
        self,
        gpu:               GPUSpec,
        max_iterations:    int   = 25,
        target_pct:        float = 100.0,
        max_explore_iters: int   = 30,
        training_weight:   float = 0.50,
        rng_seed:          Optional[int] = None,
        device:            str   = "cpu",
    ):
        self.gpu              = gpu
        self.max_iterations   = max_iterations
        self.target_pct       = target_pct
        self.max_explore_iters = max_explore_iters
        self.train_weight     = training_weight

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)

        # Sub-components
        self._refiner_a   = ArcRefiner(gpu, max_iterations=max_iterations, target_pct=target_pct)
        self._scorer      = ArcQualityScorer(gpu)
        self._proxy       = ProxyTrainer(device=device)
        self._eval        = TrainingDynamicsEvaluator(gpu)
        self._q_learner   = TrainingQLearner()
        self._perturber   = TrainingPerturbationEngine(gpu)
        self._nas_cache   = TrainingNASCache(max_size=256)

    def _evaluate_cached(self, cfg: ArchConfig) -> TrainingNASResult:
        """Evaluasi training dengan cache untuk hindari re-training config sama."""
        fp  = self._proxy._fingerprint(cfg)
        hit = self._nas_cache.get(fp)
        if hit is not None:
            return hit

        proxy_result = self._proxy.train(cfg)
        nas_result   = self._eval.evaluate(cfg, proxy_result)
        self._nas_cache.put(fp, nas_result)
        return nas_result

    def _compute_combined(self, hw_score: float, train_score: float) -> float:
        """Combined 50/50: hardware + training."""
        return round(
            self.train_weight * train_score +
            (1.0 - self.train_weight) * hw_score,
            5,
        )

    def refine(
        self,
        cfg:        ArchConfig,
        hw_score:   float = 0.0,
    ) -> Tuple[ArchConfig, TrainingAdaptiveLog]:
        """
        Refine satu ArchConfig.

        Setiap arc WAJIB dievaluasi NAS training.
        RL terus iterate sampai training_score optimal.

        Args:
            cfg:      ArchConfig
            hw_score: [0,1] dari hardware_refine.py (default 0 = standalone)
        """
        alog = TrainingAdaptiveLog(
            arch_id   = cfg.arch_id,
            arch_name = cfg.arch_name,
        )

        # ── Phase A: heuristic quality fixes ─────────────────────────────────
        base_cfg, base_log = self._refiner_a.refine(cfg)
        alog.base_log    = base_log
        alog.quality_start = base_log.initial_pct
        alog.quality_end   = base_log.final_pct
        alog.fitness_start = base_cfg.fitness_score

        # ── Initial NAS training evaluation ──────────────────────────────────
        init_nas = self._evaluate_cached(base_cfg)
        alog.nas_evaluations     += 1
        alog.nas_training_ms_total += init_nas.training_time_ms
        if init_nas.proxy_result and init_nas.proxy_result.nan_detected:
            alog.nas_nan_count += 1

        alog.train_score_start  = init_nas.training_score
        alog.combined_start     = self._compute_combined(hw_score, init_nas.training_score)

        # ── Phase B: RL Training NAS ─────────────────────────────────────────
        best_cfg        = copy.deepcopy(base_cfg)
        best_nas        = init_nas
        best_quality    = alog.quality_end
        best_combined   = alog.combined_start

        action_fail:  Dict[str, int] = {}
        action_tried: set            = set()
        no_improve   = 0
        # MAX_PAT lebih besar: training signal butuh lebih banyak percobaan
        MAX_PAT = 12
        T       = self.max_explore_iters

        for step in range(T):
            # Cek apakah sudah optimal
            if best_nas.training_score >= TRAINING_SCORE_TARGET and best_quality >= 99.9:
                alog.status = f"✓ OPTIMAL (ts={best_nas.training_score:.3f}≥{TRAINING_SCORE_TARGET})"
                break

            # Select action via RL
            act_idx = self._q_learner.select_action(
                best_nas.training_score,
                best_nas.gradient_risk,
                best_nas.stability_score,
                best_cfg.arch_family,
                action_fail,
            )
            action = TRAIN_ACTIONS[act_idx]
            alog.perturbation_tries += 1
            action_tried.add(action)

            # Apply perturbation
            new_cfg, desc = self._perturber.apply(best_cfg, action)
            if new_cfg is None:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1
                # Penalti Q-value langsung jika sudah 3× gagal
                if action_fail.get(action, 0) >= 3:
                    key = self._q_learner._state_key(
                        best_nas.training_score, best_nas.gradient_risk,
                        best_nas.stability_score, best_cfg.arch_family
                    )
                    self._q_learner._q[key][act_idx] = -2.0
                if no_improve >= MAX_PAT:
                    # Coba aksi yang belum pernah dicoba
                    untried = [a for a in TRAIN_ACTIONS if a not in action_tried]
                    if untried and step < T - 5:
                        no_improve   = 0
                        action_fail  = {k: max(0, v-1) for k, v in action_fail.items()}
                        continue
                    if best_quality >= 99.9:
                        action_fail = {}
                        no_improve  = 0
                    else:
                        break
                continue

            # Evaluasi NAS training untuk config baru
            new_nas = self._evaluate_cached(new_cfg)
            alog.nas_evaluations     += 1
            alog.nas_training_ms_total += new_nas.training_time_ms

            # Check cache hit
            fp = self._proxy._fingerprint(new_cfg)
            if self._nas_cache.get(fp) is not None:
                alog.nas_cache_hits += 1

            if new_nas.proxy_result and new_nas.proxy_result.nan_detected:
                alog.nas_nan_count += 1

            # Quality score dari ArcQualityScorer
            report       = self._scorer.score(new_cfg)
            new_q        = report.pct
            new_combined = self._compute_combined(hw_score, new_nas.training_score)

            # RL reward multidimensi — PRIORITAS ke konvergensi (T1)
            delta_ts    = new_nas.training_score    - best_nas.training_score
            delta_conv  = new_nas.convergence_score - best_nas.convergence_score   # T1 paling penting
            delta_stab  = new_nas.stability_score   - best_nas.stability_score
            delta_grad  = new_nas.gradient_health   - best_nas.gradient_health
            nan_penalty = -8.0 if (new_nas.proxy_result and
                                   new_nas.proxy_result.nan_detected) else 0.0

            reward = (delta_ts    * 10.0 +
                      delta_conv  *  7.0 +   # konvergensi paling penting
                      delta_stab  *  4.0 +
                      delta_grad  *  3.0 +
                      nan_penalty)

            # Q update
            self._q_learner.update(
                best_nas.training_score, best_nas.gradient_risk, best_nas.stability_score,
                best_cfg.arch_family,
                act_idx, reward,
                new_nas.training_score, new_nas.gradient_risk, new_nas.stability_score,
            )
            alog.rl_replay_updates += self._q_learner.replay_update(12)

            # ── Accept criterion: training-score-first ────────────────────
            # MASALAH LAMA: accept berdasarkan combined (hw+train).
            # Karena hw_score FIXED dari stage 7A dan biasanya sudah tinggi (0.87),
            # delta_combined = 0.5×delta_ts yang kecil → banyak improvement ditolak.
            #
            # FIX: di training NAS, accept jika training_score membaik (ANY positive delta_ts).
            # combined tetap dihitung untuk reporting, tapi bukan penentu accept/reject.
            delta_ts   = new_nas.training_score - best_nas.training_score
            delta_t2   = new_nas.stability_score - best_nas.stability_score
            delta_t4   = new_nas.generalization_score - best_nas.generalization_score
            delta      = new_combined - best_combined

            # Accept jika ANY of:
            #   1. training_score meningkat (utama)
            #   2. T2 atau T4 meningkat signifikan (dimensi yang paling bermasalah)
            #   3. combined meningkat (fallback)
            accept = (
                delta_ts > 5e-4 or                              # training score naik
                delta_t2 > 0.02 or                              # stability naik signifikan
                delta_t4 > 0.02 or                              # generalization naik signifikan
                (delta > 1e-6 and delta_ts >= -1e-4)            # combined naik tanpa training turun
            )

            if accept and delta_ts > -0.01:   # pastikan training tidak turun terlalu jauh
                best_cfg      = new_cfg
                best_nas      = new_nas
                best_quality  = new_q
                best_combined = new_combined
                alog.perturbations_accepted += 1

                if new_nas.convergence_score > init_nas.convergence_score + 0.04:
                    alog.convergence_improvements += 1
                if new_nas.stability_score > init_nas.stability_score + 0.04:
                    alog.stability_improvements += 1
                if new_nas.gradient_health > init_nas.gradient_health + 0.04:
                    alog.gradient_improvements += 1

                # Log improvement dengan detail T2/T4
                alog.improvement_events.append(
                    f"[step{step+1}] {action} → ts={new_nas.training_score:.4f}"
                    f" T1={new_nas.convergence_score:.3f}"
                    f" T2={new_nas.stability_score:.3f}(Δ{delta_t2:+.3f})"
                    f" T4={new_nas.generalization_score:.3f}(Δ{delta_t4:+.3f})"
                    f" combined→{best_combined:.5f}"
                    f" [{new_nas.training_time_ms:.0f}ms]"
                )
                action_fail[action] = 0
                no_improve          = 0
            else:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1
                if no_improve >= MAX_PAT:
                    untried = [a for a in TRAIN_ACTIONS if a not in action_tried]
                    if untried and step < T - 5:
                        no_improve   = 0
                        action_fail  = {k: max(0, v-1) for k, v in action_fail.items()}
                        continue
                    if best_quality >= 99.9:
                        action_fail = {}
                        no_improve  = 0
                        if step >= int(T * 0.75):
                            break
                    else:
                        break

        # Final state
        alog.fitness_end        = best_cfg.fitness_score
        alog.quality_end        = best_quality
        alog.train_score_end    = best_nas.training_score
        alog.combined_end       = self._compute_combined(hw_score, best_nas.training_score)

        # Rule effectiveness
        alog.rule_effectiveness = self._q_learner.best_q_values(
            best_nas.training_score,
            best_nas.gradient_risk,
            best_nas.stability_score,
            best_cfg.arch_family,
        )

        if not alog.status:
            if alog.perturbations_accepted > 0:
                alog.status = f"↑ IMPROVED (ts Δ{alog.train_delta:+.4f})"
            else:
                alog.status = "~ STAGNATED"

        return best_cfg, alog

    def refine_batch(
        self,
        archs:      List[ArchConfig],
        hw_scores:  Optional[Dict[str, float]] = None,
    ) -> Tuple[List[ArchConfig], List[TrainingAdaptiveLog], Dict[str, float]]:
        """
        Refine batch ARCs dengan training NAS.
        Returns (sorted_archs, logs, training_score_map).
        """
        if hw_scores is None:
            hw_scores = {}

        refined   = []
        logs      = []
        train_map = {}

        for cfg in archs:
            hs = hw_scores.get(cfg.arch_id, 0.0)
            new_cfg, log = self.refine(cfg, hw_score=hs)
            refined.append(new_cfg)
            logs.append(log)
            train_map[new_cfg.arch_id] = log.train_score_end

        # Sort by combined_end
        log_by_id = {l.arch_id: l for l in logs}
        refined.sort(
            key=lambda c: log_by_id.get(c.arch_id, TrainingAdaptiveLog()).combined_end,
            reverse=True,
        )
        return refined, logs, train_map


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_idx(value: float, buckets: List[Tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


def _grad_risk_to_bucket(risk: str) -> int:
    mapping = {"critical": 0, "high": 1, "moderate": 2, "low": 3, "unknown": 2}
    return mapping.get(risk, 2)


def compute_training_score(cfg: ArchConfig, gpu: GPUSpec, device: str = "cpu") -> float:
    """Convenience: hitung training_score dari satu ArchConfig."""
    trainer = ProxyTrainer(device=device)
    proxy   = trainer.train(cfg)
    evaluator = TrainingDynamicsEvaluator(gpu)
    result  = evaluator.evaluate(cfg, proxy)
    return result.training_score


def training_refine_archs(
    archs:             List[ArchConfig],
    gpu:               GPUSpec,
    max_iterations:    int   = 25,
    target_pct:        float = 100.0,
    max_explore_iters: int   = 30,
    rng_seed:          Optional[int] = None,
    hw_scores:         Optional[Dict[str, float]] = None,
    device:            str   = "cpu",
) -> Tuple[List[ArchConfig], List[TrainingAdaptiveLog], Dict[str, float]]:
    """
    Drop-in untuk pipeline.py.
    Returns (sorted_archs, train_logs, training_score_map).
    """
    refiner = TrainingNASRefiner(
        gpu,
        max_iterations    = max_iterations,
        target_pct        = target_pct,
        max_explore_iters = max_explore_iters,
        rng_seed          = rng_seed,
        device            = device,
    )
    return refiner.refine_batch(archs, hw_scores=hw_scores)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_training_nas_result(result: TrainingNASResult, *, console=None) -> None:
    """Print laporan Training NAS detail."""
    _p = console.print if console else print

    DIM_LABELS = {
        "T1": f"Convergence Rate     (22 pts)  score={result.convergence_score:.3f}",
        "T2": f"Training Stability   (22 pts)  score={result.stability_score:.3f}",
        "T3": f"Gradient Health      (18 pts)  score={result.gradient_health:.3f}",
        "T4": f"Generalization Gap   (15 pts)  score={result.generalization_score:.3f}",
        "T5": f"Sample Efficiency    (13 pts)  score={result.sample_efficiency:.3f}",
        "T6": f"Optimizer Compat     (10 pts)  score={result.optimizer_compat:.3f}",
    }
    pts_map = {
        "T1": result.pts_t1, "T2": result.pts_t2, "T3": result.pts_t3,
        "T4": result.pts_t4, "T5": result.pts_t5, "T6": result.pts_t6,
    }
    max_map = {"T1": 22, "T2": 22, "T3": 18, "T4": 15, "T5": 13, "T6": 10}

    _p()
    _p(f"  ┌─ Training NAS Score ─── {result.arch_id} {'─'*35}")
    _p(f"  │  Training Score: {result.training_score:.4f} ({result.training_score*100:.1f}%)   {result.grade}")
    if result.proxy_result:
        pr = result.proxy_result
        _p(f"  │  Proxy: steps={pr.steps_completed}  "
           f"L_init={pr.loss_initial:.3f}→L_final={pr.loss_final:.3f}"
           f"  conv={pr.convergence_rate:.4f}/step"
           f"  stab={pr.stability_score:.3f}"
           f"  gen_gap={pr.generalization_gap:.3f}"
           f"  NaN={'YES ⚠' if pr.nan_detected else 'no'}")
    _p(f"  │  Gradient Risk: {result.gradient_risk.upper()}  |  "
       f"LR Sensitivity: {result.lr_sensitivity.upper()}")
    _p(f"  │  Regime: {result.regime}")
    _p(f"  └{'─'*65}")

    for dim, label in DIM_LABELS.items():
        pts  = pts_map[dim]
        mx   = max_map[dim]
        bar_n = int(pts / mx * 20) if mx > 0 else 0
        bar   = "█" * bar_n + "░" * (20 - bar_n)
        _p(f"\n  [{dim}] {label}")
        _p(f"       [{bar}]  {pts:.1f}/{mx}")
    _p()


def print_training_adaptive_summary(
    logs:      List[TrainingAdaptiveLog],
    train_map: Dict[str, float],
    *,
    console=None,
) -> None:
    """Tabel ringkasan Training NAS-RL refinement."""
    _p = console.print if console else print

    ranked = sorted(logs, key=lambda l: l.combined_end, reverse=True)

    _p()
    _p("  ┌─ Training NAS-RL Summary ──────────────────────────────────────────────────────────")
    _p("  │")
    _p("  │  Training Score (6 dimensi dari proxy training + heuristik):")
    _p("  │    T1 Convergence(22) + T2 Stability(22) + T3 GradHealth(18)")
    _p("  │    + T4 GenGap(15) + T5 SampleEff(13) + T6 OptCompat(10)")
    _p("  │  Combined = 50% × hardware_score + 50% × training_score")
    _p("  │  Setiap arc wajib dites NAS training; RL iterasi hingga score optimal")
    _p("  │")
    _p(f"  │  {'Rank':<5} {'ARC-ID':<12} {'Train-Score':>14}  "
       f"{'NAS-evals':>10}  {'Combined':>12}  {'RL':>8}  Status")
    _p("  │  " + "─" * 100)

    for rank, log in enumerate(ranked, 1):
        sym     = "★" if rank == 1 else f"#{rank}"
        rl_info = f"{log.perturbation_tries}t/{log.perturbations_accepted}a"
        _p(
            f"  │  {sym:<5} {log.arch_id:<12} "
            f"{log.train_score_start:>6.4f}→{log.train_score_end:>6.4f}  "
            f"{log.nas_evaluations:>10}  "
            f"{log.combined_start:>5.4f}→{log.combined_end:>5.4f}  "
            f"{rl_info:>8}  "
            f"{log.status}"
        )

    _p("  │")
    _p("  │  t=tries · a=accepted · NAS evals = jumlah real proxy training runs")
    _p("  │  RL Actions: FIX_DEPTH_WIDTH, SWITCH_RMSNORM, INCR_BATCH, SWITCH_OPT,")
    _p("  │              ENABLE_MIXED_PREC, TIE_EMBED, ADJUST_FFN, DISABLE_DROP")
    _p("  └───────────────────────────────────────────────────────────────────────────────────")
    _p()


def print_training_adaptive_log(log: TrainingAdaptiveLog, *, console=None) -> None:
    """Detail log satu TrainingAdaptiveLog."""
    _p = console.print if console else print

    _p(f"\n  ─── Training NAS-RL Log: {log.arch_id} {'─'*40}")
    _p(f"       Quality       : {log.quality_start:.1f}% → {log.quality_end:.1f}%")
    _p(f"       Fitness       : {log.fitness_start:.4f} → {log.fitness_end:.4f}")
    _p(f"       Train Score   : {log.train_score_start:.4f} → {log.train_score_end:.4f}"
       f"   Δ={log.train_delta:+.4f}")
    _p(f"       Combined      : {log.combined_start:.5f} → {log.combined_end:.5f}"
       f"   Δ={log.combined_delta:+.5f}")
    _p(f"       NAS Stats     : {log.nas_evaluations} evals  "
       f"{log.nas_cache_hits} cache-hits  "
       f"{log.nas_nan_count} NANs  "
       f"{log.nas_training_ms_total:.0f}ms total")
    _p(f"       RL Stats      : {log.perturbation_tries} tries  "
       f"{log.perturbations_accepted} accepted  "
       f"{log.rl_replay_updates} replay")
    _p(f"       Improvements  : conv={log.convergence_improvements}  "
       f"stab={log.stability_improvements}  grad={log.gradient_improvements}")
    _p(f"       Status        : {log.status}")

    if log.improvement_events:
        _p(f"       Phase B Training Improvements ({len(log.improvement_events)}):")
        for ev in log.improvement_events:
            _p(f"         ↑ {ev}")

    if log.rule_effectiveness:
        top = sorted(log.rule_effectiveness.items(), key=lambda x: x[1], reverse=True)[:4]
        eff = "  ".join(f"{k}:{v:+.3f}" for k, v in top)
        _p(f"       Top Q-values: {eff}")
    _p()
