"""
adaptive_refiner.py  —  NAS-backed Adaptive RL Refinement Engine
═══════════════════════════════════════════════════════════════════════════════

Arsitektur Sistem:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Phase A : ArcRefiner  → koreksi formula + arch fixes (heuristic)      │
  │  Phase B : NAS + RL   → real PyTorch training + Q-learning exploration │
  └─────────────────────────────────────────────────────────────────────────┘

NAS (Neural Architecture Search) — Real Training Evaluator:
  • Membangun proxy model (scaled-down, rasio terjaga) dari ArchConfig
  • Melatih 30-50 step nyata di PyTorch dengan AdamW + LR warmup
  • Mengukur: loss trajectory, gradient norm, loss variance, NaN detection
  • Mengembalikan NASResult dengan stability_score dan convergence_rate

RL (Reinforcement Learning) — NAS-aware Q-learning:
  • State-space DIPERLUAS: (q_bucket, f_bucket, stability_bucket, fam_idx)
  • Reward NYATA dari NAS: Δstability × 4.0 + Δconvergence × 3.0
  • Experience replay lintas semua ARC (cross-arch learning)
  • UCB exploration + epsilon-greedy dengan anti-collapse mechanism
  • NAS cache: fingerprint-based LRU (hindari evaluasi ulang config sama)

Combined Score (3 sinyal):
  combined = (1 - nas_weight) × [q_weight×q_norm + f_weight×f_norm]
           + nas_weight × nas_fitness

  nas_fitness = 0.5 × convergence_norm + 0.5 × stability_score
  Default nas_weight = 0.30 (30% sinyal real, 70% formula)
"""

from __future__ import annotations

import copy
import math
import random
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_types import (
    ArchConfig, AttentionType, FFNType, OptimizerType
)
from hardware import GPUSpec
from generator import ArchitectureGenerator
from refiner import ArcRefiner, ArcQualityScorer, RefinementLog


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_FAMILY_LIST = [
    "CoT-Optimizer", "Speed-Demon", "Balanced-Pro", "MoE-Sparse",
    "Long-Horizon",  "Nano-Efficient", "Compute-Dense",
]
_FAMILY_IDX: Dict[str, int] = {f: i for i, f in enumerate(_FAMILY_LIST)}

# Quality bucket boundaries
_Q_BUCKETS   = [(0.0, 60.0), (60.0, 75.0), (75.0, 85.0),
                (85.0, 95.0), (95.0, 99.9), (99.9, 100.01)]
# Fitness (formula) bucket boundaries
_FIT_BUCKETS = [(0.0, 0.12), (0.12, 0.25), (0.25, 0.40),
                (0.40, 0.60), (0.60, 1.01)]
# NAS Stability bucket boundaries  (NEW — sinyal real training)
_STAB_BUCKETS = [(0.0, 0.20),   # 0: sangat tidak stabil (NaN/high variance)
                 (0.20, 0.40),  # 1: buruk
                 (0.40, 0.60),  # 2: sedang
                 (0.60, 0.80),  # 3: baik
                 (0.80, 1.01)]  # 4: excellent

ACTIONS = [
    "P_BATCH", "P_FFN", "P_HEAD", "P_SEQ",
    "P_KV",    "P_LAYERS", "P_EXPERT", "P_OPT",
]


# ══════════════════════════════════════════════════════════════════════════════
#  NAS CONFIGURATION & RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NASConfig:
    """
    Konfigurasi NAS Proxy Training.

    Proxy model dibangun dari ArchConfig dengan skala diperkecil
    agar training cepat (< 200 ms per arch pada CPU).
    Rasio kritis (head_dim, ffn_multiplier, kv/head ratio, MoE topology)
    dipertahankan agar sinyal stabilitas tetap representatif.
    """
    # ── Proxy Model Scale ─────────────────────────────────────────────────────
    proxy_hidden_max:  int   = 64     # max hidden dim proxy (power of 2)
    proxy_layers_max:  int   = 2      # max transformer layers proxy
    proxy_seq_len:     int   = 32     # sequence length untuk training
    proxy_vocab:       int   = 256    # vocab size (byte-level, cepat)
    proxy_batch:       int   = 2      # batch size

    # ── Training ──────────────────────────────────────────────────────────────
    train_steps:       int   = 40     # jumlah step training real
    warmup_steps:      int   = 4      # LR warmup steps
    learning_rate:     float = 3e-3   # LR awal (lebih tinggi untuk proxy)
    weight_decay:      float = 0.01
    grad_clip:         float = 1.0    # gradient clipping norm

    # ── Stability Thresholds ──────────────────────────────────────────────────
    stability_tail:    int   = 15     # jumlah langkah terakhir untuk var calc
    max_loss_for_conv: float = 8.0    # log(256) ≈ 5.5; lebih dari ini = diverged

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_size:        int   = 256    # max entry NAS cache (LRU)

    # ── Device ────────────────────────────────────────────────────────────────
    device:            str   = "cpu"  # "cpu" atau "cuda" bila tersedia


@dataclass
class NASResult:
    """
    Hasil satu sesi NAS proxy training.

    Semua metrik bersumber dari PyTorch real training:
      • losses[]      — cross-entropy per step (language modeling)
      • grad_norms[]  — L2 grad norm setelah clip per step
      • stability_score  [0,1] — 1 = sangat stabil
      • convergence_rate — rata-rata penurunan loss per step
      • nas_fitness   [0,1] — combined NAS quality (dipakai RL reward)
    """
    arch_fingerprint: str    = ""

    # Raw trajectory
    losses:      List[float] = field(default_factory=list)
    grad_norms:  List[float] = field(default_factory=list)

    # Derived metrics (dihitung via compute_derived)
    loss_initial:         float = 0.0
    loss_final:           float = 0.0
    loss_variance:        float = 999.0  # instability indicator (lower = better)
    grad_norm_mean:       float = 0.0
    grad_norm_variance:   float = 999.0  # spike indicator (lower = better)
    nan_detected:         bool  = False

    convergence_rate:     float = 0.0   # (loss_init - loss_final) / steps
    stability_score:      float = 0.0   # [0, 1]
    nas_fitness:          float = 0.0   # combined NAS score [0, 1]

    training_time_ms:     float = 0.0
    proxy_param_count:    int   = 0

    @property
    def is_stable(self) -> bool:
        return not self.nan_detected and self.stability_score > 0.40

    def compute_derived(self, train_steps: int, tail: int = 15) -> None:
        """
        Hitung semua metrik derived dari raw trajectory.
        Dipanggil oleh NASProxyTrainer setelah loop selesai.
        """
        n = len(self.losses)
        if n == 0:
            return

        self.loss_initial = self.losses[0]
        self.loss_final   = self.losses[-1]

        # Convergence rate: rata-rata penurunan loss per step
        self.convergence_rate = max(0.0, self.loss_initial - self.loss_final) / max(1, train_steps)

        # Loss variance: var dari tail terakhir (stabilitas di akhir training)
        tail_losses = self.losses[-min(tail, n):]
        if len(tail_losses) >= 2:
            self.loss_variance = float(
                torch.tensor(tail_losses, dtype=torch.float32).var().item()
            )
        else:
            self.loss_variance = 0.0

        # Gradient norm statistics
        if self.grad_norms:
            gn = torch.tensor(self.grad_norms, dtype=torch.float32)
            self.grad_norm_mean     = float(gn.mean().item())
            self.grad_norm_variance = float(gn.var().item()) if len(self.grad_norms) >= 2 else 0.0

        # Stability score: penalti tinggi untuk variance dan NaN
        if self.nan_detected:
            self.stability_score = 0.0
        else:
            # var_penalty: 1 saat variance=0, turun cepat saat variance naik
            var_penalty = 1.0 / (1.0 + self.loss_variance * 20.0)
            # gnv_penalty: 1 saat gradient variance=0
            gnv_penalty = 1.0 / (1.0 + self.grad_norm_variance * 0.5)
            # final_loss_penalty: jika loss akhir sangat tinggi → diverged
            final_pen   = max(0.0, 1.0 - max(0.0, self.loss_final - 4.0) / 4.0)
            self.stability_score = round(
                0.45 * var_penalty + 0.35 * gnv_penalty + 0.20 * final_pen, 4
            )

        # NAS fitness: gabungan konvergensi dan stabilitas
        #   convergence: normalize by max possible drop (5.5 nats ≈ log(256))
        conv_norm = min(1.0, self.convergence_rate * train_steps / 5.5)
        self.nas_fitness = round(0.50 * conv_norm + 0.50 * self.stability_score, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED SCORE  (NAS-aware)
# ══════════════════════════════════════════════════════════════════════════════

def compute_combined_score(
    quality_pct:    float,
    fitness:        float,
    *,
    quality_weight: float = 0.35,
    fitness_weight: float = 0.65,
    min_quality:    float = 70.0,
) -> float:
    """
    Skor gabungan formula-only (backward compat dengan pipeline.py).
    Bila quality < min_quality → 0.0.
    """
    if quality_pct < min_quality:
        return 0.0
    q_norm = (quality_pct - min_quality) / max(0.001, 100.0 - min_quality)
    q_norm = max(0.0, min(1.0, q_norm))
    f_norm = max(0.0, min(1.0, float(fitness)))
    return round(quality_weight * q_norm + fitness_weight * f_norm, 6)


def compute_combined_score_nas(
    quality_pct:    float,
    fitness:        float,
    nas_fitness:    float,
    *,
    quality_weight: float = 0.35,
    fitness_weight: float = 0.65,
    nas_weight:     float = 0.30,
    min_quality:    float = 70.0,
) -> float:
    """
    Skor gabungan 3-sinyal: formula quality, formula fitness, NAS fitness.

    combined = (1 - nas_w) × [q_w×q_norm + f_w×f_norm]
             + nas_w × nas_fitness

    NAS weight 0.30 → 30% sinyal real training, 70% formula architecture.
    Bila quality < min_quality → 0.0 (tidak layak).
    """
    if quality_pct < min_quality:
        return 0.0
    q_norm = (quality_pct - min_quality) / max(0.001, 100.0 - min_quality)
    q_norm = max(0.0, min(1.0, q_norm))
    f_norm = max(0.0, min(1.0, float(fitness)))
    n_norm = max(0.0, min(1.0, float(nas_fitness)))

    formula = (1.0 - nas_weight) * (quality_weight * q_norm + fitness_weight * f_norm)
    nas_comp = nas_weight * n_norm
    return round(formula + nas_comp, 6)


def select_best_arch(
    archs:          List[ArchConfig],
    quality_map:    Dict[str, float],
    *,
    quality_weight: float = 0.35,
    fitness_weight: float = 0.65,
    min_quality:    float = 70.0,
) -> Optional[ArchConfig]:
    """Pilih arsitektur terbaik (fits_gpu=True) by combined_score."""
    candidates = [a for a in archs if a.fits_gpu]
    if not candidates:
        return None

    def _score(a: ArchConfig) -> float:
        q = quality_map.get(a.arch_id, 0.0)
        return compute_combined_score(
            q, a.fitness_score,
            quality_weight=quality_weight,
            fitness_weight=fitness_weight,
            min_quality=min_quality,
        )

    return max(candidates, key=_score)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveLog:
    """Log lengkap satu sesi adaptive refinement untuk satu ARC."""
    arch_id:                str
    arch_name:              str
    base_log:               RefinementLog

    # Snapshot awal/akhir (formula)
    quality_start:          float = 0.0
    quality_end:            float = 0.0
    fitness_start:          float = 0.0
    fitness_end:            float = 0.0
    combined_start:         float = 0.0
    combined_end:           float = 0.0

    # NAS metrics (BARU)
    nas_stability_start:    float = 0.0   # stability_score before RL
    nas_stability_end:      float = 0.0   # stability_score after RL
    nas_fitness_start:      float = 0.0
    nas_fitness_end:        float = 0.0
    nas_evaluations:        int   = 0     # total NAS evaluations called
    nas_cache_hits:         int   = 0     # berapa yang di-serve dari cache
    nas_nan_count:          int   = 0     # berapa kali NaN terdeteksi
    nas_training_ms_total:  float = 0.0  # total waktu training proxy (ms)

    # Statistik Phase B RL
    perturbation_tries:     int = 0
    perturbations_accepted: int = 0
    improvement_events:     List[str] = field(default_factory=list)
    rule_effectiveness:     Dict[str, float] = field(default_factory=dict)

    # RL counters
    rl_q_updates:           int = 0
    rl_replay_updates:      int = 0
    rl_ucb_explorations:    int = 0
    rl_diversity_bonuses:   int = 0

    # Flag
    is_suspicious:          bool = False

    @property
    def combined_delta(self) -> float:
        return round(self.combined_end - self.combined_start, 6)

    @property
    def nas_stability_delta(self) -> float:
        return round(self.nas_stability_end - self.nas_stability_start, 4)

    @property
    def status(self) -> str:
        nas_tag = f"  NAS↑{self.nas_stability_delta:+.3f}" if self.nas_stability_delta != 0 else ""
        if self.is_suspicious:
            return f"⚠ SUSPICIOUS (q=100% f={self.fitness_end:.3f}){nas_tag}"
        if self.combined_end >= 0.90:
            return f"★ EXCELLENT{nas_tag}"
        if self.perturbations_accepted > 0:
            return f"↑ EXPLORED (+{self.perturbations_accepted}){nas_tag}"
        if self.quality_end >= 100.0:
            return f"✓ q=100%{nas_tag}"
        return f"~ q={self.quality_end:.1f}%{nas_tag}"


# ══════════════════════════════════════════════════════════════════════════════
#  NAS PROXY MODEL
# ══════════════════════════════════════════════════════════════════════════════

class _RMSNorm(nn.Module):
    """RMSNorm (compatible dengan semua PyTorch version ≥ 1.10)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _ProxyAttention(nn.Module):
    """
    Proxy attention: mendukung MHA (GQA head=kv), GQA (kv<heads), MQA (kv=1).
    Menggunakan scaled dot-product attention sederhana (tanpa mask).
    """
    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.h    = num_heads
        self.kv_h = num_kv_heads
        self.d    = max(1, hidden // num_heads)

        self.q_proj = nn.Linear(hidden, self.d * num_heads,    bias=False)
        self.k_proj = nn.Linear(hidden, self.d * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(hidden, self.d * num_kv_heads, bias=False)
        self.o_proj = nn.Linear(self.d * num_heads, hidden,    bias=False)
        self.scale  = self.d ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h,    self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_h, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_h, self.d).transpose(1, 2)

        # Expand KV untuk GQA/MQA
        if self.kv_h < self.h:
            g = self.h // self.kv_h
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)

        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, self.h * self.d)
        return self.o_proj(out)


class _ProxyFFN(nn.Module):
    """
    Proxy FFN:
      DENSE / GATED / GEGLU → SwiGLU (gate × up, kemudian down)
      MOE / MOE_TOPK        → Simplified sparse routing (top-K expert dispatch)

    MoE diimplementasi dengan token-routing nyata sehingga
    load-balancing instability bisa terdeteksi.
    """
    def __init__(
        self,
        hidden:      int,
        ffn_dim:     int,
        ffn_type:    FFNType,
        num_experts: int = 4,
        top_k:       int = 1,
    ):
        super().__init__()
        self.is_moe = ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
        # Cap experts untuk kecepatan proxy
        ne           = min(num_experts, 4)
        self.ne      = ne
        self.top_k   = min(top_k, ne)

        if not self.is_moe:
            self.gate = nn.Linear(hidden, ffn_dim, bias=False)
            self.up   = nn.Linear(hidden, ffn_dim, bias=False)
            self.down = nn.Linear(ffn_dim, hidden, bias=False)
        else:
            self.router = nn.Linear(hidden, ne, bias=False)
            # Setiap expert: SwiGLU mini
            self.expert_gate = nn.ModuleList(
                [nn.Linear(hidden, ffn_dim, bias=False) for _ in range(ne)]
            )
            self.expert_up   = nn.ModuleList(
                [nn.Linear(hidden, ffn_dim, bias=False) for _ in range(ne)]
            )
            self.expert_down = nn.ModuleList(
                [nn.Linear(ffn_dim, hidden, bias=False) for _ in range(ne)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_moe:
            return self.down(F.silu(self.gate(x)) * self.up(x))

        # MoE: token routing
        B, T, C = x.shape
        xf = x.reshape(-1, C)                              # (B*T, C)
        router_logits = self.router(xf)                    # (B*T, ne)
        weights = F.softmax(router_logits, dim=-1)
        topk_w, topk_i = weights.topk(self.top_k, dim=-1) # (B*T, k)

        out = torch.zeros_like(xf)
        for k in range(self.top_k):
            for e in range(self.ne):
                mask = (topk_i[:, k] == e)
                if not mask.any():
                    continue
                x_e = xf[mask]
                y_e = self.expert_down[e](
                    F.silu(self.expert_gate[e](x_e)) * self.expert_up[e](x_e)
                )
                out[mask] += topk_w[mask, k:k+1] * y_e

        return out.reshape(B, T, C)


class _ProxyBlock(nn.Module):
    """Satu transformer block: Pre-Norm → Attn + FFN."""
    def __init__(
        self,
        hidden:      int,
        num_heads:   int,
        num_kv_heads:int,
        ffn_dim:     int,
        ffn_type:    FFNType,
        num_experts: int,
        top_k:       int,
    ):
        super().__init__()
        self.norm1 = _RMSNorm(hidden)
        self.attn  = _ProxyAttention(hidden, num_heads, num_kv_heads)
        self.norm2 = _RMSNorm(hidden)
        self.ffn   = _ProxyFFN(hidden, ffn_dim, ffn_type, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _ProxyModel(nn.Module):
    """
    Proxy transformer untuk NAS evaluation.

    Scaling strategy:
      • hidden_dim   → min(cfg.hidden_dim, proxy_hidden_max)
      • Aligned ke num_heads (guarantee head_dim ≥ 8)
      • num_kv_heads dipertahankan rasionya, div-check dipenuhi
      • ffn_dim      = hidden × ffn_multiplier (kunci: rasio sama)
      • num_layers   → min(cfg.num_layers, proxy_layers_max)
      • vocab        = proxy_vocab (byte-level, sederhana)
    """
    def __init__(self, cfg: ArchConfig, pc: NASConfig):
        super().__init__()

        # ── Scale down hidden while preserving ratios ──────────────────────────
        scale  = min(1.0, pc.proxy_hidden_max / max(1, cfg.hidden_dim))

        # Derive num_heads first
        raw_h  = max(1, round(cfg.num_heads * scale))
        # head_dim minimum 8 so that hidden = heads × head_dim ≥ 8
        head_d = max(8, round((cfg.hidden_dim / cfg.num_heads)))
        # Clamp head_dim so proxy_hidden reasonable
        head_d = min(head_d, pc.proxy_hidden_max // max(1, raw_h))
        head_d = max(8, head_d)

        num_heads  = raw_h
        hidden     = num_heads * head_d

        # KV heads: maintain ratio, guarantee divisibility
        kv_ratio   = cfg.num_kv_heads / max(1, cfg.num_heads)
        num_kv     = max(1, round(num_heads * kv_ratio))
        # Ensure num_heads % num_kv == 0
        while num_heads % num_kv != 0 and num_kv > 1:
            num_kv -= 1
        num_kv_heads = max(1, num_kv)

        # FFN dim: preserve multiplier ratio
        ffn_dim    = max(hidden, int(hidden * cfg.ffn_multiplier))

        num_layers = min(pc.proxy_layers_max, cfg.num_layers)

        # ── Build model ────────────────────────────────────────────────────────
        self.embed  = nn.Embedding(pc.proxy_vocab, hidden)
        self.blocks = nn.ModuleList([
            _ProxyBlock(
                hidden, num_heads, num_kv_heads,
                ffn_dim, cfg.ffn_type,
                cfg.num_experts, cfg.top_k_experts,
            )
            for _ in range(num_layers)
        ])
        self.norm   = _RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, pc.proxy_vocab, bias=False)

        # Weight tying (mimics cfg.tie_embeddings)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.lm_head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
#  NAS PROXY TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class NASProxyTrainer:
    """
    Evaluator NAS: membangun proxy model → latih N step → kembalikan NASResult.

    Training setup:
      • Optimizer: AdamW (β₁=0.9, β₂=0.95) — standar pretraining
      • Scheduler: linear warmup + cosine decay
      • Data: synthetic randint tokens (cepat, no IO)
      • Loss: cross-entropy next-token prediction
      • Gradient clipping: 1.0 (deteksi gradient explosion)
      • NaN check setiap step

    NASCache dikelola di luar (oleh NASAdaptiveRefiner) untuk
    menghindari re-evaluasi fingerprint yang sama.
    """

    def __init__(self, nas_cfg: NASConfig):
        self._cfg    = nas_cfg
        self._device = torch.device(
            "cuda" if nas_cfg.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

    def evaluate(self, arch_cfg: ArchConfig) -> NASResult:
        """
        Run real NAS evaluation. Returns NASResult dengan trajectory lengkap.
        Thread-safe: tidak ada shared state mutable.
        """
        fp     = self._fingerprint(arch_cfg)
        result = NASResult(arch_fingerprint=fp)
        t0     = time.perf_counter()

        try:
            model = _ProxyModel(arch_cfg, self._cfg).to(self._device)
            result.proxy_param_count = model.num_params()

            # Optimizer (AdamW — sama dengan pretraining nyata)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr           = self._cfg.learning_rate,
                betas        = (0.9, 0.95),
                weight_decay = self._cfg.weight_decay,
                eps          = 1e-8,
            )

            # Cosine LR dengan warmup
            T_total  = self._cfg.train_steps
            T_warmup = self._cfg.warmup_steps

            def _lr_lambda(step: int) -> float:
                if step < T_warmup:
                    return (step + 1) / max(1, T_warmup)
                prog = (step - T_warmup) / max(1, T_total - T_warmup)
                return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

            B = self._cfg.proxy_batch
            T = self._cfg.proxy_seq_len
            V = self._cfg.proxy_vocab

            model.train()
            for step in range(T_total):
                # Input: random tokens; target: shifted by 1 (next-token)
                tokens = torch.randint(0, V, (B, T + 1), device=self._device)
                x, y   = tokens[:, :-1], tokens[:, 1:]

                optimizer.zero_grad()
                logits = model(x)                           # (B, T, V)
                loss   = F.cross_entropy(
                    logits.reshape(-1, V), y.reshape(-1)
                )

                # NaN / Inf check
                if not torch.isfinite(loss):
                    result.nan_detected = True
                    break

                loss.backward()

                # Gradient norm (sebelum clip — indikator ketidakstabilan)
                raw_gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self._cfg.grad_clip
                )
                if not torch.isfinite(raw_gn):
                    result.nan_detected = True
                    break

                optimizer.step()
                scheduler.step()

                result.losses.append(float(loss.item()))
                result.grad_norms.append(float(raw_gn.item()))

        except Exception:
            # Apapun error (OOM, dimension mismatch, dll) → anggap unstable
            result.nan_detected = True

        finally:
            result.training_time_ms = (time.perf_counter() - t0) * 1000.0

        result.compute_derived(self._cfg.train_steps, self._cfg.stability_tail)
        return result

    @staticmethod
    def _fingerprint(cfg: ArchConfig) -> str:
        """Hash dari hyperparameter yang mempengaruhi proxy architecture."""
        key = (
            cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.ffn_type.name, cfg.num_experts, cfg.top_k_experts,
            round(cfg.ffn_multiplier, 2), cfg.tie_embeddings,
        )
        return hashlib.md5(str(key).encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
#  NAS CACHE  (LRU)
# ══════════════════════════════════════════════════════════════════════════════

class NASCache:
    """
    LRU cache untuk NASResult.
    Menghindari re-training proxy yang identik selama sesi.

    Key: fingerprint dari NASProxyTrainer._fingerprint()
    Value: NASResult
    """

    def __init__(self, max_size: int = 256):
        self._max:   int                     = max_size
        self._data:  Dict[str, NASResult]    = {}
        self._order: List[str]               = []
        self.hits  = 0
        self.misses = 0

    def get(self, fp: str) -> Optional[NASResult]:
        if fp in self._data:
            # Move to end (most recently used)
            self._order.remove(fp)
            self._order.append(fp)
            self.hits += 1
            return self._data[fp]
        self.misses += 1
        return None

    def put(self, fp: str, result: NASResult) -> None:
        if fp in self._data:
            self._order.remove(fp)
        elif len(self._data) >= self._max:
            evict = self._order.pop(0)
            del self._data[evict]
        self._data[fp] = result
        self._order.append(fp)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIENCE REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ExperienceBuffer:
    """
    Ring buffer (state, action, reward, next_state).
    Dibagikan ke seluruh ARC → cross-arch learning.
    """

    def __init__(self, capacity: int = 1000):
        self._cap = capacity
        self._buf: List[Tuple] = []
        self._ptr = 0

    def push(self, state: tuple, action: str,
             reward: float, next_state: tuple) -> None:
        entry = (state, action, float(reward), next_state)
        if len(self._buf) < self._cap:
            self._buf.append(entry)
        else:
            self._buf[self._ptr] = entry
        self._ptr = (self._ptr + 1) % self._cap

    def sample(self, n: int, rng: random.Random) -> List[Tuple]:
        k = min(n, len(self._buf))
        return rng.sample(self._buf, k) if k > 0 else []

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
#  Q-TABLE  —  NAS-aware State + Reward
# ══════════════════════════════════════════════════════════════════════════════

class RLQTable:
    """
    Q-table dengan state-space NAS-aware (4-tuple).

    State = (q_bucket, f_bucket, stab_bucket, fam_idx)
           = 6 × 5 × 5 × 7 = 1050 states (vs 210 sebelumnya)
    Action = 8 (ACTIONS)

    Reward utama sekarang dari NAS:
      Δstability × 4.0 + Δconvergence × 3.0 + Δcombined_formula × 8.0

    Algorithm:
      Q(s,a) ← Q(s,a) + α(t) × [r + γ × max_a' Q(s',a') - Q(s,a)]
      Exploration: UCB + epsilon-greedy + curiosity bonus
      Replay: offline learning dari ExperienceBuffer
    """

    def __init__(
        self,
        rng:           random.Random,
        replay_buffer: ExperienceBuffer,
        *,
        alpha:         float = 0.35,
        gamma:         float = 0.92,
        ucb_c_normal:  float = 1.8,
        ucb_c_peak:    float = 3.0,
        eps_start:     float = 0.20,
        eps_end:       float = 0.05,
    ):
        self._rng        = rng
        self._replay     = replay_buffer
        self._alpha0     = alpha
        self._gamma      = gamma
        self._ucb_normal = ucb_c_normal
        self._ucb_peak   = ucb_c_peak
        self._eps_start  = eps_start
        self._eps_end    = eps_end

        self._Q:       Dict[tuple, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTIONS}
        )
        self._N:       Dict[tuple, Dict[str, int]]   = defaultdict(
            lambda: {a: 0 for a in ACTIONS}
        )
        self._N_state: Dict[tuple, int]              = defaultdict(int)
        self._seen:    set                           = set()

        self.total_updates    = 0
        self.ucb_explorations = 0

    # ── State Encoding (NAS dimension) ───────────────────────────────────────

    @staticmethod
    def encode_state(
        quality: float, fitness: float,
        stability: float, family: str,
    ) -> tuple:
        """
        Encode state ke 4-tuple dengan dimensi NAS stability.
        (q_bucket, f_bucket, stab_bucket, fam_idx)
        """
        q_idx = next(
            (i for i, (lo, hi) in enumerate(_Q_BUCKETS) if lo <= quality < hi),
            len(_Q_BUCKETS) - 1,
        )
        f_idx = next(
            (i for i, (lo, hi) in enumerate(_FIT_BUCKETS) if lo <= fitness < hi),
            len(_FIT_BUCKETS) - 1,
        )
        s_idx = next(
            (i for i, (lo, hi) in enumerate(_STAB_BUCKETS) if lo <= stability < hi),
            len(_STAB_BUCKETS) - 1,
        )
        fam = _FAMILY_IDX.get(family, 0)
        return (q_idx, f_idx, s_idx, fam)

    # ── Config Fingerprint (diversity) ────────────────────────────────────────

    @staticmethod
    def _fp(cfg: ArchConfig) -> str:
        key = (
            cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.ffn_type.name, cfg.num_experts, cfg.batch_size,
            round(cfg.ffn_multiplier, 1), cfg.attn_type.name,
        )
        return hashlib.md5(str(key).encode()).hexdigest()[:12]

    def is_novel(self, cfg: ArchConfig) -> bool:
        fp = self._fp(cfg)
        if fp not in self._seen:
            self._seen.add(fp)
            return True
        return False

    # ── Action Selection ─────────────────────────────────────────────────────

    def select_action(
        self,
        state:          tuple,
        active_actions: List[str],
        step:           int,
        total_steps:    int,
        quality:        float,
        stability:      float = 0.5,
    ) -> str:
        if not active_actions:
            return self._rng.choice(ACTIONS)

        ratio   = step / max(1, total_steps)
        epsilon = self._eps_start + (self._eps_end - self._eps_start) * ratio

        # Anti-collapse: quality peak → more exploration
        if quality >= 99.9:
            epsilon = min(0.30, epsilon + 0.10)
        # Instability detected → explore harder to escape
        if stability < 0.30:
            epsilon = min(0.35, epsilon + 0.15)

        if self._rng.random() < epsilon:
            return self._rng.choice(active_actions)

        # UCB selection
        N_total = max(1, self._N_state[state])
        ucb_c   = self._ucb_peak if quality >= 99.9 else self._ucb_normal

        best_a     = active_actions[0]
        best_score = -1e18
        for a in active_actions:
            n_sa      = self._N[state][a]
            ucb       = ucb_c * math.sqrt(math.log(N_total + 1) / (n_sa + 1))
            curiosity = 0.15 / math.sqrt(n_sa + 1)
            score     = self._Q[state][a] + ucb + curiosity
            if score > best_score:
                best_score = score
                best_a     = a

        q_greedy = max(active_actions, key=lambda a: self._Q[state][a])
        if best_a != q_greedy:
            self.ucb_explorations += 1

        return best_a

    # ── Q-Update ─────────────────────────────────────────────────────────────

    def update(
        self,
        state: tuple, action: str, reward: float,
        next_state: tuple, step: int, total_steps: int,
    ) -> None:
        self._N[state][action] += 1
        self._N_state[state]   += 1
        self.total_updates     += 1

        ratio = step / max(1, total_steps)
        alpha = max(0.08, self._alpha0 * (1.0 - 0.40 * ratio))

        max_next = max(self._Q[next_state][a] for a in ACTIONS)
        td_err   = reward + self._gamma * max_next - self._Q[state][action]
        self._Q[state][action] += alpha * td_err

        self._replay.push(state, action, reward, next_state)

    # ── Experience Replay ─────────────────────────────────────────────────────

    def replay_learn(
        self, n_samples: int = 12, step: int = 0, total_steps: int = 1
    ) -> int:
        samples = self._replay.sample(n_samples, self._rng)
        if not samples:
            return 0
        ratio        = step / max(1, total_steps)
        alpha_replay = max(0.04, self._alpha0 * 0.5 * (1.0 - 0.40 * ratio))
        for (s, a, r, ns) in samples:
            max_next = max(self._Q[ns][a2] for a2 in ACTIONS)
            td_err   = r + self._gamma * max_next - self._Q[s][a]
            self._Q[s][a] += alpha_replay * td_err
        return len(samples)

    # ── NAS-Aware Reward ──────────────────────────────────────────────────────

    @staticmethod
    def compute_reward(
        delta_combined:  float,
        is_novel:        bool,
        quality_before:  float,
        quality_after:   float,
        fitness_before:  float,
        fitness_after:   float,
        nas_current:     Optional["NASResult"]  = None,
        nas_previous:    Optional["NASResult"]  = None,
    ) -> float:
        """
        Reward multi-komponen NAS-aware:

          Formula signals (original):
            +Δcombined×8.0         — improvement skor gabungan
            +0.08 / -0.05          — novelty bonus / revisit penalty
            +0.20                  — bonus fitness naik di peak quality
            +0.10                  — recovery: quality naik dari <85%→≥85%

          NAS signals (BARU):
            +Δstability×4.0       — stability_score naik
            +Δconvergence×3.0     — convergence_rate naik
            -0.80                  — NaN detected (training diverged)
            +0.15                  — NaN dihindari & stability baik
        """
        r = delta_combined * 8.0
        r += 0.08 if is_novel else -0.05

        if quality_before >= 99.9 and fitness_after > fitness_before:
            r += 0.20
        if quality_before < 85.0 and quality_after >= 85.0:
            r += 0.10

        # NAS signals
        if nas_current is not None and nas_previous is not None:
            stab_delta = nas_current.stability_score - nas_previous.stability_score
            r += stab_delta * 4.0

            conv_delta = nas_current.convergence_rate - nas_previous.convergence_rate
            r += conv_delta * 3.0

            if nas_current.nan_detected:
                r -= 0.80
            elif nas_current.stability_score > 0.60:
                r += 0.15

        return r

    # ── Stats ─────────────────────────────────────────────────────────────────

    def action_q_values(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for a in ACTIONS:
            vals = [self._Q[s][a] for s in self._Q]
            out[a] = round(sum(vals) / max(1, len(vals)), 4)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  PERTURBATION ENGINE  (8 aksi — tidak berubah)
# ══════════════════════════════════════════════════════════════════════════════

class _PerturbEngine:
    """8 perturbasi arsitektur. Masing-masing return deskripsi (kosong = N/A)."""

    def __init__(self, rng: random.Random, gpu: GPUSpec):
        self._rng = rng
        self._gpu = gpu

    def apply(self, rule_id: str, cfg: ArchConfig) -> str:
        fn = {
            "P_BATCH":  self._batch,
            "P_FFN":    self._ffn,
            "P_HEAD":   self._head,
            "P_SEQ":    self._seq,
            "P_KV":     self._kv,
            "P_LAYERS": self._layers,
            "P_EXPERT": self._expert,
            "P_OPT":    self._opt,
        }.get(rule_id)
        return fn(cfg) if fn else ""

    def _batch(self, cfg: ArchConfig) -> str:
        mults = [2.0, 1.5, 0.5, 0.75, 3.0, 4.0, 0.25]
        self._rng.shuffle(mults)
        for m in mults:
            new_bs = max(1, int(cfg.batch_size * m))
            if new_bs != cfg.batch_size:
                old = cfg.batch_size
                cfg.batch_size = new_bs
                return f"batch {old}→{new_bs}"
        return ""

    def _ffn(self, cfg: ArchConfig) -> str:
        deltas = [-0.5, 0.5, -0.25, 0.25, -0.125, 0.125, 1.0, -1.0, 0.75, -0.75]
        self._rng.shuffle(deltas)
        is_moe = cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
        lo, hi = (0.25, 2.0) if is_moe else (1.5, 5.5)
        for d in deltas:
            nm = round(cfg.ffn_multiplier + d, 3)
            if lo <= nm <= hi:
                nf = int(cfg.hidden_dim * nm)
                if nf % 128 == 0 and nf > 0:
                    old = cfg.ffn_multiplier
                    cfg.ffn_multiplier = nm
                    return f"ffn_mult {old:.3f}→{nm:.3f} (ffn_dim={nf})"
        return ""

    def _head(self, cfg: ArchConfig) -> str:
        opts = [h for h in [32, 64, 128]
                if cfg.hidden_dim % h == 0
                and cfg.hidden_dim // h >= 4
                and h != cfg.head_dim]
        if not opts:
            return ""
        new_hd        = self._rng.choice(opts)
        old_hd        = cfg.head_dim
        cfg.head_dim  = new_hd
        cfg.num_heads = cfg.hidden_dim // new_hd
        valid_kv = [h for h in [1, 2, 4, 8]
                    if cfg.num_heads % h == 0 and h <= cfg.num_heads]
        if valid_kv and cfg.num_kv_heads not in valid_kv:
            cfg.num_kv_heads = min(valid_kv, key=lambda h: abs(h - cfg.num_kv_heads))
        return f"head_dim {old_hd}→{new_hd} (heads={cfg.num_heads})"

    def _seq(self, cfg: ArchConfig) -> str:
        if cfg.seq_len < 4096:
            return ""
        opts = []
        if cfg.attn_type != AttentionType.GQA:
            opts.append(AttentionType.GQA)
        if cfg.seq_len >= 8192 and cfg.attn_type != AttentionType.SLIDE:
            opts.append(AttentionType.SLIDE)
        if not opts:
            return ""
        old_type      = cfg.attn_type
        cfg.attn_type = self._rng.choice(opts)
        if cfg.attn_type == AttentionType.SLIDE:
            cfg.window_size = max(512, cfg.seq_len // 8)
        return f"attn {old_type.name}→{cfg.attn_type.name}"

    def _kv(self, cfg: ArchConfig) -> str:
        valid = [h for h in [1, 2, 4, 8, 16]
                 if h <= cfg.num_heads
                 and cfg.num_heads % h == 0
                 and h != cfg.num_kv_heads]
        if not valid:
            return ""
        old_kv           = cfg.num_kv_heads
        cfg.num_kv_heads = self._rng.choice(valid)
        if cfg.num_kv_heads < cfg.num_heads:
            cfg.attn_type = AttentionType.GQA
        return f"kv_heads {old_kv}→{cfg.num_kv_heads}"

    def _layers(self, cfg: ArchConfig) -> str:
        deltas = [2, -2, 4, -4, 1, -1, 6, -6]
        self._rng.shuffle(deltas)
        for d in deltas:
            new_l = cfg.num_layers + d
            if 4 <= new_l <= 64:
                old_l          = cfg.num_layers
                cfg.num_layers = new_l
                return f"layers {old_l}→{new_l}"
        return ""

    def _expert(self, cfg: ArchConfig) -> str:
        if cfg.ffn_type not in (FFNType.MOE, FFNType.MOE_TOPK):
            return ""
        opts_e  = [e for e in [4, 8, 16] if e != cfg.num_experts]
        opts_tk = [k for k in [1, 2]
                   if k != cfg.top_k_experts and k < cfg.num_experts]
        choices = []
        if opts_e:  choices.append("experts")
        if opts_tk: choices.append("topk")
        if not choices:
            return ""
        what = self._rng.choice(choices)
        if what == "experts":
            new_e = self._rng.choice(opts_e)
            old_e = cfg.num_experts
            if cfg.top_k_experts >= new_e:
                cfg.top_k_experts = max(1, new_e // 2)
            cfg.num_experts = new_e
            return f"experts {old_e}→{new_e} (top_k={cfg.top_k_experts})"
        else:
            new_tk            = self._rng.choice(opts_tk)
            old_tk            = cfg.top_k_experts
            cfg.top_k_experts = new_tk
            return f"top_k {old_tk}→{new_tk}"

    def _opt(self, cfg: ArchConfig) -> str:
        is_moe = cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
        gpu    = self._gpu
        if is_moe and gpu.vram_gb < 48:
            candidates = [OptimizerType.ADAM_8BIT, OptimizerType.LION]
        elif is_moe:
            candidates = [OptimizerType.ZERO2, OptimizerType.ZERO3,
                          OptimizerType.ADAM_8BIT]
        else:
            candidates = [
                OptimizerType.ADAM_FP32, OptimizerType.ADAM_8BIT,
                OptimizerType.LION,       OptimizerType.ADAMW_BF16,
                OptimizerType.ZERO1,      OptimizerType.ZERO2,
            ]
        opts = [o for o in candidates if o != cfg.optimizer_type]
        if not opts:
            return ""
        old_opt            = cfg.optimizer_type
        cfg.optimizer_type = self._rng.choice(opts)
        return f"optimizer {old_opt.name}→{cfg.optimizer_type.name}"


# ══════════════════════════════════════════════════════════════════════════════
#  NAS ADAPTIVE REFINER  —  MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class NASAdaptiveRefiner:
    """
    Adaptive ARC Refinement Engine dengan NAS Real Training + RL.

    Phase A: Standard convergence (ArcRefiner — formula + arch fixes)
    Phase B: NAS-guided RL exploration

    Phase B loop per step:
      1. Encode state (q, f, stability, family) — 4D dengan NAS dimension
      2. Perturbasi config via _PerturbEngine (RL UCB action selection)
      3. NAS evaluation: train proxy model → NASResult (NYATA, bukan estimasi)
      4. Compute reward: Δstability×4 + Δconvergence×3 + Δcombined×8
      5. Q-update Bellman + Experience Replay
      6. Accept/Reject: berdasarkan combined_score_nas (formula + NAS fitness)
      7. Anti-collapse: reset jika stuck di peak quality

    Cross-arch learning:
      • Q-table dibagi lintas semua ARC dalam satu batch
      • NASCache dibagi → re-evaluasi fingerprint identik dihindari
      • ExperienceBuffer → replay lintas ARC

    NAS weight = 0.30 (adjustable):
      combined = 0.70 × formula_score + 0.30 × nas_fitness
    """

    def __init__(
        self,
        gpu:               GPUSpec,
        max_iterations:    int   = 30,
        target_pct:        float = 100.0,
        max_explore_iters: int   = 35,
        quality_weight:    float = 0.35,
        fitness_weight:    float = 0.65,
        nas_weight:        float = 0.30,
        min_quality:       float = 70.0,
        suspicious_thresh: float = 0.10,
        rng_seed:          Optional[int] = None,
        nas_cfg:           Optional[NASConfig] = None,
    ):
        self.gpu               = gpu
        self.max_iterations    = max_iterations
        self.target_pct        = target_pct
        self.max_explore_iters = max_explore_iters
        self.quality_weight    = quality_weight
        self.fitness_weight    = fitness_weight
        self.nas_weight        = nas_weight
        self.min_quality       = min_quality
        self.suspicious_thresh = suspicious_thresh

        seed         = rng_seed or 42
        self._rng    = random.Random(seed)
        self._scorer = ArcQualityScorer(gpu)
        self._base   = ArcRefiner(gpu,
                                  max_iterations=max_iterations,
                                  target_pct=target_pct)
        self._perturb = _PerturbEngine(self._rng, gpu)

        # NAS components
        self._nas_cfg     = nas_cfg or NASConfig()
        self._nas_trainer = NASProxyTrainer(self._nas_cfg)
        self._nas_cache   = NASCache(self._nas_cfg.cache_size)

        # Shared RL state (cross-arch)
        self._replay = ExperienceBuffer(capacity=1000)
        self._qtable = RLQTable(
            rng           = self._rng,
            replay_buffer = self._replay,
            alpha         = 0.35,
            gamma         = 0.92,
            ucb_c_normal  = 1.8,
            ucb_c_peak    = 3.0,
            eps_start     = 0.20,
            eps_end       = 0.05,
        )

    # ── NAS Evaluation (cached) ───────────────────────────────────────────────

    def _nas_eval(
        self, cfg: ArchConfig, alog: AdaptiveLog
    ) -> NASResult:
        """
        Evaluasi NAS untuk ArchConfig dengan caching.
        Increment counters di alog.
        """
        fp     = NASProxyTrainer._fingerprint(cfg)
        cached = self._nas_cache.get(fp)
        if cached is not None:
            alog.nas_cache_hits += 1
            return cached

        alog.nas_evaluations += 1
        result = self._nas_trainer.evaluate(cfg)
        alog.nas_training_ms_total += result.training_time_ms

        if result.nan_detected:
            alog.nas_nan_count += 1

        self._nas_cache.put(fp, result)
        return result

    # ── Combined Score (NAS-enhanced) ────────────────────────────────────────

    def _combined(
        self, quality: float, fitness: float, nas_fitness: float
    ) -> float:
        return compute_combined_score_nas(
            quality, fitness, nas_fitness,
            quality_weight = self.quality_weight,
            fitness_weight = self.fitness_weight,
            nas_weight     = self.nas_weight,
            min_quality    = self.min_quality,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def refine(self, cfg: ArchConfig) -> Tuple[ArchConfig, AdaptiveLog]:
        """
        Refine satu ARC: Phase A (formula) → NAS baseline → Phase B (RL).
        """
        # ── Phase A: Standard convergence ──────────────────────────────────────
        refined_cfg, base_log = self._base.refine(cfg)

        q0 = base_log.initial_pct
        f0 = cfg.fitness_score

        q_cur = self._scorer.score(refined_cfg).pct
        f_cur = refined_cfg.fitness_score

        alog = AdaptiveLog(
            arch_id        = cfg.arch_id,
            arch_name      = cfg.arch_name,
            base_log       = base_log,
            quality_start  = q0,
            quality_end    = q_cur,
            fitness_start  = f0,
            fitness_end    = f_cur,
        )

        # ── NAS Baseline (sebelum RL) ───────────────────────────────────────────
        nas_base = self._nas_eval(refined_cfg, alog)
        alog.nas_stability_start = nas_base.stability_score
        alog.nas_fitness_start   = nas_base.nas_fitness

        c0_nas  = self._combined(q0,    f0,    nas_base.nas_fitness)
        c_cur   = self._combined(q_cur, f_cur, nas_base.nas_fitness)
        alog.combined_start = c0_nas
        alog.combined_end   = c_cur

        if q_cur >= 95.0 and f_cur < self.suspicious_thresh:
            alog.is_suspicious = True

        # ── Phase B: NAS-guided RL Exploration ─────────────────────────────────
        if q_cur >= 75.0 and self.max_explore_iters > 0:
            refined_cfg, nas_final = self._rl_explore(
                refined_cfg, nas_base, alog
            )

            q_fin  = self._scorer.score(refined_cfg).pct
            f_fin  = refined_cfg.fitness_score
            c_fin  = self._combined(q_fin, f_fin, nas_final.nas_fitness)

            alog.quality_end         = q_fin
            alog.fitness_end         = f_fin
            alog.combined_end        = c_fin
            alog.nas_stability_end   = nas_final.stability_score
            alog.nas_fitness_end     = nas_final.nas_fitness
            alog.rl_q_updates        = self._qtable.total_updates
            alog.rl_ucb_explorations = self._qtable.ucb_explorations
            alog.rule_effectiveness  = self._qtable.action_q_values()
            alog.is_suspicious = (q_fin >= 95.0 and f_fin < self.suspicious_thresh)
        else:
            alog.nas_stability_end = nas_base.stability_score
            alog.nas_fitness_end   = nas_base.nas_fitness

        return refined_cfg, alog

    def refine_batch(
        self, archs: List[ArchConfig],
    ) -> Tuple[List[ArchConfig], List[AdaptiveLog], Dict[str, float]]:
        """
        Refine batch ARCs dengan shared Q-table + NASCache.
        Returns: (sorted_archs, logs, quality_map)
        """
        refined: List[ArchConfig]  = []
        logs:    List[AdaptiveLog] = []
        qmap:    Dict[str, float]  = {}

        for cfg in archs:
            r_cfg, alog = self.refine(cfg)
            refined.append(r_cfg)
            logs.append(alog)
            qmap[r_cfg.arch_id] = alog.quality_end

        refined.sort(
            key=lambda a: self._combined(
                qmap.get(a.arch_id, 0.0),
                a.fitness_score,
                self._nas_cache.get(NASProxyTrainer._fingerprint(a)).nas_fitness
                if self._nas_cache.get(NASProxyTrainer._fingerprint(a)) else 0.0,
            ),
            reverse=True,
        )
        return refined, logs, qmap

    # ── Phase B: NAS-guided RL Exploration Loop ───────────────────────────────

    def _rl_explore(
        self,
        cfg:      ArchConfig,
        nas_init: NASResult,
        alog:     AdaptiveLog,
    ) -> Tuple[ArchConfig, NASResult]:
        """
        RL exploration loop dengan NAS-aware reward.

        Per step:
          1. Encode state 4D (quality, fitness, stability, family)
          2. UCB + ε-greedy action selection (stability-aware epsilon)
          3. Apply perturbasi
          4. Re-derive formula fields
          5. NAS evaluation (real PyTorch training, cached)
          6. Compute NAS-aware reward
          7. Q-update Bellman + Experience Replay
          8. Accept/Reject: Δcombined_nas > threshold
          9. Anti-collapse: reset streak jika stuck di peak quality

        Berhenti bila:
          • step > 75% budget DAN quality ≥ 99.9%
          • ATAU semua action fail-streak habis DAN quality < 85%
        """
        best_cfg      = copy.deepcopy(cfg)
        best_quality  = self._scorer.score(best_cfg).pct
        best_fitness  = best_cfg.fitness_score
        best_nas      = nas_init
        best_combined = self._combined(
            best_quality, best_fitness, best_nas.nas_fitness
        )

        fail_streak: Dict[str, int] = {a: 0 for a in ACTIONS}
        MAX_FAIL   = 4
        no_improve = 0
        MAX_PAT    = 12
        T          = self.max_explore_iters

        for step in range(T):
            # ── 1. Encode state (4D dengan NAS stability) ─────────────────────
            state = RLQTable.encode_state(
                best_quality, best_fitness,
                best_nas.stability_score, best_cfg.arch_family,
            )

            # ── 2. Filter aksi yang belum habis fail-streak ───────────────────
            active = [a for a in ACTIONS if fail_streak.get(a, 0) < MAX_FAIL]
            if not active:
                fail_streak = {a: 0 for a in ACTIONS}
                no_improve  = 0
                active      = list(ACTIONS)

            # ── 3. Select action (stability-aware epsilon) ────────────────────
            chosen = self._qtable.select_action(
                state, active, step, T,
                best_quality, best_nas.stability_score,
            )
            alog.perturbation_tries += 1

            # ── 4. Apply perturbasi ───────────────────────────────────────────
            candidate = copy.deepcopy(best_cfg)
            desc      = self._perturb.apply(chosen, candidate)

            if not desc:
                self._qtable.update(state, chosen, -0.02, state, step, T)
                fail_streak[chosen] = fail_streak.get(chosen, 0) + 1
                no_improve += 1
                if no_improve >= MAX_PAT and best_quality < 85.0:
                    break
                continue

            # ── 5. Re-derive formula fields ───────────────────────────────────
            try:
                self._base._full_rederive(candidate)
            except Exception:
                self._qtable.update(state, chosen, -0.05, state, step, T)
                fail_streak[chosen] = fail_streak.get(chosen, 0) + 1
                no_improve += 1
                continue

            # ── 6. NAS Evaluation (real PyTorch training, cached) ─────────────
            cand_nas     = self._nas_eval(candidate, alog)
            cand_quality = self._scorer.score(candidate).pct
            cand_fitness = candidate.fitness_score
            cand_combined = self._combined(
                cand_quality, cand_fitness, cand_nas.nas_fitness
            )

            next_state = RLQTable.encode_state(
                cand_quality, cand_fitness,
                cand_nas.stability_score, candidate.arch_family,
            )

            delta    = cand_combined - best_combined
            is_novel = self._qtable.is_novel(candidate)
            if is_novel:
                alog.rl_diversity_bonuses += 1

            # ── 7. Compute NAS-aware reward ───────────────────────────────────
            reward = RLQTable.compute_reward(
                delta_combined  = delta,
                is_novel        = is_novel,
                quality_before  = best_quality,
                quality_after   = cand_quality,
                fitness_before  = best_fitness,
                fitness_after   = cand_fitness,
                nas_current     = cand_nas,
                nas_previous    = best_nas,
            )

            # ── 8. Q-update ───────────────────────────────────────────────────
            self._qtable.update(state, chosen, reward, next_state, step, T)

            # ── 9. Experience Replay ──────────────────────────────────────────
            if len(self._replay) >= 20:
                n_rep = self._qtable.replay_learn(
                    n_samples=12, step=step, total_steps=T
                )
                alog.rl_replay_updates += n_rep

            # ── 10. Accept/Reject ─────────────────────────────────────────────
            # Terima jika combined naik (formula + NAS), ATAU jika NAS jauh lebih stabil
            stab_gain = cand_nas.stability_score - best_nas.stability_score
            accept = (delta > 0.0001) or (stab_gain > 0.15 and delta > -0.05)

            if accept:
                best_cfg      = candidate
                best_quality  = cand_quality
                best_fitness  = cand_fitness
                best_nas      = cand_nas
                best_combined = cand_combined
                alog.perturbations_accepted += 1
                alog.improvement_events.append(
                    f"[step{step+1}/{chosen}] {desc}"
                    f"  q={cand_quality:.1f}% f={cand_fitness:.4f}"
                    f"  stab={cand_nas.stability_score:.3f}"
                    f"  combined→{best_combined:.5f} (Δ{delta:+.5f})"
                    f"  NAS({cand_nas.training_time_ms:.0f}ms)"
                )
                fail_streak[chosen] = 0
                no_improve          = 0
            else:
                fail_streak[chosen] = fail_streak.get(chosen, 0) + 1
                no_improve         += 1

                # ── Anti-collapse ──────────────────────────────────────────────
                if no_improve >= MAX_PAT:
                    if best_quality >= 99.9:
                        fail_streak = {a: 0 for a in ACTIONS}
                        no_improve  = 0
                        if step >= int(T * 0.75):
                            break
                    else:
                        break

        return best_cfg, best_nas


# Alias untuk backward compatibility
AdaptiveRefiner = NASAdaptiveRefiner


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER  (drop-in untuk pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def adaptive_refine_archs(
    archs:             List[ArchConfig],
    gpu:               GPUSpec,
    max_iterations:    int   = 30,
    target_pct:        float = 100.0,
    max_explore_iters: int   = 35,
    quality_weight:    float = 0.35,
    fitness_weight:    float = 0.65,
    nas_weight:        float = 0.30,
    rng_seed:          Optional[int] = None,
    nas_cfg:           Optional[NASConfig] = None,
) -> Tuple[List[ArchConfig], List[AdaptiveLog], Dict[str, float]]:
    """
    Drop-in untuk pipeline.py.
    Returns (sorted_archs, adaptive_logs, quality_map).

    nas_weight=0.30 → 30% signal dari NAS real training, 70% formula.
    Untuk menonaktifkan NAS: set nas_weight=0.0 (fallback ke formula only).
    """
    refiner = NASAdaptiveRefiner(
        gpu,
        max_iterations    = max_iterations,
        target_pct        = target_pct,
        max_explore_iters = max_explore_iters,
        quality_weight    = quality_weight,
        fitness_weight    = fitness_weight,
        nas_weight        = nas_weight,
        rng_seed          = rng_seed,
        nas_cfg           = nas_cfg,
    )
    return refiner.refine_batch(archs)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_adaptive_summary(
    logs:        List[AdaptiveLog],
    quality_map: Dict[str, float],
    *,
    console=None,
) -> None:
    """Tabel ringkasan adaptive NAS-RL refinement."""
    _p = console.print if console else print

    ranked = sorted(logs, key=lambda l: l.combined_end, reverse=True)

    _p()
    _p("  ┌─ NAS-backed RL Refinement Summary ──────────────────────────────────────────────────────────────")
    _p("  │")
    _p("  │  Tiga dimensi skor per ARC:")
    _p("  │    Quality %  = konsistensi internal formula ARC (0–100)")
    _p("  │    Fitness    = performa training estimasi  (MFU · throughput · VRAM eff.)")
    _p("  │    NAS Stab   = stabilitas real PyTorch training  (loss variance + grad norm)")
    _p("  │    Combined   = 70%×[35%×quality + 65%×fitness] + 30%×NAS_fitness")
    _p("  │")
    _p("  │  Engine: NAS (real proxy training) → RL (Q-learning + UCB + NAS reward)")
    _p("  │  State-space: (quality_bucket, fitness_bucket, stability_bucket, family)")
    _p("  │  Reward: Δstability×4 + Δconvergence×3 + Δcombined×8 ± NaN penalty")
    _p("  │")
    _p(f"  │  {'Rank':<5} {'ARC-ID':<12} "
       f"{'Quality':>14}  {'Fitness':>14}  {'NAS-Stab':>14}  "
       f"{'Combined':>14}  {'RL+NAS':>14}  Status")
    _p("  │  " + "─" * 130)

    for rank, log in enumerate(ranked, 1):
        sym     = "★" if rank == 1 else f"#{rank}"
        suspect = " ⚠" if log.is_suspicious else ""
        rl_info = (f"{log.perturbation_tries}t/"
                   f"{log.perturbations_accepted}a/"
                   f"{log.nas_evaluations}n/"
                   f"{log.nas_cache_hits}c")
        stab_start = f"{log.nas_stability_start:.3f}"
        stab_end   = f"{log.nas_stability_end:.3f}"
        _p(
            f"  │  {sym:<5} {log.arch_id:<12} "
            f"{log.quality_start:>6.1f}%→{log.quality_end:>5.1f}%  "
            f"{log.fitness_start:>6.4f}→{log.fitness_end:>6.4f}  "
            f"{stab_start:>6}→{stab_end:>6}     "
            f"{log.combined_start:>6.4f}→{log.combined_end:>6.4f}  "
            f"{rl_info:>14}  "
            f"{log.status}{suspect}"
        )

    _p("  │")
    _p("  │  t=perturb_tries · a=accepted · n=NAS_evals · c=cache_hits")
    _p("  │  NAS: real PyTorch training (proxy scaled-down, rasio terjaga)")
    _p("  │  RL belajar langsung dari sinyal NAS: stability dan convergence.")
    _p("  └─────────────────────────────────────────────────────────────────────────────────────────────────")
    _p()


def print_adaptive_log(log: AdaptiveLog, *, console=None) -> None:
    """Print detail satu AdaptiveLog — Phase A dan Phase B NAS-RL."""
    _p = console.print if console else print

    _p(f"\n  ─── NAS-RL Log: {log.arch_id} {'─'*44}")
    _p(f"       Quality     : {log.quality_start:.1f}% → {log.quality_end:.1f}%")
    _p(f"       Fitness     : {log.fitness_start:.4f} → {log.fitness_end:.4f}")
    _p(f"       NAS Stability: {log.nas_stability_start:.4f} → {log.nas_stability_end:.4f}"
       f"   Δ={log.nas_stability_delta:+.4f}")
    _p(f"       NAS Fitness  : {log.nas_fitness_start:.4f} → {log.nas_fitness_end:.4f}")
    _p(f"       Combined    : {log.combined_start:.5f} → {log.combined_end:.5f}"
       f"   Δ={log.combined_delta:+.5f}")
    _p(f"       NAS Stats   : {log.nas_evaluations} evals  "
       f"{log.nas_cache_hits} cache-hits  "
       f"{log.nas_nan_count} NANs  "
       f"{log.nas_training_ms_total:.0f}ms total")
    _p(f"       RL Stats    : {log.perturbation_tries} tries  "
       f"{log.perturbations_accepted} accepted  "
       f"{log.rl_replay_updates} replay  "
       f"{log.rl_ucb_explorations} UCB-exp  "
       f"{log.rl_diversity_bonuses} diversity")
    _p(f"       Status      : {log.status}")

    # Phase A
    bl = log.base_log
    if bl.score_history:
        hist = " → ".join(f"{p:.1f}%" for p in bl.score_history)
        _p(f"       Phase A quality history: {hist}")
    if bl.fixes_applied:
        p1 = [f for f in bl.fixes_applied if "/P1]" in f]
        p2 = [f for f in bl.fixes_applied if "/P2]" in f]
        if p1:
            _p(f"       Phase A / P1 fixes ({len(p1)}):")
            for fx in p1[:6]:
                _p(f"         • {fx}")
        if p2:
            _p(f"       Phase A / P2 arch fixes ({len(p2)}):")
            for fx in p2[:6]:
                _p(f"         • {fx}")

    # Phase B NAS-RL improvements
    if log.improvement_events:
        _p(f"       Phase B NAS-RL improvements ({len(log.improvement_events)}):")
        for ev in log.improvement_events:
            _p(f"         ↑ {ev}")

    # Top Q-values
    if log.rule_effectiveness:
        top = sorted(log.rule_effectiveness.items(),
                     key=lambda x: x[1], reverse=True)[:4]
        eff = "  ".join(f"{k}:{v:+.3f}" for k, v in top)
        _p(f"       Top Q-values (NAS-learned): {eff}")
    _p()


def print_nas_cache_stats(refiner: NASAdaptiveRefiner, *, console=None) -> None:
    """Print statistik NAS cache — berguna untuk tuning cache_size."""
    _p = console.print if console else print
    c  = refiner._nas_cache
    _p(f"  NAS Cache: {len(c)} entries | "
       f"hits={c.hits} misses={c.misses} "
       f"hit_rate={c.hit_rate:.1%}")
