"""
hardware_refine.py — Hardware-Aware NAS + RL Refinement Engine
═══════════════════════════════════════════════════════════════════════════════

Menggantikan adaptive_refiner.py. Fokus murni pada hardware optimization
menggunakan data real dari hardware.py (GPUSpec).

Arsitektur:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  HardwareNASEvaluator  → scoring 7 dimensi hardware (total 100 poin)    │
  │  HardwareQLearner      → Q-table cross-arch dengan UCB exploration       │
  │  HardwareNASRefiner    → Phase A (heuristik) + Phase B (RL hardware)    │
  └──────────────────────────────────────────────────────────────────────────┘

Hardware Score — 7 Dimensi:
  H1  MFU Utilization        25 pts  — seberapa mendekati peak GPU throughput
  H2  Throughput Efficiency  20 pts  — tokens/sec vs GPU ceiling
  H3  VRAM Utilization       15 pts  — penggunaan budget VRAM secara efisien
  H4  TC Alignment           15 pts  — Tensor Core tile alignment per GPU gen
  H5  SM Occupancy           10 pts  — Streaming Multiprocessor occupancy
  H6  Compute Boundness      10 pts  — arithmetic intensity vs ridge point
  H7  FA Tile Feasibility     5 pts  — FlashAttention SMEM constraint
                              ─────
  TOTAL                      100 pts  → hardware_score [0.0, 1.0]

RL State Space:
  (hw_score_bucket, bottleneck_bucket, vram_bucket, family_idx)
  Buckets: 6 × 4 × 5 × 7 = 840 states

RL Actions (9 hardware-centric perturbations):
  ALIGN_HIDDEN, ALIGN_HEAD_DIM, ALIGN_FFN, INCR_BATCH, DECR_BATCH,
  ENABLE_FA, ENABLE_COMPILE, DISABLE_GC, OPT_EFFICIENT

Combined Score (50/50 seimbang dengan train_refine.py):
  combined = 0.50 × hardware_score + 0.50 × training_score
"""

from __future__ import annotations

import copy
import math
import random
import hashlib
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np

from arch_types import ArchConfig, AttentionType, FFNType, OptimizerType, NormType
from hardware import GPUSpec
from generator import ArchitectureGenerator, VRAM_LIMIT_PCT
from refiner import ArcQualityScorer, ArcRefiner, RefinementLog


# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA
# ══════════════════════════════════════════════════════════════════════════════

_FAMILY_LIST = [
    "CoT-Optimizer", "Speed-Demon", "Balanced-Pro", "MoE-Sparse",
    "Long-Horizon", "Nano-Efficient", "Compute-Dense",
]
_FAMILY_IDX: Dict[str, int] = {f: i for i, f in enumerate(_FAMILY_LIST)}

# Hardware score buckets (6 level)
_HW_BUCKETS = [
    (0.00, 0.20),   # 0: sangat buruk
    (0.20, 0.35),   # 1: buruk
    (0.35, 0.50),   # 2: di bawah rata-rata
    (0.50, 0.65),   # 3: rata-rata
    (0.65, 0.80),   # 4: baik
    (0.80, 1.01),   # 5: excellent
]

# Bottleneck type mapping
_BOTTLENECK_IDX: Dict[str, int] = {
    "compute-bound":       0,
    "memory-bound":        1,
    "kernel-launch-bound": 2,
    "balanced":            3,
}

# VRAM utilization buckets (5 level)
_VRAM_BUCKETS = [
    (0.00, 0.35),   # 0: sangat underutilized
    (0.35, 0.50),   # 1: rendah
    (0.50, 0.65),   # 2: sedang
    (0.65, 0.78),   # 3: baik
    (0.78, 1.01),   # 4: mendekati limit
]

# RL Actions — 15 total (9 lama + 6 baru untuk arch yang sudah baik)
HW_ACTIONS = [
    # ── Alignment actions (original 9) ────────────────────────────────────
    "ALIGN_HIDDEN",       # align hidden_dim ke GPU tile boundary
    "ALIGN_HEAD_DIM",     # align head_dim untuk FA feasibility
    "ALIGN_FFN",          # align FFN intermediate dim ke tile
    "INCR_BATCH",         # naikkan batch size (better utilization)
    "DECR_BATCH",         # turunkan batch size (free VRAM)
    "ENABLE_FA",          # aktifkan flash attention
    "ENABLE_COMPILE",     # aktifkan torch.compile
    "DISABLE_GC",         # nonaktifkan gradient checkpointing
    "OPT_EFFICIENT",      # switch ke optimizer hemat VRAM
    # ── Fine-tuning actions (NEW — untuk arch yang sudah alignment-optimal) ─
    "TUNE_KV_GQA",        # reduce kv_heads → GQA (kurangi KV cache, naikkan MFU)
    "TUNE_FFN_MULT_UP",   # naikkan FFN multiplier → lebih compute-bound
    "TUNE_FFN_MULT_DOWN", # turunkan FFN multiplier → throughput lebih tinggi
    "TUNE_LAYERS_SPEED",  # kurangi layers 1-2 → step time lebih cepat
    "BATCH_VRAM_FILL",    # isi sisa VRAM dengan batch size lebih besar
    "INCR_HEAD_DIM_64",   # naikkan head_dim ke 64/128 untuk FA lebih optimal
]

_N_HW_BUCKETS   = len(_HW_BUCKETS)
_N_BN_BUCKETS   = 4
_N_VRAM_BUCKETS = len(_VRAM_BUCKETS)
_N_FAM          = len(_FAMILY_LIST)
_N_ACTIONS      = len(HW_ACTIONS)   # 15


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareNASResult:
    """Hasil evaluasi hardware NAS untuk satu ArchConfig."""
    arch_id: str = ""

    # Sub-scores per dimensi [0, 1]
    mfu_score:             float = 0.0   # H1: MFU vs peak
    throughput_score:      float = 0.0   # H2: tokens/sec vs ceiling
    vram_efficiency:       float = 0.0   # H3: VRAM utilization
    tc_alignment:          float = 0.0   # H4: tensor core alignment
    sm_occupancy_score:    float = 0.0   # H5: SM occupancy
    compute_bound_score:   float = 0.0   # H6: compute vs memory bound
    flash_attn_score:      float = 0.0   # H7: FA tile feasibility

    # Aggregated
    hardware_score: float = 0.0          # [0, 1] — weighted combination

    # Detail dari GPU spec yang digunakan
    gpu_name:         str   = ""
    ridge_point:      float = 0.0
    effective_bw:     float = 0.0
    peak_flops_tf:    float = 0.0
    tc_tile_size:     int   = 16
    bottleneck_type:  str   = "unknown"

    # Per-dimensi poin (untuk display)
    pts_h1: float = 0.0
    pts_h2: float = 0.0
    pts_h3: float = 0.0
    pts_h4: float = 0.0
    pts_h5: float = 0.0
    pts_h6: float = 0.0
    pts_h7: float = 0.0

    @property
    def total_pts(self) -> float:
        return self.pts_h1 + self.pts_h2 + self.pts_h3 + self.pts_h4 + \
               self.pts_h5 + self.pts_h6 + self.pts_h7

    @property
    def grade(self) -> str:
        s = self.hardware_score
        if s >= 0.90: return "S ★★★  Excellent Hardware Fit"
        if s >= 0.80: return "A+ ★★  Very Good"
        if s >= 0.70: return "A  ★   Good"
        if s >= 0.55: return "B      Average"
        if s >= 0.40: return "C      Below Average"
        return              "F  ✗   Poor Hardware Fit"


@dataclass
class HardwareAdaptiveLog:
    """Log satu siklus hardware NAS refinement."""
    arch_id: str = ""
    arch_name: str = ""

    # Scores
    quality_start:   float = 0.0
    quality_end:     float = 0.0
    fitness_start:   float = 0.0
    fitness_end:     float = 0.0
    hw_score_start:  float = 0.0
    hw_score_end:    float = 0.0
    combined_start:  float = 0.0
    combined_end:    float = 0.0

    # RL stats
    perturbation_tries:     int = 0
    perturbations_accepted: int = 0
    rl_ucb_explorations:    int = 0
    rl_replay_updates:      int = 0

    # Alignment stats
    tc_improvements:    int = 0
    vram_improvements:  int = 0
    mfu_improvements:   int = 0

    improvement_events: List[str] = field(default_factory=list)
    rule_effectiveness: Dict[str, float] = field(default_factory=dict)
    status: str = ""

    # Base refinement log
    base_log: Optional[RefinementLog] = None

    @property
    def hw_delta(self) -> float:
        return round(self.hw_score_end - self.hw_score_start, 4)

    @property
    def combined_delta(self) -> float:
        return round(self.combined_end - self.combined_start, 5)

    @property
    def is_suspicious(self) -> bool:
        return self.quality_end >= 100.0 and self.hw_score_end < 0.30


# ══════════════════════════════════════════════════════════════════════════════
#  LRU CACHE
# ══════════════════════════════════════════════════════════════════════════════

class HardwareLRUCache:
    """LRU cache untuk hardware evaluation (fingerprint-based)."""

    def __init__(self, max_size: int = 512):
        self._cache:    OrderedDict[str, HardwareNASResult] = OrderedDict()
        self._max_size: int = max_size
        self.hits:   int = 0
        self.misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get(self, key: str) -> Optional[HardwareNASResult]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: HardwareNASResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE NAS EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class HardwareNASEvaluator:
    """
    Evaluasi hardware fitness dari ArchConfig menggunakan GPUSpec.

    Semua metrik diambil LANGSUNG dari GPUSpec (hardware.py):
      - bf16_tflops, mfu_typical_max/min, thermal_factor
      - effective_memory_bw_gbps, ecc_bw_overhead
      - optimal_tile_size, tensor_core_gen
      - shared_mem_max_kb (FA tile constraint)
      - sm_count, max_warps_per_sm
      - typical_sm_occupancy

    Setiap dimensi dinormalisasi ke [0, 1] lalu diberi bobot.
    """

    # Bobot per dimensi
    W_H1_MFU        = 0.25
    W_H2_THROUGHPUT = 0.20
    W_H3_VRAM       = 0.15
    W_H4_TC_ALIGN   = 0.15
    W_H5_SM_OCC     = 0.10
    W_H6_COMPUTE    = 0.10
    W_H7_FA         = 0.05

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu

    def _fingerprint(self, cfg: ArchConfig) -> str:
        """Hash fingerprint dari parameter yang mempengaruhi hardware score."""
        key = (
            cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads,
            cfg.head_dim, cfg.ffn_multiplier, cfg.seq_len, cfg.batch_size,
            cfg.use_flash_attn, cfg.use_torch_compile, cfg.use_gradient_checkpointing,
            cfg.optimizer_type, cfg.use_mixed_precision,
        )
        return hashlib.md5(str(key).encode()).hexdigest()[:16]

    def evaluate(self, cfg: ArchConfig) -> HardwareNASResult:
        """
        Evaluasi hardware fitness dari satu ArchConfig.
        Returns HardwareNASResult dengan hardware_score [0, 1].
        """
        gpu = self.gpu
        result = HardwareNASResult(arch_id=cfg.arch_id)
        result.gpu_name    = gpu.name
        result.tc_tile_size = gpu.optimal_tile_size

        # Informasi ridge point dari GPU spec
        peak_flops  = gpu.bf16_tflops * 1e12 * gpu.thermal_factor
        eff_bw      = gpu.effective_memory_bw_gbps * 1e9 * gpu.thermal_factor
        ridge       = peak_flops / max(1.0, eff_bw)
        result.ridge_point   = round(ridge, 1)
        result.effective_bw  = round(gpu.effective_memory_bw_gbps, 1)
        result.peak_flops_tf = round(gpu.bf16_tflops * gpu.thermal_factor, 1)
        result.bottleneck_type = cfg.bottleneck

        # ── H1: MFU Score (25 pts) ────────────────────────────────────────────
        h1 = self._score_mfu(cfg)
        result.mfu_score = h1
        result.pts_h1    = round(h1 * 25.0, 2)

        # ── H2: Throughput Efficiency (20 pts) ───────────────────────────────
        h2 = self._score_throughput(cfg)
        result.throughput_score = h2
        result.pts_h2           = round(h2 * 20.0, 2)

        # ── H3: VRAM Utilization (15 pts) ────────────────────────────────────
        h3 = self._score_vram(cfg)
        result.vram_efficiency = h3
        result.pts_h3          = round(h3 * 15.0, 2)

        # ── H4: TC Alignment (15 pts) ────────────────────────────────────────
        h4 = self._score_tc_alignment(cfg)
        result.tc_alignment = h4
        result.pts_h4       = round(h4 * 15.0, 2)

        # ── H5: SM Occupancy (10 pts) ────────────────────────────────────────
        h5 = self._score_sm_occupancy(cfg)
        result.sm_occupancy_score = h5
        result.pts_h5             = round(h5 * 10.0, 2)

        # ── H6: Compute Boundness (10 pts) ───────────────────────────────────
        h6 = self._score_compute_bound(cfg, ridge)
        result.compute_bound_score = h6
        result.pts_h6              = round(h6 * 10.0, 2)

        # ── H7: FA Tile Feasibility (5 pts) ──────────────────────────────────
        h7 = self._score_flash_attn(cfg)
        result.flash_attn_score = h7
        result.pts_h7           = round(h7 * 5.0, 2)

        # ── Weighted combination ──────────────────────────────────────────────
        result.hardware_score = float(np.clip(
            self.W_H1_MFU        * h1 +
            self.W_H2_THROUGHPUT * h2 +
            self.W_H3_VRAM       * h3 +
            self.W_H4_TC_ALIGN   * h4 +
            self.W_H5_SM_OCC     * h5 +
            self.W_H6_COMPUTE    * h6 +
            self.W_H7_FA         * h7,
            0.0, 1.0
        ))

        return result

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _score_mfu(self, cfg: ArchConfig) -> float:
        """
        H1: MFU score.
        mfu_estimate / mfu_typical_max → normalized efficiency.
        Model yang mencapai 80%+ dari max GPU MFU mendapat score penuh.
        """
        gpu    = self.gpu
        target = gpu.mfu_typical_max   # e.g. 0.55 untuk A100
        mfu    = cfg.mfu_estimate

        # Gradual scoring: 0% at mfu=0, 100% at mfu=target+
        score = mfu / max(0.001, target)

        # Bonus untuk model yang melampaui typical min dengan margin
        if mfu > gpu.mfu_typical_min * 1.5:
            score = min(1.0, score * 1.05)   # small bonus

        return float(np.clip(score, 0.0, 1.0))

    def _score_throughput(self, cfg: ArchConfig) -> float:
        """
        H2: Throughput efficiency vs GPU ceiling.
        ceiling = bf16_tflops × mfu_max / flops_per_token
        """
        gpu          = self.gpu
        flops_per_tok = max(1.0, cfg.flops_per_token_fwd + cfg.flops_per_token_bwd)
        gpu_tok_ceil = (gpu.bf16_tflops * 1e12 * gpu.mfu_typical_max) / flops_per_tok
        gpu_tok_ceil = max(10_000.0, gpu_tok_ceil)

        score = cfg.tokens_per_sec_estimate / gpu_tok_ceil
        return float(np.clip(score, 0.0, 1.0))

    def _score_vram(self, cfg: ArchConfig) -> float:
        """
        H3: VRAM utilization score.
        Target: 55–78% dari budget (terlalu rendah = underutilized, terlalu
        tinggi = OOM risk). Kurva berbentuk trapezoid.
        """
        if not cfg.fits_gpu:
            return 0.0

        pct = cfg.vram_usage_pct / 100.0  # fraction of total VRAM
        budget_pct = pct / VRAM_LIMIT_PCT  # fraction of allowed budget

        # Trapezoid scoring
        lo_ideal = 0.55   # >= 55% budget = mulai dapat nilai penuh
        hi_ideal = 0.90   # <= 90% budget = masih penuh (75% of total)
        hi_warn  = 1.00   # > 100% budget = 0

        if budget_pct < 0.20:          # sangat underutilized
            return budget_pct / 0.20 * 0.40
        elif budget_pct < lo_ideal:    # underutilized
            return 0.40 + (budget_pct - 0.20) / (lo_ideal - 0.20) * 0.45
        elif budget_pct <= hi_ideal:   # sweet spot
            return 1.0 - (budget_pct - lo_ideal) / (hi_ideal - lo_ideal) * 0.05
        elif budget_pct <= hi_warn:    # approaching limit
            return 0.95 - (budget_pct - hi_ideal) / (hi_warn - hi_ideal) * 0.95
        else:
            return 0.0

    def _score_tc_alignment(self, cfg: ArchConfig) -> float:
        """
        H4: Tensor Core alignment.
        Semua dimensi kritis harus habis dibagi GPU tile size.
        Gen4 (H100): tile=64, lebih ketat.
        Gen3 (A100): tile=32.
        Gen1/2: tile=16.
        """
        gpu  = self.gpu
        tile = gpu.optimal_tile_size   # 16, 32, atau 64

        ffn_dim = int(cfg.hidden_dim * cfg.ffn_multiplier)

        # Dimensi kritis untuk TC
        checks = [
            (cfg.hidden_dim % tile == 0,         3.0,  "hidden_dim"),
            (cfg.head_dim   % tile == 0,         2.5,  "head_dim"),
            (ffn_dim        % tile == 0,         2.5,  "ffn_dim"),
            (cfg.hidden_dim % 64   == 0,         1.0,  "hidden_64"),  # additional
            (cfg.num_heads  % 8    == 0,         0.5,  "num_heads_8"),
        ]

        total_weight  = sum(w for _, w, _ in checks)
        earned_weight = sum(w for ok, w, _ in checks if ok)

        score = earned_weight / max(0.001, total_weight)

        # Bonus: semua kritis aligned
        if checks[0][0] and checks[1][0] and checks[2][0]:
            score = min(1.0, score + 0.05)

        # Gen4 lebih ketat: tile=64 lebih sulit, jika tidak aligned penalti lebih besar
        if gpu.tensor_core_gen >= 4 and not checks[0][0]:
            score *= 0.80   # hidden_dim tidak align ke 64 → penalti signifikan

        return float(np.clip(score, 0.0, 1.0))

    def _score_sm_occupancy(self, cfg: ArchConfig) -> float:
        """
        H5: SM Occupancy dari bottleneck_factors.
        Semakin tinggi SM occupancy, semakin efisien GPU.
        """
        occ = cfg.bottleneck_factors.get("sm_occ_model", None)
        if occ is None:
            occ = cfg.sm_occupancy   # fallback ke field langsung

        # Score linear, penalti untuk occupancy < 0.5
        if occ < 0.30:
            return occ / 0.30 * 0.40
        elif occ < 0.50:
            return 0.40 + (occ - 0.30) / 0.20 * 0.20
        elif occ < 0.70:
            return 0.60 + (occ - 0.50) / 0.20 * 0.25
        else:
            return 0.85 + min(0.15, (occ - 0.70) / 0.30 * 0.15)

    def _score_compute_bound(self, cfg: ArchConfig, ridge: float) -> float:
        """
        H6: Compute-bound score.
        Compute-bound (AI > ridge) → lebih efisien karena compute units saturated.
        Skor tinggi = compute-saturated, skor rendah = memory-bound.
        """
        ai = cfg.arithmetic_intensity

        if ridge <= 0:
            ridge = 100.0   # fallback

        ratio = ai / ridge   # > 1.0 = compute-bound

        compute_sat = cfg.bottleneck_factors.get("compute_saturation", None)
        mem_sat     = cfg.bottleneck_factors.get("memory_saturation",  None)

        if compute_sat is not None and mem_sat is not None:
            # Lebih akurat: gunakan saturation langsung
            if compute_sat > 0.70:     # compute-bound dominant
                score = compute_sat * (1.0 - mem_sat * 0.25)
            elif mem_sat > 0.70:       # memory-bound: kurang baik
                score = (1.0 - mem_sat) * 0.65
            else:                      # balanced
                score = 0.55 + compute_sat * 0.25
        else:
            # Fallback: gunakan ratio AI/ridge
            if ratio >= 2.0:
                score = 1.0
            elif ratio >= 1.0:
                score = 0.70 + (ratio - 1.0) / 1.0 * 0.30
            elif ratio >= 0.5:
                score = 0.40 + (ratio - 0.5) / 0.5 * 0.30
            else:
                score = ratio / 0.5 * 0.40

        return float(np.clip(score, 0.0, 1.0))

    def _score_flash_attn(self, cfg: ArchConfig) -> float:
        """
        H7: FlashAttention tile feasibility.
        Menggunakan shared_mem_max_kb dari GPUSpec dan head_dim.
        """
        gpu = self.gpu

        if not cfg.use_flash_attn:
            # Tidak menggunakan FA: penalti ringan (mungkin valid untuk model kecil)
            return 0.30

        # Dari bottleneck_factors jika ada
        fa_tile = cfg.bottleneck_factors.get("fa_tile_feasibility", None)
        if fa_tile is not None:
            return float(np.clip(fa_tile, 0.0, 1.0))

        # Manual heuristic dari GPUSpec
        smem_kb   = gpu.shared_mem_max_kb   # max configurable SMEM per SM
        head_dim  = cfg.head_dim

        # FA tile size = 2 × block_size × head_dim × 2 bytes (BF16)
        # Standard block_size = 128, head_dim biasanya 64/128/256
        tile_kb = 2 * 128 * head_dim * 2 / 1024

        if smem_kb <= 0:
            return 0.50   # tidak ada info → neutral

        if tile_kb <= smem_kb * 0.50:
            return 1.00   # banyak ruang
        elif tile_kb <= smem_kb * 0.75:
            return 0.85
        elif tile_kb <= smem_kb:
            return 0.65
        else:
            return 0.20   # SMEM terlalu kecil untuk tile yang diperlukan


# ══════════════════════════════════════════════════════════════════════════════
#  Q-LEARNER
# ══════════════════════════════════════════════════════════════════════════════

class HardwareQLearner:
    """
    Q-learning cross-arch untuk hardware optimization.
    State: (hw_bucket, bottleneck, vram_bucket, family_idx)
    """

    def __init__(
        self,
        alpha:       float = 0.18,   # learning rate
        gamma:       float = 0.88,   # discount factor
        epsilon:     float = 0.30,   # exploration rate awal
        epsilon_min: float = 0.06,
        ucb_c:       float = 1.8,    # UCB exploration constant
    ):
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_min = epsilon_min
        self.ucb_c       = ucb_c

        # Q-table: state_key → [Q per action]
        self._q:      Dict[str, List[float]] = defaultdict(lambda: [0.0] * _N_ACTIONS)
        self._counts: Dict[str, List[int]]   = defaultdict(lambda: [0]   * _N_ACTIONS)
        self._total_steps: int = 0

        # Experience replay buffer
        self._replay: List[Tuple] = []
        self._replay_cap: int = 2000

    def _state_key(
        self,
        hw_score: float,
        bottleneck: str,
        vram_pct: float,
        family: str,
    ) -> str:
        hw_b  = _bucket_idx(hw_score,  _HW_BUCKETS)
        bn_b  = _BOTTLENECK_IDX.get(bottleneck, 3)
        vr_b  = _bucket_idx(vram_pct / 100.0, _VRAM_BUCKETS)
        fam_b = _FAMILY_IDX.get(family, 0)
        return f"{hw_b}:{bn_b}:{vr_b}:{fam_b}"

    def select_action(
        self,
        hw_score: float,
        bottleneck: str,
        vram_pct: float,
        family: str,
        fail_streak: Dict[str, int],
    ) -> int:
        """
        UCB + epsilon-greedy action selection.
        Actions dengan fail_streak tinggi dihindari.
        """
        key = self._state_key(hw_score, bottleneck, vram_pct, family)
        q   = self._q[key]
        cnt = self._counts[key]
        total_cnt = max(1, sum(cnt))

        # Epsilon decay
        eps = max(self.epsilon_min, self.epsilon * (0.995 ** self._total_steps))

        if random.random() < eps:
            # Random exploration, hindari action dengan fail_streak tinggi
            weights = []
            for i, act in enumerate(HW_ACTIONS):
                fs = fail_streak.get(act, 0)
                weights.append(max(0.1, 1.0 / (1 + fs * 0.5)))
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    return i
            return random.randrange(_N_ACTIONS)
        else:
            # UCB: Q + c × sqrt(ln(total_N) / (N_a + 1))
            ucb_vals = []
            for i in range(_N_ACTIONS):
                ucb = q[i] + self.ucb_c * math.sqrt(
                    math.log(total_cnt + 1) / (cnt[i] + 1)
                )
                # Penalti fail streak
                fs = fail_streak.get(HW_ACTIONS[i], 0)
                ucb -= fs * 0.15
                ucb_vals.append(ucb)
            return int(np.argmax(ucb_vals))

    def update(
        self,
        hw_score_old: float, bottleneck_old: str, vram_old: float, family: str,
        action_idx: int,
        reward: float,
        hw_score_new: float, bottleneck_new: str, vram_new: float,
    ) -> None:
        """Standard Q-learning update + experience replay push."""
        key_old = self._state_key(hw_score_old, bottleneck_old, vram_old, family)
        key_new = self._state_key(hw_score_new, bottleneck_new, vram_new, family)

        q_next_max = max(self._q[key_new])
        q_old      = self._q[key_old][action_idx]

        self._q[key_old][action_idx] = q_old + self.alpha * (
            reward + self.gamma * q_next_max - q_old
        )
        self._counts[key_old][action_idx] += 1
        self._total_steps += 1

        # Push to replay
        exp = (key_old, action_idx, reward, key_new)
        self._replay.append(exp)
        if len(self._replay) > self._replay_cap:
            self._replay.pop(0)

    def replay_update(self, n: int = 16) -> int:
        """Batch replay update dari experience buffer."""
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

    def best_q_values(self, hw_score: float, bottleneck: str,
                       vram_pct: float, family: str) -> Dict[str, float]:
        key = self._state_key(hw_score, bottleneck, vram_pct, family)
        q   = self._q[key]
        return {HW_ACTIONS[i]: round(q[i], 4) for i in range(_N_ACTIONS)}


# ══════════════════════════════════════════════════════════════════════════════
#  PERTURBATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class HardwarePerturbationEngine:
    """
    Mengeksekusi hardware-centric perturbasi pada ArchConfig.
    Setiap perturbasi dimotivasi oleh hardware alignment / efficiency.
    """

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu
        self._gen = ArchitectureGenerator(gpu)

    def _recompute(self, cfg: ArchConfig) -> None:
        """Recompute semua derived metrics setelah perturbasi."""
        gen = self._gen
        gpu = self.gpu

        # Fix structural constraints
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
        """
        Terapkan satu action hardware ke cfg.
        Returns (new_cfg, description) atau (None, reason_fail).

        15 aksi total:
          Original 9: ALIGN_HIDDEN, ALIGN_HEAD_DIM, ALIGN_FFN, INCR/DECR_BATCH,
                       ENABLE_FA, ENABLE_COMPILE, DISABLE_GC, OPT_EFFICIENT
          New 6:      TUNE_KV_GQA, TUNE_FFN_MULT_UP, TUNE_FFN_MULT_DOWN,
                      TUNE_LAYERS_SPEED, BATCH_VRAM_FILL, INCR_HEAD_DIM_64
        """
        gpu  = self.gpu
        tile = gpu.optimal_tile_size
        new  = copy.deepcopy(cfg)

        # ── Original 9 actions ──────────────────────────────────────────────

        if action == "ALIGN_HIDDEN":
            target = ((cfg.hidden_dim + tile - 1) // tile) * tile
            if target == cfg.hidden_dim:
                target += tile
            if cfg.num_heads > 0:
                if target % cfg.head_dim == 0:
                    new.hidden_dim = target
                    new.num_heads  = target // cfg.head_dim
                else:
                    # Cari head_dim baru yang align ke target
                    candidates = [hd for hd in [32, 64, 128] if target % hd == 0]
                    if candidates:
                        new.head_dim  = candidates[0]
                        new.hidden_dim = target
                        new.num_heads  = target // candidates[0]
                    else:
                        return None, f"ALIGN_HIDDEN: tidak ada head_dim valid untuk {target}"
            else:
                return None, "ALIGN_HIDDEN: num_heads=0"
            desc = f"ALIGN_HIDDEN {cfg.hidden_dim}→{new.hidden_dim} (tile={tile})"

        elif action == "ALIGN_HEAD_DIM":
            # Coba align ke 64 dulu (universal FA-friendly)
            for target_hd in [64, 128, 32]:
                if target_hd != cfg.head_dim and cfg.hidden_dim % target_hd == 0:
                    new.head_dim   = target_hd
                    new.num_heads  = cfg.hidden_dim // target_hd
                    desc = f"ALIGN_HEAD_DIM {cfg.head_dim}→{target_hd}"
                    break
            else:
                target = ((cfg.head_dim + tile - 1) // tile) * tile
                if target == cfg.head_dim:
                    target += tile
                new.head_dim   = target
                new.hidden_dim = cfg.num_heads * target
                desc = f"ALIGN_HEAD_DIM {cfg.head_dim}→{target}"

        elif action == "ALIGN_FFN":
            ffn_dim = int(cfg.hidden_dim * cfg.ffn_multiplier)
            aligned = ((ffn_dim + 127) // 128) * 128
            if aligned == ffn_dim:
                return None, "ALIGN_FFN: sudah aligned ke 128"
            new.ffn_multiplier = round(aligned / max(1, cfg.hidden_dim), 4)
            desc = f"ALIGN_FFN ffn_dim {ffn_dim}→{aligned}"

        elif action == "INCR_BATCH":
            new.batch_size = cfg.batch_size + 1
            desc = f"INCR_BATCH {cfg.batch_size}→{new.batch_size}"

        elif action == "DECR_BATCH":
            if cfg.batch_size <= 1:
                return None, "DECR_BATCH: batch_size sudah minimum"
            new.batch_size = cfg.batch_size - 1
            desc = f"DECR_BATCH {cfg.batch_size}→{new.batch_size}"

        elif action == "ENABLE_FA":
            if cfg.use_flash_attn:
                return None, "ENABLE_FA: FA sudah aktif"
            new.use_flash_attn = True
            desc = "ENABLE_FA: aktifkan FlashAttention"

        elif action == "ENABLE_COMPILE":
            if cfg.use_torch_compile:
                return None, "ENABLE_COMPILE: sudah aktif"
            new.use_torch_compile = True
            desc = "ENABLE_COMPILE: aktifkan torch.compile"

        elif action == "DISABLE_GC":
            if not cfg.use_gradient_checkpointing:
                return None, "DISABLE_GC: GC sudah nonaktif"
            if cfg.vram_usage_pct > 72:
                return None, f"DISABLE_GC: VRAM {cfg.vram_usage_pct:.1f}% terlalu penuh"
            new.use_gradient_checkpointing = False
            desc = "DISABLE_GC: nonaktifkan gradient checkpointing"

        elif action == "OPT_EFFICIENT":
            if cfg.optimizer_type == OptimizerType.ADAM_8BIT:
                return None, "OPT_EFFICIENT: sudah 8-bit"
            if cfg.vram_usage_pct < 60:
                return None, "OPT_EFFICIENT: VRAM tidak tertekan, tidak perlu"
            new.optimizer_type = OptimizerType.ADAM_8BIT
            desc = f"OPT_EFFICIENT {cfg.optimizer_type}→ADAM_8BIT"

        # ── New 6 fine-tuning actions (untuk arch yang sudah alignment-optimal) ──

        elif action == "TUNE_KV_GQA":
            # Kurangi kv_heads untuk GQA — kurangi KV cache, naikkan MFU
            if cfg.num_kv_heads <= 1:
                return None, "TUNE_KV_GQA: kv_heads sudah minimum"
            if cfg.num_kv_heads == cfg.num_heads:
                # MHA → GQA: kurangi ke setengahnya (cari divisor valid)
                target_kv = cfg.num_kv_heads // 2
            else:
                target_kv = max(1, cfg.num_kv_heads - 1)
            # Cari kv_heads yang valid (harus membagi num_heads)
            valid_kv = [h for h in range(1, cfg.num_heads + 1)
                        if cfg.num_heads % h == 0 and h < cfg.num_kv_heads]
            if not valid_kv:
                return None, "TUNE_KV_GQA: tidak ada kv_heads valid lebih kecil"
            new.num_kv_heads = max(valid_kv, key=lambda h: abs(h - target_kv) * -1
                                   if abs(h - target_kv) > 0 else 1)
            # Pilih kv_heads yang paling dekat ke target
            new.num_kv_heads = min(valid_kv, key=lambda h: abs(h - target_kv))
            desc = f"TUNE_KV_GQA {cfg.num_kv_heads}→{new.num_kv_heads} kv_heads (GQA)"

        elif action == "TUNE_FFN_MULT_UP":
            # Naikkan FFN multiplier sedikit → lebih compute-bound, AI naik
            current = cfg.ffn_multiplier
            if current >= 6.0:
                return None, "TUNE_FFN_MULT_UP: multiplier sudah maksimum"
            # Naikkan ke kelipatan 0.25 berikutnya
            new_mult = round((current + 0.25) * 4) / 4
            if new_mult == current:
                new_mult = current + 0.25
            # Pastikan FFN dim align ke 128
            ffn_dim_new = int(cfg.hidden_dim * new_mult)
            ffn_aligned = ((ffn_dim_new + 127) // 128) * 128
            new.ffn_multiplier = round(ffn_aligned / max(1, cfg.hidden_dim), 4)
            desc = f"TUNE_FFN_MULT_UP {current:.3f}→{new.ffn_multiplier:.3f} (ffn={ffn_aligned})"

        elif action == "TUNE_FFN_MULT_DOWN":
            # Turunkan FFN multiplier → throughput lebih tinggi, token/s naik
            current = cfg.ffn_multiplier
            if current <= 2.0:
                return None, "TUNE_FFN_MULT_DOWN: multiplier sudah minimum"
            new_mult = round((current - 0.25) * 4) / 4
            if new_mult == current:
                new_mult = current - 0.25
            new_mult = max(2.0, new_mult)
            # Align ke 128
            ffn_dim_new = int(cfg.hidden_dim * new_mult)
            ffn_aligned = ((ffn_dim_new + 127) // 128) * 128
            ffn_aligned = max(ffn_aligned, 128)  # minimum 128
            new.ffn_multiplier = round(ffn_aligned / max(1, cfg.hidden_dim), 4)
            if abs(new.ffn_multiplier - current) < 0.01:
                return None, "TUNE_FFN_MULT_DOWN: perubahan terlalu kecil"
            desc = f"TUNE_FFN_MULT_DOWN {current:.3f}→{new.ffn_multiplier:.3f}"

        elif action == "TUNE_LAYERS_SPEED":
            # Kurangi layers 1-2 → step time lebih cepat, throughput naik
            if cfg.num_layers <= 4:
                return None, "TUNE_LAYERS_SPEED: layers terlalu sedikit"
            if cfg.vram_usage_pct > 70:
                return None, "TUNE_LAYERS_SPEED: VRAM penuh, hapus layers berisiko"
            new.num_layers = cfg.num_layers - 2
            desc = f"TUNE_LAYERS_SPEED {cfg.num_layers}→{new.num_layers} layers"

        elif action == "BATCH_VRAM_FILL":
            # Isi sisa VRAM budget dengan batch lebih besar
            # Hanya jika VRAM tersisa >20% dari budget
            if cfg.vram_usage_pct > 65:
                return None, f"BATCH_VRAM_FILL: VRAM {cfg.vram_usage_pct:.1f}% sudah tinggi"
            # Estimasi: setiap +1 batch ≈ +activations_gb
            vram_headroom = gpu.vram_gb * VRAM_LIMIT_PCT - cfg.vram_total_gb
            extra_batches = max(1, int(vram_headroom / max(0.01, cfg.vram_activations_gb) * 0.5))
            extra_batches = min(extra_batches, 3)
            if extra_batches == 0:
                return None, "BATCH_VRAM_FILL: tidak ada headroom VRAM"
            new.batch_size = cfg.batch_size + extra_batches
            desc = f"BATCH_VRAM_FILL {cfg.batch_size}→{new.batch_size} (headroom={vram_headroom:.2f}GB)"

        elif action == "INCR_HEAD_DIM_64":
            # Naikkan head_dim ke 64 atau 128 untuk FA lebih optimal
            if cfg.head_dim >= 128:
                return None, "INCR_HEAD_DIM_64: head_dim sudah ≥128"
            # Coba target 64 atau 128
            for target_hd in [128, 64]:
                if target_hd > cfg.head_dim and cfg.hidden_dim % target_hd == 0:
                    new.head_dim   = target_hd
                    new.num_heads  = cfg.hidden_dim // target_hd
                    # Sesuaikan kv_heads
                    valid_kv = [h for h in range(1, new.num_heads + 1)
                                if new.num_heads % h == 0]
                    if valid_kv:
                        new.num_kv_heads = min(valid_kv,
                                               key=lambda h: abs(h - cfg.num_kv_heads))
                    desc = f"INCR_HEAD_DIM_64 {cfg.head_dim}→{target_hd} (heads {cfg.num_heads}→{new.num_heads})"
                    break
            else:
                return None, f"INCR_HEAD_DIM_64: hidden_dim={cfg.hidden_dim} tidak divisible oleh 64/128"

        else:
            return None, f"Unknown action: {action}"

        # Recompute metrics
        self._recompute(new)

        # Pastikan masih valid
        if not new.fits_gpu:
            return None, f"{action}: OOM setelah perturbasi"

        return new, desc


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE NAS REFINER (MAIN CLASS)
# ══════════════════════════════════════════════════════════════════════════════

class HardwareNASRefiner:
    """
    Hardware-Aware NAS + RL Refinement Engine.

    Phase A: heuristik fixes via ArcRefiner (quality corrections)
    Phase B: RL hardware exploration via HardwareQLearner + HardwarePerturbationEngine

    Combined Score = 0.50 × hardware_score + 0.50 × training_score
    (training_score disuplai dari train_refine.py saat pipeline berjalan)
    """

    def __init__(
        self,
        gpu:               GPUSpec,
        max_iterations:    int   = 25,
        target_pct:        float = 100.0,
        max_explore_iters: int   = 30,
        hardware_weight:   float = 0.50,   # 50% hardware, 50% training
        rng_seed:          Optional[int] = None,
    ):
        self.gpu              = gpu
        self.max_iterations   = max_iterations
        self.target_pct       = target_pct
        self.max_explore_iters = max_explore_iters
        self.hw_weight        = hardware_weight

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Sub-components
        self._evaluator   = HardwareNASEvaluator(gpu)
        self._refiner_a   = ArcRefiner(gpu, max_iterations=max_iterations, target_pct=target_pct)
        self._scorer      = ArcQualityScorer(gpu)
        self._q_learner   = HardwareQLearner()
        self._perturber   = HardwarePerturbationEngine(gpu)
        self._hw_cache    = HardwareLRUCache(max_size=512)

    def _evaluate_cached(self, cfg: ArchConfig) -> HardwareNASResult:
        """Evaluasi hardware score dengan cache."""
        fp  = self._evaluator._fingerprint(cfg)
        hit = self._hw_cache.get(fp)
        if hit:
            return hit
        result = self._evaluator.evaluate(cfg)
        self._hw_cache.put(fp, result)
        return result

    def _compute_combined(self, quality_pct: float, hw_score: float,
                           training_score: float = 0.0) -> float:
        """
        Combined score 50/50: hardware + training.
        Saat training_score belum tersedia (=0), combined = hardware_score.
        """
        q_norm = quality_pct / 100.0
        return round(self.hw_weight * hw_score +
                     (1.0 - self.hw_weight) * training_score, 5)

    def refine(
        self,
        cfg: ArchConfig,
        training_score: float = 0.0,
    ) -> Tuple[ArchConfig, HardwareAdaptiveLog]:
        """
        Refine satu ArchConfig melalui Phase A + Phase B hardware RL.

        Args:
            cfg: ArchConfig dari generator
            training_score: [0,1] dari train_refine.py (default 0 = standalone)

        Returns:
            (refined_cfg, HardwareAdaptiveLog)
        """
        alog = HardwareAdaptiveLog(
            arch_id   = cfg.arch_id,
            arch_name = cfg.arch_name,
        )

        # ── Phase A: Quality heuristic fixes ─────────────────────────────────
        base_cfg, base_log = self._refiner_a.refine(cfg)
        alog.base_log    = base_log
        alog.quality_start = base_log.initial_pct
        alog.quality_end   = base_log.final_pct

        # Initial hardware evaluation
        hw_init            = self._evaluate_cached(base_cfg)
        alog.hw_score_start = hw_init.hardware_score
        alog.fitness_start  = base_cfg.fitness_score
        alog.combined_start = self._compute_combined(
            alog.quality_end, hw_init.hardware_score, training_score
        )

        # ── Phase B: RL Hardware Exploration ─────────────────────────────────
        best_cfg      = copy.deepcopy(base_cfg)
        best_hw       = hw_init
        best_quality  = alog.quality_end
        best_combined = alog.combined_start

        # Deteksi "already optimal": jika hw_score sudah tinggi DAN
        # semua alignment aksi kemungkinan besar gagal, eksplorasi lebih panjang
        already_high = hw_init.hardware_score >= 0.82
        # MAX_PAT lebih besar untuk arch yang sudah baik (butuh eksplorasi lebih)
        MAX_PAT = 15 if already_high else 10
        # Per-action patience (bukan global counter)
        action_fail:  Dict[str, int]   = {}
        action_tried: set              = set()
        no_improve  = 0
        T           = self.max_explore_iters

        # Simulated annealing temperature untuk hardware NAS
        sa_temp_init = 0.03 if already_high else 0.01
        sa_temp      = sa_temp_init

        for step in range(T):
            # Decay SA temperature
            sa_temp = sa_temp_init * (0.95 ** step)

            # Select action via RL
            act_idx = self._q_learner.select_action(
                best_hw.hardware_score,
                best_cfg.bottleneck,
                best_cfg.vram_usage_pct,
                best_cfg.arch_family,
                action_fail,
            )
            action = HW_ACTIONS[act_idx]
            alog.perturbation_tries += 1
            action_tried.add(action)

            # Apply perturbation
            new_cfg, desc = self._perturber.apply(best_cfg, action)
            if new_cfg is None:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1
                # Per-action patience: skip aksi ini jika sudah 3x gagal berturut
                if action_fail.get(action, 0) >= 3:
                    self._q_learner._q[
                        self._q_learner._state_key(
                            best_hw.hardware_score, best_cfg.bottleneck,
                            best_cfg.vram_usage_pct, best_cfg.arch_family
                        )
                    ][act_idx] = -2.0   # penalti Q-value langsung
                # Global patience
                if no_improve >= MAX_PAT:
                    # Cek apakah masih ada aksi yang belum dicoba
                    untried = [a for a in HW_ACTIONS if a not in action_tried]
                    if untried and step < T - 5:
                        # Reset dan paksa coba aksi yang belum dicoba
                        no_improve  = 0
                        action_fail = {k: max(0, v-1) for k, v in action_fail.items()}
                        continue
                    if best_quality >= 99.9:
                        action_fail = {}
                        no_improve  = 0
                    else:
                        break
                continue

            # Evaluate new config
            new_hw = self._evaluate_cached(new_cfg)
            report = self._scorer.score(new_cfg)
            new_q  = report.pct

            new_combined = self._compute_combined(new_q, new_hw.hardware_score, training_score)

            # RL reward: multidimensi — bukan hanya hw_score total
            delta_hw     = new_hw.hardware_score    - best_hw.hardware_score
            delta_mfu    = new_hw.mfu_score          - best_hw.mfu_score
            delta_thru   = new_hw.throughput_score   - best_hw.throughput_score
            delta_vram   = new_hw.vram_efficiency     - best_hw.vram_efficiency
            delta_tc     = new_hw.tc_alignment        - best_hw.tc_alignment
            align_bonus  = delta_tc * 0.5

            reward = (delta_hw   * 10.0 +
                      delta_mfu  *  4.0 +
                      delta_thru *  3.0 +
                      delta_vram *  2.0 +
                      align_bonus * 2.0)

            # Q-table update
            self._q_learner.update(
                best_hw.hardware_score, best_cfg.bottleneck, best_cfg.vram_usage_pct,
                best_cfg.arch_family,
                act_idx, reward,
                new_hw.hardware_score, new_cfg.bottleneck, new_cfg.vram_usage_pct,
            )
            alog.rl_replay_updates += self._q_learner.replay_update(12)

            # Accept criterion: combined naik ATAU (simulated annealing untuk eksplorasi)
            delta    = new_combined - best_combined
            # SA: terima perubahan kecil negatif dengan probabilitas exp(Δ/T)
            import math as _math
            sa_accept = (delta > -1e-7 or
                         (sa_temp > 0 and random.random() < _math.exp(delta / max(sa_temp, 1e-6))))

            if delta > 1e-6:
                # Strict improvement
                best_cfg      = new_cfg
                best_hw       = new_hw
                best_quality  = new_q
                best_combined = new_combined
                alog.perturbations_accepted += 1

                if new_hw.tc_alignment > hw_init.tc_alignment + 0.03:
                    alog.tc_improvements += 1
                if new_hw.vram_efficiency > hw_init.vram_efficiency + 0.03:
                    alog.vram_improvements += 1
                if new_hw.mfu_score > hw_init.mfu_score + 0.03:
                    alog.mfu_improvements += 1

                alog.improvement_events.append(
                    f"[step{step+1}] {action} → hw={new_hw.hardware_score:.4f}"
                    f" mfu={new_hw.mfu_score:.3f} tc={new_hw.tc_alignment:.3f}"
                    f" thru={new_hw.throughput_score:.3f}"
                    f" combined→{best_combined:.5f} (Δ{delta:+.5f})"
                )
                action_fail[action] = 0
                no_improve          = 0

            elif sa_accept and delta > -0.005:
                # SA exploration: terima perubahan kecil untuk escape local optima
                best_cfg      = new_cfg
                best_hw       = new_hw
                best_quality  = new_q
                best_combined = new_combined
                alog.improvement_events.append(
                    f"[step{step+1}] {action} SA-explore hw={new_hw.hardware_score:.4f}"
                    f" (Δ{delta:+.5f} T={sa_temp:.4f})"
                )
                action_fail[action] = 0
                no_improve += 1

            else:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1
                if no_improve >= MAX_PAT:
                    untried = [a for a in HW_ACTIONS if a not in action_tried]
                    if untried and step < T - 3:
                        no_improve  = 0
                        action_fail = {k: max(0, v-1) for k, v in action_fail.items()}
                        continue
                    if best_quality >= 99.9:
                        action_fail = {}
                        no_improve  = 0
                        if step >= int(T * 0.75):
                            break
                    else:
                        break

        # Final evaluation
        final_hw = self._evaluate_cached(best_cfg)
        alog.hw_score_end  = final_hw.hardware_score
        alog.fitness_end   = best_cfg.fitness_score
        alog.quality_end   = best_quality
        alog.combined_end  = self._compute_combined(
            best_quality, final_hw.hardware_score, training_score
        )

        # Rule effectiveness
        alog.rule_effectiveness = self._q_learner.best_q_values(
            final_hw.hardware_score,
            best_cfg.bottleneck,
            best_cfg.vram_usage_pct,
            best_cfg.arch_family,
        )

        # Status
        if best_quality >= 100.0 and final_hw.hardware_score >= 0.70:
            alog.status = "✓ OPTIMAL"
        elif alog.perturbations_accepted > 0:
            alog.status = f"↑ IMPROVED (hw Δ{alog.hw_delta:+.4f})"
        else:
            alog.status = "~ STAGNATED"

        return best_cfg, alog

    def refine_batch(
        self,
        archs: List[ArchConfig],
        training_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[ArchConfig], List[HardwareAdaptiveLog], Dict[str, float]]:
        """
        Refine batch ARCs.
        Returns (sorted_archs, logs, hw_score_map).
        training_scores: dict arch_id → training_score (opsional)
        """
        if training_scores is None:
            training_scores = {}

        refined = []
        logs    = []
        hw_map  = {}

        for cfg in archs:
            ts  = training_scores.get(cfg.arch_id, 0.0)
            new_cfg, log = self.refine(cfg, training_score=ts)
            refined.append(new_cfg)
            logs.append(log)
            hw_map[new_cfg.arch_id] = log.hw_score_end

        # Sort by combined_end descending
        log_by_id = {l.arch_id: l for l in logs}
        refined.sort(
            key=lambda c: log_by_id.get(c.arch_id, HardwareAdaptiveLog()).combined_end,
            reverse=True,
        )
        return refined, logs, hw_map


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_idx(value: float, buckets: List[Tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


def compute_hardware_score(cfg: ArchConfig, gpu: GPUSpec) -> float:
    """Convenience: hitung hardware_score dari satu ArchConfig."""
    return HardwareNASEvaluator(gpu).evaluate(cfg).hardware_score


def hardware_refine_archs(
    archs:             List[ArchConfig],
    gpu:               GPUSpec,
    max_iterations:    int   = 25,
    target_pct:        float = 100.0,
    max_explore_iters: int   = 30,
    rng_seed:          Optional[int] = None,
    training_scores:   Optional[Dict[str, float]] = None,
) -> Tuple[List[ArchConfig], List[HardwareAdaptiveLog], Dict[str, float]]:
    """
    Drop-in untuk pipeline.py.
    Returns (sorted_archs, hw_logs, hw_score_map).
    """
    refiner = HardwareNASRefiner(
        gpu,
        max_iterations    = max_iterations,
        target_pct        = target_pct,
        max_explore_iters = max_explore_iters,
        rng_seed          = rng_seed,
    )
    return refiner.refine_batch(archs, training_scores=training_scores)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_hardware_nas_result(
    result: HardwareNASResult,
    *,
    console=None,
) -> None:
    """Print laporan Hardware NAS detail."""
    _p = console.print if console else print

    DIM_LABELS = {
        "H1": f"MFU Utilization      (25 pts)  score={result.mfu_score:.3f}",
        "H2": f"Throughput Efficiency(20 pts)  score={result.throughput_score:.3f}",
        "H3": f"VRAM Utilization     (15 pts)  score={result.vram_efficiency:.3f}",
        "H4": f"TC Alignment         (15 pts)  score={result.tc_alignment:.3f}",
        "H5": f"SM Occupancy         (10 pts)  score={result.sm_occupancy_score:.3f}",
        "H6": f"Compute Boundness    (10 pts)  score={result.compute_bound_score:.3f}",
        "H7": f"FA Tile Feasibility   (5 pts)  score={result.flash_attn_score:.3f}",
    }
    pts_map = {
        "H1": result.pts_h1, "H2": result.pts_h2, "H3": result.pts_h3,
        "H4": result.pts_h4, "H5": result.pts_h5, "H6": result.pts_h6,
        "H7": result.pts_h7,
    }

    _p()
    _p(f"  ┌─ Hardware NAS Score ─── {result.arch_id} {'─'*35}")
    _p(f"  │  GPU: {result.gpu_name}  |  TC Tile={result.tc_tile_size}  "
       f"|  Ridge={result.ridge_point:.0f} FLOP/B  |  BW={result.effective_bw:.0f} GB/s")
    _p(f"  │  Hardware Score: {result.hardware_score:.4f}  "
       f"({result.hardware_score*100:.1f}%)   {result.grade}")
    _p(f"  └{'─'*65}")

    for dim, label in DIM_LABELS.items():
        pts  = pts_map[dim]
        mx   = {"H1":25,"H2":20,"H3":15,"H4":15,"H5":10,"H6":10,"H7":5}[dim]
        bar_n = int(pts / mx * 20) if mx > 0 else 0
        bar   = "█" * bar_n + "░" * (20 - bar_n)
        _p(f"\n  [{dim}] {label}")
        _p(f"       [{bar}]  {pts:.1f}/{mx}")
    _p()


def print_hardware_adaptive_summary(
    logs:    List[HardwareAdaptiveLog],
    hw_map:  Dict[str, float],
    *,
    console=None,
) -> None:
    """Tabel ringkasan Hardware NAS-RL refinement."""
    _p = console.print if console else print

    ranked = sorted(logs, key=lambda l: l.combined_end, reverse=True)

    _p()
    _p("  ┌─ Hardware NAS-RL Summary ──────────────────────────────────────────────────────────")
    _p("  │")
    _p("  │  Hardware Score (7 dimensi dari GPUSpec):")
    _p("  │    H1 MFU(25) + H2 Throughput(20) + H3 VRAM(15) + H4 TC-Align(15)")
    _p("  │    + H5 SM-Occ(10) + H6 Compute-Bound(10) + H7 FA-Tile(5)")
    _p("  │  Combined = 50% × hardware_score + 50% × training_score")
    _p("  │")
    _p(f"  │  {'Rank':<5} {'ARC-ID':<12} {'Quality':>14}  "
       f"{'HW-Score':>12}  {'Combined':>12}  {'RL':>10}  Status")
    _p("  │  " + "─" * 100)

    for rank, log in enumerate(ranked, 1):
        sym     = "★" if rank == 1 else f"#{rank}"
        suspect = " ⚠" if log.is_suspicious else ""
        rl_info = f"{log.perturbation_tries}t/{log.perturbations_accepted}a"
        _p(
            f"  │  {sym:<5} {log.arch_id:<12} "
            f"{log.quality_start:>6.1f}%→{log.quality_end:>5.1f}%  "
            f"{log.hw_score_start:>5.4f}→{log.hw_score_end:>5.4f}  "
            f"{log.combined_start:>5.4f}→{log.combined_end:>5.4f}  "
            f"{rl_info:>10}  "
            f"{log.status}{suspect}"
        )

    _p("  │")
    _p("  │  t=perturb_tries · a=accepted")
    _p("  │  RL Actions: ALIGN_HIDDEN/HEAD/FFN, INCR/DECR_BATCH, ENABLE_FA/COMPILE, DISABLE_GC, OPT_EFFICIENT")
    _p("  └───────────────────────────────────────────────────────────────────────────────────")
    _p()


def print_hardware_adaptive_log(log: HardwareAdaptiveLog, *, console=None) -> None:
    """Detail log satu HardwareAdaptiveLog."""
    _p = console.print if console else print

    _p(f"\n  ─── Hardware NAS-RL Log: {log.arch_id} {'─'*40}")
    _p(f"       Quality      : {log.quality_start:.1f}% → {log.quality_end:.1f}%")
    _p(f"       Fitness      : {log.fitness_start:.4f} → {log.fitness_end:.4f}")
    _p(f"       HW Score     : {log.hw_score_start:.4f} → {log.hw_score_end:.4f}"
       f"   Δ={log.hw_delta:+.4f}")
    _p(f"       Combined     : {log.combined_start:.5f} → {log.combined_end:.5f}"
       f"   Δ={log.combined_delta:+.5f}")
    _p(f"       RL Stats     : {log.perturbation_tries} tries  "
       f"{log.perturbations_accepted} accepted  "
       f"{log.rl_replay_updates} replay")
    _p(f"       Improvements : TC={log.tc_improvements}  "
       f"VRAM={log.vram_improvements}  MFU={log.mfu_improvements}")
    _p(f"       Status       : {log.status}")

    if log.improvement_events:
        _p(f"       Phase B Hardware Improvements ({len(log.improvement_events)}):")
        for ev in log.improvement_events:
            _p(f"         ↑ {ev}")

    if log.rule_effectiveness:
        top = sorted(log.rule_effectiveness.items(), key=lambda x: x[1], reverse=True)[:4]
        eff = "  ".join(f"{k}:{v:+.3f}" for k, v in top)
        _p(f"       Top Q-values: {eff}")
    _p()
