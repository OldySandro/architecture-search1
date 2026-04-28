"""
combination_nas.py — Combination Architecture NAS System
═══════════════════════════════════════════════════════════════════════════════

Sistem khusus untuk menghasilkan dan mengevaluasi arsitektur KOMBINASI dari
dua atau lebih family yang berbeda. Tidak random/ngasal — setiap kombinasi
divalidasi oleh rule engine sebelum diproses NAS.

FILOSOFI SISTEM:
  User memilih 2–4 AI type yang mau di-blend → sistem membuat 1 arsitektur
  TERKUAT dari blend tersebut. Bukan banyak arsitektur — satu yang paling
  optimal.

  Untuk RL, ada DUA modul yang tidak bertabrakan:
    • adaptive_refiner.py  → RL untuk single-type architecture
    • combination_refiner.py → RL khusus combination (file baru)

Arsitektur:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  CombinationRuleEngine  → validasi apakah N family bisa dikombinasikan   │
  │  CombinationBlender     → 3 strategi blend: Interleaved/Staged/Weighted  │
  │  CombinationNASEvaluator→ 5 dimensi khusus kombinasi (100 pts)           │
  │  CombinationQLearner    → Q-learning dasar (dipakai CombinationNASRefiner)│
  │  CombinationNASRefiner  → Legacy refiner (kompatibilitas pipeline lama)  │
  │                                                                           │
  │  ► Untuk RL yang lebih powerful, gunakan combination_refiner.py          │
  │    CombinationRefiner   → 16 aksi, prioritized replay, anti-bias reward  │
  └──────────────────────────────────────────────────────────────────────────┘

Combination NAS Score — 5 Dimensi:
  C1  Family Coherence       25 pts  — apakah N family secara arsitektur
                                       saling melengkapi (bukan bertentangan)
  C2  Blend Balance          20 pts  — apakah rasio blend seimbang & valid
  C3  Architectural Synergy  20 pts  — apakah depth/width/FFN hybrid masuk akal
  C4  Hardware Compatibility 20 pts  — hardware_score after blend
  C5  Training Synergy       15 pts  — apakah multi-family tidak konflik training
                              ─────
  TOTAL                      100 pts → combination_score [0.0, 1.0]

Blend Strategies:
  INTERLEAVED — alternasi per-layer antara family A dan B
                (misal: A-B-A-B-A-B, cocok untuk CoT+MoE)
  STAGED      — N layers pertama dari A, sisa dari B
                (misal: A-A-A-B-B-B, cocok untuk Speed+Balanced)
  WEIGHTED    — hyperparameter di-blend proporsional (0–1)
                (misal: hidden=0.6×A+0.4×B, cocok untuk Nano+Efficient)

Valid Combinations (28 dari 49 pairs):
  STRONGLY_VALID (×1.0 synergy):
    CoT-Optimizer  + Long-Horizon    (deep reasoning + context)
    CoT-Optimizer  + MoE-Sparse      (deep reasoning + sparse capacity)
    Balanced-Pro   + Long-Horizon    (general purpose + long context)
    Balanced-Pro   + MoE-Sparse      (general + sparse experts)
    Speed-Demon    + Nano-Efficient  (throughput + efficiency, both shallow)
    MoE-Sparse     + Compute-Dense   (sparse + dense compute)
    Nano-Efficient + Balanced-Pro    (small + balanced)
    Compute-Dense  + CoT-Optimizer   (compute heavy + reasoning)

  COMPATIBLE (×0.85 synergy):
    Speed-Demon    + Balanced-Pro
    Speed-Demon    + Compute-Dense
    CoT-Optimizer  + Balanced-Pro
    Long-Horizon   + MoE-Sparse
    Nano-Efficient + CoT-Optimizer
    Compute-Dense  + Long-Horizon
    Compute-Dense  + Balanced-Pro
    Nano-Efficient + MoE-Sparse

  MARGINAL (×0.70 synergy — butuh careful tuning):
    Speed-Demon    + CoT-Optimizer   (shallow vs deep — ratio kritis)
    Speed-Demon    + Long-Horizon    (low throughput + high context)
    Speed-Demon    + MoE-Sparse      (throughput loss dari MoE overhead)
    Nano-Efficient + Compute-Dense   (conflicting param budgets)
    Nano-Efficient + Long-Horizon    (memory constraint vs context)

  INVALID (kombinasi tidak diperbolehkan — saling bertentangan):
    + semua same-family pairs (tidak bisa blend family yang sama)

ask_combination_type():
  Fungsi UI ini menampilkan SATU daftar kombinasi valid dari N type yang
  dipilih user (semua valid pair dari pool). User memilih SATU strategi
  blend yang akan digunakan.

  Output: List[CombinationSpec] — 1 spec yang akan diproses NAS menjadi
  1 arsitektur terkuat via RL combination refinement.

Usage (dari pipeline.py):
  from combination_nas import (
      ask_combination_type,
      CombinationNASRefiner,
      run_combination_pipeline,
      print_combination_summary,
  )

  # Untuk RL yang lebih powerful:
  from combination_refiner import CombinationRefiner
  refiner = CombinationRefiner(gpu)
  best_cfg, best_spec, log = refiner.refine_to_best(specs)
"""

from __future__ import annotations

import copy
import math
import random
import hashlib
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, FrozenSet

import numpy as np

from arch_types import ArchConfig, AttentionType, FFNType, OptimizerType, NormType, PosEncType
from hardware import GPUSpec
from generator import ArchitectureGenerator, VRAM_LIMIT_PCT
from refiner import ArcQualityScorer, ArcRefiner, RefinementLog
from hardware_refine import HardwareNASEvaluator, HardwareNASResult
from train_refine import (
    ProxyTrainer, TrainingDynamicsEvaluator,
    TrainingNASResult, ProxyTrainingResult,
)


# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA
# ══════════════════════════════════════════════════════════════════════════════

ALL_FAMILIES = [
    "CoT-Optimizer", "Speed-Demon", "Balanced-Pro", "MoE-Sparse",
    "Long-Horizon", "Nano-Efficient", "Compute-Dense",
]

FAMILY_DESCRIPTIONS = {
    "CoT-Optimizer":  "Deep narrow — chain-of-thought reasoning",
    "Speed-Demon":    "Wide shallow — maximum throughput",
    "Balanced-Pro":   "Balanced depth/width — general purpose",
    "MoE-Sparse":     "Mixture-of-Experts — sparse capacity",
    "Long-Horizon":   "Extended context — long dependencies",
    "Nano-Efficient": "Ultra-small — max quality per VRAM byte",
    "Compute-Dense":  "Compute heavy — high arithmetic intensity",
}

# ── Synergy matrix ────────────────────────────────────────────────────────────
# (family_a, family_b) → (compatibility, synergy_mult, best_strategy, rationale)
# Key selalu disimpan dengan urutan alphabetical (family_a < family_b)
_SYNERGY_DB: Dict[Tuple[str, str], Dict] = {
    # ── STRONGLY VALID ────────────────────────────────────────────────────────
    # Synergy mult diturunkan: STRONGLY_VALID max = 0.92 (bukan 1.00)
    # Kombinasi selalu lebih sulit dari pure single-type, mult mencerminkan ini
    ("CoT-Optimizer", "Long-Horizon"): {
        "compat": "STRONGLY_VALID", "synergy": 0.92,
        "best_strategy": "STAGED",
        "rationale": "Deep reasoning layers dulu, lalu panjang konteks — seperti Qwen2.5",
        "recommended_ratio": 0.60,   # 60% CoT, 40% Long-Horizon
        "param_constraint": "hidden_dim match wajib",
    },
    ("CoT-Optimizer", "MoE-Sparse"): {
        "compat": "STRONGLY_VALID", "synergy": 0.88,
        "best_strategy": "INTERLEAVED",
        "rationale": "Reasoning dense + sparse expert capacity — seperti DeepSeek-R1",
        "recommended_ratio": 0.50,
        "param_constraint": "MoE expert_count ≥ 4",
    },
    ("Balanced-Pro", "Long-Horizon"): {
        "compat": "STRONGLY_VALID", "synergy": 0.86,
        "best_strategy": "STAGED",
        "rationale": "General pretraining + long-context attention — solid combo",
        "recommended_ratio": 0.55,
        "param_constraint": "seq_len ≥ 4096",
    },
    ("Balanced-Pro", "MoE-Sparse"): {
        "compat": "STRONGLY_VALID", "synergy": 0.84,
        "best_strategy": "INTERLEAVED",
        "rationale": "Dense baseline + MoE capacity untuk specialized tasks",
        "recommended_ratio": 0.60,
        "param_constraint": "MoE top_k=2",
    },
    ("Nano-Efficient", "Speed-Demon"): {
        "compat": "STRONGLY_VALID", "synergy": 0.82,
        "best_strategy": "WEIGHTED",
        "rationale": "Keduanya shallow — blend parameter cukup harmonis",
        "recommended_ratio": 0.50,
        "param_constraint": "total params < 200M",
    },
    ("Compute-Dense", "MoE-Sparse"): {
        "compat": "STRONGLY_VALID", "synergy": 0.81,
        "best_strategy": "INTERLEAVED",
        "rationale": "Dense compute layer diselingi sparse expert — arithmetic intensity tinggi",
        "recommended_ratio": 0.55,
        "param_constraint": "arithmetic intensity > 500",
    },
    ("Balanced-Pro", "Nano-Efficient"): {
        "compat": "STRONGLY_VALID", "synergy": 0.80,
        "best_strategy": "WEIGHTED",
        "rationale": "Balanced + small — baik untuk edge device dengan capability umum",
        "recommended_ratio": 0.65,
        "param_constraint": "total params < 500M",
    },
    ("CoT-Optimizer", "Compute-Dense"): {
        "compat": "STRONGLY_VALID", "synergy": 0.79,
        "best_strategy": "STAGED",
        "rationale": "High compute + reasoning depth — like Llama-3.3 style",
        "recommended_ratio": 0.55,
        "param_constraint": "ffn_mult ≥ 4.0",
    },
    # ── COMPATIBLE ────────────────────────────────────────────────────────────
    # COMPATIBLE synergy range: 0.62–0.72 (diturunkan dari 0.72–0.82)
    ("Balanced-Pro", "Speed-Demon"): {
        "compat": "COMPATIBLE", "synergy": 0.72,
        "best_strategy": "WEIGHTED",
        "rationale": "Blend balanced profile dengan throughput emphasis",
        "recommended_ratio": 0.55,
        "param_constraint": None,
    },
    ("Compute-Dense", "Speed-Demon"): {
        "compat": "COMPATIBLE", "synergy": 0.70,
        "best_strategy": "WEIGHTED",
        "rationale": "Dense compute + fast inference — throughput-oriented dense model",
        "recommended_ratio": 0.50,
        "param_constraint": None,
    },
    ("Balanced-Pro", "CoT-Optimizer"): {
        "compat": "COMPATIBLE", "synergy": 0.70,
        "best_strategy": "STAGED",
        "rationale": "Pondasi balanced + reasoning capacity atas",
        "recommended_ratio": 0.50,
        "param_constraint": None,
    },
    ("Long-Horizon", "MoE-Sparse"): {
        "compat": "COMPATIBLE", "synergy": 0.68,
        "best_strategy": "INTERLEAVED",
        "rationale": "Long context + MoE sparse — memory-intensive, perlu VRAM besar",
        "recommended_ratio": 0.55,
        "param_constraint": "VRAM > 20GB recommended",
    },
    ("CoT-Optimizer", "Nano-Efficient"): {
        "compat": "COMPATIBLE", "synergy": 0.65,
        "best_strategy": "STAGED",
        "rationale": "Reasoning dengan constraint VRAM — trade-off quality vs size",
        "recommended_ratio": 0.60,
        "param_constraint": "total params < 400M",
    },
    ("Compute-Dense", "Long-Horizon"): {
        "compat": "COMPATIBLE", "synergy": 0.65,
        "best_strategy": "WEIGHTED",
        "rationale": "High compute + long context — VRAM besar diperlukan",
        "recommended_ratio": 0.50,
        "param_constraint": "seq_len ≥ 4096",
    },
    ("Balanced-Pro", "Compute-Dense"): {
        "compat": "COMPATIBLE", "synergy": 0.68,
        "best_strategy": "WEIGHTED",
        "rationale": "Upgrade compute density dari balanced baseline",
        "recommended_ratio": 0.55,
        "param_constraint": None,
    },
    ("MoE-Sparse", "Nano-Efficient"): {
        "compat": "COMPATIBLE", "synergy": 0.62,
        "best_strategy": "WEIGHTED",
        "rationale": "MoE dengan constraint small model — hanya 2-4 experts",
        "recommended_ratio": 0.60,
        "param_constraint": "num_experts ≤ 4",
    },
    # ── MARGINAL ──────────────────────────────────────────────────────────────
    # MARGINAL synergy range: 0.40–0.55 (diturunkan dari 0.55–0.65)
    # Kombinasi marginal harus terlihat jelas susahnya
    ("CoT-Optimizer", "Speed-Demon"): {
        "compat": "MARGINAL", "synergy": 0.52,
        "best_strategy": "STAGED",
        "rationale": "Konflik depth vs throughput — ratio kritis, hati-hati",
        "recommended_ratio": 0.65,   # lebih banyak CoT untuk dominasi
        "param_constraint": "ratio wajib ≥ 0.60",
    },
    ("Long-Horizon", "Speed-Demon"): {
        "compat": "MARGINAL", "synergy": 0.46,
        "best_strategy": "WEIGHTED",
        "rationale": "Attention quadratic vs throughput — bisa OOM",
        "recommended_ratio": 0.70,   # lebih banyak Speed-Demon
        "param_constraint": "seq_len ≤ 4096",
    },
    ("MoE-Sparse", "Speed-Demon"): {
        "compat": "MARGINAL", "synergy": 0.48,
        "best_strategy": "STAGED",
        "rationale": "MoE routing overhead mengurangi throughput Speed-Demon",
        "recommended_ratio": 0.60,
        "param_constraint": "top_k=1 untuk minimize routing penalty",
    },
    ("Compute-Dense", "Nano-Efficient"): {
        "compat": "MARGINAL", "synergy": 0.44,
        "best_strategy": "WEIGHTED",
        "rationale": "Sangat bertentangan: high compute vs ultra-small",
        "recommended_ratio": 0.70,   # dominasi satu family
        "param_constraint": "total params < 300M wajib",
    },
    ("Long-Horizon", "Nano-Efficient"): {
        "compat": "MARGINAL", "synergy": 0.40,
        "best_strategy": "STAGED",
        "rationale": "Memory constraint vs context length — trade-off berat",
        "recommended_ratio": 0.65,
        "param_constraint": "seq_len ≤ 2048",
    },
}

# Normalisasi key ke alphabetical order
_SYNERGY_NORMALIZED: Dict[Tuple[str, str], Dict] = {}
for (a, b), v in _SYNERGY_DB.items():
    key = (min(a, b), max(a, b))
    _SYNERGY_NORMALIZED[key] = v

# ── Blend strategy constants ──────────────────────────────────────────────────
BLEND_INTERLEAVED = "INTERLEAVED"
BLEND_STAGED      = "STAGED"
BLEND_WEIGHTED    = "WEIGHTED"

# ── RL Actions untuk combination tuning ──────────────────────────────────────
COMBO_ACTIONS = [
    "SHIFT_RATIO_A",       # naikkan ratio family A (−B)
    "SHIFT_RATIO_B",       # naikkan ratio family B (−A)
    "SWITCH_INTERLEAVED",  # ganti ke interleaved strategy
    "SWITCH_STAGED",       # ganti ke staged strategy
    "SWITCH_WEIGHTED",     # ganti ke weighted strategy
    "TUNE_BLEND_DEPTH",    # sesuaikan total layers hasil blend
    "TUNE_BLEND_FFN",      # sesuaikan FFN multiplier hybrid
    "FIX_ATTENTION_UNITY", # samakan attn type ke satu jenis
    "ENABLE_SHARED_EMBED", # tie embeddings antar family
    "BALANCE_KV_HEADS",    # seimbangkan GQA ratio di hybrid
    "REDUCE_MOE_EXPERTS",  # kurangi experts jika MoE overloads VRAM
    "ADJUST_SEQ_LEN",      # sesuaikan seq_len untuk hybrid
]

_N_COMBO_ACTIONS = len(COMBO_ACTIONS)

# ── Scoring buckets ───────────────────────────────────────────────────────────
_COMBO_SCORE_BUCKETS = [
    (0.00, 0.20),  # 0: invalid/terrible
    (0.20, 0.40),  # 1: poor
    (0.40, 0.55),  # 2: marginal
    (0.55, 0.70),  # 3: acceptable
    (0.70, 0.85),  # 4: good
    (0.85, 1.01),  # 5: excellent
]

# Bucket untuk HW dan Train score — dipakai di RL state encoding
# Harus sama dengan yang di combination_refiner.py untuk cross-compatibility
_HW_SCORE_BUCKETS = [
    (0.00, 0.30),  # 0: poor hardware fit
    (0.30, 0.50),  # 1: below average
    (0.50, 0.65),  # 2: acceptable
    (0.65, 0.80),  # 3: good
    (0.80, 1.01),  # 4: excellent
]
_TRAIN_SCORE_BUCKETS = [
    (0.00, 0.30),  # 0: poor training dynamics
    (0.30, 0.50),  # 1: below average
    (0.50, 0.65),  # 2: acceptable
    (0.65, 0.80),  # 3: good
    (0.80, 1.01),  # 4: excellent
]

# Reward weights — SEIMBANG, tidak ada dimensi yang dominasi
# W_COMBO sedikit lebih kecil dari W_HW karena combo_score sudah aggregat 5 dimensi
_W_COMBO_RL  = 0.34
_W_HW_RL     = 0.33
_W_TRAIN_RL  = 0.33
# Balance penalty threshold — penalti bila satu dimensi jauh di bawah rata-rata
_BALANCE_PENALTY_THRESHOLD_NAS = 0.15


def _balanced_combo_reward(
    delta_combo: float,
    delta_hw:    float,
    delta_train: float,
    new_combo:   float,
    new_hw:      float,
    new_train:   float,
) -> float:
    """
    Reward function seimbang untuk combination RL.

    TIDAK bias ke satu dimensi. Formula:
      base      = W_COMBO×Δcombo + W_HW×Δhw + W_TRAIN×Δtrain
      synergy   = +0.05 jika semua 3 naik, +0.02 jika 2 dari 3 naik
      balance   = −penalti jika satu dimensi jauh tertinggal dari rata-rata

    Dengan W_COMBO ≈ W_HW ≈ W_TRAIN ≈ 0.33, tidak ada dimensi dominan.
    synergy bonus mendorong improvement yang merata.
    balance penalty mencegah arsitektur "bagus di satu sisi, buruk di lain".
    """
    base = _W_COMBO_RL * delta_combo + _W_HW_RL * delta_hw + _W_TRAIN_RL * delta_train

    # Synergy bonus: semua atau sebagian besar dimensi naik serentak
    n_improved = sum(1 for d in [delta_combo, delta_hw, delta_train] if d > 1e-4)
    synergy_bonus = 0.05 if n_improved >= 3 else (0.02 if n_improved >= 2 else 0.0)

    # Balance penalty: hukum jika satu dimensi sangat tertinggal
    mean_new = (new_combo + new_hw + new_train) / 3.0
    balance_penalty = 0.0
    for s in [new_combo, new_hw, new_train]:
        gap = mean_new - s
        if gap > _BALANCE_PENALTY_THRESHOLD_NAS:
            balance_penalty += (gap - _BALANCE_PENALTY_THRESHOLD_NAS) * 0.30

    return base + synergy_bonus - balance_penalty


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CombinationSpec:
    """
    Spesifikasi kombinasi N family (2 atau lebih).

    Perubahan dari versi sebelumnya:
      - families: List[str] — mendukung 2, 3, atau lebih family
      - ratios:   List[float] — kontribusi tiap family (sum=1.0, auto-normalized)
      - strategy: str — INTERLEAVED / STAGED / WEIGHTED
      - blend_order: urutan blending (SEQUENTIAL = A→B→C, TREE = (A+B)→C)
    """
    families:    List[str]
    ratios:      List[float]     # len == len(families), sum ≈ 1.0
    strategy:    str             # INTERLEAVED / STAGED / WEIGHTED
    blend_order: str = "SEQUENTIAL"   # SEQUENTIAL atau TREE
    synergy_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Normalize ratios agar sum = 1.0
        total = sum(self.ratios)
        if total > 0:
            self.ratios = [round(r / total, 4) for r in self.ratios]
        # Pastikan jumlah sama
        while len(self.ratios) < len(self.families):
            self.ratios.append(0.0)
        self.ratios = self.ratios[:len(self.families)]

    # ── Legacy 2-family compatibility ─────────────────────────────────────────
    @property
    def family_a(self) -> str:
        return self.families[0] if self.families else ""

    @property
    def family_b(self) -> str:
        return self.families[1] if len(self.families) > 1 else ""

    @property
    def ratio_a(self) -> float:
        return self.ratios[0] if self.ratios else 0.5

    @property
    def ratio_b(self) -> float:
        return self.ratios[1] if len(self.ratios) > 1 else round(1.0 - self.ratio_a, 4)

    @property
    def n_families(self) -> int:
        return len(self.families)

    @property
    def is_multi(self) -> bool:
        return self.n_families > 2

    @property
    def combo_key(self) -> Tuple[str, ...]:
        """Sorted tuple dari semua family untuk lookup."""
        return tuple(sorted(set(self.families)))

    @property
    def pair_keys(self) -> List[Tuple[str, str]]:
        """Semua pasangan 2-family dari N families untuk synergy lookup."""
        pairs = []
        fams = list(dict.fromkeys(self.families))   # deduplicate, preserve order
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                key = (min(fams[i], fams[j]), max(fams[i], fams[j]))
                pairs.append(key)
        return pairs

    @property
    def synergy_mult(self) -> float:
        """Synergy rata-rata semua pasangan (penalti untuk N-way vs 2-way)."""
        pairs = self.pair_keys
        if not pairs:
            return 0.60
        mults = [_SYNERGY_NORMALIZED.get(p, {}).get("synergy", 0.55) for p in pairs]
        base = sum(mults) / len(mults)
        # Penalti kecil untuk setiap family tambahan di atas 2
        penalty = max(0.0, (self.n_families - 2) * 0.05)
        return round(max(0.40, base - penalty), 3)

    @property
    def compatibility(self) -> str:
        """Compatibility level paling rendah dari semua pasangan."""
        pairs = self.pair_keys
        if not pairs:
            return "UNKNOWN"
        compat_rank = {"STRONGLY_VALID": 3, "COMPATIBLE": 2, "MARGINAL": 1, "UNKNOWN": 0}
        rank_name   = {3: "STRONGLY_VALID", 2: "COMPATIBLE", 1: "MARGINAL", 0: "UNKNOWN"}
        min_rank = min(
            compat_rank.get(_SYNERGY_NORMALIZED.get(p, {}).get("compat", "UNKNOWN"), 0)
            for p in pairs
        )
        return rank_name.get(min_rank, "UNKNOWN")

    @property
    def is_valid(self) -> bool:
        if len(set(self.families)) < 2:
            return False
        # Semua pasangan harus ada di synergy DB
        return all(p in _SYNERGY_NORMALIZED for p in self.pair_keys)

    @property
    def label(self) -> str:
        parts = []
        for fam, ratio in zip(self.families, self.ratios):
            short = fam.replace("-", "").replace("Optimizer", "CoT")[:5]
            parts.append(f"{short}{int(ratio*100)}")
        return "+".join(parts) + f"[{self.strategy[:3]}]"

    @classmethod
    def from_two(
        cls,
        family_a: str,
        family_b: str,
        ratio_a:  float,
        strategy: str,
        synergy_info: Dict = None,
    ) -> "CombinationSpec":
        """Factory untuk membuat CombinationSpec 2-family (backward compat)."""
        return cls(
            families     = [family_a, family_b],
            ratios       = [ratio_a, 1.0 - ratio_a],
            strategy     = strategy,
            synergy_info = synergy_info or {},
        )


@dataclass
class CombinationNASResult:
    """Hasil evaluasi NAS combination-specific."""
    arch_id:       str = ""
    spec:          Optional[CombinationSpec] = None

    # Sub-scores [0, 1]
    coherence_score:    float = 0.0   # C1
    balance_score:      float = 0.0   # C2
    synergy_score:      float = 0.0   # C3
    hw_compat_score:    float = 0.0   # C4
    train_synergy_score: float = 0.0  # C5

    # Aggregated
    combination_score:  float = 0.0   # [0, 1] weighted

    # Dimensional pts
    pts_c1: float = 0.0
    pts_c2: float = 0.0
    pts_c3: float = 0.0
    pts_c4: float = 0.0
    pts_c5: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)

    @property
    def total_pts(self) -> float:
        return self.pts_c1 + self.pts_c2 + self.pts_c3 + self.pts_c4 + self.pts_c5

    @property
    def grade(self) -> str:
        """
        Grade berdasarkan combination_score yang sudah di-calibrate.

        Skala baru (setelah redesign scoring):
          S ★★★  : ≥ 0.80  — Exceptional combo, semua C1-C5 sangat baik
          A+ ★★  : ≥ 0.70  — Very Good, hampir sempurna
          A  ★   : ≥ 0.60  — Good, layak dipakai produksi
          B      : ≥ 0.48  — Acceptable, ada ruang improvement
          C      : ≥ 0.35  — Marginal, bisa diandalkan tapi tidak optimal
          D      : ≥ 0.22  — Poor, butuh banyak tuning
          F  ✗   : < 0.22  — Kombinasi tidak layak / bermasalah fundamental
        """
        s = self.combination_score
        if s >= 0.80: return "S ★★★  Exceptional Combination"
        if s >= 0.70: return "A+ ★★  Very Good"
        if s >= 0.60: return "A  ★   Good"
        if s >= 0.48: return "B      Acceptable"
        if s >= 0.35: return "C      Marginal"
        if s >= 0.22: return "D      Poor"
        return              "F  ✗   Combination Not Viable"


@dataclass
class CombinationAdaptiveLog:
    """Log satu siklus combination NAS refinement."""
    arch_id:        str = ""
    arch_name:      str = ""
    spec_label:     str = ""

    # Scores
    combo_score_start:  float = 0.0
    combo_score_end:    float = 0.0
    quality_start:      float = 0.0
    quality_end:        float = 0.0
    hw_score_start:     float = 0.0
    hw_score_end:       float = 0.0
    train_score_start:  float = 0.0
    train_score_end:    float = 0.0
    combined_start:     float = 0.0
    combined_end:       float = 0.0

    # RL stats
    perturbation_tries:     int = 0
    perturbations_accepted: int = 0
    ratio_adjustments:      int = 0
    strategy_switches:      int = 0

    improvement_events: List[str] = field(default_factory=list)
    warnings:           List[str] = field(default_factory=list)
    status:             str = ""

    base_log: Optional[RefinementLog] = None

    @property
    def combo_delta(self) -> float:
        return round(self.combo_score_end - self.combo_score_start, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION RULE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CombinationRuleEngine:
    """
    Validasi apakah kombinasi N family diperbolehkan.
    Untuk N>2: semua pasangan 2-family harus valid di synergy database.
    """

    @staticmethod
    def get_synergy(family_a: str, family_b: str) -> Optional[Dict]:
        if family_a == family_b:
            return None
        key = (min(family_a, family_b), max(family_a, family_b))
        return _SYNERGY_NORMALIZED.get(key)

    @staticmethod
    def is_valid_pair(family_a: str, family_b: str) -> bool:
        if family_a == family_b:
            return False
        key = (min(family_a, family_b), max(family_a, family_b))
        return key in _SYNERGY_NORMALIZED

    @staticmethod
    def is_valid_combination(spec: "CombinationSpec") -> bool:
        """Cek apakah semua pasangan dalam spec valid."""
        return spec.is_valid

    @staticmethod
    def get_all_valid_combos() -> List[Tuple[str, str, str, float]]:
        """Return semua kombinasi 2-family valid: (family_a, family_b, compat, synergy)."""
        result = []
        for (a, b), info in sorted(_SYNERGY_NORMALIZED.items(),
                                    key=lambda x: -x[1]["synergy"]):
            result.append((a, b, info["compat"], info["synergy"]))
        return result

    @staticmethod
    def get_valid_for_families(
        selected: List[str],
        min_families: int = 2,
        max_families: int = 3,
    ) -> List[Tuple[Tuple[str, ...], str, float]]:
        """
        Dapatkan semua kombinasi valid dari daftar `selected` families.
        Support 2-way dan 3-way combinations.

        Returns: List of (family_tuple, compat_label, avg_synergy)
          sorted by avg_synergy descending.
        """
        import itertools
        selected_clean = list(dict.fromkeys(selected))   # deduplicate, preserve order
        results = []

        for n in range(min_families, min(max_families + 1, len(selected_clean) + 1)):
            for combo in itertools.combinations(selected_clean, n):
                # Cek semua pasangan
                pairs = [(min(a, b), max(a, b))
                         for i, a in enumerate(combo)
                         for b in combo[i+1:]]
                if not all(p in _SYNERGY_NORMALIZED for p in pairs):
                    continue

                # Hitung avg synergy
                synergies  = [_SYNERGY_NORMALIZED[p]["synergy"] for p in pairs]
                avg_syn    = sum(synergies) / len(synergies)
                # Penalti untuk 3-way
                if n == 3:
                    avg_syn = round(avg_syn - 0.05, 3)

                # Compat: paling rendah dari semua pasangan
                compat_rank = {"STRONGLY_VALID": 3, "COMPATIBLE": 2,
                                "MARGINAL": 1, "UNKNOWN": 0}
                rank_name   = {3: "STRONGLY_VALID", 2: "COMPATIBLE",
                                1: "MARGINAL", 0: "UNKNOWN"}
                min_rank    = min(
                    compat_rank.get(_SYNERGY_NORMALIZED[p]["compat"], 0)
                    for p in pairs
                )
                compat_label = rank_name.get(min_rank, "UNKNOWN")
                results.append((combo, compat_label, avg_syn))

        results.sort(key=lambda x: -x[2])
        return results

    @staticmethod
    def recommend_strategy(families: List[str]) -> Tuple[str, List[float]]:
        """
        Return (best_strategy, recommended_ratios) untuk N families.
        Ratios sum ke 1.0, uniform jika tidak ada info spesifik.
        """
        if len(families) == 2:
            fa, fb = families[0], families[1]
            info = _SYNERGY_NORMALIZED.get((min(fa, fb), max(fa, fb)), {})
            strat    = info.get("best_strategy", BLEND_WEIGHTED)
            ratio_a  = info.get("recommended_ratio", 0.50)
            return strat, [ratio_a, round(1.0 - ratio_a, 2)]
        else:
            # N>2: default ke WEIGHTED dengan ratio uniform
            n       = len(families)
            uniform = round(1.0 / n, 4)
            ratios  = [uniform] * (n - 1) + [round(1.0 - uniform * (n-1), 4)]
            return BLEND_WEIGHTED, ratios

    @staticmethod
    def validate_spec(spec: "CombinationSpec", gpu: GPUSpec) -> List[str]:
        warnings = []

        if not spec.is_valid:
            missing_pairs = [
                f"{a}+{b}" for a, b in spec.pair_keys
                if (a, b) not in _SYNERGY_NORMALIZED
            ]
            if missing_pairs:
                warnings.append(
                    f"INVALID pairs: {', '.join(missing_pairs)}"
                )
            return warnings

        if spec.compatibility == "MARGINAL":
            warnings.append("⚠ MARGINAL combination — perlu careful tuning")

        # N-way warning
        if spec.n_families >= 3:
            warnings.append(
                f"ℹ {spec.n_families}-way combination: lebih kompleks, "
                f"butuh VRAM lebih besar dan training lebih hati-hati"
            )

        # MoE check
        if any("MoE" in f for f in spec.families):
            if gpu.vram_gb < 12:
                warnings.append(f"⚠ MoE family butuh ≥12GB VRAM, GPU={gpu.vram_gb:.0f}GB")

        # Long-Horizon check
        if any("Long" in f for f in spec.families):
            if gpu.vram_gb < 16:
                warnings.append("⚠ Long-Horizon kombinasi butuh ≥16GB VRAM")

        return warnings


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION BLENDER
# ══════════════════════════════════════════════════════════════════════════════

class CombinationBlender:
    """
    Menghasilkan ArchConfig hybrid dari dua family menggunakan salah satu
    dari 3 strategi blend.
    """

    def __init__(self, gpu: GPUSpec, rng_seed: Optional[int] = None):
        self.gpu  = gpu
        self.rng  = random.Random(rng_seed)
        self._gen = ArchitectureGenerator(gpu, rng_seed=rng_seed)

    def blend(self, spec: CombinationSpec) -> ArchConfig:
        """
        Generate ArchConfig hybrid dari CombinationSpec.

        Untuk 2-family: gunakan strategi langsung.
        Untuk N-family (N>2): chain sequential — blend A+B dulu,
          hasilnya blend lagi dengan C, dan seterusnya.
          Rasio disesuaikan proporsional di setiap langkah.

        Contoh 3-way (A=40%, B=35%, C=25%):
          Step 1: blend(A, B) dengan ratio_a = 40/(40+35) = 0.533
          Step 2: blend(A+B, C) dengan ratio_a = 75/(75+25) = 0.75
        """
        families = spec.families
        ratios   = spec.ratios

        if len(families) < 2:
            return self._gen.generate_one(families[0] if families else "Balanced-Pro")

        if len(families) == 2:
            cfg_a = self._gen.generate_one(families[0])
            cfg_b = self._gen.generate_one(families[1])
            return self._blend_two(cfg_a, cfg_b, ratios[0], spec.strategy, spec)

        # N > 2: chain blending
        # Mulai dari dua family pertama
        running_cfg   = self._gen.generate_one(families[0])
        running_ratio = ratios[0]   # accumulated weight so far

        for i in range(1, len(families)):
            next_cfg   = self._gen.generate_one(families[i])
            next_ratio = ratios[i]

            # Ratio A dalam pasangan ini = accumulated / (accumulated + next)
            total = running_ratio + next_ratio
            pair_ratio_a = running_ratio / max(1e-6, total)

            # Buat spec sementara untuk pasangan ini
            pair_spec = CombinationSpec.from_two(
                family_a    = running_cfg.arch_family,
                family_b    = families[i],
                ratio_a     = pair_ratio_a,
                strategy    = spec.strategy,
                synergy_info= {},
            )

            running_cfg   = self._blend_two(running_cfg, next_cfg, pair_ratio_a, spec.strategy, pair_spec)
            running_ratio = running_ratio + next_ratio

        # Update arch identity untuk N-way
        n    = len(families)
        fams = "+".join(f.split("-")[0] for f in families)
        rats = "/".join(f"{int(r*100)}" for r in ratios)
        running_cfg.arch_family = "+".join(families)
        running_cfg.arch_name   = (f"{fams} | {spec.strategy}-{rats} | "
                                    f"L{running_cfg.num_layers}×D{running_cfg.hidden_dim}")

        return self._finalize(running_cfg)

    def _blend_two(
        self,
        cfg_a:    ArchConfig,
        cfg_b:    ArchConfig,
        ratio_a:  float,
        strategy: str,
        spec:     CombinationSpec,
    ) -> ArchConfig:
        """Internal: blend dua ArchConfig dengan satu strategi."""
        if strategy == BLEND_INTERLEAVED:
            return self._blend_interleaved(cfg_a, cfg_b, spec, ratio_a)
        elif strategy == BLEND_STAGED:
            return self._blend_staged(cfg_a, cfg_b, spec, ratio_a)
        else:
            return self._blend_weighted(cfg_a, cfg_b, spec, ratio_a)

    # ── Interleaved: alternasi per-layer ──────────────────────────────────────
    def _blend_interleaved(
        self, cfg_a: ArchConfig, cfg_b: ArchConfig,
        spec: CombinationSpec, ratio_a: float = None,
    ) -> ArchConfig:
        """Alternasi layers antara family A dan B."""
        if ratio_a is None:
            ratio_a = spec.ratio_a
        ratio_b = round(1.0 - ratio_a, 4)

        hybrid = copy.deepcopy(cfg_a)

        # Total layers: weighted average, rounded ke genap untuk interleave
        total_L = int(round(ratio_a * cfg_a.num_layers + ratio_b * cfg_b.num_layers))
        if total_L % 2 != 0:
            total_L += 1
        total_L = max(4, total_L)
        hybrid.num_layers = total_L

        # Hidden dim: harus sama untuk interleaving → pilih berdasarkan dominasi NYATA
        # Fix: gunakan threshold 0.52 agar 50/50 tidak selalu bias ke family A.
        # Jika benar-benar seimbang (< 52% vs 48%), pilih yang hidden_dim lebih
        # optimal (aligned 64 dan num_heads lebih seimbang).
        _DOM_THRESH = 0.52
        if ratio_a >= _DOM_THRESH:
            hybrid.hidden_dim   = cfg_a.hidden_dim
            hybrid.num_heads    = cfg_a.num_heads
            hybrid.head_dim     = cfg_a.head_dim
            hybrid.num_kv_heads = cfg_a.num_kv_heads
        elif ratio_b >= _DOM_THRESH:
            hybrid.hidden_dim   = cfg_b.hidden_dim
            hybrid.num_heads    = cfg_b.num_heads
            hybrid.head_dim     = cfg_b.head_dim
            hybrid.num_kv_heads = cfg_b.num_kv_heads
        else:
            # Benar-benar seimbang: pilih yang alignment lebih baik
            score_a = int(cfg_a.hidden_dim % 64 == 0) + int(cfg_a.num_heads % 4 == 0)
            score_b = int(cfg_b.hidden_dim % 64 == 0) + int(cfg_b.num_heads % 4 == 0)
            if score_a >= score_b:
                hybrid.hidden_dim   = cfg_a.hidden_dim
                hybrid.num_heads    = cfg_a.num_heads
                hybrid.head_dim     = cfg_a.head_dim
                hybrid.num_kv_heads = cfg_a.num_kv_heads
            else:
                hybrid.hidden_dim   = cfg_b.hidden_dim
                hybrid.num_heads    = cfg_b.num_heads
                hybrid.head_dim     = cfg_b.head_dim
                hybrid.num_kv_heads = cfg_b.num_kv_heads

        # FFN: weighted average multiplier
        hybrid.ffn_multiplier = round(
            ratio_a * cfg_a.ffn_multiplier + ratio_b * cfg_b.ffn_multiplier, 3
        )

        # Attention type: ambil dari dominant family tapi preferensikan GQA/MQA
        attn_preference_order = [
            AttentionType.GQA, AttentionType.ROPE, AttentionType.MHA,
            AttentionType.MQA, AttentionType.SLIDE, AttentionType.HYBRID,
        ]
        for attn in attn_preference_order:
            if cfg_a.attn_type == attn or cfg_b.attn_type == attn:
                hybrid.attn_type = attn
                break
        else:
            hybrid.attn_type = cfg_a.attn_type

        # FFN type: preferensikan GeGLU/GATED untuk stabilitas
        if cfg_a.ffn_type in (FFNType.GEGLU, FFNType.GATED):
            hybrid.ffn_type = cfg_a.ffn_type
        elif cfg_b.ffn_type in (FFNType.GEGLU, FFNType.GATED):
            hybrid.ffn_type = cfg_b.ffn_type
        else:
            hybrid.ffn_type = cfg_a.ffn_type if spec.ratio_a >= 0.5 else cfg_b.ffn_type

        # MoE: aktif jika salah satu family adalah MoE
        if cfg_a.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            hybrid.ffn_type = cfg_a.ffn_type
            hybrid.num_experts = cfg_a.num_experts
            hybrid.top_k_experts = cfg_a.top_k_experts
        elif cfg_b.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            hybrid.ffn_type = cfg_b.ffn_type
            hybrid.num_experts = cfg_b.num_experts
            hybrid.top_k_experts = cfg_b.top_k_experts

        # Seq len: ambil max tapi cap untuk VRAM
        raw_seq = max(cfg_a.seq_len, cfg_b.seq_len)
        hybrid.seq_len = min(raw_seq, self._max_seq_for_vram(hybrid))

        # Optimizer: pilih yang lebih stabil
        hybrid.optimizer_type = self._pick_optimizer(cfg_a, cfg_b, spec)

        # Flags: OR dari kedua family (aktifkan jika salah satu aktif)
        hybrid.use_flash_attn               = cfg_a.use_flash_attn or cfg_b.use_flash_attn
        hybrid.use_mixed_precision          = cfg_a.use_mixed_precision or cfg_b.use_mixed_precision
        hybrid.tie_embeddings               = cfg_a.tie_embeddings and cfg_b.tie_embeddings
        hybrid.use_gradient_checkpointing   = cfg_a.use_gradient_checkpointing or cfg_b.use_gradient_checkpointing
        hybrid.norm_type                    = NormType.RMSNORM   # selalu RMSNorm untuk hybrid
        hybrid.dropout                      = 0.0

        # Arch identity
        hybrid.arch_family = f"{spec.family_a}+{spec.family_b}"
        hybrid.arch_name   = (f"{spec.family_a}+{spec.family_b} | "
                               f"INTERLEAVED-{int(spec.ratio_a*100)}/{int(spec.ratio_b*100)} | "
                               f"L{hybrid.num_layers}×D{hybrid.hidden_dim}")

        return self._finalize(hybrid)

    # ── Staged: N layers A kemudian M layers B ────────────────────────────────
    def _blend_staged(
        self, cfg_a: ArchConfig, cfg_b: ArchConfig,
        spec: CombinationSpec, ratio_a: float = None,
    ) -> ArchConfig:
        """Stage-1 layers dari family A, kemudian sisanya dari family B."""
        if ratio_a is None:
            ratio_a = spec.ratio_a
        ratio_b = round(1.0 - ratio_a, 4)

        hybrid = copy.deepcopy(cfg_a)

        total_L = int(round(ratio_a * cfg_a.num_layers + ratio_b * cfg_b.num_layers))
        total_L = max(4, total_L)
        hybrid.num_layers = total_L

        layers_a = max(1, round(total_L * ratio_a))
        layers_b = total_L - layers_a

        # Fix: sama seperti interleaved, gunakan threshold 0.52 agar tidak bias ke A.
        _DOM_THRESH = 0.52
        if ratio_a >= _DOM_THRESH:
            hybrid.hidden_dim   = cfg_a.hidden_dim
            hybrid.num_heads    = cfg_a.num_heads
            hybrid.head_dim     = cfg_a.head_dim
            hybrid.num_kv_heads = cfg_a.num_kv_heads
        elif ratio_b >= _DOM_THRESH:
            hybrid.hidden_dim   = cfg_b.hidden_dim
            hybrid.num_heads    = cfg_b.num_heads
            hybrid.head_dim     = cfg_b.head_dim
            hybrid.num_kv_heads = cfg_b.num_kv_heads
        else:
            score_a = int(cfg_a.hidden_dim % 64 == 0) + int(cfg_a.num_heads % 4 == 0)
            score_b = int(cfg_b.hidden_dim % 64 == 0) + int(cfg_b.num_heads % 4 == 0)
            if score_a >= score_b:
                hybrid.hidden_dim   = cfg_a.hidden_dim
                hybrid.num_heads    = cfg_a.num_heads
                hybrid.head_dim     = cfg_a.head_dim
                hybrid.num_kv_heads = cfg_a.num_kv_heads
            else:
                hybrid.hidden_dim   = cfg_b.hidden_dim
                hybrid.num_heads    = cfg_b.num_heads
                hybrid.head_dim     = cfg_b.head_dim
                hybrid.num_kv_heads = cfg_b.num_kv_heads

        if layers_a >= layers_b:
            hybrid.ffn_type       = cfg_a.ffn_type
            hybrid.ffn_multiplier = cfg_a.ffn_multiplier
            hybrid.num_experts    = cfg_a.num_experts
            hybrid.top_k_experts  = cfg_a.top_k_experts
        else:
            hybrid.ffn_type       = cfg_b.ffn_type
            hybrid.ffn_multiplier = cfg_b.ffn_multiplier
            hybrid.num_experts    = cfg_b.num_experts
            hybrid.top_k_experts  = cfg_b.top_k_experts

        # Long-Horizon special handling
        fam_a = cfg_a.arch_family if hasattr(cfg_a, 'arch_family') else spec.family_a
        fam_b = spec.family_b
        if any("Long" in f for f in spec.families):
            lh_cfg = cfg_a if any("Long" in f for f in [fam_a]) else cfg_b
            hybrid.seq_len            = lh_cfg.seq_len
            hybrid.window_size        = lh_cfg.window_size
            hybrid.global_attn_layers = max(2, layers_b // 3)
            hybrid.attn_type          = AttentionType.HYBRID
        else:
            hybrid.seq_len   = max(cfg_a.seq_len, cfg_b.seq_len)
            hybrid.attn_type = cfg_a.attn_type if layers_a >= layers_b else cfg_b.attn_type

        hybrid.seq_len                    = min(hybrid.seq_len, self._max_seq_for_vram(hybrid))
        hybrid.optimizer_type             = self._pick_optimizer(cfg_a, cfg_b, spec)
        hybrid.use_flash_attn             = True
        hybrid.use_mixed_precision        = True
        hybrid.tie_embeddings             = cfg_a.tie_embeddings and cfg_b.tie_embeddings
        hybrid.use_gradient_checkpointing = (cfg_a.use_gradient_checkpointing or
                                              cfg_b.use_gradient_checkpointing)
        hybrid.norm_type                  = NormType.RMSNORM
        hybrid.dropout                    = 0.0

        fams = "+".join(spec.families[:2])
        hybrid.arch_family = fams
        hybrid.arch_name   = (f"{fams} | STAGED-{int(ratio_a*100)}/{int(ratio_b*100)} | "
                               f"L{hybrid.num_layers}×D{hybrid.hidden_dim}")

        return self._finalize(hybrid)

    # ── Weighted: interpolasi semua hyperparameter ────────────────────────────
    def _blend_weighted(
        self, cfg_a: ArchConfig, cfg_b: ArchConfig,
        spec: CombinationSpec, ratio_a: float = None,
    ) -> ArchConfig:
        """Interpolasi proporsional semua hyperparameter antara family A dan B."""
        if ratio_a is None:
            ratio_a = spec.ratio_a
        ra = ratio_a
        rb = round(1.0 - ra, 4)

        hybrid = copy.deepcopy(cfg_a)

        # Blend scalar params
        raw_layers = ra * cfg_a.num_layers + rb * cfg_b.num_layers
        hybrid.num_layers = max(2, round(raw_layers))

        # Hidden dim: interpolasi lalu snap ke head_dim boundary
        raw_hidden    = ra * cfg_a.hidden_dim + rb * cfg_b.hidden_dim
        base_head_dim = cfg_a.head_dim if ra >= rb else cfg_b.head_dim
        hybrid.head_dim   = base_head_dim
        hybrid.num_heads  = max(1, round(raw_hidden / base_head_dim))
        hybrid.hidden_dim = hybrid.num_heads * base_head_dim

        # KV heads
        raw_kv    = ra * cfg_a.num_kv_heads + rb * cfg_b.num_kv_heads
        kv_target = max(1, round(raw_kv))
        valid_kv  = [h for h in range(1, hybrid.num_heads + 1)
                     if hybrid.num_heads % h == 0]
        if valid_kv:
            hybrid.num_kv_heads = min(valid_kv, key=lambda h: abs(h - kv_target))

        # FFN
        hybrid.ffn_multiplier = round(ra * cfg_a.ffn_multiplier + rb * cfg_b.ffn_multiplier, 3)
        if any(c.ffn_type == FFNType.GEGLU for c in [cfg_a, cfg_b]):
            hybrid.ffn_type = FFNType.GEGLU
        elif any(c.ffn_type == FFNType.GATED for c in [cfg_a, cfg_b]):
            hybrid.ffn_type = FFNType.GATED
        else:
            hybrid.ffn_type = cfg_a.ffn_type if ra >= rb else cfg_b.ffn_type

        # MoE
        moe_ratio = 0.0
        if cfg_a.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            moe_ratio = ra
        elif cfg_b.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            moe_ratio = rb
        if moe_ratio >= 0.40:
            moe_cfg = cfg_a if cfg_a.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK) else cfg_b
            hybrid.ffn_type      = moe_cfg.ffn_type
            hybrid.num_experts   = moe_cfg.num_experts
            hybrid.top_k_experts = moe_cfg.top_k_experts

        # Seq len
        raw_seq = ra * cfg_a.seq_len + rb * cfg_b.seq_len
        seq_candidates = [512, 1024, 2048, 4096, 8192]
        hybrid.seq_len = min(seq_candidates, key=lambda s: abs(s - raw_seq))
        hybrid.seq_len = min(hybrid.seq_len, self._max_seq_for_vram(hybrid))

        # Attention
        attn_priority = {
            AttentionType.GQA: 5, AttentionType.ROPE: 4, AttentionType.MHA: 3,
            AttentionType.MQA: 2, AttentionType.ALIBI: 1,
        }
        hybrid.attn_type = (cfg_a.attn_type
                            if attn_priority.get(cfg_a.attn_type, 0) >=
                               attn_priority.get(cfg_b.attn_type, 0)
                            else cfg_b.attn_type)

        hybrid.optimizer_type             = self._pick_optimizer(cfg_a, cfg_b, spec)
        hybrid.use_flash_attn             = True
        hybrid.use_mixed_precision        = True
        hybrid.tie_embeddings             = cfg_a.tie_embeddings and cfg_b.tie_embeddings
        hybrid.use_gradient_checkpointing = (cfg_a.use_gradient_checkpointing or
                                              cfg_b.use_gradient_checkpointing)
        hybrid.norm_type                  = NormType.RMSNORM
        hybrid.dropout                    = 0.0
        hybrid.batch_size                 = max(1, round(ra * cfg_a.batch_size + rb * cfg_b.batch_size))

        fams = "+".join(spec.families[:2])
        hybrid.arch_family = fams
        hybrid.arch_name   = (f"{fams} | WEIGHTED-{int(ra*100)}/{int(rb*100)} | "
                               f"L{hybrid.num_layers}×D{hybrid.hidden_dim}")

        return self._finalize(hybrid)

    # ── Helper methods ─────────────────────────────────────────────────────────

    def _max_seq_for_vram(self, cfg: ArchConfig) -> int:
        """Estimasi seq_len maksimum berdasarkan VRAM sisa."""
        # Rough estimate: KV cache ≈ 2×kv_heads×head_dim×layers×batch bytes per token
        kv_per_token = (2 * max(1, cfg.num_kv_heads) * max(64, cfg.head_dim) *
                        max(1, cfg.num_layers) * max(1, cfg.batch_size) * 2) / 1e9
        vram_budget   = self.gpu.vram_gb * VRAM_LIMIT_PCT * 0.25   # 25% for KV
        max_tokens    = int(vram_budget / max(1e-9, kv_per_token))
        for seq in reversed([512, 1024, 2048, 4096, 8192, 16384]):
            if seq <= max_tokens:
                return seq
        return 512

    def _pick_optimizer(
        self, cfg_a: ArchConfig, cfg_b: ArchConfig, spec: CombinationSpec
    ) -> OptimizerType:
        """Pilih optimizer yang paling aman untuk hybrid."""
        # Adam FP32 selalu aman untuk kombinasi
        stable_opts = {OptimizerType.ADAM_FP32, OptimizerType.ADAMW_BF16}
        if cfg_a.optimizer_type in stable_opts:
            return cfg_a.optimizer_type
        if cfg_b.optimizer_type in stable_opts:
            return cfg_b.optimizer_type
        return OptimizerType.ADAM_FP32

    def _finalize(self, cfg: ArchConfig) -> ArchConfig:
        """Recompute semua derived metrics setelah blend."""
        gen = self._gen
        gpu = self.gpu

        # Fix structural: num_heads × head_dim == hidden_dim
        if cfg.num_heads > 0 and cfg.head_dim > 0:
            cfg.hidden_dim = cfg.num_heads * cfg.head_dim

        # Fix KV heads
        valid_kv = [h for h in range(1, cfg.num_heads + 1)
                    if cfg.num_heads % h == 0]
        if cfg.num_kv_heads not in valid_kv and valid_kv:
            cfg.num_kv_heads = min(valid_kv, key=lambda h: abs(h - cfg.num_kv_heads))

        # Fix MoE
        if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            cfg.num_experts    = max(2, cfg.num_experts)
            cfg.top_k_experts  = max(1, min(cfg.top_k_experts, cfg.num_experts - 1))
        else:
            cfg.num_experts   = 1
            cfg.top_k_experts = 1

        # Recompute all metrics
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

        return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION NAS EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class CombinationNASEvaluator:
    """
    Evaluasi combination-specific score untuk ArchConfig hybrid.

    5 Dimensi (100 pts total):
      C1  Family Coherence      (25 pts) — do the families complement each other?
      C2  Blend Balance         (20 pts) — is the blend ratio and strategy valid?
      C3  Architectural Synergy (20 pts) — depth/width/FFN hybrid coherence
      C4  Hardware Compatibility (20 pts) — hw_score setelah blend
      C5  Training Synergy      (15 pts) — stability prediction untuk hybrid training
    """

    W_C1 = 0.25
    W_C2 = 0.20
    W_C3 = 0.20
    W_C4 = 0.20
    W_C5 = 0.15

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu
        self._hw_eval = HardwareNASEvaluator(gpu)

    def evaluate(
        self,
        cfg:  ArchConfig,
        spec: CombinationSpec,
    ) -> CombinationNASResult:
        result      = CombinationNASResult(arch_id=cfg.arch_id, spec=spec)
        warnings    = CombinationRuleEngine.validate_spec(spec, self.gpu)
        result.warnings = warnings

        # C1: Family Coherence (25 pts)
        c1 = self._score_coherence(spec)
        result.coherence_score = c1
        result.pts_c1          = round(c1 * 25.0, 2)

        # C2: Blend Balance (20 pts)
        c2 = self._score_balance(cfg, spec)
        result.balance_score = c2
        result.pts_c2        = round(c2 * 20.0, 2)

        # C3: Architectural Synergy (20 pts)
        c3 = self._score_arch_synergy(cfg, spec)
        result.synergy_score = c3
        result.pts_c3        = round(c3 * 20.0, 2)

        # C4: Hardware Compatibility (20 pts)
        hw_result = self._hw_eval.evaluate(cfg)
        c4 = hw_result.hardware_score
        result.hw_compat_score = c4
        result.pts_c4          = round(c4 * 20.0, 2)

        # C5: Training Synergy (15 pts)
        c5 = self._score_training_synergy(cfg, spec)
        result.train_synergy_score = c5
        result.pts_c5              = round(c5 * 15.0, 2)

        # Weighted combination
        raw = (self.W_C1 * c1 + self.W_C2 * c2 + self.W_C3 * c3 +
               self.W_C4 * c4 + self.W_C5 * c5)

        # Apply synergy multiplier dari database (penalti untuk marginal combos)
        # synergy_mult values: STRONGLY_VALID=0.92, COMPATIBLE=0.80, MARGINAL=0.62
        # (lihat _SYNERGY_DB — nilainya sudah disesuaikan di sana)
        synergy_mult = spec.synergy_mult

        # Combination difficulty penalty: arsitektur hybrid SELALU lebih sulit
        # dari pure single-type, sehingga combination_score diberi cap yang
        # mencerminkan realita ini.
        #   n=2: cap 0.88 (single-type bisa 1.0, combination max 0.88)
        #   n=3: cap 0.82
        #   n=4: cap 0.76
        n_cap = {2: 0.88, 3: 0.82, 4: 0.76}.get(spec.n_families, 0.72)

        # Nilai min(raw * synergy_mult, n_cap) memastikan skor tidak
        # pernah melampaui kemungkinan terbaik yang bisa dicapai kombinasi N-family
        result.combination_score = float(np.clip(
            min(raw * synergy_mult, n_cap),
            0.0, 1.0
        ))

        return result

    # ── C1: Family Coherence ──────────────────────────────────────────────────

    def _score_coherence(self, spec: CombinationSpec) -> float:
        """
        C1: Seberapa cocok N family secara nature.

        Redesign — nilai tidak bisa tinggi tanpa bukti nyata:
          • Base score diturunkan drastis:
              STRONGLY_VALID = 0.62  (bukan 0.95)
              COMPATIBLE     = 0.44  (bukan 0.80)
              MARGINAL       = 0.24  (bukan 0.55)
              UNKNOWN        = 0.10  (bukan 0.30)
          • Bonus untuk strategy + ratio yang tepat bisa naikkan hingga:
              STRONGLY_VALID max = 0.62+0.14+0.09 = 0.85
              COMPATIBLE max     = 0.44+0.14+0.09 = 0.67
              MARGINAL max       = 0.24+0.14+0.09 = 0.47
          • Max teoretis ≤ 0.85 dari C1 — tidak pernah 1.0.
            Skor sempurna butuh C2-C5 semua juga bagus.
          • Penalti ratio_err sekarang lebih agresif.
          • N-way penalti lebih besar (-0.06 per extra family).
        """
        pairs = spec.pair_keys
        if not pairs:
            return 0.0

        # Base score yang realistis — kombinasi SELALU lebih sulit dari single-type
        base_scores = {
            "STRONGLY_VALID": 0.62,
            "COMPATIBLE":     0.44,
            "MARGINAL":       0.24,
            "UNKNOWN":        0.10,
        }

        total_score = 0.0
        for pair_key in pairs:
            info   = _SYNERGY_NORMALIZED.get(pair_key, {})
            compat = info.get("compat", "UNKNOWN")
            ps     = base_scores.get(compat, 0.15)

            # Strategy bonus: lebih besar agar ada reward yang jelas
            if spec.n_families == 2:
                recommended = info.get("best_strategy", BLEND_WEIGHTED)
                if spec.strategy == recommended:
                    ps = min(0.85, ps + 0.14)   # max ditambah hanya sampai 0.85
                elif spec.strategy != BLEND_WEIGHTED:
                    ps -= 0.06   # penalti untuk strategy yang salah

                # Ratio penalty/bonus yang lebih ketat
                rec_ratio = info.get("recommended_ratio", 0.50)
                ratio_err = abs(spec.ratio_a - rec_ratio)
                if ratio_err <= 0.05:
                    ps = min(0.85, ps + 0.09)   # tepat sasaran
                elif ratio_err <= 0.10:
                    ps = min(0.85, ps + 0.04)   # mendekati
                elif ratio_err <= 0.20:
                    ps -= ratio_err * 0.25       # mulai kena penalti
                else:
                    ps -= ratio_err * 0.40       # penalti besar jika jauh

            total_score += ps

        avg = total_score / len(pairs)

        # N-way penalti lebih besar — setiap family tambahan lebih sulit diintegrasikan
        if spec.n_families >= 3:
            extra = spec.n_families - 2
            avg = max(0.0, avg - 0.06 * extra)

        return float(np.clip(avg, 0.0, 1.0))

    # ── C2: Blend Balance ─────────────────────────────────────────────────────

    def _score_balance(self, cfg: ArchConfig, spec: CombinationSpec) -> float:
        """C2: Apakah blend balance valid dari sisi parameter dan VRAM.

        Redesign:
          - VRAM target range diperketat: 50-72% (bukan 45-78%)
          - Gunakan geometric mean sub-skor agar tidak ada sub-score yang
            bisa "menebus" sub-score lain yang rendah.
          - Hard cap C2 di 0.88 — blend balance tidak pernah sempurna 1.0
            untuk arsitektur hybrid.
          - Ratio balance lebih agresif: max_ratio > 0.75 sudah mendapat penalti.
          - Param count: range lebih ketat (70–600M untuk score 1.0).
        """
        if not cfg.fits_gpu:
            return 0.08   # OOM penalty (lebih rendah dari sebelumnya)

        # VRAM balance: target 50–72% (tighter dari sebelumnya 45–78%)
        vram_pct = cfg.vram_usage_pct
        if 50 <= vram_pct <= 72:
            vram_score = 1.0
        elif 42 <= vram_pct < 50:
            vram_score = 0.55 + (vram_pct - 42) / 8 * 0.45
        elif 72 < vram_pct <= 80:
            vram_score = max(0.30, 1.0 - (vram_pct - 72) / 8 * 0.70)
        elif 80 < vram_pct <= 88:
            vram_score = max(0.10, 0.30 - (vram_pct - 80) / 8 * 0.20)
        else:
            vram_score = max(0.05, 1.0 - abs(vram_pct - 61) / 61)

        # Ratio balance: lebih agresif
        max_ratio = max(spec.ratios) if spec.ratios else 0.5
        if max_ratio > 0.80:
            ratio_balance = 0.25   # sangat dominan satu family
        elif max_ratio > 0.72:
            ratio_balance = 0.50
        elif max_ratio > 0.65:
            ratio_balance = 0.75
        else:
            ratio_balance = 1.0

        # N-way entropy check
        if spec.n_families >= 3:
            ent = -sum(r * math.log(r + 1e-9) for r in spec.ratios if r > 0)
            max_ent = math.log(spec.n_families)
            entropy_ratio = ent / max(1e-9, max_ent)
            if entropy_ratio < 0.55:
                ratio_balance *= max(0.35, entropy_ratio / 0.55)
            elif entropy_ratio >= 0.85:
                ratio_balance = min(1.0, ratio_balance + 0.04)

        # Parameter count: range lebih ketat
        params_m = cfg.param_count / 1e6
        if 70 <= params_m <= 600:
            param_score = 1.0
        elif 40 <= params_m < 70:
            param_score = 0.50 + (params_m - 40) / 30 * 0.50
        elif 600 < params_m <= 1200:
            param_score = max(0.40, 1.0 - (params_m - 600) / 600 * 0.60)
        elif params_m < 40:
            param_score = max(0.15, params_m / 40 * 0.50)
        else:
            param_score = max(0.20, 1.0 - (params_m - 1200) / 2000)

        # Geometric mean: semua harus bagus, tidak bisa satu kompensasi yang lain
        geo = (vram_score * ratio_balance * param_score) ** (1.0 / 3.0)

        # Hard cap: blend balance tidak pernah sempurna
        return float(np.clip(min(geo, 0.88), 0.0, 1.0))

    # ── C3: Architectural Synergy ─────────────────────────────────────────────

    def _score_arch_synergy(self, cfg: ArchConfig, spec: CombinationSpec) -> float:
        """
        C3: Apakah hybrid arsitektur secara struktural masuk akal.

        Redesign — model berbasis BUKTI (bukan penalti dari 1.0):
          • Mulai dari 0, bangun skor dari evidence positif
          • Gunakan geometric mean sub-skor agar tidak ada komponen yang bisa
            "gratis" — semua harus bagus untuk skor tinggi
          • Bonus INTERLEAVED dihapus — interleaved bukan keunggulan otomatis
          • Penalti MoE expert terlalu kecil diperketat

        Sub-komponen (semua harus bagus):
          dw_score  : depth/width ratio dalam range reasonable
          ffn_score : FFN multiplier range
          head_score: head config coherence
          moe_score : MoE config (1.0 jika tidak pakai MoE)
          compat_bonus: attn type sesuai strategy
        """
        # ── dw_ratio: layers / sqrt(hidden_dim) ──────────────────────────────
        dw_ratio = cfg.num_layers / max(1.0, math.sqrt(max(1, cfg.hidden_dim)))
        if 0.12 <= dw_ratio <= 0.55:
            dw_score = 1.0
        elif 0.08 <= dw_ratio < 0.12:
            dw_score = (dw_ratio - 0.08) / 0.04 * 0.70 + 0.30
        elif 0.55 < dw_ratio <= 0.80:
            dw_score = max(0.30, 1.0 - (dw_ratio - 0.55) / 0.25 * 0.70)
        else:
            # Terlalu extreme di kedua ujung
            dw_score = max(0.10, 0.30 - abs(dw_ratio - 0.33) * 0.5)

        # ── FFN multiplier ────────────────────────────────────────────────────
        ffn = cfg.ffn_multiplier
        if 2.67 <= ffn <= 4.5:
            ffn_score = 1.0   # sweet spot
        elif 2.0 <= ffn < 2.67:
            ffn_score = 0.60 + (ffn - 2.0) / 0.67 * 0.40
        elif 4.5 < ffn <= 6.0:
            ffn_score = max(0.40, 1.0 - (ffn - 4.5) / 1.5 * 0.60)
        else:
            ffn_score = max(0.15, 0.40 - abs(ffn - 3.5) * 0.10)

        # ── Head coherence ────────────────────────────────────────────────────
        # head_dim harus kelipatan 32, num_heads harus kelipatan 4
        head_ok  = (cfg.head_dim % 32 == 0) and (cfg.num_heads % 4 == 0)
        # Untuk hybrid: GQA atau MHA lebih stabil daripada pure MQA
        attn_str = cfg.attn_type.value if hasattr(cfg.attn_type, "value") else str(cfg.attn_type)
        attn_ok  = any(x in attn_str.upper() for x in ("GQA", "MHA", "HYBRID"))
        head_score = 1.0 if (head_ok and attn_ok) else (0.70 if head_ok else 0.45)

        # ── Attn–Strategy compatibility ───────────────────────────────────────
        if "HYBRID" in attn_str.upper():
            if spec.strategy == BLEND_STAGED:
                compat_bonus = 0.10   # STAGED + HYBRID = good fit
            elif spec.strategy == BLEND_INTERLEAVED:
                compat_bonus = 0.05   # OK
            else:
                compat_bonus = -0.05  # HYBRID attn di WEIGHTED = kurang natural
        else:
            compat_bonus = 0.0

        # ── MoE config ────────────────────────────────────────────────────────
        if cfg.num_experts > 1:
            expert_hidden = cfg.hidden_dim / max(1, cfg.num_experts)
            if expert_hidden >= 256:
                moe_score = 1.0
            elif expert_hidden >= 128:
                moe_score = 0.65
            else:
                moe_score = 0.30   # expert terlalu kecil
            # top_k penalty
            if cfg.top_k_experts > cfg.num_experts // 2:
                moe_score *= 0.80
        else:
            moe_score = 1.0   # tidak pakai MoE = tidak ada masalah MoE

        # ── Geometric mean: semua komponen harus bagus ────────────────────────
        # Geometric mean mencegah satu sub-skor tinggi "menebus" yang lain rendah
        geo = (dw_score * ffn_score * head_score * moe_score) ** (1.0 / 4.0)

        # Terapkan compat_bonus setelah geometric mean (bukan di dalamnya)
        result = float(np.clip(geo + compat_bonus, 0.0, 1.0))

        # Hard cap: C3 tidak bisa lebih dari 0.88 untuk kombinasi hybrid
        # Perfect arch synergy hanya untuk pure single-type
        return min(result, 0.88)

    # ── C5: Training Synergy ──────────────────────────────────────────────────

    def _score_training_synergy(self, cfg: ArchConfig, spec: CombinationSpec) -> float:
        """
        C5: Prediksi stabilitas training untuk hybrid architecture.

        Redesign — mulai dari 0.40 (hybrid selalu lebih sulit ditraining),
        bangun skor dari bukti positif:
          • Optimizer yang tepat untuk hybrid: max +0.30
          • Mixed precision: +0.15
          • Strategy sesuai depth: +0.10
          • Tidak ada kondisi yang bermasalah: tidak kena penalti

        Max achievable C5 ≈ 0.40+0.30+0.15+0.10 = 0.95, dengan semua kondisi terpenuhi.
        Default tanpa kondisi apapun terpenuhi = 0.40 (sangat rendah).
        """
        # Base: hybrid architecture lebih sulit ditraining daripada pure
        base = 0.40

        # Optimizer: seberapa cocok optimizer dengan hybrid
        opt = cfg.optimizer_type
        if opt == OptimizerType.ADAMW_BF16:
            opt_score = 0.30    # best untuk hybrid (BF16 lebih stabil)
        elif opt == OptimizerType.ADAM_FP32:
            opt_score = 0.26    # good tapi lebih lambat
        elif opt == OptimizerType.ADAM_8BIT:
            opt_score = 0.16    # less stable untuk hybrid
        elif opt == OptimizerType.LION:
            opt_score = 0.10    # Lion + hybrid = unstable di banyak kasus
        else:
            opt_score = 0.18    # unknown = moderate

        # Mixed precision: sangat penting untuk hybrid training
        mp_score = 0.15 if cfg.use_mixed_precision else 0.0

        # Strategy-depth coherence: strategi harus masuk akal untuk jumlah layer
        # (deep model + INTERLEAVED = butuh careful tuning)
        strat_score = 0.0
        if spec.strategy == BLEND_WEIGHTED:
            # WEIGHTED paling stabil untuk semua depth
            strat_score = 0.08
        elif spec.strategy == BLEND_STAGED and 16 <= cfg.num_layers <= 48:
            # STAGED optimal untuk medium depth
            strat_score = 0.06
        elif spec.strategy == BLEND_INTERLEAVED and cfg.num_layers <= 32:
            # INTERLEAVED lebih aman untuk model yang tidak terlalu dalam
            strat_score = 0.04
        elif spec.strategy == BLEND_INTERLEAVED and cfg.num_layers > 40:
            # INTERLEAVED + deep = sulit, penalti
            strat_score = -0.06

        # Penalti dari kondisi buruk — tetap ada tapi lebih terukur
        penalties = 0.0

        # Deep model + LION optimizer
        if cfg.num_layers > 40 and opt == OptimizerType.LION:
            penalties += 0.12

        # Deep model + ADAM_8BIT
        if cfg.num_layers > 30 and opt == OptimizerType.ADAM_8BIT:
            penalties += 0.06

        # MoE training stability
        if cfg.num_experts > 1:
            if cfg.top_k_experts > cfg.num_experts // 2:
                penalties += 0.08   # terlalu banyak experts aktif = overhead

        # Dropout terlalu tinggi → slower convergence + instability di hybrid
        if cfg.dropout > 0.15:
            penalties += (cfg.dropout - 0.15) * 0.5
        elif cfg.dropout > 0.10:
            penalties += 0.03

        raw = base + opt_score + mp_score + strat_score - penalties
        # Hard cap: C5 tidak bisa melebihi 0.92
        return float(np.clip(raw, 0.0, 0.92))


# ══════════════════════════════════════════════════════════════════════════════
#  LRU CACHE
# ══════════════════════════════════════════════════════════════════════════════

class CombinationNASCache:
    """LRU cache untuk CombinationNASResult."""

    def __init__(self, max_size: int = 256):
        self._cache:    OrderedDict[str, CombinationNASResult] = OrderedDict()
        self._max_size: int = max_size
        self.hits:   int = 0
        self.misses: int = 0

    def _key(self, cfg: ArchConfig, spec: CombinationSpec) -> str:
        arch_fp = hashlib.md5(
            f"{cfg.hidden_dim}{cfg.num_layers}{cfg.num_heads}{cfg.ffn_multiplier}".encode()
        ).hexdigest()[:8]
        spec_fp = f"{spec.family_a}_{spec.family_b}_{spec.strategy}_{spec.ratio_a:.2f}"
        return f"{arch_fp}_{spec_fp}"

    def get(self, cfg: ArchConfig, spec: CombinationSpec) -> Optional[CombinationNASResult]:
        k = self._key(cfg, spec)
        if k in self._cache:
            self._cache.move_to_end(k)
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, cfg: ArchConfig, spec: CombinationSpec, val: CombinationNASResult) -> None:
        k = self._key(cfg, spec)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = val
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Q-LEARNER UNTUK COMBINATION
# ══════════════════════════════════════════════════════════════════════════════

class CombinationQLearner:
    """Q-learning untuk tuning blend ratio dan strategy.

    Perbaikan vs versi sebelumnya:
      1. State-key 6-dim: (combo_b, hw_b, train_b, compat, strategy, ratio_b)
         Tidak hanya combo — hw dan train juga masuk state agar Q-table
         bisa membedakan situasi yang berbeda secara holistik.
      2. Reward: _balanced_combo_reward() — bobot seimbang W≈0.33 per dimensi.
         Tidak bias ke satu dimensi. Synergy bonus + balance penalty.
      3. TD-priority replay: transisi yang lebih informatif (TD error besar)
         punya probabilitas lebih tinggi di-replay. Konvergensi lebih cepat.
      4. select_action: menerima hw_score dan train_score sebagai input
         untuk state encoding yang benar.
    """

    def __init__(
        self,
        alpha:       float = 0.15,   # diturunkan sedikit dari 0.18 untuk stabilitas
        gamma:       float = 0.88,
        epsilon:     float = 0.30,
        epsilon_min: float = 0.06,
        ucb_c:       float = 2.0,
        replay_cap:  int   = 1200,
    ):
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.epsilon_min = epsilon_min
        self.ucb_c       = ucb_c

        self._q:      Dict[str, List[float]] = defaultdict(
            lambda: [0.0] * _N_COMBO_ACTIONS
        )
        self._counts: Dict[str, List[int]] = defaultdict(
            lambda: [0] * _N_COMBO_ACTIONS
        )
        self._total_steps: int = 0

        # Replay buffer: (state_key, action_idx, reward, next_key, priority)
        self._replay: List[Tuple[str, int, float, str, float]] = []
        self._replay_cap: int = replay_cap
        self._max_priority: float = 1.0

    def _state_key(
        self,
        combo_score: float,
        hw_score:    float,
        train_score: float,
        spec:        CombinationSpec,
    ) -> str:
        """
        Encode state menjadi string key 6-dimensi.

        Dimensi:
          combo_b  : bucket combo_score   (0-5 dari _COMBO_SCORE_BUCKETS)
          hw_b     : bucket hw_score      (0-4 dari _HW_SCORE_BUCKETS)
          tr_b     : bucket train_score   (0-4 dari _TRAIN_SCORE_BUCKETS)
          compat_i : compatibility index  (0=STRONGLY_VALID … 3=UNKNOWN)
          strat_i  : strategy index       (0=WEIGHTED, 1=STAGED, 2=INTERLEAVED)
          ratio_b  : ratio_a bucket       (0-5)

        6-dim state menghindari Q-table yang terlalu sparse dan memastikan
        policy berbasis kondisi semua 3 dimensi, bukan hanya combo.
        """
        cb = _bucket_idx(combo_score, _COMBO_SCORE_BUCKETS)
        hb = _bucket_idx(hw_score,    _HW_SCORE_BUCKETS)
        tb = _bucket_idx(train_score, _TRAIN_SCORE_BUCKETS)
        compat_idx = {
            "STRONGLY_VALID": 0, "COMPATIBLE": 1, "MARGINAL": 2, "UNKNOWN": 3,
        }.get(spec.compatibility, 3)
        strat_idx  = {
            BLEND_WEIGHTED: 0, BLEND_STAGED: 1, BLEND_INTERLEAVED: 2,
        }.get(spec.strategy, 0)
        ratio_b = min(5, int(spec.ratio_a * 6))   # 0-5 bucket (finer granularity)
        return f"{cb}:{hb}:{tb}:{compat_idx}:{strat_idx}:{ratio_b}"

    def select_action(
        self,
        combo_score: float,
        hw_score:    float,
        train_score: float,
        spec:        CombinationSpec,
        action_fail: Dict[str, int],
        *,
        force_explore: bool = False,
    ) -> int:
        """
        Pilih aksi menggunakan epsilon-greedy + UCB.

        Perbaikan:
          - State 6-dim (hw + train ikut state)
          - Weighted random exploration: aksi yang sering gagal punya
            probabilitas lebih kecil
          - force_explore: jika True, paksa pilih aksi yang paling jarang dicoba
        """
        key = self._state_key(combo_score, hw_score, train_score, spec)
        q   = self._q[key]
        cnt = self._counts[key]
        total_cnt = max(1, sum(cnt))

        eps = max(self.epsilon_min, self.epsilon * (0.995 ** self._total_steps))

        # Force explore: pilih aksi paling jarang dipakai (anti-stagnasi)
        if force_explore:
            untried = [i for i, c in enumerate(cnt) if c == 0
                       and action_fail.get(COMBO_ACTIONS[i], 0) < 3]
            if untried:
                return random.choice(untried)
            # Semua sudah dicoba: pilih yang paling jarang
            return int(np.argmin(cnt))

        if random.random() < eps:
            # Weighted random: aksi yang sering gagal punya bobot lebih kecil
            weights = [
                max(0.05, 1.0 / (1 + action_fail.get(a, 0) * 0.5))
                for a in COMBO_ACTIONS
            ]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    return i
            return random.randrange(_N_COMBO_ACTIONS)
        else:
            # UCB exploration
            ucb_vals = []
            for i in range(_N_COMBO_ACTIONS):
                ucb = q[i] + self.ucb_c * math.sqrt(
                    math.log(total_cnt + 1) / (cnt[i] + 1)
                )
                fail_count = action_fail.get(COMBO_ACTIONS[i], 0)
                ucb -= fail_count * 0.20   # penalti aksi yang sering gagal
                ucb_vals.append(ucb)
            return int(np.argmax(ucb_vals))

    def update(
        self,
        combo_old:   float,
        hw_old:      float,
        train_old:   float,
        spec_old:    CombinationSpec,
        action_idx:  int,
        reward:      float,
        combo_new:   float,
        hw_new:      float,
        train_new:   float,
        spec_new:    CombinationSpec,
    ) -> float:
        """
        Q-update Bellman. Returns TD-error untuk priority update.

        Perbaikan: signature diperluas dengan hw/train untuk state encoding
        6-dim yang benar.
        """
        key_old = self._state_key(combo_old, hw_old, train_old, spec_old)
        key_new = self._state_key(combo_new, hw_new, train_new, spec_new)
        q_next  = max(self._q[key_new]) if self._q[key_new] else 0.0
        q_old   = self._q[key_old][action_idx]
        td_target = reward + self.gamma * q_next
        td_error  = td_target - q_old
        self._q[key_old][action_idx] = q_old + self.alpha * td_error
        self._counts[key_old][action_idx] += 1
        self._total_steps += 1

        # Push ke replay dengan TD-priority
        priority = abs(td_error) + 1e-5
        self._push_replay(key_old, action_idx, reward, key_new, priority)

        return td_error

    def _push_replay(
        self,
        state_key:  str,
        action_idx: int,
        reward:     float,
        next_key:   str,
        priority:   float,
    ) -> None:
        """Push ke replay buffer dengan priority. Hapus entry priority terendah jika penuh."""
        if len(self._replay) >= self._replay_cap:
            # Hapus entry dengan priority terendah
            min_idx = int(np.argmin([e[4] for e in self._replay]))
            self._replay.pop(min_idx)
        self._max_priority = max(self._max_priority, priority)
        self._replay.append((state_key, action_idx, reward, next_key, priority))

    def replay_update(self, n: int = 12) -> int:
        """
        Mini-batch replay dengan TD-priority sampling.

        Perbaikan: menggunakan prioritized sampling (bukan uniform random).
        Transisi dengan TD error besar lebih sering di-replay.
        alpha_prio = 0.6 — seberapa kuat pengaruh priority vs uniform.
        """
        if len(self._replay) < max(4, n // 2):
            return 0
        n = min(n, len(self._replay))

        # Prioritized sampling
        priorities = np.array([e[4] for e in self._replay], dtype=np.float64)
        probs      = priorities ** 0.6
        probs     /= probs.sum()

        indices = np.random.choice(len(self._replay), size=n, replace=False, p=probs)
        updated = 0
        for idx in indices:
            k_old, ai, rew, k_new, _ = self._replay[idx]
            q_next = max(self._q[k_new]) if self._q[k_new] else 0.0
            q_old  = self._q[k_old][ai]
            td     = rew + self.gamma * q_next - q_old
            self._q[k_old][ai] = q_old + self.alpha * td
            # Update priority
            new_prio = abs(td) + 1e-5
            self._replay[idx] = (k_old, ai, rew, k_new, new_prio)
            updated += 1
        return updated


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION PERTURBATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CombinationPerturbEngine:
    """Eksekusi perturbasi pada (ArchConfig, CombinationSpec)."""

    def __init__(self, gpu: GPUSpec, rng_seed: Optional[int] = None):
        self.gpu     = gpu
        self._blender = CombinationBlender(gpu, rng_seed=rng_seed)

    def apply(
        self,
        cfg:    ArchConfig,
        spec:   CombinationSpec,
        action: str,
    ) -> Tuple[Optional[ArchConfig], Optional[CombinationSpec], str]:
        """
        Apply action. Returns (new_cfg, new_spec, desc) atau (None, None, reason).
        """
        new_spec = copy.deepcopy(spec)
        gen      = self._blender._gen

        if action == "SHIFT_RATIO_A":
            if spec.ratio_a >= 0.85:
                return None, None, "SHIFT_RATIO_A: ratio_a sudah maksimum"
            new_ratio_a = round(min(0.85, spec.ratio_a + 0.10), 2)
            old_ratio_a = spec.ratio_a
            # Distribusikan sisa ratio ke semua family selain index 0
            n = len(new_spec.ratios)
            if n >= 2:
                new_spec.ratios[0] = new_ratio_a
                remainder = round(1.0 - new_ratio_a, 4)
                if n == 2:
                    new_spec.ratios[1] = remainder
                else:
                    # Proporsional ke family lain
                    old_others_sum = sum(new_spec.ratios[1:]) or 1.0
                    for i in range(1, n):
                        new_spec.ratios[i] = round(
                            new_spec.ratios[i] / old_others_sum * remainder, 4
                        )
            new_cfg = self._blender.blend(new_spec)
            desc = f"SHIFT_RATIO_A {old_ratio_a:.2f}→{new_spec.ratio_a:.2f}"

        elif action == "SHIFT_RATIO_B":
            if spec.ratio_a <= 0.15:
                return None, None, "SHIFT_RATIO_B: ratio_b sudah maksimum"
            new_ratio_a = round(max(0.15, spec.ratio_a - 0.10), 2)
            old_ratio_b = spec.ratio_b
            n = len(new_spec.ratios)
            if n >= 2:
                new_spec.ratios[0] = new_ratio_a
                remainder = round(1.0 - new_ratio_a, 4)
                if n == 2:
                    new_spec.ratios[1] = remainder
                else:
                    old_others_sum = sum(new_spec.ratios[1:]) or 1.0
                    for i in range(1, n):
                        new_spec.ratios[i] = round(
                            new_spec.ratios[i] / old_others_sum * remainder, 4
                        )
            new_cfg = self._blender.blend(new_spec)
            desc = f"SHIFT_RATIO_B {old_ratio_b:.2f}→{new_spec.ratio_b:.2f}"

        elif action == "SWITCH_INTERLEAVED":
            if spec.strategy == BLEND_INTERLEAVED:
                return None, None, "SWITCH_INTERLEAVED: sudah INTERLEAVED"
            new_spec.strategy = BLEND_INTERLEAVED
            new_cfg = self._blender.blend(new_spec)
            desc = f"SWITCH_INTERLEAVED {spec.strategy}→INTERLEAVED"

        elif action == "SWITCH_STAGED":
            if spec.strategy == BLEND_STAGED:
                return None, None, "SWITCH_STAGED: sudah STAGED"
            new_spec.strategy = BLEND_STAGED
            new_cfg = self._blender.blend(new_spec)
            desc = f"SWITCH_STAGED {spec.strategy}→STAGED"

        elif action == "SWITCH_WEIGHTED":
            if spec.strategy == BLEND_WEIGHTED:
                return None, None, "SWITCH_WEIGHTED: sudah WEIGHTED"
            new_spec.strategy = BLEND_WEIGHTED
            new_cfg = self._blender.blend(new_spec)
            desc = f"SWITCH_WEIGHTED {spec.strategy}→WEIGHTED"

        elif action == "TUNE_BLEND_DEPTH":
            new_cfg = copy.deepcopy(cfg)
            if cfg.num_layers <= 4:
                return None, None, "TUNE_BLEND_DEPTH: layers terlalu sedikit"
            # Sesuaikan depth ke depth-width ratio optimal
            dw = cfg.num_layers / math.sqrt(max(1, cfg.hidden_dim))
            if dw > 0.45:
                new_cfg.num_layers = max(4, cfg.num_layers - 2)
                desc = f"TUNE_BLEND_DEPTH ↓ {cfg.num_layers}→{new_cfg.num_layers}"
            elif dw < 0.15:
                new_cfg.num_layers = cfg.num_layers + 2
                desc = f"TUNE_BLEND_DEPTH ↑ {cfg.num_layers}→{new_cfg.num_layers}"
            else:
                return None, None, "TUNE_BLEND_DEPTH: depth sudah optimal"
            new_cfg = self._blender._finalize(new_cfg)

        elif action == "TUNE_BLEND_FFN":
            new_cfg = copy.deepcopy(cfg)
            ffn = cfg.ffn_multiplier
            if 3.0 <= ffn <= 4.5:
                return None, None, "TUNE_BLEND_FFN: FFN sudah dalam range optimal"
            target = max(3.0, min(4.5, ffn))
            # Snap ke aligned value
            ffn_dim  = int(cfg.hidden_dim * target)
            aligned  = ((ffn_dim + 127) // 128) * 128
            new_cfg.ffn_multiplier = round(aligned / max(1, cfg.hidden_dim), 4)
            desc = f"TUNE_BLEND_FFN {ffn:.3f}→{new_cfg.ffn_multiplier:.3f}"
            new_cfg = self._blender._finalize(new_cfg)

        elif action == "FIX_ATTENTION_UNITY":
            new_cfg = copy.deepcopy(cfg)
            # Unifikasi ke GQA (paling stabil untuk hybrid)
            attn_val = cfg.attn_type
            attn_str = attn_val.value if hasattr(attn_val, 'value') else str(attn_val)
            if "GQA" in attn_str.upper():
                return None, None, "FIX_ATTENTION_UNITY: sudah GQA"
            new_cfg.attn_type = AttentionType.GQA
            # Pastikan kv_heads valid
            valid_kv = [h for h in range(1, cfg.num_heads + 1)
                        if cfg.num_heads % h == 0 and h <= cfg.num_heads // 2]
            if valid_kv:
                new_cfg.num_kv_heads = valid_kv[-1]   # terbesar yang masih ≤ heads/2
            desc = f"FIX_ATTENTION_UNITY {attn_str}→GQA"
            new_cfg = self._blender._finalize(new_cfg)

        elif action == "ENABLE_SHARED_EMBED":
            if cfg.tie_embeddings:
                return None, None, "ENABLE_SHARED_EMBED: sudah tie_embeddings"
            new_cfg = copy.deepcopy(cfg)
            new_cfg.tie_embeddings = True
            desc = "ENABLE_SHARED_EMBED: aktifkan tied embeddings"
            new_cfg = self._blender._finalize(new_cfg)

        elif action == "BALANCE_KV_HEADS":
            new_cfg = copy.deepcopy(cfg)
            # Cari GQA ratio yang optimal: kv_heads = num_heads / 4
            target_kv = max(1, cfg.num_heads // 4)
            valid_kv  = [h for h in range(1, cfg.num_heads + 1)
                         if cfg.num_heads % h == 0]
            if not valid_kv:
                return None, None, "BALANCE_KV_HEADS: tidak ada valid kv_heads"
            best_kv = min(valid_kv, key=lambda h: abs(h - target_kv))
            if best_kv == cfg.num_kv_heads:
                return None, None, "BALANCE_KV_HEADS: sudah balanced"
            new_cfg.num_kv_heads = best_kv
            desc = f"BALANCE_KV_HEADS {cfg.num_kv_heads}→{best_kv}"
            new_cfg = self._blender._finalize(new_cfg)

        elif action == "REDUCE_MOE_EXPERTS":
            if cfg.num_experts <= 1:
                return None, None, "REDUCE_MOE_EXPERTS: bukan MoE atau sudah 1"
            if cfg.vram_usage_pct <= 65:
                return None, None, "REDUCE_MOE_EXPERTS: VRAM tidak tertekan"
            new_cfg = copy.deepcopy(cfg)
            new_cfg.num_experts  = max(2, cfg.num_experts - 2)
            new_cfg.top_k_experts = max(1, min(new_cfg.top_k_experts, new_cfg.num_experts - 1))
            desc = f"REDUCE_MOE_EXPERTS {cfg.num_experts}→{new_cfg.num_experts}"
            new_cfg = self._blender._finalize(new_cfg)

        elif action == "ADJUST_SEQ_LEN":
            new_cfg = copy.deepcopy(cfg)
            # Turunkan seq_len jika VRAM tinggi
            if cfg.vram_usage_pct > 70 and cfg.seq_len > 1024:
                target_seq = cfg.seq_len // 2
                new_cfg.seq_len = max(512, target_seq)
                desc = f"ADJUST_SEQ_LEN ↓ {cfg.seq_len}→{new_cfg.seq_len}"
            elif cfg.vram_usage_pct < 50 and cfg.seq_len < 4096:
                new_cfg.seq_len = min(cfg.seq_len * 2, 4096)
                desc = f"ADJUST_SEQ_LEN ↑ {cfg.seq_len}→{new_cfg.seq_len}"
            else:
                return None, None, "ADJUST_SEQ_LEN: seq_len sudah optimal"
            new_cfg = self._blender._finalize(new_cfg)

        else:
            return None, None, f"Unknown action: {action}"

        if not new_cfg.fits_gpu:
            return None, None, f"{action}: OOM setelah perturbasi"

        return new_cfg, new_spec, desc


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION NAS REFINER (MAIN CLASS)
# ══════════════════════════════════════════════════════════════════════════════

class CombinationNASRefiner:
    """
    Combination NAS + RL Refinement Engine.

    Phase A: Generate initial blend dari CombinationSpec
    Phase B: Quality heuristic fixes (ArcRefiner)
    Phase C: RL combination tuning (ratio + strategy optimization)
    Phase D: Training NAS evaluation (dari train_refine.py)

    Combined Score = 33% combo_score + 34% hardware_score + 33% training_score
    """

    def __init__(
        self,
        gpu:               GPUSpec,
        max_iterations:    int   = 25,
        max_explore_iters: int   = 30,
        rng_seed:          Optional[int] = None,
        device:            str   = "cpu",
    ):
        self.gpu              = gpu
        self.max_iterations   = max_iterations
        self.max_explore_iters = max_explore_iters

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self._blender     = CombinationBlender(gpu, rng_seed=rng_seed)
        self._combo_eval  = CombinationNASEvaluator(gpu)
        self._hw_eval     = HardwareNASEvaluator(gpu)
        self._proxy       = ProxyTrainer(device=device)
        self._train_eval  = TrainingDynamicsEvaluator(gpu)
        self._quality_scorer = ArcQualityScorer(gpu)
        self._arc_refiner = ArcRefiner(gpu, max_iterations=max_iterations)
        self._q_learner   = CombinationQLearner()
        self._perturber   = CombinationPerturbEngine(gpu, rng_seed=rng_seed)
        self._cache       = CombinationNASCache()

    def _evaluate_cached(
        self, cfg: ArchConfig, spec: CombinationSpec
    ) -> CombinationNASResult:
        hit = self._cache.get(cfg, spec)
        if hit is not None:
            return hit
        result = self._combo_eval.evaluate(cfg, spec)
        self._cache.put(cfg, spec, result)
        return result

    def _compute_combined(
        self,
        combo_score: float,
        hw_score:    float,
        train_score: float,
    ) -> float:
        """Combined = 33% combo + 34% hardware + 33% training."""
        return round(
            0.33 * combo_score + 0.34 * hw_score + 0.33 * train_score,
            5
        )

    def refine(
        self,
        spec: CombinationSpec,
        n_candidates: int = 3,
    ) -> Tuple[ArchConfig, CombinationSpec, CombinationAdaptiveLog]:
        """
        Generate dan refine satu kombinasi arsitektur.

        Phase A: Generate n_candidates blends, pilih yang terbaik
        Phase B: Quality heuristic fixes
        Phase C: RL tuning (ratio, strategy, structural)
        Phase D: Training NAS evaluation

        Returns: (best_cfg, best_spec, log)
        """
        alog = CombinationAdaptiveLog(
            spec_label = spec.label,
        )

        # ── Phase A: Generate candidates ──────────────────────────────────────
        candidates = []
        for _ in range(n_candidates):
            try:
                cfg = self._blender.blend(spec)
                if cfg.fits_gpu:
                    combo_res = self._evaluate_cached(cfg, spec)
                    candidates.append((cfg, spec, combo_res))
            except Exception:
                continue

        if not candidates:
            # Fallback: generate dengan weighted strategy
            fallback_spec          = copy.deepcopy(spec)
            fallback_spec.strategy = BLEND_WEIGHTED
            try:
                cfg       = self._blender.blend(fallback_spec)
                combo_res = self._evaluate_cached(cfg, fallback_spec)
                candidates.append((cfg, fallback_spec, combo_res))
            except Exception:
                pass

        if not candidates:
            # Absolute fallback: generate satu family saja
            cfg = self._blender._gen.generate_one(spec.family_a)
            combo_res = self._evaluate_cached(cfg, spec)
            candidates.append((cfg, spec, combo_res))

        # Pilih kandidat terbaik
        candidates.sort(key=lambda x: x[2].combination_score, reverse=True)
        best_cfg, best_spec, best_combo = candidates[0]

        # ── Phase B: Quality heuristic fixes ─────────────────────────────────
        best_cfg, base_log = self._arc_refiner.refine(best_cfg)
        alog.base_log    = base_log
        alog.arch_id     = best_cfg.arch_id
        alog.arch_name   = best_cfg.arch_name
        alog.quality_start = base_log.initial_pct
        alog.quality_end   = base_log.final_pct

        # Initial scores
        best_combo    = self._evaluate_cached(best_cfg, best_spec)
        hw_result     = self._hw_eval.evaluate(best_cfg)
        proxy_result  = self._proxy.train(best_cfg)
        train_result  = self._train_eval.evaluate(best_cfg, proxy_result)

        alog.combo_score_start  = best_combo.combination_score
        alog.hw_score_start     = hw_result.hardware_score
        alog.train_score_start  = train_result.training_score
        alog.combined_start     = self._compute_combined(
            best_combo.combination_score,
            hw_result.hardware_score,
            train_result.training_score,
        )

        best_hw_score    = hw_result.hardware_score
        best_train_score = train_result.training_score
        best_combined    = alog.combined_start

        # ── Phase C: RL Combination Tuning ────────────────────────────────────
        action_fail:  Dict[str, int] = {}
        action_tried: set            = set()
        no_improve   = 0
        burst_count  = 0
        MAX_PAT      = 10   # diturunkan dari 12 agar lebih responsif
        BURST_EVERY  = 8    # burst setelah MAX_PAT/BURST_EVERY non-improve steps
        T            = self.max_explore_iters

        for step in range(T):
            # Anti-stagnation: burst explore jika mandek
            force_explore = (no_improve >= MAX_PAT)
            if force_explore:
                burst_count += 1
                no_improve = 0   # reset setelah trigger burst

            act_idx = self._q_learner.select_action(
                best_combo.combination_score,
                best_hw_score,
                best_train_score,
                best_spec,
                action_fail,
                force_explore=force_explore,
            )
            action = COMBO_ACTIONS[act_idx]
            alog.perturbation_tries += 1
            action_tried.add(action)

            new_cfg, new_spec, desc = self._perturber.apply(best_cfg, best_spec, action)
            if new_cfg is None:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1
                # Penalti Q-value untuk aksi yang sering gagal
                if action_fail.get(action, 0) >= 3:
                    key = self._q_learner._state_key(
                        best_combo.combination_score,
                        best_hw_score,
                        best_train_score,
                        best_spec,
                    )
                    self._q_learner._q[key][act_idx] = max(
                        self._q_learner._q[key][act_idx] - 0.5, -3.0
                    )
                if no_improve >= MAX_PAT * 2:
                    break
                continue

            # Evaluate new combo
            new_combo   = self._evaluate_cached(new_cfg, new_spec)
            new_hw      = self._hw_eval.evaluate(new_cfg)

            # Training eval lebih mahal — lakukan setiap 3 steps atau jika structural change
            is_structural = action in (
                "TUNE_BLEND_DEPTH", "SHIFT_RATIO_A", "SHIFT_RATIO_B",
                "SWITCH_INTERLEAVED", "SWITCH_STAGED", "SWITCH_WEIGHTED",
            )
            run_train_eval = (is_structural or step % 3 == 0 or
                              new_combo.combination_score > best_combo.combination_score + 0.04)

            if run_train_eval:
                new_proxy = self._proxy.train(new_cfg)
                new_train = self._train_eval.evaluate(new_cfg, new_proxy)
                new_train_score = new_train.training_score
            else:
                new_train_score = best_train_score   # pakai nilai terakhir

            new_combined = self._compute_combined(
                new_combo.combination_score,
                new_hw.hardware_score,
                new_train_score,
            )

            # Reward seimbang: menggunakan _balanced_combo_reward()
            # Tidak bias ke satu dimensi
            delta_combo = new_combo.combination_score - best_combo.combination_score
            delta_hw    = new_hw.hardware_score - best_hw_score
            delta_train = new_train_score - best_train_score
            reward = _balanced_combo_reward(
                delta_combo, delta_hw, delta_train,
                new_combo.combination_score,
                new_hw.hardware_score,
                new_train_score,
            )

            # Q-update dengan 6-dim state (bukan hanya combo_score)
            td_error = self._q_learner.update(
                best_combo.combination_score, best_hw_score, best_train_score, best_spec,
                act_idx, reward,
                new_combo.combination_score, new_hw.hardware_score, new_train_score, new_spec,
            )
            self._q_learner.replay_update(10)

            # Accept criterion seimbang: combined naik, ATAU satu dimensi naik
            # TANPA collapse di dimensi lain (threshold -0.03)
            delta_comb = new_combined - best_combined
            accept = (
                delta_comb > 5e-5 or
                (delta_combo > 3e-3 and delta_hw > -0.03 and delta_train > -0.03) or
                (delta_hw    > 3e-3 and delta_combo > -0.03 and delta_train > -0.03) or
                (delta_train > 3e-3 and delta_combo > -0.03 and delta_hw > -0.03)
            )
            # Anti-collapse: jangan accept jika satu dimensi drop drastis
            has_collapse = (
                new_combo.combination_score < best_combo.combination_score - 0.05 or
                new_hw.hardware_score       < best_hw_score - 0.05 or
                new_train_score             < best_train_score - 0.05
            )

            if accept and not has_collapse:
                best_cfg      = new_cfg
                best_spec     = new_spec
                best_combo    = new_combo
                best_hw_score = new_hw.hardware_score
                best_train_score = new_train_score
                best_combined = new_combined
                alog.perturbations_accepted += 1
                action_fail[action] = 0
                no_improve          = 0

                if "RATIO" in action:
                    alog.ratio_adjustments += 1
                if "SWITCH" in action:
                    alog.strategy_switches += 1

                alog.improvement_events.append(
                    f"[step{step+1:02d}] {action:<22} → "
                    f"combo={best_combo.combination_score:.4f} "
                    f"hw={best_hw_score:.4f} "
                    f"train={best_train_score:.4f} "
                    f"combined={best_combined:.5f} (Δ{delta_comb:+.5f})"
                )
            else:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1

        # ── Phase D: Final Training NAS evaluation ────────────────────────────
        final_proxy = self._proxy.train(best_cfg)
        final_train = self._train_eval.evaluate(best_cfg, final_proxy)
        final_hw    = self._hw_eval.evaluate(best_cfg)
        final_combo = self._evaluate_cached(best_cfg, best_spec)

        alog.combo_score_end  = final_combo.combination_score
        alog.hw_score_end     = final_hw.hardware_score
        alog.train_score_end  = final_train.training_score
        alog.quality_end      = self._quality_scorer.score(best_cfg).pct
        alog.combined_end     = self._compute_combined(
            final_combo.combination_score,
            final_hw.hardware_score,
            final_train.training_score,
        )
        alog.warnings = final_combo.warnings

        if alog.perturbations_accepted > 0:
            alog.status = f"↑ IMPROVED (combo Δ{alog.combo_delta:+.4f})"
        else:
            alog.status = "~ STAGNATED"

        return best_cfg, best_spec, alog

    def refine_batch(
        self,
        specs:        List[CombinationSpec],
        n_candidates: int = 3,
    ) -> Tuple[List[ArchConfig], List[CombinationSpec], List[CombinationAdaptiveLog]]:
        """
        Refine batch CombinationSpec.
        Returns (archs, specs, logs) sorted by combined_end descending.
        """
        archs_out = []
        specs_out = []
        logs_out  = []

        for spec in specs:
            cfg, final_spec, log = self.refine(spec, n_candidates=n_candidates)
            archs_out.append(cfg)
            specs_out.append(final_spec)
            logs_out.append(log)

        # Sort by combined_end
        triplets = sorted(
            zip(archs_out, specs_out, logs_out),
            key=lambda x: x[2].combined_end,
            reverse=True,
        )
        if triplets:
            archs_out, specs_out, logs_out = zip(*triplets)
            return list(archs_out), list(specs_out), list(logs_out)
        return [], [], []


# ══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ask_combination_type(
    selected_families: Optional[List[str]] = None,
) -> Optional[List["CombinationSpec"]]:
    """
    Tampilkan semua kombinasi valid dari N family yang dipilih user,
    lalu minta user memilih SATU strategi blend yang akan diproses NAS.

    Filosofi:
      User sudah memilih 2–4 family yang ingin di-blend.
      Sistem menampilkan SEMUA kombinasi valid (pasangan 2-way dan 3-way)
      dari pool family tersebut — diurutkan synergy tertinggi dulu.

      User kemudian memilih SATU kombinasi (1 nomor) yang paling cocok
      dengan tujuannya. Sistem akan menghasilkan 1 arsitektur terkuat
      dari blend tersebut via RL combination refinement.

      Mengapa hanya 1?
        Karena tujuan combination mode adalah menemukan SATU arsitektur
        terkuat yang menggabungkan kekuatan beberapa type. Bukan membuat
        banyak arsitektur. RL dijalankan untuk menyempurnakan 1 kombinasi
        ini terus-menerus hingga optimal.

    Args:
      selected_families: list family yang dipilih user (2–4 family)

    Returns:
      List[CombinationSpec] berisi tepat 1 spec, atau None jika dibatalkan.
      (Dikembalikan sebagai List untuk kompatibilitas dengan pipeline.py)
    """
    # Tentukan pool family yang bisa dipilih
    pool = list(selected_families) if selected_families else list(ALL_FAMILIES)
    pool = [f for f in pool if f in ALL_FAMILIES]   # pastikan valid

    if len(pool) < 2:
        print(f"  ⚠ Hanya {len(pool)} family dipilih, butuh minimal 2 untuk kombinasi.")
        return None

    # Dapatkan SEMUA kombinasi valid dari pool (2-way + 3-way)
    # Untuk 4-way pool: semua 2-way + semua 3-way yang valid
    max_nway = min(len(pool), 3)   # max 3-way untuk display (4-way terlalu kompleks)
    combos_all = CombinationRuleEngine.get_valid_for_families(
        pool, min_families=2, max_families=max_nway
    )

    if not combos_all:
        print()
        print("  ⚠ Tidak ada kombinasi valid antar family yang dipilih:")
        for f in pool:
            print(f"    • {f}")
        print()
        expand = input("▶  Tampilkan semua kombinasi tersedia dari 7 family? (Y/n): ").strip().lower()
        if expand in ("n", "no"):
            return None
        pool      = list(ALL_FAMILIES)
        combos_all = CombinationRuleEngine.get_valid_for_families(pool, 2, 3)
        print("  ℹ Menampilkan semua kombinasi valid (7 family).\n")

    if not combos_all:
        print("  ✗ Tidak ada kombinasi tersedia sama sekali.")
        return None

    # ── Tampilkan tabel kombinasi valid ──────────────────────────────────────
    sel_str = f"dari: {', '.join(pool)}" if len(pool) <= 5 else f"{len(pool)} families"
    print()
    print(f"  ┌─ Kombinasi Valid ({sel_str}) {'─'*30}")
    print("  │  Sistem akan menghasilkan 1 arsitektur terkuat dari blend yang dipilih.")
    print("  │  RL combination menyempurnakan terus hingga optimal.")
    print("  │  Urutan: synergy tertinggi dulu")
    print("  │")

    all_display = []   # (idx_display, families_tuple, compat, avg_syn, n_way)

    # Pisahkan 2-way dan 3-way untuk display yang lebih jelas
    combos_2 = [(f, c, s) for f, c, s in combos_all if len(f) == 2]
    combos_3 = [(f, c, s) for f, c, s in combos_all if len(f) == 3]

    if combos_2:
        print("  │  ── 2-Way Combinations ──────────────────────────────────────")
        for fams_tuple, compat, syn in combos_2:
            i = len(all_display) + 1
            sym = {
                "STRONGLY_VALID": "✓✓",
                "COMPATIBLE":     "✓ ",
                "MARGINAL":       "~ ",
            }.get(compat, "? ")
            key  = (min(fams_tuple[0], fams_tuple[1]), max(fams_tuple[0], fams_tuple[1]))
            info = _SYNERGY_NORMALIZED.get(key, {})
            strat_abbr = info.get("best_strategy", BLEND_WEIGHTED)[:3]
            rat        = info.get("rationale", "")[:42]
            rec_ratio  = info.get("recommended_ratio", 0.5)
            print(
                f"  │  [{i:2d}] {sym}  {fams_tuple[0]:<20} + {fams_tuple[1]:<20}  "
                f"syn={syn:.2f}  [{strat_abbr}]  ratio={int(rec_ratio*100)}/{int((1-rec_ratio)*100)}"
            )
            if rat:
                print(f"  │       └ {rat}")
            all_display.append((i, fams_tuple, compat, syn, 2))

    if combos_3:
        print("  │")
        print("  │  ── 3-Way Combinations ──────────────────────────────────────")
        for fams_tuple, compat, syn in combos_3:
            i   = len(all_display) + 1
            sym = {
                "STRONGLY_VALID": "✓✓",
                "COMPATIBLE":     "✓ ",
                "MARGINAL":       "~ ",
            }.get(compat, "? ")
            fstr = " + ".join(f.split("-")[0][:8] for f in fams_tuple)
            print(f"  │  [{i:2d}] {sym}  {fstr:<48}  syn={syn:.2f}  (3-way)")
            all_display.append((i, fams_tuple, compat, syn, 3))

    print("  │")
    print("  └" + "─" * 70)
    print()
    print("  ℹ Pilih 1 kombinasi → sistem buat arsitektur terkuat dari blend itu.")
    print("    RL terus menyempurnakan ratio, strategy, dan struktur secara otomatis.")
    print()

    # ── Input pilihan — SATU nomor ────────────────────────────────────────────
    while True:
        choice_str = input(
            f"▶  Pilih nomor kombinasi (1 pilihan saja, 1–{len(all_display)}): "
        ).strip()

        if not choice_str:
            print(f"  ✗ Masukkan 1 nomor, contoh: '1' atau '3'")
            continue

        # Hanya ambil angka pertama yang valid
        first_num = None
        for part in choice_str.replace(" ", "").split(","):
            if not part:
                continue
            try:
                idx = int(part)
                if 1 <= idx <= len(all_display):
                    first_num = idx - 1
                    break
                else:
                    print(f"  ✗ Nomor {part} tidak valid, pilih 1–{len(all_display)}.")
            except ValueError:
                print(f"  ✗ '{part}' bukan angka valid.")

        if first_num is None:
            continue

        if choice_str.replace(" ", "").count(",") > 0:
            # User masukkan lebih dari satu, tapi kita hanya ambil pertama
            print(f"  ℹ Mode combination hanya proses 1 kombinasi → diambil nomor {first_num+1}.")

        break

    di = first_num
    _, fams_tuple, compat, avg_syn, n_way = all_display[di]
    families = list(fams_tuple)
    n        = len(families)

    # ── Info kombinasi yang dipilih ───────────────────────────────────────────
    print()
    print(f"  ✓ Kombinasi [{di+1}]: {' + '.join(families)}")
    print(f"    Compatibility: {compat}  |  Avg synergy: {avg_syn:.2f}  |  {n}-way")

    if n == 2:
        key  = (min(families[0], families[1]), max(families[0], families[1]))
        info = _SYNERGY_NORMALIZED.get(key, {})
        rat  = info.get("rationale", "")
        constraint = info.get("param_constraint", "")
        if rat:
            print(f"    Rationale  : {rat}")
        if constraint:
            print(f"    Constraint : {constraint}")
    else:
        print(f"    3-way blend: lebih kompleks, butuh VRAM lebih dan training lebih hati-hati")

    # ── Rekomendasi strategy & ratio ─────────────────────────────────────────
    rec_strat, rec_ratios = CombinationRuleEngine.recommend_strategy(families)
    print()
    print(f"    Recommended: strategy={rec_strat}  "
          f"ratio={[int(r*100) for r in rec_ratios]}")
    print(f"    (RL akan otomatis fine-tune ratio dan strategy dari starting point ini)")
    print()

    # ── Pilih strategy ────────────────────────────────────────────────────────
    strat_input = input(
        f"▶  Strategy awal [I=Interleaved / S=Staged / W=Weighted"
        f", Enter={rec_strat[0]}]: "
    ).strip().upper()

    strat_map = {
        "I": BLEND_INTERLEAVED, "INTERLEAVED": BLEND_INTERLEAVED,
        "S": BLEND_STAGED,      "STAGED":      BLEND_STAGED,
        "W": BLEND_WEIGHTED,    "WEIGHTED":    BLEND_WEIGHTED,
    }
    strategy = strat_map.get(strat_input, rec_strat)

    # ── Atur ratio awal per family ────────────────────────────────────────────
    print(f"  Atur ratio awal (total = 100%, Enter = pakai rekomendasi):")
    print(f"  ℹ RL akan otomatis adjust ratio ini — ini hanya titik awal.")
    custom_ratios = []
    for i_f, fam in enumerate(families):
        default_r = rec_ratios[i_f] if i_f < len(rec_ratios) else round(1.0/n, 2)
        raw = input(
            f"▶    Ratio {fam.split('-')[0]:<12} [default={int(default_r*100)}%]: "
        ).strip()
        try:
            val = float(raw)
            val = val / 100.0 if val > 1.0 else val
            custom_ratios.append(max(0.10, min(0.90, val)))
        except ValueError:
            custom_ratios.append(default_r)

    # Normalize
    total = sum(custom_ratios)
    if total > 0:
        ratios = [round(r / total, 4) for r in custom_ratios]
    else:
        ratios = [round(1.0/n, 4)] * n

    # ── Buat CombinationSpec ──────────────────────────────────────────────────
    spec = CombinationSpec(
        families  = families,
        ratios    = ratios,
        strategy  = strategy,
    )

    ratio_str = " + ".join(
        f"{fam.split('-')[0]}:{int(r*100)}%"
        for fam, r in zip(families, ratios)
    )
    print()
    print(f"  ✓ Blend spec: {ratio_str}  [{strategy}]")
    print(f"    Compatibility: {compat}  |  Synergy: {spec.synergy_mult:.2f}")
    print()
    print(f"  ► NAS combination akan mencari 1 arsitektur terkuat dari blend ini.")
    print(f"    RL combination refiner menyempurnakan ratio, strategy, dan parameter")
    print(f"    secara otomatis hingga combined_score (combo+hw+train) maksimal.")
    print()

    return [spec]


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_idx(value: float, buckets: List[Tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


def print_combination_result(
    cfg:        ArchConfig,
    spec:       CombinationSpec,
    combo_res:  CombinationNASResult,
    hw_score:   float,
    train_score: float,
    combined:   float,
    *,
    console=None,
) -> None:
    """Print hasil evaluasi satu kombinasi."""
    _p = console.print if console else print

    W = 88
    S = "─" * W
    compat_sym = {"STRONGLY_VALID": "✓✓ STRONGLY", "COMPATIBLE": "✓ COMPATIBLE",
                   "MARGINAL": "~ MARGINAL"}.get(spec.compatibility, spec.compatibility)

    _p()
    _p(f"╭{S}╮")
    _p(f"│{'  🔀 COMBINATION ARCHITECTURE RESULT':^{W}}│")
    _p(f"├{S}┤")
    _p(f"│  ARC: {cfg.arch_id:<10}  {cfg.arch_name[:55]:<55}  │")
    _p(f"│  Combo: {spec.family_a} ({spec.ratio_a:.0%}) + "
       f"{spec.family_b} ({spec.ratio_b:.0%})  │")
    _p(f"│  Strategy: {spec.strategy:<12}  Compatibility: {compat_sym:<25}  Synergy: {spec.synergy_mult:.2f}  │")
    _p(f"├{S}┤")
    _p(f"│  {'COMBINATION SCORES (33% combo + 34% hw + 33% training)':^{W}}│")
    _p(f"│  Combined       : {combined:.5f}                                         │")
    _p(f"│  Combo Score    : {combo_res.combination_score:.4f}  ×33%               │")
    _p(f"│  Hardware Score : {hw_score:.4f}  ×34%               │")
    _p(f"│  Training Score : {train_score:.4f}  ×33%               │")
    _p(f"├{S}┤")
    _p(f"│  {'COMBINATION NAS (5 dimensi)':^{W}}│")
    _p(f"│  C1 Family Coherence      : {combo_res.pts_c1:>5.1f}/25  score={combo_res.coherence_score:.3f}  │")
    _p(f"│  C2 Blend Balance         : {combo_res.pts_c2:>5.1f}/20  score={combo_res.balance_score:.3f}  │")
    _p(f"│  C3 Architectural Synergy : {combo_res.pts_c3:>5.1f}/20  score={combo_res.synergy_score:.3f}  │")
    _p(f"│  C4 Hardware Compat       : {combo_res.pts_c4:>5.1f}/20  score={combo_res.hw_compat_score:.3f}  │")
    _p(f"│  C5 Training Synergy      : {combo_res.pts_c5:>5.1f}/15  score={combo_res.train_synergy_score:.3f}  │")
    _p(f"│  Total: {combo_res.total_pts:.1f}/100  [{combo_res.grade[:35]}]  │")
    _p(f"├{S}┤")
    _p(f"│  {'ARCHITECTURE':^{W}}│")
    _p(f"│  Params: {cfg.param_count/1e6:.1f}M  L={cfg.num_layers}  D={cfg.hidden_dim}  "
       f"H={cfg.num_heads}/{cfg.num_kv_heads}  hd={cfg.head_dim}  │")
    _p(f"│  FFN: {_enum_str(cfg.ffn_type):<15}×{cfg.ffn_multiplier:.2f}  "
       f"Attn: {_enum_str(cfg.attn_type):<15}  Seq={cfg.seq_len}  │")
    _p(f"│  VRAM: {cfg.vram_total_gb:.2f}GB ({cfg.vram_usage_pct:.1f}%)  "
       f"MFU: {cfg.mfu_estimate:.4f}  Bottleneck: {cfg.bottleneck}  │")
    if combo_res.warnings:
        _p(f"├{S}┤")
        _p(f"│  {'⚠ WARNINGS':^{W}}│")
        for w in combo_res.warnings[:3]:
            _p(f"│  {w[:W-4]:<{W-4}}  │")
    _p(f"╰{S}╯")
    _p()


def print_combination_summary(
    archs:    List[ArchConfig],
    specs:    List[CombinationSpec],
    logs:     List[CombinationAdaptiveLog],
    *,
    console=None,
) -> None:
    """Print tabel ringkasan semua kombinasi."""
    _p = console.print if console else print

    _p()
    _p("  ┌─ Combination NAS Summary ─────────────────────────────────────────────────────────")
    _p("  │  Combined = 33% combo_score + 34% hardware_score + 33% training_score")
    _p("  │  combo_score dari 5 dimensi: Coherence/Balance/Synergy/HW-Compat/Train-Synergy")
    _p("  │")
    _p(f"  │  {'Rank':<4}  {'ARC-ID':<10}  {'Combination':<40}  "
       f"{'Combo':>7}  {'HW':>7}  {'Train':>7}  {'Combined':>9}  Status")
    _p("  │  " + "─" * 110)

    ranked = sorted(enumerate(logs), key=lambda x: x[1].combined_end, reverse=True)

    for new_rank, (orig_idx, log) in enumerate(ranked, 1):
        cfg  = archs[orig_idx] if orig_idx < len(archs) else None
        spec = specs[orig_idx] if orig_idx < len(specs) else None
        if cfg is None or spec is None:
            continue

        sym     = "★" if new_rank == 1 else f"#{new_rank}"
        combo_label = f"{spec.family_a[:8]}+{spec.family_b[:8]}[{spec.strategy[:3]}]"

        _p(f"  │  {sym:<4}  {log.arch_id:<10}  {combo_label:<40}  "
           f"{log.combo_score_end:>7.4f}  {log.hw_score_end:>7.4f}  "
           f"{log.train_score_end:>7.4f}  {log.combined_end:>9.5f}  "
           f"{log.status}")

    _p("  │")
    _p("  │  ★ = Best combination berdasarkan balanced 3-way score")
    _p("  └───────────────────────────────────────────────────────────────────────────────────")
    _p()


def _enum_str(val) -> str:
    if val is None:
        return "—"
    if hasattr(val, 'value'):
        return str(val.value)
    s = str(val)
    if '.' in s:
        s = s.split('.')[-1]
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_combination_pipeline(
    specs:             List[CombinationSpec],
    gpu:               GPUSpec,
    max_iterations:    int  = 25,
    max_explore_iters: int  = 30,
    n_candidates:      int  = 3,
    seed:              Optional[int] = None,
    device:            str  = "cpu",
    console=None,
    *,
    use_rl_refiner:    bool = True,   # True → pakai combination_refiner.py (lebih kuat)
) -> Tuple[List[ArchConfig], List[CombinationSpec], List[CombinationAdaptiveLog], Dict]:
    """
    Full combination NAS pipeline.
    Dipanggil dari pipeline.py saat user memilih combination mode.

    Mode baru (use_rl_refiner=True, default):
      Menggunakan CombinationRefiner dari combination_refiner.py yang
      memiliki 16 aksi RL, prioritized replay, reward seimbang anti-bias,
      dan menghasilkan SATU arsitektur terkuat.

    Mode lama (use_rl_refiner=False):
      Menggunakan CombinationNASRefiner (legacy, kompatibilitas).

    Returns:
        (archs, final_specs, logs, score_maps)
        score_maps = {arch_id: {combo, hw, train, combined}}
    """
    _p = console.print if console else print

    _p()
    if console:
        console.rule("[bold cyan]  Combination NAS Pipeline  ")
    else:
        print("─" * 60 + "  Combination NAS Pipeline  " + "─" * 60)

    mode_label = "RL CombinationRefiner (16 aksi, anti-bias)" if use_rl_refiner \
                 else "CombinationNASRefiner (legacy)"
    _p(f"  Mode: {mode_label}")
    _p(f"  Processing {len(specs)} combination spec(s) → akan menghasilkan 1 arsitektur terkuat")
    _p()

    archs:  List[ArchConfig]          = []
    fspecs: List[CombinationSpec]     = []
    logs_out: List[CombinationAdaptiveLog] = []

    if use_rl_refiner:
        # ── Mode baru: gunakan CombinationRefiner dari combination_refiner.py ──
        try:
            from combination_refiner import (
                CombinationRefiner,
                CombinationRLConfig,
                CombinationRLLog,
                print_rl_summary,
            )

            rl_cfg = CombinationRLConfig(
                max_explore_iters = max_explore_iters,
                n_candidates      = n_candidates,
                proxy_device      = device,
            )
            rl_refiner = CombinationRefiner(gpu, cfg=rl_cfg, seed=seed)
            best_cfg, best_spec, rl_log = rl_refiner.refine_to_best(
                specs, n_candidates=n_candidates
            )

            # Bungkus rl_log ke CombinationAdaptiveLog untuk backward compat
            alog = CombinationAdaptiveLog(
                arch_id         = rl_log.arch_id,
                arch_name       = rl_log.arch_name,
                spec_label      = rl_log.spec_label,
                combo_score_start = rl_log.combo_score_start,
                combo_score_end   = rl_log.combo_score_end,
                hw_score_start    = rl_log.hw_score_start,
                hw_score_end      = rl_log.hw_score_end,
                train_score_start = rl_log.train_score_start,
                train_score_end   = rl_log.train_score_end,
                combined_start    = rl_log.combined_start,
                combined_end      = rl_log.combined_end,
                quality_start     = 0.0,
                quality_end       = rl_log.quality_end,
                perturbation_tries     = rl_log.perturbation_tries,
                perturbations_accepted = rl_log.perturbations_accepted,
                ratio_adjustments      = rl_log.ratio_adjustments,
                strategy_switches      = rl_log.strategy_switches,
                improvement_events     = rl_log.improvement_events,
                warnings               = rl_log.warnings,
                status                 = rl_log.status,
            )

            archs.append(best_cfg)
            fspecs.append(best_spec)
            logs_out.append(alog)

        except ImportError:
            _p("  ⚠ combination_refiner.py tidak ditemukan — fallback ke legacy refiner")
            use_rl_refiner = False

    if not use_rl_refiner:
        # ── Mode lama (legacy) ───────────────────────────────────────────────
        refiner = CombinationNASRefiner(
            gpu,
            max_iterations    = max_iterations,
            max_explore_iters = max_explore_iters,
            rng_seed          = seed,
            device            = device,
        )
        archs_b, fspecs_b, logs_b = refiner.refine_batch(specs, n_candidates=n_candidates)
        archs   = archs_b
        fspecs  = fspecs_b
        logs_out = logs_b

    # Build score maps
    score_maps = {}
    for cfg, spec, log in zip(archs, fspecs, logs_out):
        score_maps[cfg.arch_id] = {
            "combo":    log.combo_score_end,
            "hw":       log.hw_score_end,
            "train":    log.train_score_end,
            "combined": log.combined_end,
            "quality":  log.quality_end,
            "spec":     spec,
        }

    # Print summary
    print_combination_summary(archs, fspecs, logs_out, console=console)

    if archs:
        best_log = max(logs_out, key=lambda l: l.combined_end)
        best_idx = logs_out.index(best_log)
        best_cfg  = archs[best_idx]
        best_spec = fspecs[best_idx]
        best_sm   = score_maps[best_cfg.arch_id]

        msg = (
            f"  ★ Best Combination: {best_cfg.arch_id}  "
            f"{best_spec.family_a}+{best_spec.family_b}[{best_spec.strategy}]  "
            f"combined={best_sm['combined']:.5f}"
        )
        if console:
            console.print(f"[bold cyan]{msg}[/bold cyan]")
        else:
            print(msg)

    return archs, fspecs, logs_out, score_maps
