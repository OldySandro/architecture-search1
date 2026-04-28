"""
combination_refiner.py — Dedicated RL Refinement Engine for Combination Architecture
═══════════════════════════════════════════════════════════════════════════════════════

File ini KHUSUS untuk mode combination (blend 2-4 AI type menjadi 1 arsitektur
terkuat). Berbeda dari adaptive_refiner.py yang dipakai untuk single-type RL,
file ini menangani:

  1. State-space yang memperhitungkan SEMUA dimensi combination:
       (combo_bucket, hw_bucket, train_bucket, compat_idx, strategy_idx,
        n_families, ratio_entropy_bucket)
     → Tidak bias ke family manapun karena state di-encode dari skor
       dan properti struktural, bukan dari identitas family.

  2. Reward function yang seimbang dan anti-bias:
       reward = Δcombo×W_C + Δhw×W_H + Δtrain×W_T + synergy_bonus
     dengan W_C/W_H/W_T dikalibrasi agar tidak ada satu aspek yang dominasi.
     synergy_bonus: bonus kecil bila improvement merata di ≥2 dimensi sekaligus.

  3. 16 aksi RL yang komprehensif — lebih luas dari CombinationQLearner dasar:
       Ratio/Strategy (4), Structural (6), MoE-specific (2), Training (4)

  4. Experience replay dengan prioritized sampling:
       TD-error based priority → transisi yang paling "informatif" lebih sering
       di-replay.

  5. Anti-stagnation mechanism:
       Jika no_improve ≥ PATIENCE, reset epsilon sementara (burst explore)
       dan coba aksi yang belum pernah dipakai.

  6. NAS proxy re-evaluation:
       Setiap perubahan struktural yang signifikan → jalankan proxy PyTorch
       (bukan sekadar formula). Ini memastikan skor tidak palsu.

  7. Final selection: 1 kombinasi terkuat dari semua kandidat yang diuji,
       berdasarkan skor combined yang sudah distabilkan oleh NAS.

Arsitektur:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  CombinationRLConfig      — hyperparameter RL & NAS                   │
  │  CombinationRLState       — state representation yang kaya             │
  │  CombinationRLAction      — 16 aksi terstruktur                       │
  │  PrioritizedReplayBuffer  — experience replay dengan TD-priority       │
  │  CombinationRLAgent       — Q-table + UCB + epsilon-greedy agent      │
  │  CombinationRefiner       — main class: Phase A→B→C→D + final pick    │
  │  CombinationRLLog         — log lengkap satu sesi                     │
  └────────────────────────────────────────────────────────────────────────┘

Scoring (3 sinyal seimbang):
  combined = W_COMBO × combo_score     (33%)
           + W_HW    × hardware_score  (34%)
           + W_TRAIN × training_score  (33%)

  combo_score dari CombinationNASEvaluator (5 dimensi C1–C5, 100 pts total).
  Tidak ada multiplier bias keluarga — setiap family diperlakukan setara
  oleh evaluator yang score-based (bukan identity-based).

Integrasi:
  from combination_refiner import CombinationRefiner, CombinationRLLog
  refiner = CombinationRefiner(gpu, device="cpu")
  best_cfg, best_spec, log = refiner.refine_to_best(specs, n_candidates=3)
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

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from arch_types import ArchConfig, AttentionType, FFNType, OptimizerType, NormType
from hardware import GPUSpec
from combination_nas import (
    CombinationSpec,
    CombinationNASResult,
    CombinationNASEvaluator,
    CombinationBlender,
    CombinationRuleEngine,
    CombinationNASCache,
    BLEND_INTERLEAVED,
    BLEND_STAGED,
    BLEND_WEIGHTED,
    COMBO_ACTIONS,
    _SYNERGY_NORMALIZED,
    _COMBO_SCORE_BUCKETS,
    _bucket_idx,
    _balanced_combo_reward,   # ← reward function seimbang dari combination_nas.py
)
from hardware_refine import HardwareNASEvaluator
from train_refine import ProxyTrainer, TrainingDynamicsEvaluator
from refiner import ArcQualityScorer, ArcRefiner


# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA & AKSI RL
# ══════════════════════════════════════════════════════════════════════════════

# 16 aksi RL — superset dari COMBO_ACTIONS dasar, ditambah aksi structural
# dan training yang lebih granular.
COMB_RL_ACTIONS: List[str] = [
    # ── Ratio & Strategy (4) ──────────────────────────────────────────────────
    "SHIFT_RATIO_TOWARD_A",      # naikkan ratio family pertama +0.08
    "SHIFT_RATIO_TOWARD_B",      # turunkan ratio family pertama -0.08
    "SWITCH_TO_INTERLEAVED",     # ganti ke INTERLEAVED strategy
    "SWITCH_TO_STAGED",          # ganti ke STAGED strategy
    # ── Structural — depth/width (4) ─────────────────────────────────────────
    "INCR_LAYERS",               # tambah 2 layer
    "DECR_LAYERS",               # kurangi 2 layer
    "GROW_HIDDEN",               # naikkan hidden_dim satu step (×1.15, aligned 64)
    "SHRINK_HIDDEN",             # turunkan hidden_dim satu step (×0.87, aligned 64)
    # ── Structural — FFN & Attn (2) ──────────────────────────────────────────
    "TUNE_FFN_UP",               # naikkan ffn_multiplier ke range optimal (→4.0)
    "TUNE_FFN_DOWN",             # turunkan ffn_multiplier ke range optimal (→3.0)
    # ── Attention & KV (2) ───────────────────────────────────────────────────
    "BALANCE_KV_HEADS",          # seimbangkan kv_heads ke num_heads//4
    "FIX_ATTN_GQA",              # unifikasi ke GQA (paling stabil hybrid)
    # ── MoE-specific (2) ─────────────────────────────────────────────────────
    "REDUCE_MOE_EXPERTS",        # kurangi experts jika VRAM tertekan
    "INCR_MOE_TOPK",             # naikkan top-k jika underutilized
    # ── Training flags (2) ───────────────────────────────────────────────────
    "ENABLE_MIXED_PREC",         # aktifkan mixed precision
    "TIE_EMBEDDINGS",            # aktifkan tied embeddings
]

_N_COMB_RL_ACTIONS = len(COMB_RL_ACTIONS)

# Bucket boundaries untuk state encoding
_HW_BUCKETS = [
    (0.00, 0.30),  # 0: poor hardware
    (0.30, 0.50),  # 1: below average
    (0.50, 0.65),  # 2: acceptable
    (0.65, 0.80),  # 3: good
    (0.80, 1.01),  # 4: excellent
]

_TRAIN_BUCKETS = [
    (0.00, 0.30),
    (0.30, 0.50),
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 1.01),
]

# Reward weights — SEIMBANG, tidak ada satu aspek yang dominasi
# W_COMBO tidak lebih dari W_HW/W_TRAIN karena combo_score sudah aggregat 5 sub-dim
_W_COMBO = 0.33
_W_HW    = 0.34
_W_TRAIN = 0.33

# Anti-bias: reward penalti kalau ada dimensi yang jauh tertinggal
# Diturunkan dari 0.15 ke 0.12 agar lebih sensitif mendeteksi imbalance lebih awal
_BALANCE_PENALTY_THRESHOLD = 0.12


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG & LOG DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CombinationRLConfig:
    """
    Hyperparameter untuk RL combination refinement.

    Catatan desain anti-bias:
      - alpha 0.14 (diturunkan dari 0.15) untuk konvergensi lebih stabil
        pada state-space yang lebih besar.
      - epsilon_decay 0.990 (lebih lambat dari sebelumnya) agar eksplorasi
        lebih lama sebelum mengeksploitasi — ini penting agar RL tidak
        terlalu cepat konverge ke solusi lokal yang bias ke tipe tertentu.
      - max_no_improve 7 (diturunkan dari 8) untuk lebih agresif melakukan
        burst explore ketika stagnan.
      - Bobot reward seimbang: tidak ada dimensi yang lebih "berharga".
    """
    alpha:              float = 0.14     # learning rate Q-update
    gamma:              float = 0.90     # discount factor
    epsilon:            float = 0.38     # initial exploration rate (naik sedikit)
    epsilon_min:        float = 0.05     # minimum epsilon (exploitation floor)
    epsilon_decay:      float = 0.990    # per-step decay (lebih lambat = eksplorasi lebih lama)
    ucb_c:              float = 2.5      # UCB exploration constant
    replay_cap:         int   = 2000     # max replay buffer size
    replay_batch:       int   = 16       # mini-batch per replay update
    max_explore_iters:  int   = 40       # max RL steps per refinement
    max_no_improve:     int   = 7        # patience sebelum burst explore
    n_candidates:       int   = 3        # jumlah kandidat awal yang di-generate
    proxy_every:        int   = 3        # jalankan NAS proxy setiap N steps
    # Structural probe: setiap berapa step lakukan full re-blend
    reblend_every:      int   = 8

    # NAS proxy settings
    proxy_train_steps:  int   = 40
    proxy_device:       str   = "cpu"


@dataclass
class CombinationRLLog:
    """
    Log lengkap satu sesi CombinationRefiner.

    Berisi:
      • Skor awal dan akhir untuk 3 dimensi + combined
      • Statistik RL: tries, accepted, ratio adj, strategy switches
      • Riwayat improvement events
      • Info kombinasi: families, strategy, ratios
      • Status akhir
    """
    arch_id:            str = ""
    arch_name:          str = ""
    spec_label:         str = ""

    # Families & ratios akhir
    families:           List[str] = field(default_factory=list)
    ratios:             List[float] = field(default_factory=list)
    strategy:           str = ""
    compatibility:      str = ""
    synergy_mult:       float = 0.0

    # Skor awal
    combo_score_start:  float = 0.0
    hw_score_start:     float = 0.0
    train_score_start:  float = 0.0
    combined_start:     float = 0.0

    # Skor akhir
    combo_score_end:    float = 0.0
    hw_score_end:       float = 0.0
    train_score_end:    float = 0.0
    combined_end:       float = 0.0
    quality_end:        float = 0.0

    # Sub-dimensi combo akhir
    pts_c1:             float = 0.0   # Family Coherence /25
    pts_c2:             float = 0.0   # Blend Balance /20
    pts_c3:             float = 0.0   # Arch Synergy /20
    pts_c4:             float = 0.0   # HW Compat /20
    pts_c5:             float = 0.0   # Training Synergy /15

    # RL statistik
    perturbation_tries:     int = 0
    perturbations_accepted: int = 0
    ratio_adjustments:      int = 0
    strategy_switches:      int = 0
    structural_changes:     int = 0
    burst_explore_count:    int = 0
    replay_updates:         int = 0

    # Riwayat
    improvement_events: List[str] = field(default_factory=list)
    warnings:           List[str] = field(default_factory=list)

    # NAS proxy
    proxy_eval_count:   int   = 0
    proxy_nan_count:    int   = 0
    proxy_ms_total:     float = 0.0

    status:             str = ""

    @property
    def combined_delta(self) -> float:
        return round(self.combined_end - self.combined_start, 5)

    @property
    def grade(self) -> str:
        """Grade berdasarkan combined_end (3-way: combo+hw+train).

        Skala disesuaikan — combined score combo arsitektur hybrid lebih rendah
        dari pure single-type karena combination_score-nya sudah di-cap.
        """
        s = self.combined_end
        if s >= 0.76: return "S ★★★  Exceptional Combination"
        if s >= 0.66: return "A+ ★★  Very Good"
        if s >= 0.56: return "A  ★   Good"
        if s >= 0.44: return "B      Acceptable"
        if s >= 0.32: return "C      Marginal"
        if s >= 0.20: return "D      Poor"
        return              "F  ✗   Not Viable"

    @property
    def accept_rate(self) -> float:
        if self.perturbation_tries == 0:
            return 0.0
        return round(self.perturbations_accepted / self.perturbation_tries, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE REPRESENTATION
# ══════════════════════════════════════════════════════════════════════════════

class CombinationRLState:
    """
    Representasi state yang kaya dan tidak bias ke family tertentu.

    State-key adalah tuple integer yang di-encode dari:
      (combo_b, hw_b, train_b, compat_idx, strat_idx, n_fam_b, ratio_ent_b)

    Semua encoding berbasis SKOR dan PROPERTI STRUKTURAL, bukan nama family.
    Ini memastikan policy bisa general ke kombinasi family apapun.
    """

    _COMPAT_IDX = {
        "STRONGLY_VALID": 0,
        "COMPATIBLE":     1,
        "MARGINAL":       2,
        "UNKNOWN":        3,
    }
    _STRAT_IDX = {
        BLEND_WEIGHTED:    0,
        BLEND_STAGED:      1,
        BLEND_INTERLEAVED: 2,
    }

    @classmethod
    def encode(
        cls,
        combo_score:  float,
        hw_score:     float,
        train_score:  float,
        spec:         CombinationSpec,
    ) -> str:
        """
        Encode state menjadi string key untuk Q-table.
        8 dimensi (diperluas dari 7):
          combo_b : bucket combo_score  (0-5)
          hw_b    : bucket hw_score     (0-4)
          train_b : bucket train_score  (0-4)
          compat  : compatibility index (0-3)
          strat   : strategy index      (0-2)
          nf_b    : n_families bucket   (0-2)
          reb     : ratio entropy bucket (0-3)
          balance : imbalance index antara 3 dimensi (0-3, 0=seimbang)

        Dimensi ke-8 (balance) membantu policy membedakan situasi
        "skor tinggi tapi imbalanced" vs "skor sedang tapi seimbang".
        Ini penting agar aksi dipilih berdasarkan konteks yang lengkap.
        """
        cb = _bucket_idx(combo_score, _COMBO_SCORE_BUCKETS)   # 0-5
        hb = _bucket_idx(hw_score,    _HW_BUCKETS)             # 0-4
        tb = _bucket_idx(train_score, _TRAIN_BUCKETS)          # 0-4

        compat_i = cls._COMPAT_IDX.get(spec.compatibility, 3)  # 0-3
        strat_i  = cls._STRAT_IDX.get(spec.strategy, 0)        # 0-2

        # n_families bucket: 2→0, 3→1, 4→2
        nf_b = min(2, max(0, spec.n_families - 2))             # 0-2

        # Ratio entropy bucket: ukuran seberapa "seimbang" rasio
        ratios = [r for r in spec.ratios if r > 0]
        if len(ratios) >= 2:
            ent = -sum(r * math.log(r + 1e-9) for r in ratios)
            max_ent = math.log(len(ratios))
            ratio_ent = ent / max(1e-9, max_ent)   # [0, 1]
        else:
            ratio_ent = 0.0
        reb = min(3, int(ratio_ent * 4))   # 0-3

        # Balance bucket: seberapa seimbang 3 dimensi skor?
        # std dev ternormalisasi: 0=sangat seimbang, 3=sangat imbalanced
        scores_3 = [combo_score, hw_score, train_score]
        mean_3   = sum(scores_3) / 3.0
        std_3    = math.sqrt(sum((s - mean_3) ** 2 for s in scores_3) / 3.0)
        balance_norm = min(1.0, std_3 / 0.4)   # normalize ke [0,1]
        balance_b    = min(3, int(balance_norm * 4))   # 0-3

        return f"{cb}:{hb}:{tb}:{compat_i}:{strat_i}:{nf_b}:{reb}:{balance_b}"


# ══════════════════════════════════════════════════════════════════════════════
#  PRIORITIZED REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ReplayEntry:
    state_key:  str
    action_idx: int
    reward:     float
    next_key:   str
    priority:   float = 1.0


class PrioritizedReplayBuffer:
    """
    Experience replay dengan TD-error based priority.

    Transisi dengan TD-error besar (lebih "mengejutkan") punya probabilitas
    lebih tinggi untuk di-sample. Ini mempercepat konvergensi Q-values
    untuk state-action yang jarang tapi informatif.

    alpha_prio = 0.6  → seberapa kuat pengaruh priority vs uniform
    beta_is    = 0.4  → importance sampling correction (bias correction)
    """

    def __init__(
        self,
        cap:        int   = 2000,
        alpha_prio: float = 0.60,
        beta_is:    float = 0.40,
    ):
        self._buf:        List[_ReplayEntry] = []
        self._cap         = cap
        self._alpha_prio  = alpha_prio
        self._beta_is     = beta_is
        self._max_priority = 1.0

    def push(self, entry: _ReplayEntry) -> None:
        entry.priority = self._max_priority
        if len(self._buf) >= self._cap:
            # Hapus entry dengan priority terendah
            self._buf.sort(key=lambda e: e.priority)
            self._buf.pop(0)
        self._buf.append(entry)

    def sample(self, n: int) -> List[Tuple[_ReplayEntry, float]]:
        """
        Sample n entries.
        Returns list of (entry, importance_weight).
        """
        if len(self._buf) < n:
            n = len(self._buf)
        if n == 0:
            return []

        probs = np.array([e.priority ** self._alpha_prio for e in self._buf],
                          dtype=np.float64)
        probs /= probs.sum()

        indices = np.random.choice(len(self._buf), size=n, replace=False, p=probs)
        # Importance sampling weights
        n_total = len(self._buf)
        weights = (n_total * probs[indices]) ** (-self._beta_is)
        weights /= weights.max()   # normalize

        return [(self._buf[i], float(weights[j]))
                for j, i in enumerate(indices)]

    def update_priority(self, entry: _ReplayEntry, td_error: float) -> None:
        entry.priority = abs(td_error) + 1e-6
        self._max_priority = max(self._max_priority, entry.priority)

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
#  RL AGENT
# ══════════════════════════════════════════════════════════════════════════════

class CombinationRLAgent:
    """
    Q-learning agent untuk combination architecture tuning.

    Fitur:
      • Q-table tabular dengan key dari CombinationRLState.encode()
      • UCB exploration (bonus √(log N / n_i)) + epsilon-greedy
      • Anti-bias: aksi yang terlalu sering gagal mendapat penalti UCB
      • Prioritized experience replay (PrioritizedReplayBuffer)
      • Cross-combination learning: semua experience dikumpulkan
        (tidak reset antar kombinasi) untuk transfer learning
    """

    def __init__(self, cfg: CombinationRLConfig):
        self._cfg    = cfg
        self._q:     Dict[str, List[float]] = defaultdict(
            lambda: [0.0] * _N_COMB_RL_ACTIONS
        )
        self._counts: Dict[str, List[int]] = defaultdict(
            lambda: [0] * _N_COMB_RL_ACTIONS
        )
        self._total_steps = 0
        self._replay = PrioritizedReplayBuffer(
            cap=cfg.replay_cap
        )
        # Track aksi yang sudah dicoba dalam episode ini
        self._tried_this_episode: set = set()

    def reset_episode(self) -> None:
        """Reset per-episode tracking (dipanggil setiap mulai refine baru)."""
        self._tried_this_episode.clear()

    def select_action(
        self,
        combo_score:  float,
        hw_score:     float,
        train_score:  float,
        spec:         CombinationSpec,
        action_fail:  Dict[str, int],
        force_untried: bool = False,
    ) -> int:
        """
        Pilih aksi berdasarkan epsilon-greedy + UCB.

        Perbaikan anti-bias:
          1. force_untried: saat burst, prioritaskan aksi dari KATEGORI berbeda
             yang belum dicoba — bukan hanya aksi random yang belum dicoba.
             Ini mencegah RL terjebak di satu kategori aksi (mis: hanya ratio).
          2. Action type diversity weight: dalam exploration, aksi dari
             kategori yang jarang dipakai mendapat bobot lebih tinggi.
          3. Weak-dimension boost: jika satu dimensi skor jauh di bawah rata-rata,
             aksi yang relevan untuk dimensi itu mendapat UCB boost.
        """
        state_key = CombinationRLState.encode(combo_score, hw_score, train_score, spec)
        q         = self._q[state_key]
        cnt       = self._counts[state_key]
        total_cnt = max(1, sum(cnt))

        # Epsilon decay
        eps = max(
            self._cfg.epsilon_min,
            self._cfg.epsilon * (self._cfg.epsilon_decay ** self._total_steps)
        )

        # ── Burst explore: pilih dari kategori yang paling kurang dieksplorasi ──
        if force_untried:
            # Tentukan kategori aksi yang dicoba tiap jenis
            ratio_tried    = any(a in self._tried_this_episode
                                 for a in ("SHIFT_RATIO_TOWARD_A", "SHIFT_RATIO_TOWARD_B"))
            strategy_tried = any(a in self._tried_this_episode
                                 for a in ("SWITCH_TO_INTERLEAVED", "SWITCH_TO_STAGED"))
            struct_tried   = any(a in self._tried_this_episode
                                 for a in ("INCR_LAYERS", "DECR_LAYERS",
                                           "GROW_HIDDEN", "SHRINK_HIDDEN"))
            ffn_tried      = any(a in self._tried_this_episode
                                 for a in ("TUNE_FFN_UP", "TUNE_FFN_DOWN"))
            train_tried    = any(a in self._tried_this_episode
                                 for a in ("ENABLE_MIXED_PREC", "TIE_EMBEDDINGS"))

            # Pilih dari kategori yang belum dicoba, prioritas sesuai dimensi lemah
            scores_3  = [combo_score, hw_score, train_score]
            mean_3    = sum(scores_3) / 3.0
            weak_combo = combo_score < mean_3 - 0.10
            weak_hw    = hw_score    < mean_3 - 0.10
            weak_train = train_score < mean_3 - 0.10

            candidate_sets = []
            if not ratio_tried:
                candidate_sets.append([0, 1])   # SHIFT_RATIO
            if not strategy_tried:
                candidate_sets.append([2, 3])   # SWITCH_TO_*
            if not struct_tried and (weak_combo or weak_hw):
                candidate_sets.append([4, 5, 6, 7])   # depth/width
            if not ffn_tried and weak_combo:
                candidate_sets.append([8, 9])   # FFN
            if not train_tried and weak_train:
                candidate_sets.append([14, 15])  # training flags

            if candidate_sets:
                chosen_set = random.choice(candidate_sets)
                valid_from_set = [
                    i for i in chosen_set
                    if action_fail.get(COMB_RL_ACTIONS[i], 0) < 4
                ]
                if valid_from_set:
                    return random.choice(valid_from_set)

            # Fallback: aksi yang belum dicoba sama sekali
            untried = [i for i, a in enumerate(COMB_RL_ACTIONS)
                       if a not in self._tried_this_episode
                       and action_fail.get(a, 0) < 4]
            if untried:
                return random.choice(untried)
            # Semua dicoba: pilih paling jarang dipakai
            return int(np.argmin(cnt))

        # ── Epsilon-greedy (bukan burst) ─────────────────────────────────────
        if random.random() < eps:
            # Weighted random: aksi yang sering gagal punya bobot lebih kecil
            # + action type diversity: kategori yang jarang dipakai dapat bobot lebih
            category_count = {
                "ratio":    sum(cnt[i] for i in [0, 1]),
                "strategy": sum(cnt[i] for i in [2, 3]),
                "struct":   sum(cnt[i] for i in [4, 5, 6, 7]),
                "ffn":      sum(cnt[i] for i in [8, 9]),
                "attn":     sum(cnt[i] for i in [10, 11]),
                "moe":      sum(cnt[i] for i in [12, 13]),
                "train":    sum(cnt[i] for i in [14, 15]),
            }
            category_map = {
                0: "ratio", 1: "ratio", 2: "strategy", 3: "strategy",
                4: "struct", 5: "struct", 6: "struct", 7: "struct",
                8: "ffn", 9: "ffn", 10: "attn", 11: "attn",
                12: "moe", 13: "moe", 14: "train", 15: "train",
            }
            max_cat_cnt = max(category_count.values()) + 1
            weights = []
            for i, a in enumerate(COMB_RL_ACTIONS):
                fail_count = action_fail.get(a, 0)
                # Base weight: inversely proportional to fail count
                w = max(0.05, 1.0 / (1.0 + fail_count * 0.5))
                # Diversity bonus: kategori yang jarang dipakai dapat boost
                cat = category_map.get(i, "")
                cat_cnt = category_count.get(cat, 0)
                diversity_bonus = max(0.0, (max_cat_cnt - cat_cnt) / max_cat_cnt) * 0.5
                w += diversity_bonus
                weights.append(w)
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    return i
            return random.randrange(_N_COMB_RL_ACTIONS)
        else:
            # UCB + weak-dimension boost
            # Dimensi yang paling lemah mendorong preferensi ke aksi terkait
            scores_3 = [combo_score, hw_score, train_score]
            mean_3   = sum(scores_3) / 3.0
            weak_combo = combo_score < mean_3 - 0.08
            weak_hw    = hw_score    < mean_3 - 0.08
            weak_train = train_score < mean_3 - 0.08

            # Aksi yang relevan untuk tiap dimensi lemah
            _COMBO_ACTIONS_IDX  = {0, 1, 2, 3, 8, 9, 10, 11}   # ratio/strategy/FFN/attn
            _HW_ACTIONS_IDX     = {6, 7, 8, 9, 12}              # hidden/FFN/MoE
            _TRAIN_ACTIONS_IDX  = {14, 15, 10, 11}              # training flags/attn

            ucb_vals = []
            for i, a in enumerate(COMB_RL_ACTIONS):
                ucb = q[i] + self._cfg.ucb_c * math.sqrt(
                    math.log(total_cnt + 1) / (cnt[i] + 1)
                )
                # Penalti aksi gagal
                fail_count = action_fail.get(a, 0)
                ucb -= fail_count * 0.25

                # Weak-dim boost: kecil tapi konsisten
                if weak_combo and i in _COMBO_ACTIONS_IDX:
                    ucb += 0.08
                if weak_hw and i in _HW_ACTIONS_IDX:
                    ucb += 0.08
                if weak_train and i in _TRAIN_ACTIONS_IDX:
                    ucb += 0.08

                ucb_vals.append(ucb)
            return int(np.argmax(ucb_vals))

    def update(
        self,
        state_key:  str,
        action_idx: int,
        reward:     float,
        next_key:   str,
    ) -> float:
        """
        Q-update Bellman. Returns TD-error untuk priority update.
        """
        q_next   = max(self._q[next_key]) if self._q[next_key] else 0.0
        q_old    = self._q[state_key][action_idx]
        td_target = reward + self._cfg.gamma * q_next
        td_error  = td_target - q_old
        self._q[state_key][action_idx] = q_old + self._cfg.alpha * td_error
        self._counts[state_key][action_idx] += 1
        self._total_steps += 1
        self._tried_this_episode.add(COMB_RL_ACTIONS[action_idx])
        return td_error

    def push_replay(
        self,
        state_key:  str,
        action_idx: int,
        reward:     float,
        next_key:   str,
        td_error:   float,
    ) -> None:
        entry = _ReplayEntry(state_key, action_idx, reward, next_key)
        self._replay.push(entry)
        self._replay.update_priority(entry, td_error)

    def replay_update(self, n: int = 16) -> int:
        """Mini-batch replay dengan importance sampling."""
        samples = self._replay.sample(n)
        if not samples:
            return 0
        for entry, is_weight in samples:
            q_next = max(self._q[entry.next_key]) if self._q[entry.next_key] else 0.0
            q_old  = self._q[entry.state_key][entry.action_idx]
            td     = entry.reward + self._cfg.gamma * q_next - q_old
            # IS-weighted update
            self._q[entry.state_key][entry.action_idx] = (
                q_old + self._cfg.alpha * is_weight * td
            )
            self._replay.update_priority(entry, td)
        return len(samples)


# ══════════════════════════════════════════════════════════════════════════════
#  PERTURBATION ENGINE (EXTENDED)
# ══════════════════════════════════════════════════════════════════════════════

class CombinationRLPerturber:
    """
    Eksekusi 16 aksi RL pada (ArchConfig, CombinationSpec).

    Prinsip:
      - Setiap aksi mengembalikan (new_cfg, new_spec, desc) atau (None, None, reason)
      - Aksi yang menghasilkan OOM langsung ditolak
      - Aksi yang tidak berpengaruh (sudah di state optimal) ditolak
      - Structural actions (INCR/DECR/GROW/SHRINK) selalu re-blend untuk
        menjamin konsistensi internal arsitektur hybrid
    """

    def __init__(self, gpu: GPUSpec, rng_seed: Optional[int] = None):
        self.gpu      = gpu
        self._blender = CombinationBlender(gpu, rng_seed=rng_seed)

    def apply(
        self,
        cfg:    ArchConfig,
        spec:   CombinationSpec,
        action: str,
    ) -> Tuple[Optional[ArchConfig], Optional[CombinationSpec], str]:

        new_spec = copy.deepcopy(spec)

        # ── Ratio & Strategy ──────────────────────────────────────────────────

        if action == "SHIFT_RATIO_TOWARD_A":
            if spec.ratio_a >= 0.82:
                return None, None, "SHIFT_RATIO_TOWARD_A: ratio_a sudah maksimum"
            new_spec.ratios[0] = round(min(0.82, spec.ratio_a + 0.08), 3)
            remainder = round(1.0 - new_spec.ratios[0], 4)
            _redistribute_ratios(new_spec, 0, remainder)
            new_cfg = self._blender.blend(new_spec)
            desc = f"SHIFT_RATIO_TOWARD_A {spec.ratio_a:.2f}→{new_spec.ratio_a:.2f}"

        elif action == "SHIFT_RATIO_TOWARD_B":
            if spec.ratio_a <= 0.18:
                return None, None, "SHIFT_RATIO_TOWARD_B: ratio_b sudah maksimum"
            new_spec.ratios[0] = round(max(0.18, spec.ratio_a - 0.08), 3)
            remainder = round(1.0 - new_spec.ratios[0], 4)
            _redistribute_ratios(new_spec, 0, remainder)
            new_cfg = self._blender.blend(new_spec)
            desc = f"SHIFT_RATIO_TOWARD_B {spec.ratio_a:.2f}→{new_spec.ratio_a:.2f}"

        elif action == "SWITCH_TO_INTERLEAVED":
            if spec.strategy == BLEND_INTERLEAVED:
                return None, None, "SWITCH_TO_INTERLEAVED: sudah INTERLEAVED"
            new_spec.strategy = BLEND_INTERLEAVED
            new_cfg = self._blender.blend(new_spec)
            desc = f"SWITCH_TO_INTERLEAVED {spec.strategy}→INTERLEAVED"

        elif action == "SWITCH_TO_STAGED":
            if spec.strategy == BLEND_STAGED:
                return None, None, "SWITCH_TO_STAGED: sudah STAGED"
            new_spec.strategy = BLEND_STAGED
            new_cfg = self._blender.blend(new_spec)
            desc = f"SWITCH_TO_STAGED {spec.strategy}→STAGED"

        # ── Structural ────────────────────────────────────────────────────────

        elif action == "INCR_LAYERS":
            new_cfg = copy.deepcopy(cfg)
            if cfg.num_layers >= 64:
                return None, None, "INCR_LAYERS: sudah 64 layers"
            new_cfg.num_layers = cfg.num_layers + 2
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"INCR_LAYERS {cfg.num_layers}→{new_cfg.num_layers}"

        elif action == "DECR_LAYERS":
            new_cfg = copy.deepcopy(cfg)
            if cfg.num_layers <= 4:
                return None, None, "DECR_LAYERS: layers sudah minimal"
            dw = cfg.num_layers / math.sqrt(max(1, cfg.hidden_dim))
            if dw <= 0.15:
                return None, None, "DECR_LAYERS: depth-width ratio sudah rendah"
            new_cfg.num_layers = max(4, cfg.num_layers - 2)
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"DECR_LAYERS {cfg.num_layers}→{new_cfg.num_layers}"

        elif action == "GROW_HIDDEN":
            new_cfg = copy.deepcopy(cfg)
            target = _align64(int(cfg.hidden_dim * 1.15))
            if target == cfg.hidden_dim:
                target = cfg.hidden_dim + 64
            # Validasi head divisibility
            valid_heads = [h for h in [cfg.num_heads, cfg.num_heads + 1, cfg.num_heads - 1]
                           if h > 0 and target % h == 0]
            if not valid_heads:
                # Cari jumlah head terdekat yang valid
                for h in range(cfg.num_heads, max(1, cfg.num_heads - 8), -1):
                    if target % h == 0:
                        valid_heads = [h]
                        break
            if not valid_heads:
                return None, None, "GROW_HIDDEN: tidak bisa align head"
            new_cfg.hidden_dim = target
            new_cfg.num_heads  = valid_heads[0]
            new_cfg.head_dim   = target // new_cfg.num_heads
            # KV heads
            valid_kv = [h for h in range(1, new_cfg.num_heads + 1)
                        if new_cfg.num_heads % h == 0 and h <= new_cfg.num_heads]
            new_cfg.num_kv_heads = min(valid_kv,
                key=lambda h: abs(h - max(1, new_cfg.num_heads // 4)))
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"GROW_HIDDEN {cfg.hidden_dim}→{new_cfg.hidden_dim}"

        elif action == "SHRINK_HIDDEN":
            new_cfg = copy.deepcopy(cfg)
            target = _align64(int(cfg.hidden_dim * 0.87))
            if target == cfg.hidden_dim or target < 64:
                return None, None, "SHRINK_HIDDEN: hidden sudah minimal"
            valid_heads = [h for h in [cfg.num_heads, cfg.num_heads - 1, cfg.num_heads - 2]
                           if h > 0 and target % h == 0]
            if not valid_heads:
                for h in range(cfg.num_heads, 0, -1):
                    if target % h == 0:
                        valid_heads = [h]
                        break
            if not valid_heads:
                return None, None, "SHRINK_HIDDEN: tidak bisa align head"
            new_cfg.hidden_dim = target
            new_cfg.num_heads  = valid_heads[0]
            new_cfg.head_dim   = target // new_cfg.num_heads
            valid_kv = [h for h in range(1, new_cfg.num_heads + 1)
                        if new_cfg.num_heads % h == 0 and h <= new_cfg.num_heads]
            new_cfg.num_kv_heads = min(valid_kv,
                key=lambda h: abs(h - max(1, new_cfg.num_heads // 4)))
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"SHRINK_HIDDEN {cfg.hidden_dim}→{new_cfg.hidden_dim}"

        # ── FFN ───────────────────────────────────────────────────────────────

        elif action == "TUNE_FFN_UP":
            new_cfg = copy.deepcopy(cfg)
            if cfg.ffn_multiplier >= 4.25:
                return None, None, "TUNE_FFN_UP: FFN sudah optimal"
            target_ffn = min(4.25, cfg.ffn_multiplier + 0.25)
            ffn_dim = _align128(int(cfg.hidden_dim * target_ffn))
            new_cfg.ffn_multiplier = round(ffn_dim / max(1, cfg.hidden_dim), 4)
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"TUNE_FFN_UP {cfg.ffn_multiplier:.3f}→{new_cfg.ffn_multiplier:.3f}"

        elif action == "TUNE_FFN_DOWN":
            new_cfg = copy.deepcopy(cfg)
            if cfg.ffn_multiplier <= 2.75:
                return None, None, "TUNE_FFN_DOWN: FFN sudah minimal"
            target_ffn = max(2.75, cfg.ffn_multiplier - 0.25)
            ffn_dim = _align128(int(cfg.hidden_dim * target_ffn))
            new_cfg.ffn_multiplier = round(ffn_dim / max(1, cfg.hidden_dim), 4)
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"TUNE_FFN_DOWN {cfg.ffn_multiplier:.3f}→{new_cfg.ffn_multiplier:.3f}"

        # ── Attention ─────────────────────────────────────────────────────────

        elif action == "BALANCE_KV_HEADS":
            new_cfg = copy.deepcopy(cfg)
            target_kv = max(1, cfg.num_heads // 4)
            valid_kv  = [h for h in range(1, cfg.num_heads + 1)
                         if cfg.num_heads % h == 0]
            if not valid_kv:
                return None, None, "BALANCE_KV_HEADS: tidak ada valid kv_heads"
            best_kv = min(valid_kv, key=lambda h: abs(h - target_kv))
            if best_kv == cfg.num_kv_heads:
                return None, None, "BALANCE_KV_HEADS: sudah balanced"
            new_cfg.num_kv_heads = best_kv
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"BALANCE_KV_HEADS {cfg.num_kv_heads}→{best_kv}"

        elif action == "FIX_ATTN_GQA":
            new_cfg = copy.deepcopy(cfg)
            attn_str = (cfg.attn_type.value if hasattr(cfg.attn_type, 'value')
                        else str(cfg.attn_type).split('.')[-1])
            if "GQA" in attn_str.upper():
                return None, None, "FIX_ATTN_GQA: sudah GQA"
            new_cfg.attn_type = AttentionType.GQA
            valid_kv = [h for h in range(1, cfg.num_heads + 1)
                        if cfg.num_heads % h == 0 and h <= cfg.num_heads // 2]
            if valid_kv:
                new_cfg.num_kv_heads = valid_kv[-1]
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"FIX_ATTN_GQA {attn_str}→GQA kv={new_cfg.num_kv_heads}"

        # ── MoE ───────────────────────────────────────────────────────────────

        elif action == "REDUCE_MOE_EXPERTS":
            if cfg.num_experts <= 1:
                return None, None, "REDUCE_MOE_EXPERTS: bukan MoE"
            if cfg.vram_usage_pct <= 60:
                return None, None, "REDUCE_MOE_EXPERTS: VRAM tidak tertekan"
            new_cfg = copy.deepcopy(cfg)
            new_cfg.num_experts   = max(2, cfg.num_experts - 2)
            new_cfg.top_k_experts = max(1, min(new_cfg.top_k_experts, new_cfg.num_experts - 1))
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"REDUCE_MOE_EXPERTS {cfg.num_experts}→{new_cfg.num_experts}"

        elif action == "INCR_MOE_TOPK":
            if cfg.num_experts <= 1:
                return None, None, "INCR_MOE_TOPK: bukan MoE"
            if cfg.top_k_experts >= cfg.num_experts:
                return None, None, "INCR_MOE_TOPK: top_k sudah maksimum"
            new_cfg = copy.deepcopy(cfg)
            new_cfg.top_k_experts = min(cfg.num_experts, cfg.top_k_experts + 1)
            new_cfg = self._blender._finalize(new_cfg)
            desc = f"INCR_MOE_TOPK {cfg.top_k_experts}→{new_cfg.top_k_experts}"

        # ── Training flags ────────────────────────────────────────────────────

        elif action == "ENABLE_MIXED_PREC":
            if cfg.use_mixed_precision:
                return None, None, "ENABLE_MIXED_PREC: sudah aktif"
            new_cfg = copy.deepcopy(cfg)
            new_cfg.use_mixed_precision = True
            new_cfg = self._blender._finalize(new_cfg)
            desc = "ENABLE_MIXED_PREC: aktifkan BF16 mixed precision"

        elif action == "TIE_EMBEDDINGS":
            if cfg.tie_embeddings:
                return None, None, "TIE_EMBEDDINGS: sudah aktif"
            new_cfg = copy.deepcopy(cfg)
            new_cfg.tie_embeddings = True
            new_cfg = self._blender._finalize(new_cfg)
            desc = "TIE_EMBEDDINGS: tie embedding + LM head"

        else:
            return None, None, f"Unknown action: {action}"

        if not new_cfg.fits_gpu:
            return None, None, f"{action}: OOM setelah perturbasi"

        return new_cfg, new_spec, desc


# ══════════════════════════════════════════════════════════════════════════════
#  REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _compute_reward(
    delta_combo: float,
    delta_hw:    float,
    delta_train: float,
    old_scores:  Tuple[float, float, float],
    new_scores:  Tuple[float, float, float],
) -> float:
    """
    Reward function yang seimbang dan anti-bias.

    Rumus:
      base          = W_C×Δcombo + W_H×Δhw + W_T×Δtrain       (W ≈ 0.33 semua)
      synergy_bonus = +0.05 jika semua 3 naik, +0.02 jika 2 dari 3 naik
      balance_pen   = −penalti jika satu dimensi tertinggal > threshold (0.12)
      direction_pen = −penalti kecil jika reward negatif DAN improvement single-dim
                      (mencegah single-dim greediness)

    Prinsip:
      1. Bobot seimbang W=0.33: tidak ada dimensi yang lebih "berharga".
      2. synergy_bonus mendorong improvement yang merata di semua dimensi.
      3. balance_penalty mencegah arsitektur yang bagus di satu sisi
         tapi sangat buruk di dimensi lain.
      4. direction_penalty: jika satu dimensi sangat naik tapi lainnya
         turun signifikan, reward diperkecil untuk mencegah greedy
         single-dimension optimization.
    """
    # Dasar weighted reward — bobot benar-benar seimbang
    base = _W_COMBO * delta_combo + _W_HW * delta_hw + _W_TRAIN * delta_train

    # Synergy bonus: mendorong improvement yang merata
    n_improved = sum(1 for d in [delta_combo, delta_hw, delta_train] if d > 1e-4)
    if n_improved >= 3:
        synergy_bonus = 0.05
    elif n_improved >= 2:
        synergy_bonus = 0.02
    else:
        synergy_bonus = 0.0

    # Balance penalty: hukum jika satu dimensi jauh tertinggal dari rata-rata
    new_c, new_h, new_t = new_scores
    mean_new = (new_c + new_h + new_t) / 3.0
    balance_penalty = 0.0
    for s in [new_c, new_h, new_t]:
        gap = mean_new - s
        if gap > _BALANCE_PENALTY_THRESHOLD:
            balance_penalty += (gap - _BALANCE_PENALTY_THRESHOLD) * 0.35

    # Direction penalty: jika satu dim turun signifikan saat yang lain naik,
    # tambahkan penalti kecil untuk mencegah single-dim greediness
    n_dropped = sum(1 for d in [delta_combo, delta_hw, delta_train] if d < -0.03)
    direction_penalty = 0.0
    if n_dropped >= 1 and n_improved >= 1:
        direction_penalty = 0.01 * n_dropped   # kecil tapi konsisten

    return base + synergy_bonus - balance_penalty - direction_penalty


def _compute_combined(
    combo_score: float,
    hw_score:    float,
    train_score: float,
) -> float:
    """Combined = 33% combo + 34% hardware + 33% training."""
    return round(0.33 * combo_score + 0.34 * hw_score + 0.33 * train_score, 5)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _align64(x: int) -> int:
    """Round up ke kelipatan 64."""
    return max(64, ((x + 63) // 64) * 64)

def _align128(x: int) -> int:
    """Round up ke kelipatan 128."""
    return max(128, ((x + 127) // 128) * 128)

def _redistribute_ratios(spec: CombinationSpec, fixed_idx: int, remainder: float) -> None:
    """
    Distribusikan remainder ke semua family selain fixed_idx,
    proporsional terhadap rasio sebelumnya.
    """
    n = len(spec.ratios)
    if n <= 1:
        return
    others_sum = sum(spec.ratios[i] for i in range(n) if i != fixed_idx)
    others_sum = max(1e-9, others_sum)
    for i in range(n):
        if i != fixed_idx:
            spec.ratios[i] = round(spec.ratios[i] / others_sum * remainder, 4)
    # Re-normalize untuk pastikan sum = 1.0
    total = sum(spec.ratios)
    if abs(total - 1.0) > 1e-3:
        spec.ratios = [round(r / total, 4) for r in spec.ratios]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN REFINER
# ══════════════════════════════════════════════════════════════════════════════

class CombinationRefiner:
    """
    RL Refinement Engine khusus untuk Combination Architecture.

    Alur:
      Phase A: Generate n_candidates blends per spec, pilih yang terbaik
               berdasarkan combo_score awal.
      Phase B: ArcRefiner heuristic — koreksi formula + arsitektur fixes.
      Phase C: RL Combination Tuning (CombinationRLAgent + CombinationRLPerturber)
               → max_explore_iters steps per spec
               → cross-combination learning (agent tidak di-reset antar spec)
               → anti-stagnation burst explore
               → NAS proxy re-evaluation setiap proxy_every steps
      Phase D: Final NAS evaluation lengkap (hardware + training)
      Phase E: Final selection dari semua spec → 1 kombinasi terkuat

    Tidak bias ke family tertentu karena:
      1. State encoding berbasis skor (bukan nama family)
      2. Reward seimbang 3 dimensi + balance penalty
      3. Evaluator (CombinationNASEvaluator) berbasis skor formula,
         bukan hard-coded per family
    """

    def __init__(
        self,
        gpu:     GPUSpec,
        cfg:     Optional[CombinationRLConfig] = None,
        seed:    Optional[int] = None,
    ):
        self.gpu     = gpu
        self._cfg    = cfg or CombinationRLConfig()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._blender     = CombinationBlender(gpu, rng_seed=seed)
        self._combo_eval  = CombinationNASEvaluator(gpu)
        self._hw_eval     = HardwareNASEvaluator(gpu)
        self._proxy       = ProxyTrainer(device=self._cfg.proxy_device)
        self._train_eval  = TrainingDynamicsEvaluator(gpu)
        self._quality     = ArcQualityScorer(gpu)
        self._arc_refiner = ArcRefiner(gpu, max_iterations=25)
        self._perturber   = CombinationRLPerturber(gpu, rng_seed=seed)
        self._cache       = CombinationNASCache(max_size=512)

        # Agent SATU untuk semua kombinasi — cross-combination learning
        self._agent = CombinationRLAgent(self._cfg)

    # ── Cached evaluation ─────────────────────────────────────────────────────

    def _eval_combo_cached(
        self, cfg: ArchConfig, spec: CombinationSpec
    ) -> CombinationNASResult:
        hit = self._cache.get(cfg, spec)
        if hit is not None:
            return hit
        res = self._combo_eval.evaluate(cfg, spec)
        self._cache.put(cfg, spec, res)
        return res

    # ── Score triplet ─────────────────────────────────────────────────────────

    def _get_scores(
        self,
        cfg:  ArchConfig,
        spec: CombinationSpec,
        log:  CombinationRLLog,
        *,
        run_proxy: bool = True,
    ) -> Tuple[float, float, float, float]:
        """
        Dapatkan (combo_score, hw_score, train_score, combined).
        Jika run_proxy=False, pakai hanya combo + hw (lebih cepat).
        """
        combo_res = self._eval_combo_cached(cfg, spec)
        hw_res    = self._hw_eval.evaluate(cfg)

        if run_proxy and _TORCH_OK:
            t0 = time.perf_counter()
            proxy_res  = self._proxy.train(cfg)
            train_res  = self._train_eval.evaluate(cfg, proxy_res)
            log.proxy_eval_count += 1
            log.proxy_ms_total   += (time.perf_counter() - t0) * 1000
            if proxy_res.nan_detected:
                log.proxy_nan_count += 1
            train_score = train_res.training_score
        else:
            train_score = 0.50   # fallback netral

        combined = _compute_combined(
            combo_res.combination_score,
            hw_res.hardware_score,
            train_score,
        )
        return (
            combo_res.combination_score,
            hw_res.hardware_score,
            train_score,
            combined,
        )

    # ── Refine satu spec ──────────────────────────────────────────────────────

    def refine(
        self,
        spec: CombinationSpec,
    ) -> Tuple[ArchConfig, CombinationSpec, CombinationRLLog]:
        """
        Refine satu CombinationSpec menjadi arsitektur terkuat.

        Returns: (best_cfg, best_spec, log)
        """
        log = CombinationRLLog(spec_label=spec.label)
        self._agent.reset_episode()

        # ── Phase A: Generate candidates ──────────────────────────────────────
        candidates: List[Tuple[ArchConfig, CombinationSpec, CombinationNASResult]] = []
        for _ in range(self._cfg.n_candidates):
            try:
                c = self._blender.blend(spec)
                if c.fits_gpu:
                    cr = self._eval_combo_cached(c, spec)
                    candidates.append((c, spec, cr))
            except Exception:
                continue

        # Fallback ke WEIGHTED jika semua gagal
        if not candidates:
            fallback = copy.deepcopy(spec)
            fallback.strategy = BLEND_WEIGHTED
            try:
                c  = self._blender.blend(fallback)
                cr = self._eval_combo_cached(c, fallback)
                candidates.append((c, fallback, cr))
            except Exception:
                pass

        if not candidates:
            c  = self._blender._gen.generate_one(spec.family_a)
            cr = self._eval_combo_cached(c, spec)
            candidates.append((c, spec, cr))

        candidates.sort(key=lambda x: x[2].combination_score, reverse=True)
        best_cfg, best_spec, _ = candidates[0]

        # ── Phase B: Heuristic quality fixes ─────────────────────────────────
        best_cfg, base_log = self._arc_refiner.refine(best_cfg)
        log.arch_id  = best_cfg.arch_id
        log.arch_name = best_cfg.arch_name

        # Score awal
        c0, h0, t0, comb0 = self._get_scores(best_cfg, best_spec, log, run_proxy=True)
        log.combo_score_start  = c0
        log.hw_score_start     = h0
        log.train_score_start  = t0
        log.combined_start     = comb0

        best_c, best_h, best_t, best_comb = c0, h0, t0, comb0

        # State encoding awal
        best_state = CombinationRLState.encode(best_c, best_h, best_t, best_spec)

        # ── Phase C: RL Tuning ────────────────────────────────────────────────
        action_fail:  Dict[str, int] = {}
        no_improve    = 0
        T             = self._cfg.max_explore_iters

        for step in range(T):
            # Pilih aksi
            burst = (no_improve >= self._cfg.max_no_improve)
            act_idx = self._agent.select_action(
                best_c, best_h, best_t, best_spec,
                action_fail, force_untried=burst
            )
            action = COMB_RL_ACTIONS[act_idx]
            log.perturbation_tries += 1

            if burst:
                log.burst_explore_count += 1
                no_improve = 0   # reset setelah burst

            # Apply perturbasi
            new_cfg, new_spec, desc = self._perturber.apply(best_cfg, best_spec, action)

            if new_cfg is None:
                action_fail[action] = action_fail.get(action, 0) + 1
                # Hard penalti ke Q-table untuk aksi yang gagal berkali-kali
                if action_fail.get(action, 0) >= 3:
                    self._agent._q[best_state][act_idx] = max(
                        self._agent._q[best_state][act_idx] - 0.5, -3.0
                    )
                no_improve += 1
                if no_improve >= self._cfg.max_no_improve * 2:
                    break
                continue

            # Evaluasi — run proxy setiap N steps atau jika structural change
            is_structural = action in (
                "INCR_LAYERS", "DECR_LAYERS", "GROW_HIDDEN", "SHRINK_HIDDEN",
                "SWITCH_TO_INTERLEAVED", "SWITCH_TO_STAGED",
            )
            run_proxy_now = (
                is_structural or
                step % self._cfg.proxy_every == 0
            )

            new_c, new_h, new_t, new_comb = self._get_scores(
                new_cfg, new_spec, log, run_proxy=run_proxy_now
            )
            if not run_proxy_now:
                new_t = best_t   # pakai nilai training terakhir

            # Reward
            delta_c = new_c - best_c
            delta_h = new_h - best_h
            delta_t = new_t - best_t
            reward = _compute_reward(
                delta_c, delta_h, delta_t,
                (best_c, best_h, best_t),
                (new_c, new_h, new_t),
            )

            # Q-update
            new_state = CombinationRLState.encode(new_c, new_h, new_t, new_spec)
            td_error  = self._agent.update(best_state, act_idx, reward, new_state)
            self._agent.push_replay(best_state, act_idx, reward, new_state, td_error)
            n_rep = self._agent.replay_update(self._cfg.replay_batch)
            log.replay_updates += n_rep

            # Accept criterion — simetris untuk semua 3 dimensi:
            #   1. Combined naik secara keseluruhan (mode utama)
            #   2. Satu dimensi naik SIGNIFIKAN tanpa merusak dimensi lain > -0.03
            #      (threshold -0.03 SAMA untuk semua dimensi — tidak ada yang diistimewakan)
            #   3. Dua dimensi naik walau yang ketiga turun sedikit
            delta_comb = new_comb - best_comb

            # Mode 1: combined naik
            accept_combined = delta_comb > 5e-5

            # Mode 2: single-dim improvement (threshold simetris untuk semua dim)
            _tol = -0.03
            accept_single = (
                (delta_c > 3e-3 and delta_h > _tol and delta_t > _tol) or
                (delta_h > 3e-3 and delta_c > _tol and delta_t > _tol) or
                (delta_t > 3e-3 and delta_c > _tol and delta_h > _tol)
            )

            # Mode 3: dua dimensi naik bersama
            n_improved_now = sum(1 for d in [delta_c, delta_h, delta_t] if d > 1e-3)
            accept_multi   = (n_improved_now >= 2 and delta_comb > -1e-4)

            accept = accept_combined or accept_single or accept_multi

            # Anti-collapse: jangan accept jika satu dimensi turun drastis
            # Threshold SAMA untuk semua dimensi (-0.05) — tidak ada yang lebih dilindungi
            _collapse_tol = 0.05
            has_collapse = (
                new_c < best_c - _collapse_tol or
                new_h < best_h - _collapse_tol or
                new_t < best_t - _collapse_tol
            )

            if accept and not has_collapse:
                best_cfg, best_spec  = new_cfg, new_spec
                best_c, best_h, best_t, best_comb = new_c, new_h, new_t, new_comb
                best_state = new_state
                action_fail[action] = 0
                no_improve          = 0
                log.perturbations_accepted += 1

                # Counter per jenis aksi — tambah dimension-specific counter
                if "RATIO" in action:
                    log.ratio_adjustments += 1
                if "SWITCH" in action:
                    log.strategy_switches += 1
                if action in ("INCR_LAYERS", "DECR_LAYERS", "GROW_HIDDEN", "SHRINK_HIDDEN"):
                    log.structural_changes += 1

                # Catat dimensi mana yang paling benefit dari aksi ini
                dim_gains = []
                if delta_c > 1e-3: dim_gains.append(f"combo+{delta_c:.4f}")
                if delta_h > 1e-3: dim_gains.append(f"hw+{delta_h:.4f}")
                if delta_t > 1e-3: dim_gains.append(f"train+{delta_t:.4f}")
                dim_str = " | ".join(dim_gains) if dim_gains else "combined+"

                log.improvement_events.append(
                    f"[step{step+1:02d}] {action:<30} "
                    f"combo={best_c:.4f} hw={best_h:.4f} train={best_t:.4f} "
                    f"combined={best_comb:.5f} (Δ{delta_comb:+.5f}) [{dim_str}]"
                )
            else:
                action_fail[action] = action_fail.get(action, 0) + 1
                no_improve += 1

        # ── Phase D: Final NAS evaluation ─────────────────────────────────────
        # Evaluasi penuh — tidak pakai cache untuk akurasi maksimal
        final_combo = self._combo_eval.evaluate(best_cfg, best_spec)
        final_hw    = self._hw_eval.evaluate(best_cfg)

        if _TORCH_OK:
            t_start = time.perf_counter()
            final_proxy = self._proxy.train(best_cfg)
            final_train = self._train_eval.evaluate(best_cfg, final_proxy)
            log.proxy_eval_count += 1
            log.proxy_ms_total   += (time.perf_counter() - t_start) * 1000
            if final_proxy.nan_detected:
                log.proxy_nan_count += 1
            final_train_score = final_train.training_score
        else:
            final_train_score = best_t

        final_combined = _compute_combined(
            final_combo.combination_score,
            final_hw.hardware_score,
            final_train_score,
        )

        # Isi log akhir
        log.families      = list(best_spec.families)
        log.ratios        = list(best_spec.ratios)
        log.strategy      = best_spec.strategy
        log.compatibility = best_spec.compatibility
        log.synergy_mult  = best_spec.synergy_mult

        log.combo_score_end  = final_combo.combination_score
        log.hw_score_end     = final_hw.hardware_score
        log.train_score_end  = final_train_score
        log.combined_end     = final_combined
        log.quality_end      = self._quality.score(best_cfg).pct

        log.pts_c1 = final_combo.pts_c1
        log.pts_c2 = final_combo.pts_c2
        log.pts_c3 = final_combo.pts_c3
        log.pts_c4 = final_combo.pts_c4
        log.pts_c5 = final_combo.pts_c5

        log.warnings = final_combo.warnings

        if log.perturbations_accepted > 0:
            log.status = f"↑ IMPROVED Δ{log.combined_delta:+.5f} ({log.perturbations_accepted} accepted)"
        else:
            log.status = "~ STAGNATED (no improvement found)"

        return best_cfg, best_spec, log

    # ── Refine batch + final selection ────────────────────────────────────────

    def refine_batch(
        self,
        specs: List[CombinationSpec],
    ) -> Tuple[List[ArchConfig], List[CombinationSpec], List[CombinationRLLog]]:
        """
        Refine semua specs.
        Returns (archs, specs, logs) diurut descending combined_end.
        """
        archs_out: List[ArchConfig]       = []
        specs_out: List[CombinationSpec]  = []
        logs_out:  List[CombinationRLLog] = []

        for spec in specs:
            cfg, final_spec, log = self.refine(spec)
            archs_out.append(cfg)
            specs_out.append(final_spec)
            logs_out.append(log)

        # Sort descending
        triplets = sorted(
            zip(archs_out, specs_out, logs_out),
            key=lambda x: x[2].combined_end,
            reverse=True,
        )
        if triplets:
            archs_out, specs_out, logs_out = zip(*triplets)
            return list(archs_out), list(specs_out), list(logs_out)
        return [], [], []

    def refine_to_best(
        self,
        specs:        List[CombinationSpec],
        n_candidates: int = 3,
    ) -> Tuple[ArchConfig, CombinationSpec, CombinationRLLog]:
        """
        Refine semua specs dan kembalikan 1 kombinasi terkuat.

        Ini adalah entry point utama dari pipeline.py untuk mode combination
        yang menghasilkan SATU arsitektur final terbaik.

        Returns: (best_cfg, best_spec, best_log)
        """
        self._cfg.n_candidates = n_candidates
        archs, final_specs, logs = self.refine_batch(specs)

        if not archs:
            # Absolute fallback
            spec = specs[0] if specs else CombinationSpec(
                families=["Balanced-Pro", "CoT-Optimizer"],
                ratios=[0.5, 0.5],
                strategy=BLEND_WEIGHTED,
            )
            cfg = self._blender._gen.generate_one(spec.family_a)
            log = CombinationRLLog(
                arch_id=cfg.arch_id,
                arch_name=cfg.arch_name,
                spec_label=spec.label,
                status="FALLBACK: semua spec gagal",
            )
            return cfg, spec, log

        # Sudah di-sort, ambil #1
        return archs[0], final_specs[0], logs[0]


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_rl_log(log: CombinationRLLog, *, console=None) -> None:
    """Print detail log RL refinement untuk satu combination."""
    _p = console.print if console else print
    W  = 88
    S  = "─" * W

    _p()
    _p(f"╭{S}╮")
    _p(f"│{'  🔀 COMBINATION RL LOG':^{W}}│")
    _p(f"├{S}┤")
    _p(f"│  ARC: {log.arch_id:<10}  {log.arch_name[:56]:<56}  │")
    fam_str = " + ".join(
        f"{f.split('-')[0]}:{int(r*100)}%"
        for f, r in zip(log.families, log.ratios)
    )
    _p(f"│  Combo: {fam_str[:75]:<75}  │")
    _p(f"│  Strategy: {log.strategy:<12}  Compat: {log.compatibility:<18}  "
       f"Syn: {log.synergy_mult:.3f}  │")
    _p(f"├{S}┤")
    _p(f"│  {'SCORES':^{W}}│")
    _p(f"│  {'':4}{'Combo':>10}{'HW':>10}{'Train':>10}{'Combined':>12}  │")
    _p(f"│  {'Start':4}{log.combo_score_start:>10.4f}{log.hw_score_start:>10.4f}"
       f"{log.train_score_start:>10.4f}{log.combined_start:>12.5f}  │")
    _p(f"│  {'End':4}{log.combo_score_end:>10.4f}{log.hw_score_end:>10.4f}"
       f"{log.train_score_end:>10.4f}{log.combined_end:>12.5f}  │")
    delta_c = log.combo_score_end - log.combo_score_start
    delta_h = log.hw_score_end    - log.hw_score_start
    delta_t = log.train_score_end - log.train_score_start
    _p(f"│  {'Δ':4}{delta_c:>+10.4f}{delta_h:>+10.4f}"
       f"{delta_t:>+10.4f}{log.combined_delta:>+12.5f}  │")
    _p(f"├{S}┤")
    _p(f"│  {'COMBINATION SCORE (5 dimensi)':^{W}}│")
    _p(f"│  C1 Family Coherence      : {log.pts_c1:>5.1f}/25  │")
    _p(f"│  C2 Blend Balance         : {log.pts_c2:>5.1f}/20  │")
    _p(f"│  C3 Architectural Synergy : {log.pts_c3:>5.1f}/20  │")
    _p(f"│  C4 Hardware Compat       : {log.pts_c4:>5.1f}/20  │")
    _p(f"│  C5 Training Synergy      : {log.pts_c5:>5.1f}/15  │")
    total_pts = log.pts_c1 + log.pts_c2 + log.pts_c3 + log.pts_c4 + log.pts_c5
    _p(f"│  Total: {total_pts:.1f}/100  Grade: {log.grade:<35}  │")
    _p(f"├{S}┤")
    _p(f"│  {'RL STATISTIK':^{W}}│")
    _p(f"│  Tries: {log.perturbation_tries:<5}  Accepted: {log.perturbations_accepted:<5}"
       f"  Accept-rate: {log.accept_rate:.1%}  │")
    _p(f"│  Ratio-adj: {log.ratio_adjustments:<4}  Strategy-sw: {log.strategy_switches:<4}"
       f"  Structural: {log.structural_changes:<4}  Burst: {log.burst_explore_count:<4}  │")
    _p(f"│  Replay-updates: {log.replay_updates:<6}  "
       f"Proxy-evals: {log.proxy_eval_count:<5}  "
       f"NaN: {log.proxy_nan_count:<3}  "
       f"Proxy-ms: {log.proxy_ms_total:.0f}  │")
    _p(f"│  Quality: {log.quality_end:.1f}%  │")
    if log.improvement_events:
        _p(f"├{S}┤")
        _p(f"│  {'IMPROVEMENT EVENTS (top 6)':^{W}}│")
        for ev in log.improvement_events[:6]:
            _p(f"│  {ev[:W-4]:<{W-4}}  │")
    if log.warnings:
        _p(f"├{S}┤")
        _p(f"│  {'⚠ WARNINGS':^{W}}│")
        for w in log.warnings[:3]:
            _p(f"│  {w[:W-4]:<{W-4}}  │")
    _p(f"│  Status: {log.status[:W-12]:<{W-12}}  │")
    _p(f"╰{S}╯")
    _p()


def print_rl_summary(
    archs:  List[ArchConfig],
    specs:  List[CombinationSpec],
    logs:   List[CombinationRLLog],
    *,
    console=None,
) -> None:
    """Print tabel ringkasan semua combination results."""
    _p = console.print if console else print

    _p()
    _p("  ┌─ Combination RL Summary ─────────────────────────────────────────────")
    _p("  │  Combined = 33% combo + 34% hardware + 33% training")
    _p("  │")
    _p(f"  │  {'Rank':<4}  {'ARC-ID':<10}  {'Families':<36}  "
       f"{'Combo':>7}  {'HW':>7}  {'Train':>7}  {'Combined':>9}  Status")
    _p("  │  " + "─" * 105)

    # Sudah di-sort descending
    for rank, (cfg, spec, log) in enumerate(zip(archs, specs, logs), 1):
        sym     = "★" if rank == 1 else f"#{rank}"
        fam_lbl = "+".join(f.split("-")[0][:5] for f in spec.families)
        fam_lbl += f"[{spec.strategy[:3]}]"

        _p(f"  │  {sym:<4}  {log.arch_id:<10}  {fam_lbl:<36}  "
           f"{log.combo_score_end:>7.4f}  {log.hw_score_end:>7.4f}  "
           f"{log.train_score_end:>7.4f}  {log.combined_end:>9.5f}  "
           f"{log.status[:30]}")

    _p("  │")
    _p("  │  ★ = Best combination — highest balanced 3-way score")
    _p("  └─────────────────────────────────────────────────────────────────────")
    _p()
