"""
arc_type_generator.py — Interactive Arc Type Selector & NAS Combination Evaluator
═══════════════════════════════════════════════════════════════════════════════════════

File baru pelengkap combination_nas.py & combination_refiner.py yang menyediakan:

  1. ask_arc_type(gpu)
       Menu interaktif: user memilih 1 dari 7 AI type yang tersedia.
       Sistem generate ArchConfig untuk type tersebut, lalu evaluasi
       lengkap (HW + Training + Kualitas) → hasilnya bisa dipakai sebagai
       baseline atau di-blend ke CombinationSpec.

  2. generate_arc_from_type(family, gpu, *, device, seed)
       Generate ArchConfig optimal untuk 1 AI type via ArchitectureGenerator
       + ArcRefiner quality heuristic fixes.

  3. NASCombinationResultScorer
       Scorer komprehensif untuk menilai HASIL combination NAS.
       Tidak bias ke satu type — semua penilaian berbasis metrik struktural
       dan skor hardware/training, bukan identitas family.
       5 dimensi yang diperluas dengan sub-breakdown per dimensi.

  4. compare_combination_results(results, gpu)
       Bandingkan beberapa (cfg, spec) hasil combination NAS dan pilih
       yang terbaik berdasarkan combined score yang seimbang.

  5. print_arc_type_report(cfg, hw_res, train_res, combo_res, spec)
       Print laporan lengkap satu arsitektur + evaluasi 3-way.

Prinsip anti-bias:
  - Semua scoring berbasis properti struktural dan kinerja, BUKAN nama family.
  - Reward kombinasi menggunakan 3 bobot yang seimbang (W_C=W_H=W_T ≈ 0.33).
  - Tidak ada hard-coded bonus/penalti per family.

Usage:
  from arc_type_generator import ask_arc_type, NASCombinationResultScorer
  cfg = ask_arc_type(gpu)                             # interactive
  scorer = NASCombinationResultScorer(gpu)
  report = scorer.score_full(cfg, spec)               # full eval
  scorer.print_report(report)
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from arch_types import ArchConfig, AttentionType, FFNType, OptimizerType, NormType
from hardware import GPUSpec
from generator import ArchitectureGenerator
from hardware_refine import HardwareNASEvaluator, HardwareNASResult
from train_refine import (
    ProxyTrainer, TrainingDynamicsEvaluator,
    TrainingNASResult, ProxyTrainingResult,
)
from combination_nas import (
    CombinationSpec,
    CombinationNASResult,
    CombinationNASEvaluator,
    CombinationBlender,
    ALL_FAMILIES,
    FAMILY_DESCRIPTIONS,
    BLEND_INTERLEAVED,
    BLEND_STAGED,
    BLEND_WEIGHTED,
    _SYNERGY_NORMALIZED,
)
from refiner import ArcQualityScorer, ArcRefiner, RefinementLog

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA
# ══════════════════════════════════════════════════════════════════════════════

# Bobot combined score — SEIMBANG, tidak ada dominasi
W_COMBO  = 0.33
W_HW     = 0.34
W_TRAIN  = 0.33

# Threshold grade — disesuaikan dengan skala scoring yang lebih ketat
# Kombinasi TIDAK PERNAH bisa mendapat "S" dari combination_score saja
# (max combination_score ≈ 0.82 untuk n=2, 0.76 untuk n=3)
# Grade ini berlaku untuk combined 3-way score (hw+train+combo)
_GRADE_THRESHOLDS = [
    (0.80, "S  ★★★  Exceptional"),
    (0.70, "A+ ★★   Excellent"),
    (0.60, "A  ★    Very Good"),
    (0.48, "B       Good"),
    (0.36, "C       Acceptable"),
    (0.24, "D       Marginal"),
]

# Extended family metadata: guidance tanpa bias apapun
_FAMILY_GUIDE: Dict[str, Dict] = {
    "CoT-Optimizer": {
        "abbr": "CoT",
        "icon": "🧠",
        "tagline": "Deep narrow — chain-of-thought & multi-step reasoning",
        "strengths": ["reasoning depth", "complex tasks", "logical chains"],
        "tradeoffs": ["slower throughput", "higher VRAM per token"],
        "typical_params": "3B–30B",
        "use_case": "math, code, logic, research",
    },
    "Speed-Demon": {
        "abbr": "Spd",
        "icon": "⚡",
        "tagline": "Wide shallow — maximum throughput & low latency",
        "strengths": ["tokens/sec", "batch efficiency", "real-time tasks"],
        "tradeoffs": ["lower reasoning depth", "less context handling"],
        "typical_params": "500M–7B",
        "use_case": "chat, classification, summarization at scale",
    },
    "Balanced-Pro": {
        "abbr": "Bal",
        "icon": "⚖️",
        "tagline": "Balanced depth/width — best general-purpose baseline",
        "strengths": ["versatility", "stable training", "broad capability"],
        "tradeoffs": ["not specialized for any one task"],
        "typical_params": "1B–13B",
        "use_case": "general assistant, instruction following, QA",
    },
    "MoE-Sparse": {
        "abbr": "MoE",
        "icon": "🔀",
        "tagline": "Mixture-of-Experts — sparse activation, high capacity",
        "strengths": ["high param count at low inference cost", "specialization"],
        "tradeoffs": ["complex routing", "needs more VRAM for experts"],
        "typical_params": "7B–70B (sparse params)",
        "use_case": "specialized tasks, multi-domain, large-scale serving",
    },
    "Long-Horizon": {
        "abbr": "LH",
        "icon": "📜",
        "tagline": "Extended context — long sequences & document understanding",
        "strengths": ["long context window", "document analysis", "memory"],
        "tradeoffs": ["attention is O(n²)", "high KV cache VRAM"],
        "typical_params": "3B–13B",
        "use_case": "RAG, document Q&A, code review, long conversations",
    },
    "Nano-Efficient": {
        "abbr": "Nano",
        "icon": "🔬",
        "tagline": "Ultra-small — max quality per VRAM byte for edge/mobile",
        "strengths": ["minimal VRAM", "fast on CPU/mobile", "deployable anywhere"],
        "tradeoffs": ["limited capacity", "lower ceiling on complex tasks"],
        "typical_params": "50M–1B",
        "use_case": "on-device AI, edge deployment, embedded systems",
    },
    "Compute-Dense": {
        "abbr": "Cmp",
        "icon": "💪",
        "tagline": "Compute heavy — high arithmetic intensity & FLOPs utilization",
        "strengths": ["GPU compute saturation", "high MFU", "dense matrix ops"],
        "tradeoffs": ["high training cost", "needs powerful GPU"],
        "typical_params": "7B–65B",
        "use_case": "pre-training, compute-bound fine-tuning, research",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArcTypeEvalResult:
    """
    Hasil evaluasi lengkap satu ArchConfig dari arc type yang dipilih.

    Berisi 3 sinyal evaluasi yang seimbang:
      hw_score    — dari HardwareNASEvaluator
      train_score — dari TrainingDynamicsEvaluator + ProxyTrainer
      quality_pct — dari ArcQualityScorer (heuristik formula)
    Plus:
      combined    — W_C×hw + W_H×train + W_T×quality_normalized
      grade       — S/A+/A/B/C/D
    """
    arch_id:        str   = ""
    arch_name:      str   = ""
    family:         str   = ""

    # 3 sinyal utama
    hw_score:       float = 0.0
    train_score:    float = 0.0
    quality_pct:    float = 0.0   # 0–100

    # Combined
    combined:       float = 0.0
    grade:          str   = ""

    # Sub-breakdown HW
    hw_vram_pct:    float = 0.0
    hw_mfu:         float = 0.0
    hw_tps:         float = 0.0
    hw_bottleneck:  str   = ""

    # Sub-breakdown training
    train_stability: float = 0.0
    train_convergence: float = 0.0
    proxy_nan:       bool  = False

    # Arch details
    param_count_m:  float = 0.0
    num_layers:     int   = 0
    hidden_dim:     int   = 0
    num_heads:      int   = 0
    ffn_multiplier: float = 0.0
    seq_len:        int   = 0
    fits_gpu:       bool  = True

    # Timing
    eval_ms:        float = 0.0
    warnings:       List[str] = field(default_factory=list)

    @property
    def quality_norm(self) -> float:
        """Quality score dinormalisasi ke [0, 1]."""
        return self.quality_pct / 100.0

    def _compute_grade(self) -> str:
        for thresh, label in _GRADE_THRESHOLDS:
            if self.combined >= thresh:
                return label
        return "F  ✗    Poor"


@dataclass
class CombinationScoreReport:
    """
    Laporan lengkap skor satu hasil combination NAS.

    Dihasilkan oleh NASCombinationResultScorer.score_full().
    Berisi:
      • Skor 5 dimensi (C1–C5) dengan breakdown per sub-komponen
      • 3-way score (combo, hw, train)
      • Combined score dan grade
      • Rekomendasi improvement
    """
    arch_id:    str = ""
    spec_label: str = ""

    # 3-way scores
    combo_score:  float = 0.0
    hw_score:     float = 0.0
    train_score:  float = 0.0
    combined:     float = 0.0
    grade:        str   = ""

    # C1–C5 breakdown
    pts_c1:  float = 0.0    # Family Coherence /25
    pts_c2:  float = 0.0    # Blend Balance    /20
    pts_c3:  float = 0.0    # Arch Synergy     /20
    pts_c4:  float = 0.0    # HW Compat        /20
    pts_c5:  float = 0.0    # Train Synergy    /15
    total_pts: float = 0.0  # /100

    # Sub-breakdown C2
    c2_vram_score:    float = 0.0
    c2_ratio_balance: float = 0.0
    c2_param_score:   float = 0.0

    # Sub-breakdown C3
    c3_dw_score:  float = 0.0
    c3_ffn_score: float = 0.0
    c3_attn_ok:   bool  = True

    # Sub-breakdown C5
    c5_opt_score:   float = 0.0
    c5_mp_score:    float = 0.0
    c5_strat_bonus: float = 0.0

    # Balance indicator: apakah 3 dimensi seimbang?
    balance_ok:       bool  = True
    weakest_dim:      str   = ""
    weakest_score:    float = 0.0
    improvement_tips: List[str] = field(default_factory=list)
    warnings:         List[str] = field(default_factory=list)

    # Arch details
    param_count_m: float = 0.0
    num_layers:    int   = 0
    hidden_dim:    int   = 0
    vram_pct:      float = 0.0
    families:      List[str] = field(default_factory=list)
    ratios:        List[float] = field(default_factory=list)
    strategy:      str   = ""
    synergy_mult:  float = 0.0
    compatibility: str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE ARC TYPE MENU
# ══════════════════════════════════════════════════════════════════════════════

def ask_arc_type(
    gpu:    GPUSpec,
    *,
    device: str  = "cpu",
    seed:   Optional[int] = None,
    run_eval: bool = True,
) -> Optional[ArchConfig]:
    """
    Tampilkan menu interaktif 7 AI type. User pilih 1 type →
    generate ArchConfig → evaluasi NAS → return cfg.

    Args:
      gpu       : GPUSpec target
      device    : device untuk proxy training ('cpu'/'cuda')
      seed      : random seed (None = non-deterministic)
      run_eval  : jika True, jalankan full evaluation dan print laporan

    Returns:
      ArchConfig yang sudah di-generate dan di-evaluate,
      atau None jika user membatalkan.
    """
    print()
    print("  ╭─ Arc Type Generator ─────────────────────────────────────────────────────╮")
    print("  │  Pilih 1 AI Architecture Type untuk di-generate dan di-evaluasi NAS.     │")
    print("  │  Sistem akan menghasilkan ArchConfig terkuat dari type yang dipilih.      │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print(f"  │  GPU target: {gpu.name:<20}  VRAM: {gpu.vram_gb:.0f}GB                          │")
    print("  ├──────────────────────────────────────────────────────────────────────────┤")
    print("  │  No  Icon  Type              Tagline                                      │")
    print("  │  ──  ────  ────────────────  ────────────────────────────────────────    │")

    for i, fam in enumerate(ALL_FAMILIES, 1):
        g = _FAMILY_GUIDE.get(fam, {})
        icon    = g.get("icon", "•")
        tagline = g.get("tagline", "")[:46]
        abbr    = g.get("abbr", fam[:4])
        print(f"  │  [{i:1d}]  {icon}   {fam:<18}  {tagline:<46}  │")

    print("  ╰──────────────────────────────────────────────────────────────────────────╯")
    print()

    # Input loop
    while True:
        raw = input(
            f"  ▶  Pilih nomor type (1–{len(ALL_FAMILIES)}, atau Enter untuk batal): "
        ).strip()

        if not raw:
            print("  ✗ Dibatalkan.")
            return None

        try:
            choice = int(raw)
            if 1 <= choice <= len(ALL_FAMILIES):
                break
            print(f"  ✗ Nomor tidak valid. Masukkan 1–{len(ALL_FAMILIES)}.")
        except ValueError:
            print(f"  ✗ Masukkan angka. Contoh: '3'")

    family = ALL_FAMILIES[choice - 1]
    guide  = _FAMILY_GUIDE.get(family, {})

    print()
    print(f"  ✓ Type dipilih: {guide.get('icon', '')}  {family}")
    print(f"    {guide.get('tagline', '')}")
    print(f"    Strengths  : {', '.join(guide.get('strengths', []))}")
    print(f"    Tradeoffs  : {', '.join(guide.get('tradeoffs', []))}")
    print(f"    Use case   : {guide.get('use_case', '')}")
    print(f"    Typical params: {guide.get('typical_params', '?')}")
    print()
    print(f"  ► Generating {family} architecture...")

    cfg = generate_arc_from_type(family, gpu, device=device, seed=seed)

    if run_eval:
        print(f"  ► Evaluating...")
        t0  = time.perf_counter()
        res = evaluate_arc_type(cfg, gpu, family=family, device=device)
        res.eval_ms = (time.perf_counter() - t0) * 1000
        print_arc_type_report(res, cfg)

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_arc_from_type(
    family: str,
    gpu:    GPUSpec,
    *,
    device:       str  = "cpu",
    seed:         Optional[int] = None,
    apply_refine: bool = True,
    n_candidates: int  = 3,
) -> ArchConfig:
    """
    Generate ArchConfig terkuat untuk 1 AI type.

    Steps:
      1. Generate n_candidates via ArchitectureGenerator
      2. Pilih kandidat dengan fitness_score tertinggi
      3. Opsional: apply ArcRefiner untuk quality heuristic fixes
      4. Return ArchConfig final

    Args:
      family        : nama family dari ALL_FAMILIES
      gpu           : GPUSpec target
      device        : tidak dipakai di sini tapi diteruskan ke trainer bila perlu
      seed          : random seed
      apply_refine  : True → jalankan ArcRefiner (heuristic quality fixes)
      n_candidates  : jumlah kandidat yang dihasilkan sebelum memilih terbaik

    Returns:
      ArchConfig yang fits GPU dan memiliki skor terbaik.
    """
    if family not in ALL_FAMILIES:
        family = "Balanced-Pro"   # safe fallback

    gen = ArchitectureGenerator(gpu, rng_seed=seed)

    # Generate n_candidates dan pilih terbaik berdasarkan fitness_score
    best_cfg:   Optional[ArchConfig] = None
    best_score: float = -1.0

    for _ in range(n_candidates):
        try:
            cfg = gen.generate_one(family)
            if cfg.fits_gpu and cfg.fitness_score > best_score:
                best_cfg   = cfg
                best_score = cfg.fitness_score
        except Exception:
            continue

    # Fallback: generate sekali tanpa seleksi
    if best_cfg is None:
        try:
            best_cfg = gen.generate_one(family)
        except Exception:
            best_cfg = gen.generate_one("Balanced-Pro")

    # Optional quality fixes via ArcRefiner
    if apply_refine:
        try:
            refiner = ArcRefiner(gpu, max_iterations=20)
            best_cfg, _ = refiner.refine(best_cfg)
        except Exception:
            pass   # silently skip jika ArcRefiner gagal

    return best_cfg


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_arc_type(
    cfg:    ArchConfig,
    gpu:    GPUSpec,
    *,
    family: str  = "",
    device: str  = "cpu",
) -> ArcTypeEvalResult:
    """
    Evaluasi NAS lengkap untuk satu ArchConfig.

    Menjalankan 3 evaluator secara berurutan:
      1. HardwareNASEvaluator   → hardware_score
      2. ProxyTrainer + TrainingDynamicsEvaluator → training_score
      3. ArcQualityScorer       → quality_pct

    Semua skor dinormalisasi ke [0, 1] sebelum digabung.
    Combined = W_HW × hw + W_TRAIN × train + W_QUALITY × quality_norm
    """
    res          = ArcTypeEvalResult()
    res.arch_id  = cfg.arch_id
    res.arch_name = cfg.arch_name
    res.family   = family or getattr(cfg, "arch_family", "Unknown")

    # Arch details
    res.param_count_m = round(cfg.param_count / 1e6, 2)
    res.num_layers    = cfg.num_layers
    res.hidden_dim    = cfg.hidden_dim
    res.num_heads     = cfg.num_heads
    res.ffn_multiplier = cfg.ffn_multiplier
    res.seq_len       = cfg.seq_len
    res.fits_gpu      = cfg.fits_gpu
    res.hw_vram_pct   = cfg.vram_usage_pct
    res.hw_mfu        = cfg.mfu_estimate
    res.hw_tps        = cfg.tokens_per_sec_estimate
    res.hw_bottleneck = getattr(cfg, "bottleneck", "unknown")

    # ── 1. Hardware eval ──────────────────────────────────────────────────────
    try:
        hw_eval      = HardwareNASEvaluator(gpu)
        hw_result    = hw_eval.evaluate(cfg)
        res.hw_score = hw_result.hardware_score
    except Exception as e:
        res.warnings.append(f"HW eval gagal: {e}")
        res.hw_score = 0.40   # fallback netral

    # ── 2. Training eval ─────────────────────────────────────────────────────
    if _TORCH_OK:
        try:
            proxy       = ProxyTrainer(device=device)
            proxy_res   = proxy.train(cfg)
            train_eval  = TrainingDynamicsEvaluator(gpu)
            train_res   = train_eval.evaluate(cfg, proxy_res)
            res.train_score  = train_res.training_score
            res.train_stability   = getattr(train_res, "stability_score", res.train_score)
            res.train_convergence = getattr(train_res, "convergence_score", res.train_score)
            res.proxy_nan        = proxy_res.nan_detected
        except Exception as e:
            res.warnings.append(f"Training eval gagal: {e}")
            res.train_score = 0.50
    else:
        res.train_score = 0.50
        res.warnings.append("PyTorch tidak tersedia — training score: fallback 0.50")

    # ── 3. Quality scorer ─────────────────────────────────────────────────────
    try:
        quality     = ArcQualityScorer(gpu)
        q_result    = quality.score(cfg)
        res.quality_pct = q_result.pct
    except Exception as e:
        res.warnings.append(f"Quality scorer gagal: {e}")
        res.quality_pct = 60.0   # fallback netral

    # ── Combined — seimbang 3-way ─────────────────────────────────────────────
    # Catatan: quality_norm sudah [0,1]; combined memakai bobot W_COMBO/W_HW/W_TRAIN
    # Di sini: combo = quality_norm (karena tidak ada CombinationSpec)
    res.combined = round(
        W_HW    * res.hw_score +
        W_TRAIN * res.train_score +
        W_COMBO * res.quality_norm,
        5
    )
    res.grade = res._compute_grade()

    return res


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINATION RESULT SCORER (TIDAK BIAS KE TYPE APAPUN)
# ══════════════════════════════════════════════════════════════════════════════

class NASCombinationResultScorer:
    """
    Scorer komprehensif untuk menilai hasil combination NAS.

    Tidak bias ke type manapun karena semua penilaian berbasis:
      • Properti struktural arsitektur (depth/width ratio, FFN range, dst)
      • Skor performa terukur (hardware, training)
      • Keseimbangan 3-way: combo + hw + train harus merata

    Usage:
      scorer = NASCombinationResultScorer(gpu)
      report = scorer.score_full(cfg, spec, device="cpu")
      scorer.print_report(report)
    """

    def __init__(self, gpu: GPUSpec):
        self.gpu        = gpu
        self._hw_eval   = HardwareNASEvaluator(gpu)
        self._combo_eval = CombinationNASEvaluator(gpu)

    # ── Main entry point ──────────────────────────────────────────────────────

    def score_full(
        self,
        cfg:    ArchConfig,
        spec:   CombinationSpec,
        *,
        device: str = "cpu",
    ) -> CombinationScoreReport:
        """
        Evaluasi lengkap satu kombinasi.
        Menjalankan semua 3 evaluator dan mengisi CombinationScoreReport.
        """
        report             = CombinationScoreReport()
        report.arch_id     = cfg.arch_id
        report.spec_label  = spec.label
        report.families    = list(spec.families)
        report.ratios      = list(spec.ratios)
        report.strategy    = spec.strategy
        report.synergy_mult = spec.synergy_mult
        report.compatibility = spec.compatibility
        report.param_count_m = round(cfg.param_count / 1e6, 2)
        report.num_layers  = cfg.num_layers
        report.hidden_dim  = cfg.hidden_dim
        report.vram_pct    = cfg.vram_usage_pct

        # ── Combination NAS eval ──────────────────────────────────────────────
        combo_res = self._combo_eval.evaluate(cfg, spec)
        report.combo_score = combo_res.combination_score
        report.pts_c1      = combo_res.pts_c1
        report.pts_c2      = combo_res.pts_c2
        report.pts_c3      = combo_res.pts_c3
        report.pts_c4      = combo_res.pts_c4
        report.pts_c5      = combo_res.pts_c5
        report.total_pts   = combo_res.total_pts
        report.warnings    = list(combo_res.warnings)

        # ── Hardware eval ─────────────────────────────────────────────────────
        hw_res = self._hw_eval.evaluate(cfg)
        report.hw_score = hw_res.hardware_score

        # ── Training eval ─────────────────────────────────────────────────────
        if _TORCH_OK:
            try:
                proxy      = ProxyTrainer(device=device)
                proxy_res  = proxy.train(cfg)
                train_eval = TrainingDynamicsEvaluator(self.gpu)
                train_res  = train_eval.evaluate(cfg, proxy_res)
                report.train_score = train_res.training_score
            except Exception as e:
                report.warnings.append(f"Training eval gagal: {e}")
                report.train_score = 0.50
        else:
            report.train_score = 0.50

        # ── Sub-breakdown C2 ─────────────────────────────────────────────────
        report.c2_vram_score    = self._sub_c2_vram(cfg)
        report.c2_ratio_balance = self._sub_c2_ratio(spec)
        report.c2_param_score   = self._sub_c2_param(cfg)

        # ── Sub-breakdown C3 ─────────────────────────────────────────────────
        dw = cfg.num_layers / max(1.0, math.sqrt(max(1, cfg.hidden_dim)))
        report.c3_dw_score  = 1.0 if 0.10 <= dw <= 0.60 else max(0.3, 1.0 - abs(dw - 0.35))
        ffn = cfg.ffn_multiplier
        report.c3_ffn_score = 1.0 if 2.5 <= ffn <= 5.5 else (0.70 if 1.5 <= ffn < 2.5 else 0.50)
        attn_str = cfg.attn_type.value if hasattr(cfg.attn_type, "value") else str(cfg.attn_type)
        report.c3_attn_ok   = "GQA" in attn_str.upper() or "HYBRID" in attn_str.upper()

        # ── Sub-breakdown C5 ─────────────────────────────────────────────────
        opt = cfg.optimizer_type
        report.c5_opt_score = (
            1.0 if opt in (OptimizerType.ADAM_FP32, OptimizerType.ADAMW_BF16) else
            0.75 if opt == OptimizerType.ADAM_8BIT else 0.65
        )
        report.c5_mp_score    = 1.0 if cfg.use_mixed_precision else 0.60
        strat_bonus = {BLEND_WEIGHTED: 0.05, BLEND_STAGED: 0.03, BLEND_INTERLEAVED: 0.0}
        report.c5_strat_bonus = strat_bonus.get(spec.strategy, 0.0)

        # ── Combined — 3 bobot seimbang ───────────────────────────────────────
        report.combined = round(
            W_COMBO * report.combo_score +
            W_HW    * report.hw_score    +
            W_TRAIN * report.train_score,
            5
        )

        # ── Grade ─────────────────────────────────────────────────────────────
        for thresh, label in _GRADE_THRESHOLDS:
            if report.combined >= thresh:
                report.grade = label
                break
        else:
            report.grade = "F  ✗    Poor"

        # ── Balance analysis ─────────────────────────────────────────────────
        scores_3 = {
            "combo":    report.combo_score,
            "hardware": report.hw_score,
            "training": report.train_score,
        }
        mean_3 = sum(scores_3.values()) / 3.0
        gaps   = {k: (mean_3 - v) for k, v in scores_3.items()}
        report.weakest_dim   = max(gaps, key=lambda k: gaps[k])
        report.weakest_score = scores_3[report.weakest_dim]
        max_gap = max(gaps.values())
        report.balance_ok = max_gap <= 0.15

        # ── Improvement tips ─────────────────────────────────────────────────
        report.improvement_tips = self._generate_tips(report, cfg, spec)

        return report

    def score_from_results(
        self,
        cfg:       ArchConfig,
        spec:      CombinationSpec,
        combo_res: CombinationNASResult,
        hw_score:  float,
        train_score: float,
    ) -> CombinationScoreReport:
        """
        Buat report dari hasil evaluasi yang sudah ada (tanpa re-evaluate).
        Berguna untuk scoring pasca-NAS tanpa menjalankan ulang proxy.
        """
        report             = CombinationScoreReport()
        report.arch_id     = cfg.arch_id
        report.spec_label  = spec.label
        report.families    = list(spec.families)
        report.ratios      = list(spec.ratios)
        report.strategy    = spec.strategy
        report.synergy_mult = spec.synergy_mult
        report.compatibility = spec.compatibility
        report.param_count_m = round(cfg.param_count / 1e6, 2)
        report.num_layers  = cfg.num_layers
        report.hidden_dim  = cfg.hidden_dim
        report.vram_pct    = cfg.vram_usage_pct

        report.combo_score = combo_res.combination_score
        report.hw_score    = hw_score
        report.train_score = train_score
        report.pts_c1      = combo_res.pts_c1
        report.pts_c2      = combo_res.pts_c2
        report.pts_c3      = combo_res.pts_c3
        report.pts_c4      = combo_res.pts_c4
        report.pts_c5      = combo_res.pts_c5
        report.total_pts   = combo_res.total_pts
        report.warnings    = list(combo_res.warnings)

        report.combined = round(
            W_COMBO * report.combo_score +
            W_HW    * report.hw_score    +
            W_TRAIN * report.train_score,
            5
        )
        for thresh, label in _GRADE_THRESHOLDS:
            if report.combined >= thresh:
                report.grade = label
                break
        else:
            report.grade = "F  ✗    Poor"

        scores_3 = {"combo": report.combo_score, "hardware": hw_score, "training": train_score}
        mean_3   = sum(scores_3.values()) / 3.0
        gaps     = {k: (mean_3 - v) for k, v in scores_3.items()}
        report.weakest_dim   = max(gaps, key=lambda k: gaps[k])
        report.weakest_score = scores_3[report.weakest_dim]
        report.balance_ok    = max(gaps.values()) <= 0.15
        report.improvement_tips = self._generate_tips(report, cfg, spec)

        return report

    # ── Sub-score helpers ─────────────────────────────────────────────────────

    def _sub_c2_vram(self, cfg: ArchConfig) -> float:
        vram_pct = cfg.vram_usage_pct
        if 45 <= vram_pct <= 78:
            return 1.0
        elif 35 <= vram_pct < 45:
            return 0.70 + (vram_pct - 35) / 10 * 0.30
        elif 78 < vram_pct <= 85:
            return max(0.40, 1.0 - (vram_pct - 78) / 7 * 0.60)
        else:
            return max(0.10, 1.0 - abs(vram_pct - 60) / 60)

    def _sub_c2_ratio(self, spec: CombinationSpec) -> float:
        """Ratio balance: tidak terlalu dominan satu family."""
        max_r = max(spec.ratios)
        if max_r > 0.85:
            return 0.40
        elif max_r > 0.75:
            return 0.70
        # N-way: tambahan entropy check
        if spec.n_families >= 3:
            ent     = -sum(r * math.log(r + 1e-9) for r in spec.ratios if r > 0)
            max_ent = math.log(spec.n_families)
            entropy_ratio = ent / max(1e-9, max_ent)
            return round(min(1.0, 0.80 + entropy_ratio * 0.20), 3)
        return 1.0

    def _sub_c2_param(self, cfg: ArchConfig) -> float:
        params_m = cfg.param_count / 1e6
        if 50 <= params_m <= 800:
            return 1.0
        elif params_m < 50:
            return params_m / 50
        else:
            return max(0.5, 1.0 - (params_m - 800) / 2000)

    def _generate_tips(
        self,
        report: CombinationScoreReport,
        cfg:    ArchConfig,
        spec:   CombinationSpec,
    ) -> List[str]:
        """Generate actionable improvement tips berdasarkan skor."""
        tips = []

        # Balance tip
        if not report.balance_ok:
            tips.append(
                f"⚖ Dimensi '{report.weakest_dim}' tertinggal "
                f"({report.weakest_score:.3f}). "
                + {
                    "hardware": "Coba SHRINK_HIDDEN atau TUNE_FFN_DOWN untuk turunkan VRAM.",
                    "training": "Aktifkan mixed precision & gunakan ADAMW_BF16.",
                    "combo": "Coba strategy INTERLEAVED atau sesuaikan ratio mendekati recommended.",
                }.get(report.weakest_dim, "Tune parameter arsitektur.")
            )

        # VRAM tip
        if cfg.vram_usage_pct > 85:
            tips.append(f"🔴 VRAM {cfg.vram_usage_pct:.1f}% terlalu tinggi. Coba SHRINK_HIDDEN atau DECR_LAYERS.")
        elif cfg.vram_usage_pct < 35:
            tips.append(f"🟡 VRAM {cfg.vram_usage_pct:.1f}% terlalu rendah. Model bisa diperbesar (GROW_HIDDEN).")

        # Ratio tip
        if report.c2_ratio_balance < 0.70:
            tips.append(
                f"📊 Ratio blend tidak seimbang (max ratio: {max(spec.ratios):.0%}). "
                f"Pertimbangkan SHIFT_RATIO untuk mendekati 50–60% / 40–50%."
            )

        # FFN tip
        if not (2.5 <= cfg.ffn_multiplier <= 5.5):
            tips.append(
                f"🔧 FFN multiplier {cfg.ffn_multiplier:.2f} di luar range optimal (2.5–5.5). "
                f"Gunakan TUNE_FFN_{'UP' if cfg.ffn_multiplier < 2.5 else 'DOWN'}."
            )

        # Mixed precision tip
        if not cfg.use_mixed_precision:
            tips.append("⚙ Mixed precision belum aktif. Aktifkan untuk training stability +20%.")

        # Attn tip
        if not report.c3_attn_ok:
            tips.append("🎯 Pertimbangkan FIX_ATTN_GQA untuk attention lebih stabil pada arsitektur hybrid.")

        # Strategy tip
        if spec.strategy == BLEND_INTERLEAVED and spec.compatibility == "MARGINAL":
            tips.append(
                "⚠ INTERLEAVED + MARGINAL compatibility bisa tidak stabil. "
                "Coba SWITCH_STAGED untuk lebih predictable."
            )

        return tips[:5]   # max 5 tips

    # ── Print report ──────────────────────────────────────────────────────────

    def print_report(
        self,
        report: CombinationScoreReport,
        *,
        console=None,
    ) -> None:
        """Print laporan lengkap CombinationScoreReport."""
        _p = console.print if console else print
        W  = 90
        S  = "─" * W

        _p()
        _p(f"╭{S}╮")
        _p(f"│{'  📊 COMBINATION NAS RESULT SCORE REPORT':^{W}}│")
        _p(f"├{S}┤")
        _p(f"│  ARC: {report.arch_id:<10}  Spec: {report.spec_label:<50}  │")

        fam_str = " + ".join(
            f"{f.split('-')[0][:6]}:{int(r*100)}%"
            for f, r in zip(report.families, report.ratios)
        )
        _p(f"│  Families: {fam_str[:72]:<72}  │")
        _p(f"│  Strategy: {report.strategy:<14}  Compat: {report.compatibility:<18}  "
           f"Synergy: {report.synergy_mult:.3f}  │")
        _p(f"│  Arch: {report.param_count_m:.1f}M params  "
           f"L={report.num_layers}  D={report.hidden_dim}  VRAM={report.vram_pct:.1f}%  │")
        _p(f"├{S}┤")
        _p(f"│  {'SCORES SEIMBANG (33% combo + 34% hw + 33% training)':^{W}}│")
        _p(f"│  {'':4}{'Combo':>10}{'Hardware':>10}{'Training':>10}{'Combined':>12}  Grade  │")
        _p(f"│  {'':4}{report.combo_score:>10.4f}{report.hw_score:>10.4f}"
           f"{report.train_score:>10.4f}{report.combined:>12.5f}  "
           f"{report.grade[:12]}  │")
        bal_icon = "✓" if report.balance_ok else "⚠"
        _p(f"│  {bal_icon} Balance: weakest={report.weakest_dim} ({report.weakest_score:.3f})  "
           f"{'Seimbang 3-way OK' if report.balance_ok else 'Perlu penyeimbangan'}  │")
        _p(f"├{S}┤")
        _p(f"│  {'COMBINATION NAS — 5 DIMENSI':^{W}}│")
        _p(f"│  C1 Family Coherence      : {report.pts_c1:>5.1f}/25  "
           f"{'Excellent' if report.pts_c1 >= 20 else 'Good' if report.pts_c1 >= 15 else 'Low'}  │")
        _p(f"│  C2 Blend Balance         : {report.pts_c2:>5.1f}/20  "
           f"VRAM={report.c2_vram_score:.2f}  Ratio={report.c2_ratio_balance:.2f}  "
           f"Param={report.c2_param_score:.2f}  │")
        _p(f"│  C3 Architectural Synergy : {report.pts_c3:>5.1f}/20  "
           f"DW={report.c3_dw_score:.2f}  FFN={report.c3_ffn_score:.2f}  "
           f"Attn={'GQA✓' if report.c3_attn_ok else 'needs fix'}  │")
        _p(f"│  C4 Hardware Compat       : {report.pts_c4:>5.1f}/20  │")
        _p(f"│  C5 Training Synergy      : {report.pts_c5:>5.1f}/15  "
           f"Opt={report.c5_opt_score:.2f}  MP={report.c5_mp_score:.2f}  "
           f"Bonus={report.c5_strat_bonus:.2f}  │")
        _p(f"│  Total: {report.total_pts:.1f}/100  [{report.grade}]  │")

        if report.improvement_tips:
            _p(f"├{S}┤")
            _p(f"│  {'💡 IMPROVEMENT TIPS':^{W}}│")
            for tip in report.improvement_tips:
                _p(f"│  {tip[:W-4]:<{W-4}}  │")

        if report.warnings:
            _p(f"├{S}┤")
            _p(f"│  {'⚠ WARNINGS':^{W}}│")
            for w in report.warnings[:3]:
                _p(f"│  {w[:W-4]:<{W-4}}  │")

        _p(f"╰{S}╯")
        _p()


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE & RANK
# ══════════════════════════════════════════════════════════════════════════════

def compare_combination_results(
    candidates: List[Tuple[ArchConfig, CombinationSpec]],
    gpu:        GPUSpec,
    *,
    device:     str = "cpu",
    console=None,
) -> Tuple[ArchConfig, CombinationSpec, CombinationScoreReport]:
    """
    Bandingkan beberapa (cfg, spec) hasil combination NAS.
    Evaluasi semua secara penuh, pilih yang terbaik berdasarkan combined score.

    Args:
      candidates  : list of (ArchConfig, CombinationSpec) untuk dibandingkan
      gpu         : GPUSpec target
      device      : device untuk proxy training
      console     : Rich console (opsional)

    Returns:
      (best_cfg, best_spec, best_report) — kombinasi dengan combined tertinggi
    """
    _p  = console.print if console else print
    scorer = NASCombinationResultScorer(gpu)

    _p()
    _p(f"  ► Comparing {len(candidates)} combination results...")

    reports: List[Tuple[ArchConfig, CombinationSpec, CombinationScoreReport]] = []
    for cfg, spec in candidates:
        try:
            report = scorer.score_full(cfg, spec, device=device)
            reports.append((cfg, spec, report))
        except Exception as e:
            _p(f"  ⚠ Evaluasi gagal untuk {cfg.arch_id}: {e}")
            continue

    if not reports:
        # Emergency fallback
        cfg, spec = candidates[0]
        report    = CombinationScoreReport(arch_id=cfg.arch_id, spec_label=spec.label)
        return cfg, spec, report

    # Sort descending by combined
    reports.sort(key=lambda x: x[2].combined, reverse=True)

    # Print summary table
    _p()
    _p("  ┌─ Comparison Summary ──────────────────────────────────────────────────────")
    _p(f"  │  {'Rank':<4}  {'ARC-ID':<10}  {'Families':<30}  "
       f"{'Combo':>7}  {'HW':>7}  {'Train':>7}  {'Combined':>9}  Grade")
    _p("  │  " + "─" * 90)

    for rank, (cfg, spec, rpt) in enumerate(reports, 1):
        sym     = "★" if rank == 1 else f"#{rank}"
        fam_lbl = "+".join(f.split("-")[0][:5] for f in spec.families)
        _p(f"  │  {sym:<4}  {rpt.arch_id:<10}  {fam_lbl:<30}  "
           f"{rpt.combo_score:>7.4f}  {rpt.hw_score:>7.4f}  "
           f"{rpt.train_score:>7.4f}  {rpt.combined:>9.5f}  {rpt.grade[:14]}")

    _p("  │")
    _p("  │  ★ = Best combination — highest balanced 3-way score")
    _p("  └──────────────────────────────────────────────────────────────────────────")
    _p()

    best_cfg, best_spec, best_report = reports[0]
    return best_cfg, best_spec, best_report


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_arc_type_report(
    res:    ArcTypeEvalResult,
    cfg:    ArchConfig,
    *,
    console=None,
) -> None:
    """Print laporan evaluasi single arc type."""
    _p = console.print if console else print
    W  = 88
    S  = "─" * W

    guide = _FAMILY_GUIDE.get(res.family, {})
    icon  = guide.get("icon", "•")

    _p()
    _p(f"╭{S}╮")
    _p(f"│{f'  {icon} ARC TYPE EVALUATION — {res.family}':^{W}}│")
    _p(f"├{S}┤")
    _p(f"│  ARC: {res.arch_id:<10}  {res.arch_name[:56]:<56}  │")
    _p(f"│  Family: {res.family:<20}  Params: {res.param_count_m:.1f}M  "
       f"L={res.num_layers}  D={res.hidden_dim}  H={res.num_heads}  │")
    _p(f"│  VRAM: {res.hw_vram_pct:.1f}%  MFU: {res.hw_mfu:.4f}  "
       f"TPS: {res.hw_tps:,.0f}  Bottleneck: {res.hw_bottleneck}  │")
    _p(f"├{S}┤")
    _p(f"│  {'SCORES (Balanced: HW + Training + Quality)':^{W}}│")
    _p(f"│  {'':4}{'HW':>10}{'Training':>10}{'Quality':>10}{'Combined':>12}  │")
    _p(f"│  {'':4}{res.hw_score:>10.4f}{res.train_score:>10.4f}"
       f"{res.quality_norm:>10.4f}{res.combined:>12.5f}  │")
    _p(f"│  Quality: {res.quality_pct:.1f}%  Grade: {res.grade:<35}  │")
    if res.proxy_nan:
        _p(f"│  ⚠ NaN detected in proxy training — training score mungkin tidak akurat  │")
    _p(f"├{S}┤")
    _p(f"│  {'ARCHITECTURE DETAILS':^{W}}│")
    attn_str = cfg.attn_type.value if hasattr(cfg.attn_type, "value") else str(cfg.attn_type)
    ffn_str  = cfg.ffn_type.value  if hasattr(cfg.ffn_type, "value")  else str(cfg.ffn_type)
    _p(f"│  Attn: {attn_str:<12}  FFN: {ffn_str:<12}×{res.ffn_multiplier:.2f}  "
       f"Seq: {res.seq_len}  │")
    _p(f"│  Mixed prec: {'YES' if cfg.use_mixed_precision else 'NO'}  "
       f"Tie embed: {'YES' if cfg.tie_embeddings else 'NO'}  "
       f"Flash attn: {'YES' if cfg.use_flash_attn else 'NO'}  │")
    if res.warnings:
        _p(f"├{S}┤")
        for w in res.warnings[:3]:
            _p(f"│  ⚠ {w[:W-6]:<{W-6}}  │")
    _p(f"│  Eval time: {res.eval_ms:.0f}ms  │")
    _p(f"╰{S}╯")
    _p()
