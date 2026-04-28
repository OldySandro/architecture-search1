"""
training_aware.py  —  Training-Aware NAS Scorer
════════════════════════════════════════════════════════════════════════════════

FILOSOFI
─────────
File ini melengkapi hardware_aware (fitness_score) dengan dimensi yang selama
ini diabaikan: seberapa baik suatu arsitektur akan BERLATIH dalam praktik.

GPU-aware menjawab: "Apakah arsitektur ini muat dan efisien secara komputasi?"
Training-aware menjawab:
  1. Apakah gradien akan mengalir dengan sehat selama ribuan steps?
  2. Seberapa cepat model akan konvergen ke perplexity rendah?
  3. Apakah optimizer cocok dengan kedalaman/lebar arsitektur?
  4. Seberapa efisien setiap token data digunakan untuk belajar?
  5. Seberapa stabil proses training tanpa divergence / loss spike?

DIMENSI SCORING
──────────────────────────────────────────────────────────────────────────────
  T1  Gradient Flow Health      25 pts  — Kesehatan aliran gradien
  T2  Convergence Dynamics      25 pts  — Kecepatan & efisiensi konvergensi
  T3  Training Stability        20 pts  — Stabilitas loss, precision, overflow
  T4  Sample Efficiency         15 pts  — Kualitas sinyal gradient per token
  T5  Optimizer Compatibility   15 pts  — Kecocokan optimizer–arsitektur
                                ──────
  TOTAL                        100 pts

INTEGRASI
──────────
  • Standalone: TrainingAwareScorer(gpu).score(cfg)
  • Pipeline:   compute_combined_score_v2(quality_pct, fitness, training_score)
  • Refiner:    patch ke AdaptiveRefiner via training_refine_hints(cfg)
  
FORMULA SKOR GABUNGAN BARU
──────────────────────────────────────────────────────────────────────────────
  OLD: combined = 0.35 × quality + 0.65 × fitness
  NEW: combined = 0.25 × quality + 0.40 × fitness + 0.35 × training_score

  Bobot training dinaikkan dari ~0% ke 35% karena inilah yang paling
  menentukan apakah arsitektur benar-benar berguna untuk pretraining nyata.

════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from arch_types import (
    ArchConfig, FFNType, AttentionType, OptimizerType, NormType, PosEncType
)
from hardware import GPUSpec


# ═══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

# Chinchilla scaling law: C ≈ 6 × N × D  (compute = 6 × params × tokens)
# Optimal training: D* = sqrt(C / 6) / sqrt(N) → tokens ≈ params untuk fixed C
CHINCHILLA_COEFF: float = 6.0      # FLOPs per token per param (fwd+bwd approx)

# Gradient noise threshold: batch_size × seq_len < threshold → noisy gradients
GRAD_NOISE_THRESHOLD: int = 4096   # token equivalents per step

# Depth/Width optimal ratio dari literature (Transformer scaling):
# D/sqrt(W) ≈ 0.2–0.5 untuk model seimbang. Terlalu dalam → vanishing.
OPTIMAL_DW_RATIO_LO: float = 0.15
OPTIMAL_DW_RATIO_HI: float = 0.55

# Maximum relative gradient norm degradation per layer sebelum dianggap berisiko
GRAD_ATTENUATION_PER_LAYER: float = 0.015   # 1.5% per layer → 43 layer = 48% attenuasi

# LR sensitivity threshold berdasarkan kedalaman (layers > threshold perlu LR lebih kecil)
LR_DEPTH_THRESHOLD: int = 24

# Chinchilla token multiplier untuk training yang "adequate" (bukan over-trained)
CHINCHILLA_TOKEN_MULT: float = 20.0  # tokens = params × multiplier untuk Chinchilla-optimal


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingCheck:
    """Hasil satu audit check training-aware."""
    name:           str
    dimension:      str       # T1 / T2 / T3 / T4 / T5
    passed:         bool
    points_earned:  float
    points_max:     float
    detail:         str       # penjelasan pass/fail dengan nilai nyata
    insight:        str = ""  # rekomendasi training praktis
    severity:       str = "info"  # "critical" / "warning" / "info"

    @property
    def pct(self) -> float:
        return self.points_earned / max(0.001, self.points_max) * 100

    @property
    def partial(self) -> bool:
        return 0.0 < self.points_earned < self.points_max


@dataclass
class TrainingScoreReport:
    """Laporan lengkap training-aware scoring."""
    arch_id:       str
    arch_name:     str
    total_score:   float          # 0–100
    max_score:     float = 100.0
    checks:        List[TrainingCheck] = field(default_factory=list)

    # Metrik training turunan (untuk display dan export)
    gradient_health_score:    float = 0.0   # 0–1
    convergence_score:        float = 0.0   # 0–1
    stability_score:          float = 0.0   # 0–1
    sample_efficiency_score:  float = 0.0   # 0–1
    optimizer_compat_score:   float = 0.0   # 0–1

    chinchilla_optimal_tokens: float = 0.0  # token ideal untuk model ini (Millions)
    estimated_steps_to_target: float = 0.0  # steps estimasi ke target perplexity
    gradient_attenuation_risk: str   = ""   # "low" / "moderate" / "high" / "critical"
    lr_sensitivity_class:      str   = ""   # "robust" / "sensitive" / "fragile"
    training_regime:           str   = ""   # deskripsi singkat karakteristik training

    @property
    def pct(self) -> float:
        return self.total_score / max(0.001, self.max_score) * 100

    @property
    def grade(self) -> str:
        p = self.pct
        if p >= 95:  return "S ★★★  Excellent Trainability"
        if p >= 85:  return "A+ ★★★ Very Good"
        if p >= 75:  return "A  ★★  Good"
        if p >= 60:  return "B  ★   Acceptable"
        if p >= 45:  return "C      Marginal"
        return              "F  ✗   Poor Trainability"

    @property
    def grade_color(self) -> str:
        p = self.pct
        if p >= 95:  return "bold green"
        if p >= 85:  return "green"
        if p >= 75:  return "cyan"
        if p >= 60:  return "yellow"
        if p >= 45:  return "dark_orange"
        return              "red"

    @property
    def failed_checks(self) -> List[TrainingCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def critical_issues(self) -> List[TrainingCheck]:
        return [c for c in self.checks if c.severity == "critical" and not c.passed]

    def by_dimension(self) -> Dict[str, List[TrainingCheck]]:
        out: Dict[str, List[TrainingCheck]] = {}
        for c in self.checks:
            out.setdefault(c.dimension, []).append(c)
        return out

    def dimension_score(self, dim: str) -> Tuple[float, float]:
        checks = self.by_dimension().get(dim, [])
        return (
            sum(c.points_earned for c in checks),
            sum(c.points_max    for c in checks),
        )

    def refine_hints(self) -> List[str]:
        """Daftar perubahan konkret untuk meningkatkan training score."""
        hints = []
        for c in self.checks:
            if not c.passed and c.insight:
                hints.append(f"[{c.dimension}] {c.insight}")
        return hints


@dataclass
class TrainingFitnessBreakdown:
    """
    Breakdown gabungan untuk rekomendasi final.
    Menggantikan combined_score lama dengan bobot yang training-aware.
    """
    arch_id:         str
    quality_pct:     float   # dari ArcQualityScorer (0–100)
    fitness_score:   float   # dari _fitness_score (0–1)
    training_pct:    float   # dari TrainingAwareScorer (0–100)

    # Bobot baru yang lebih seimbang
    W_QUALITY:  float = 0.25
    W_FITNESS:  float = 0.40
    W_TRAINING: float = 0.35

    @property
    def combined(self) -> float:
        """
        Skor gabungan yang seimbang antara GPU-aware dan training-aware.
        Range: 0.0 – 1.0
        """
        q_norm = self.quality_pct  / 100.0
        t_norm = self.training_pct / 100.0
        return round(
            self.W_QUALITY  * q_norm       +
            self.W_FITNESS  * self.fitness_score +
            self.W_TRAINING * t_norm,
            5
        )

    @property
    def verdict(self) -> str:
        c = self.combined
        if c >= 0.75: return "★★★ Highly Recommended"
        if c >= 0.60: return "★★  Recommended"
        if c >= 0.45: return "★   Acceptable"
        if c >= 0.30: return "~   Marginal"
        return               "✗   Not Recommended"

    def summary(self) -> str:
        return (
            f"Combined={self.combined:.5f}  "
            f"(quality={self.quality_pct:.1f}%×{self.W_QUALITY:.0%} + "
            f"fitness={self.fitness_score:.4f}×{self.W_FITNESS:.0%} + "
            f"training={self.training_pct:.1f}%×{self.W_TRAINING:.0%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING-AWARE SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingAwareScorer:
    """
    Mengaudit ArchConfig dari perspektif proses training nyata.

    Tidak seperti ArcQualityScorer yang memeriksa konsistensi internal,
    dan generator._fitness_score yang mengukur GPU efficiency,
    kelas ini mengukur apakah arsitektur ini akan:
      1) Converge dengan gradient yang sehat
      2) Belajar secara efisien dari data
      3) Stabil secara numerik selama ribuan training steps
      4) Cocok dengan optimizer yang dipilih
      5) Menggunakan setiap token data secara produktif

    Semua check menggunakan PARTIAL CREDIT kontinu — bukan binary.
    Nilai didasarkan pada paper empiris + scaling law literature.
    """

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu

    # ── Helpers Internal ──────────────────────────────────────────────────────

    def _dw_ratio(self, cfg: ArchConfig) -> float:
        """
        Depth-Width ratio: L / sqrt(D).
        Dari Transformer scaling literature: optimal ~0.2–0.5.
        Terlalu tinggi (>0.6) → gradient vanishing lebih parah.
        Terlalu rendah (<0.1) → under-parameterized per layer.
        """
        return cfg.num_layers / max(1.0, math.sqrt(cfg.hidden_dim))

    def _total_training_tokens(self, cfg: ArchConfig) -> float:
        """
        Estimasi total token yang dibutuhkan untuk Chinchilla-optimal training.
        Formula: T_opt = CHINCHILLA_TOKEN_MULT × N (params)
        """
        return CHINCHILLA_TOKEN_MULT * cfg.param_count

    def _steps_for_tokens(self, cfg: ArchConfig, total_tokens: float) -> float:
        """
        Konversi total training tokens ke jumlah optimizer steps.
        tokens_per_step = batch_size × seq_len
        """
        tokens_per_step = max(1, cfg.batch_size * cfg.seq_len)
        return total_tokens / tokens_per_step

    def _gradient_attenuation_factor(self, cfg: ArchConfig) -> float:
        """
        Estimasi faktor attenuasi gradien dari output ke input.
        Menggunakan model attenuasi eksponensial per layer.

        Model: attenuation = (1 - r)^L dimana r = per-layer loss rate.
        RMSNorm pre-norm mengurangi r secara signifikan.
        """
        # Base per-layer gradient loss rate
        base_rate = GRAD_ATTENUATION_PER_LAYER

        # RMSNorm pre-norm mengurangi gradient loss ~40%
        if cfg.norm_type == NormType.RMSNORM:
            base_rate *= 0.60
        elif cfg.norm_type == NormType.LAYERNORM:
            base_rate *= 0.75  # sedikit lebih buruk dari RMSNorm

        # Flash attention mengurangi numerical issues di backward pass
        if cfg.use_flash_attn:
            base_rate *= 0.90

        # GQA lebih stabil dibanding MHA (lebih sedikit attention head → lebih stabil)
        if cfg.attn_type == AttentionType.GQA:
            base_rate *= 0.95
        elif cfg.attn_type == AttentionType.MQA:
            base_rate *= 0.93

        # Dropout 0 → tidak ada stochastic noise, gradient lebih bersih
        if cfg.dropout == 0.0:
            base_rate *= 0.95

        return (1.0 - base_rate) ** cfg.num_layers

    def _lr_sensitivity_score(self, cfg: ArchConfig) -> float:
        """
        Estimasi seberapa sensitif model terhadap learning rate.
        Return: 1.0 = robust, mendekati 0 = sangat fragile.

        Factor:
        - Depth: lebih dalam → gradient conditioning lebih buruk → LR range sempit
        - Norm type: RMSNorm lebih stabil dari LayerNorm
        - Optimizer type: Adam FP32 lebih robust dari Lion untuk model dalam
        - Tied embeddings: slight stabilization (shared gradient signals)
        """
        sensitivity = 1.0

        # Depth penalty: linier hingga threshold, kemudian lebih agresif
        if cfg.num_layers > LR_DEPTH_THRESHOLD:
            excess = cfg.num_layers - LR_DEPTH_THRESHOLD
            depth_pen = min(0.40, 0.012 * excess)   # max 40% penalty
            sensitivity -= depth_pen

        # Norm type benefit
        if cfg.norm_type == NormType.RMSNORM:
            sensitivity += 0.05
        elif cfg.norm_type == NormType.LAYERNORM:
            pass  # neutral
        elif cfg.norm_type == NormType.GROUPNORM:
            sensitivity -= 0.10   # worse for transformers

        # Optimizer stability
        if cfg.optimizer_type in (OptimizerType.ADAM_FP32, OptimizerType.ZERO1,
                                   OptimizerType.ZERO2, OptimizerType.ZERO3):
            sensitivity += 0.05   # Adam FP32 most robust
        elif cfg.optimizer_type == OptimizerType.ADAMW_BF16:
            sensitivity += 0.02   # slightly less robust
        elif cfg.optimizer_type == OptimizerType.LION:
            # Lion needs more careful LR tuning, especially for deep models
            lion_depth_pen = max(0.0, (cfg.num_layers - 12) * 0.008)
            sensitivity -= lion_depth_pen
        elif cfg.optimizer_type == OptimizerType.ADAM_8BIT:
            # 8-bit quantization adds noise to optimizer states
            sensitivity -= 0.07

        # Tied embeddings: shared gradient → slight regularization effect
        if cfg.tie_embeddings:
            sensitivity += 0.02

        # Mixed precision: BF16 has wider range than FP16 → more stable
        if cfg.use_mixed_precision:
            sensitivity += 0.03

        return float(min(1.0, max(0.0, sensitivity)))

    def _batch_gradient_quality(self, cfg: ArchConfig) -> float:
        """
        Kualitas estimasi gradien berdasarkan effective batch size.
        Effective tokens per step = batch_size × seq_len

        Signal-to-noise ratio: SNR ∝ sqrt(effective_tokens / NOISE_FLOOR)
        Skor: 1.0 jika effective_tokens ≥ 4096, degradasi cepat di bawah ini.
        """
        effective_tokens = cfg.batch_size * cfg.seq_len
        if effective_tokens >= 8192:
            return 1.0
        elif effective_tokens >= 4096:
            return 0.90 + 0.10 * (effective_tokens - 4096) / 4096
        elif effective_tokens >= 2048:
            return 0.70 + 0.20 * (effective_tokens - 2048) / 2048
        elif effective_tokens >= 1024:
            return 0.45 + 0.25 * (effective_tokens - 1024) / 1024
        else:
            return max(0.20, 0.45 * effective_tokens / 1024)

    def _mixed_precision_stability(self, cfg: ArchConfig) -> float:
        """
        Probabilitas stabilitas numerik dengan konfigurasi presisi yang dipilih.
        BF16 lebih baik dari FP16 untuk training panjang karena dynamic range lebih lebar.
        """
        score = 1.0

        if not cfg.use_mixed_precision:
            # FP32 penuh: paling stabil tapi lambat
            score = 0.95   # slight penalty karena bukan best practice

        # Optimizer state precision matters a lot
        if cfg.optimizer_type == OptimizerType.ADAM_FP32:
            score += 0.05   # FP32 optimizer states → most stable
        elif cfg.optimizer_type == OptimizerType.ADAMW_BF16:
            score -= 0.05   # BF16 states → potential underflow for small gradients
        elif cfg.optimizer_type == OptimizerType.ADAM_8BIT:
            score -= 0.12   # 8-bit quantization → gradient precision loss

        # Deep models dengan BF16 activation: ada risiko overflow di layer akhir
        if cfg.num_layers > 36 and cfg.use_mixed_precision:
            overflow_risk = max(0.0, (cfg.num_layers - 36) * 0.003)
            score -= overflow_risk

        # FFN type impact on numerical stability
        if cfg.ffn_type == FFNType.GEGLU:
            score += 0.02   # GeGLU lebih smooth, lebih stabil dari ReLU variants

        # GQA: lebih stabil dari MHA (lebih sedikit attention parameter → smoother optim)
        if cfg.attn_type == AttentionType.GQA:
            score += 0.02

        return float(min(1.0, max(0.0, score)))

    def _chinchilla_param_efficiency(self, cfg: ArchConfig) -> float:
        """
        Seberapa efisien parameter model digunakan untuk kapasitas belajar.

        Chinchilla: N × D = C/6  →  D_opt = C/(6N)
        Untuk compute budget TETAP (GPU kami), apakah model ini terlalu besar
        (under-trained) atau justru Chinchilla-optimal?

        Return: 1.0 = optimal, <1.0 = under/over-parameterized relatif terhadap
                compute yang tersedia di GPU ini.
        """
        gpu = self.gpu

        # Estimasi compute budget GPU dalam FLOPs untuk satu training run (1 hari referensi)
        # 1 hari × GPU peak BF16 × MFU typical
        one_day_flops = (gpu.bf16_tflops * 1e12) * gpu.mfu_typical_max * 86400.0

        # Chinchilla-optimal N untuk budget ini: N* = sqrt(C / (6 × 20))
        # D* = 20 × N*  →  N* = sqrt(C / 120)
        N_optimal = math.sqrt(one_day_flops / (CHINCHILLA_COEFF * CHINCHILLA_TOKEN_MULT))
        N_actual  = cfg.param_count

        if N_actual <= 0:
            return 0.0

        # Ratio: 1.0 = perfectly chinchilla optimal
        ratio = N_actual / max(1.0, N_optimal)

        # Gaussian-like scoring: peak at ratio=1.0, degrades symmetrically
        # Model 2× terlalu besar atau 2× terlalu kecil → skor ~0.60
        log_ratio = math.log(max(0.01, ratio))
        score = math.exp(-0.5 * (log_ratio / 0.7) ** 2)

        return float(min(1.0, max(0.0, score)))

    def _moe_routing_quality(self, cfg: ArchConfig) -> Optional[float]:
        """
        Untuk MoE: kualitas routing dan load balance dari sudut pandang training.
        Return None jika bukan MoE.
        """
        if cfg.ffn_type not in (FFNType.MOE, FFNType.MOE_TOPK):
            return None

        score = 1.0

        # Capacity factor < 1.0: experts akan overflow → token dropping
        if cfg.expert_capacity_factor < 1.0:
            score -= 0.30   # critical: tokens di-drop → gradient signal hilang
        elif cfg.expert_capacity_factor < 1.25:
            score -= 0.10   # marginal overflow risk
        elif cfg.expert_capacity_factor > 2.0:
            score -= 0.05   # wasteful padding

        # Top-k selection: k=1 → specializasi tinggi, gradient update sparser
        # k=2 → lebih robust, load balance lebih baik
        if cfg.top_k_experts == 1:
            score -= 0.08   # gradient sparsity issue
        elif cfg.top_k_experts >= 2:
            score += 0.0    # optimal

        # Expert count vs model size: terlalu banyak expert untuk model kecil
        # → tiap expert terlalu sedikit dilatih → expert collapse risk
        params_per_expert = cfg.param_count / max(1, cfg.num_experts)
        if params_per_expert < 5e6:   # <5M params per expert → collapse risk
            score -= 0.15
        elif params_per_expert < 10e6:
            score -= 0.07

        # Gradient checkpointing dengan MoE: routing mask tidak bisa di-checkpoint
        # dengan mudah → memory spike saat backward
        if cfg.use_gradient_checkpointing:
            score -= 0.05

        return float(min(1.0, max(0.0, score)))

    def _pos_enc_training_quality(self, cfg: ArchConfig) -> float:
        """
        Kualitas positional encoding dari sudut pandang training.
        """
        score = 0.0
        pe = cfg.pos_enc

        if pe == PosEncType.ROPE:
            # RoPE: length extrapolation, gradient-friendly, modern best practice
            score = 1.0
        elif pe == PosEncType.ALIBI:
            # ALiBi: gradient-friendly, extrapolates well but fixed bias
            score = 0.90
        elif pe == PosEncType.LEARNED:
            # Learned: dapat overfit ke training sequence length
            # Extra parameters → sedikit lebih lambat konvergen di awal
            score = 0.70
        elif pe == PosEncType.SINCOS:
            # Sinusoidal: fixed, ok tapi tidak sebaik RoPE untuk training panjang
            score = 0.75
        elif pe == PosEncType.NONE:
            # No pos enc: hanya ok untuk model ALiBi-style (sudah tercakup ALiBi)
            score = 0.65

        # Bonus: RoPE + GQA kombinasi optimal untuk modern training
        if pe == PosEncType.ROPE and cfg.attn_type in (AttentionType.GQA, AttentionType.ROPE):
            score = min(1.0, score + 0.05)

        return score

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, cfg: ArchConfig) -> TrainingScoreReport:
        """
        Audit lengkap 5 dimensi training-aware.
        Returns TrainingScoreReport dengan partial credit kontinu.
        """
        checks: List[TrainingCheck] = []

        # Run semua dimensi
        t1_checks = self._dim_t1_gradient_flow(cfg)
        t2_checks = self._dim_t2_convergence(cfg)
        t3_checks = self._dim_t3_stability(cfg)
        t4_checks = self._dim_t4_sample_efficiency(cfg)
        t5_checks = self._dim_t5_optimizer_compat(cfg)

        checks.extend(t1_checks)
        checks.extend(t2_checks)
        checks.extend(t3_checks)
        checks.extend(t4_checks)
        checks.extend(t5_checks)

        total = sum(c.points_earned for c in checks)

        # Hitung skor per-dimensi
        def _dim_pct(dim: str) -> float:
            dc = [c for c in checks if c.dimension == dim]
            e = sum(c.points_earned for c in dc)
            m = sum(c.points_max for c in dc)
            return (e / max(0.001, m)) if m > 0 else 0.0

        # Attenuation risk label
        att = self._gradient_attenuation_factor(cfg)
        if att >= 0.80:   att_risk = "low"
        elif att >= 0.60: att_risk = "moderate"
        elif att >= 0.40: att_risk = "high"
        else:             att_risk = "critical"

        # LR sensitivity label
        lr_s = self._lr_sensitivity_score(cfg)
        if lr_s >= 0.85:   lr_class = "robust"
        elif lr_s >= 0.70: lr_class = "sensitive"
        else:              lr_class = "fragile"

        # Training regime deskripsi
        regime = _describe_training_regime(cfg)

        # Chinchilla optimal tokens (dalam jutaan)
        chin_tokens_m = self._total_training_tokens(cfg) / 1e6
        steps_est = self._steps_for_tokens(cfg, self._total_training_tokens(cfg))

        report = TrainingScoreReport(
            arch_id       = cfg.arch_id,
            arch_name     = cfg.arch_name,
            total_score   = round(total, 2),
            max_score     = 100.0,
            checks        = checks,
            gradient_health_score   = _dim_pct("T1"),
            convergence_score       = _dim_pct("T2"),
            stability_score         = _dim_pct("T3"),
            sample_efficiency_score = _dim_pct("T4"),
            optimizer_compat_score  = _dim_pct("T5"),
            chinchilla_optimal_tokens = round(chin_tokens_m, 1),
            estimated_steps_to_target = round(steps_est),
            gradient_attenuation_risk = att_risk,
            lr_sensitivity_class      = lr_class,
            training_regime           = regime,
        )
        return report

    # ══════════════════════════════════════════════════════════════════════════
    #  T1: GRADIENT FLOW HEALTH  (25 pts)
    # ══════════════════════════════════════════════════════════════════════════
    #   T1.1  Depth-Width ratio optimal          6 pts  (graduated)
    #   T1.2  Gradient attenuation magnitude     6 pts  (graduated)
    #   T1.3  Normalization type & placement     5 pts  (graduated)
    #   T1.4  Residual stream health             4 pts  (graduated)
    #   T1.5  Positional encoding trainability   4 pts  (graduated)
    #                                           ──────
    #                                           25 pts

    def _dim_t1_gradient_flow(self, cfg: ArchConfig) -> List[TrainingCheck]:
        results = []

        # ── T1.1: Depth-Width Ratio (6 pts) ──────────────────────────────────
        # L / sqrt(D): optimal 0.15–0.55 dari scaling law literature.
        # Terlalu besar → vanishing gradients, terlalu kecil → wasteful width.
        dw = self._dw_ratio(cfg)

        if OPTIMAL_DW_RATIO_LO <= dw <= OPTIMAL_DW_RATIO_HI:
            # Dalam zona optimal
            t11_pts = 6.0
            t11_det = (f"✓ D/W ratio={dw:.3f} (optimal range "
                       f"{OPTIMAL_DW_RATIO_LO}–{OPTIMAL_DW_RATIO_HI})")
            t11_ins = ""
            t11_sev = "info"
            t11_ok  = True
        elif dw < OPTIMAL_DW_RATIO_LO:
            # Terlalu shallow relative to width: parameter-heavy per layer
            deficit  = OPTIMAL_DW_RATIO_LO - dw
            t11_pts  = max(2.0, 6.0 - 20 * deficit)
            t11_det  = (f"~ D/W ratio={dw:.3f} < {OPTIMAL_DW_RATIO_LO} "
                        f"(too wide/shallow — L={cfg.num_layers}, D={cfg.hidden_dim})  "
                        f"[{t11_pts:.1f}/6 pts]")
            t11_ins  = f"Tambah num_layers ke ≥{int(OPTIMAL_DW_RATIO_LO * math.sqrt(cfg.hidden_dim))} untuk D/W optimal"
            t11_sev  = "warning"
            t11_ok   = False
        else:
            # Terlalu deep relative to width: vanishing gradient risk
            excess   = dw - OPTIMAL_DW_RATIO_HI
            t11_pts  = max(0.0, 6.0 - 15 * excess)
            t11_det  = (f"✗ D/W ratio={dw:.3f} > {OPTIMAL_DW_RATIO_HI} "
                        f"(too deep/narrow — gradient vanishing risk!)  "
                        f"[{t11_pts:.1f}/6 pts]")
            t11_ins  = (f"Kurangi num_layers ke ≤{int(OPTIMAL_DW_RATIO_HI * math.sqrt(cfg.hidden_dim))} "
                        f"ATAU naikkan hidden_dim ke ≥{int((cfg.num_layers / OPTIMAL_DW_RATIO_HI)**2)}")
            t11_sev  = "critical" if dw > 0.70 else "warning"
            t11_ok   = False

        results.append(TrainingCheck(
            name          = "T1.1: Depth-Width ratio",
            dimension     = "T1",
            passed        = t11_ok,
            points_earned = t11_pts,
            points_max    = 6.0,
            detail        = t11_det,
            insight       = t11_ins,
            severity      = t11_sev,
        ))

        # ── T1.2: Gradient Attenuation (6 pts) ────────────────────────────────
        # Berapa banyak sinyal gradien yang tersisa saat mencapai layer pertama.
        # Nilai 1.0 = tidak ada attenuasi, 0.0 = gradien hilang total.
        att = self._gradient_attenuation_factor(cfg)
        att_pct = att * 100

        if att >= 0.80:
            t12_pts = 6.0
            t12_det = f"✓ Gradient residual={att_pct:.1f}% (healthy flow to early layers)"
            t12_ins = ""
            t12_sev = "info"
            t12_ok  = True
        elif att >= 0.60:
            t12_pts = 4.5 - (0.80 - att) * 7.5
            t12_pts = max(3.0, t12_pts)
            t12_det = (f"~ Gradient residual={att_pct:.1f}% (moderate attenuation — "
                       f"monitor gradient norms during training)  [{t12_pts:.1f}/6 pts]")
            t12_ins = "Pertimbangkan gradient clipping agresif (max_norm=0.5) dan warmup panjang"
            t12_sev = "warning"
            t12_ok  = False
        elif att >= 0.40:
            t12_pts = max(1.0, 3.0 - (0.60 - att) * 10)
            t12_det = (f"✗ Gradient residual={att_pct:.1f}% (high attenuation — "
                       f"early layers underfit risk)  [{t12_pts:.1f}/6 pts]")
            t12_ins = ("Gunakan RMSNorm, kurangi num_layers, atau tambah residual connection "
                       "lebih sering. Pertimbangkan gradient clipping max_norm=0.3")
            t12_sev = "critical"
            t12_ok  = False
        else:
            t12_pts = 0.0
            t12_det = (f"✗ Gradient residual={att_pct:.1f}% (CRITICAL — "
                       f"severe vanishing gradient, model likely won't train)")
            t12_ins = ("KRITIS: Arsitektur terlalu dalam untuk dimensi ini. "
                       "Kurangi layers DRASTIS atau naikkan hidden_dim minimal 2×")
            t12_sev = "critical"
            t12_ok  = False

        results.append(TrainingCheck(
            name          = "T1.2: Gradient attenuation",
            dimension     = "T1",
            passed        = t12_ok,
            points_earned = t12_pts,
            points_max    = 6.0,
            detail        = t12_det,
            insight       = t12_ins,
            severity      = t12_sev,
        ))

        # ── T1.3: Normalization Type & Placement (5 pts) ──────────────────────
        # Pre-norm (RMSNorm/LN sebelum sublayer) adalah best practice modern.
        # RMSNorm lebih cepat dan lebih stabil dari LayerNorm.
        norm = cfg.norm_type
        if norm == NormType.RMSNORM:
            t13_pts = 5.0
            t13_det = ("✓ RMSNorm: best-in-class gradient stability, "
                       "lower compute than LayerNorm")
            t13_ins = ""
            t13_ok  = True
        elif norm == NormType.LAYERNORM:
            t13_pts = 3.5
            t13_det = ("~ LayerNorm: good stability, slightly more compute "
                       "and parameter (bias+scale) vs RMSNorm  [3.5/5 pts]")
            t13_ins = "Upgrade ke RMSNorm untuk gradient stability yang lebih baik"
            t13_ok  = True  # masih ok, hanya suboptimal
        elif norm == NormType.GROUPNORM:
            t13_pts = 1.5
            t13_det = ("✗ GroupNorm: tidak ideal untuk Transformer — "
                       "designed for spatial data, gradient flow lebih noisy  [1.5/5 pts]")
            t13_ins = "Ganti ke RMSNorm untuk training stability yang signifikan"
            t13_ok  = False
        else:
            t13_pts = 0.0
            t13_det = "✗ Unknown norm type: tidak dapat menilai gradient stability"
            t13_ins = "Pilih RMSNorm"
            t13_ok  = False

        results.append(TrainingCheck(
            name          = "T1.3: Normalization quality",
            dimension     = "T1",
            passed        = t13_ok,
            points_earned = t13_pts,
            points_max    = 5.0,
            detail        = t13_det,
            insight       = t13_ins,
            severity      = "warning" if not t13_ok else "info",
        ))

        # ── T1.4: Residual Stream Health (4 pts) ──────────────────────────────
        # Residual connection yang sehat = gradien mengalir langsung tanpa melewati
        # semua transformasi. Kunci: head_dim × num_heads == hidden_dim (A1 check)
        # PLUS KV head ratio tidak terlalu ekstrem.
        product_ok   = (cfg.num_heads * cfg.head_dim == cfg.hidden_dim)
        kv_ratio     = cfg.num_kv_heads / max(1, cfg.num_heads)

        t14_pts = 0.0
        t14_issues = []

        # +2.0 pts: residual dimensi konsisten
        if product_ok:
            t14_pts += 2.0
        else:
            t14_issues.append(f"heads×head_dim≠hidden_dim ({cfg.num_heads}×{cfg.head_dim}≠{cfg.hidden_dim})")

        # +1.0 pts: KV compression ratio tidak terlalu ekstrem (>= 0.125 = 1/8)
        if kv_ratio >= 0.125:
            t14_pts += 1.0
        else:
            t14_issues.append(f"KV compression {kv_ratio:.3f} < 1/8 — terlalu aggressive")

        # +1.0 pts: FlashAttention aktif (gradient checkpointing di attention lebih stabil)
        if cfg.use_flash_attn:
            t14_pts += 1.0
        else:
            t14_issues.append("Flash Attention disabled — standard attention menambah numerical noise")

        t14_ok  = len(t14_issues) == 0
        t14_det = (f"✓ Residual stream healthy (kv_ratio={kv_ratio:.3f}, FA=on)"
                   if t14_ok else
                   f"~ Residual issues: {' | '.join(t14_issues)}  [{t14_pts:.1f}/4 pts]")
        t14_ins = ("Perbaiki: " + "; ".join(t14_issues)) if t14_issues else ""

        results.append(TrainingCheck(
            name          = "T1.4: Residual stream health",
            dimension     = "T1",
            passed        = t14_ok,
            points_earned = t14_pts,
            points_max    = 4.0,
            detail        = t14_det,
            insight       = t14_ins,
            severity      = "warning" if not t14_ok else "info",
        ))

        # ── T1.5: Positional Encoding Trainability (4 pts) ────────────────────
        pe_score = self._pos_enc_training_quality(cfg)
        t15_pts  = round(4.0 * pe_score, 2)
        t15_ok   = pe_score >= 0.80

        pe_names = {
            PosEncType.ROPE:    "RoPE (extrapolation-capable, gradient-friendly)",
            PosEncType.ALIBI:   "ALiBi (length extrapolation, fixed bias — good)",
            PosEncType.LEARNED: "Learned (may overfit to training seq_len)",
            PosEncType.SINCOS:  "Sinusoidal (fixed, no length extrapolation)",
            PosEncType.NONE:    "None (ALiBi-style, limited without explicit bias)",
        }
        pe_label = pe_names.get(cfg.pos_enc, str(cfg.pos_enc))

        results.append(TrainingCheck(
            name          = "T1.5: Positional encoding",
            dimension     = "T1",
            passed        = t15_ok,
            points_earned = t15_pts,
            points_max    = 4.0,
            detail        = (f"{'✓' if t15_ok else '~'} {pe_label}  "
                             f"[{t15_pts:.1f}/4 pts]"),
            insight       = ("Pertimbangkan RoPE untuk training stability dan length extrapolation"
                             if not t15_ok else ""),
            severity      = "info",
        ))

        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  T2: CONVERGENCE DYNAMICS  (25 pts)
    # ══════════════════════════════════════════════════════════════════════════
    #   T2.1  Chinchilla param efficiency        7 pts  (graduated)
    #   T2.2  LR sensitivity / robustness        6 pts  (graduated)
    #   T2.3  Convergence step estimate          5 pts  (graduated)
    #   T2.4  Depth-appropriate LR range         4 pts  (graduated)
    #   T2.5  Warmup budget adequacy             3 pts  (graduated)
    #                                           ──────
    #                                           25 pts

    def _dim_t2_convergence(self, cfg: ArchConfig) -> List[TrainingCheck]:
        results = []

        # ── T2.1: Chinchilla Parameter Efficiency (7 pts) ─────────────────────
        chin_eff = self._chinchilla_param_efficiency(cfg)
        t21_pts  = round(7.0 * chin_eff, 2)
        t21_ok   = chin_eff >= 0.70

        gpu       = self.gpu
        one_day_F = (gpu.bf16_tflops * 1e12) * gpu.mfu_typical_max * 86400.0
        N_opt     = math.sqrt(one_day_F / (CHINCHILLA_COEFF * CHINCHILLA_TOKEN_MULT))
        ratio     = cfg.param_count / max(1.0, N_opt)

        if ratio < 0.5:
            chin_desc = "under-parameterized"
            chin_rec  = f"Model terlalu kecil untuk compute budget GPU. Target ≥{N_opt/1e6:.0f}M params"
        elif ratio > 2.0:
            chin_desc = "over-parameterized"
            chin_rec  = f"Model terlalu besar → akan under-trained. Chinchilla-optimal: {N_opt/1e6:.0f}M params"
        else:
            chin_desc = "near-optimal"
            chin_rec  = ""

        results.append(TrainingCheck(
            name          = "T2.1: Chinchilla param efficiency",
            dimension     = "T2",
            passed        = t21_ok,
            points_earned = t21_pts,
            points_max    = 7.0,
            detail        = (f"{'✓' if t21_ok else '~'} Chinchilla efficiency={chin_eff:.3f} "
                             f"({chin_desc}) | N_actual={cfg.param_count/1e6:.1f}M, "
                             f"N_optimal≈{N_opt/1e6:.1f}M (ratio={ratio:.2f}×)  "
                             f"[{t21_pts:.1f}/7 pts]"),
            insight       = chin_rec,
            severity      = "warning" if ratio > 3.0 or ratio < 0.3 else "info",
        ))

        # ── T2.2: LR Sensitivity / Robustness (6 pts) ─────────────────────────
        lr_s     = self._lr_sensitivity_score(cfg)
        t22_pts  = round(6.0 * lr_s, 2)
        t22_ok   = lr_s >= 0.70

        if lr_s >= 0.85:
            lr_label = "robust (wide LR range, forgiving)"
        elif lr_s >= 0.70:
            lr_label = "sensitive (needs careful LR tuning)"
        else:
            lr_label = "fragile (narrow LR window, easy to diverge)"

        results.append(TrainingCheck(
            name          = "T2.2: LR sensitivity",
            dimension     = "T2",
            passed        = t22_ok,
            points_earned = t22_pts,
            points_max    = 6.0,
            detail        = (f"{'✓' if t22_ok else '✗'} LR sensitivity={lr_s:.3f} "
                             f"({lr_label})  [{t22_pts:.1f}/6 pts]"),
            insight       = (_lr_recommendation(cfg) if not t22_ok else ""),
            severity      = "critical" if lr_s < 0.50 else ("warning" if not t22_ok else "info"),
        ))

        # ── T2.3: Convergence Step Estimate (5 pts) ───────────────────────────
        # Apakah steps yang dibutuhkan masuk akal untuk GPU ini?
        # Reference: 100K–500K steps adalah range training yang umum.
        # Terlalu sedikit (<50K): under-trained. Terlalu banyak (>2M): tidak praktis.
        chin_tokens = self._total_training_tokens(cfg)
        steps_est   = self._steps_for_tokens(cfg, chin_tokens)

        STEPS_LOW  = 50_000
        STEPS_HIGH = 1_000_000
        STEPS_OPT  = 200_000

        if STEPS_LOW <= steps_est <= STEPS_HIGH:
            # Semakin dekat ke optimal, semakin tinggi skor
            log_dist = abs(math.log(steps_est) - math.log(STEPS_OPT))
            t23_pts  = max(2.0, 5.0 - log_dist * 1.5)
            t23_pts  = min(5.0, t23_pts)
            t23_ok   = True
            t23_det  = (f"✓ Steps≈{steps_est:,.0f} (practical training horizon)  "
                        f"[{t23_pts:.1f}/5 pts]")
            t23_ins  = ""
        elif steps_est < STEPS_LOW:
            deficit  = STEPS_LOW / max(1, steps_est)
            t23_pts  = max(0.5, 5.0 - deficit * 2)
            t23_ok   = False
            t23_det  = (f"✗ Steps≈{steps_est:,.0f} < {STEPS_LOW:,} (too few — "
                        f"model likely under-trained)  [{t23_pts:.1f}/5 pts]")
            t23_ins  = ("Naikkan batch_size atau tambah params untuk menjamin "
                        f"≥{STEPS_LOW:,} training steps")
        else:
            excess   = steps_est / STEPS_HIGH
            t23_pts  = max(0.0, 5.0 - math.log(excess) * 3)
            t23_ok   = False
            t23_det  = (f"~ Steps≈{steps_est:,.0f} > {STEPS_HIGH:,} (very long training — "
                        f"consider larger batch_size)  [{t23_pts:.1f}/5 pts]")
            t23_ins  = f"Naikkan batch_size untuk mengurangi steps ke sekitar {STEPS_OPT:,}"

        results.append(TrainingCheck(
            name          = "T2.3: Convergence steps estimate",
            dimension     = "T2",
            passed        = t23_ok,
            points_earned = round(t23_pts, 2),
            points_max    = 5.0,
            detail        = t23_det,
            insight       = t23_ins,
            severity      = "warning" if not t23_ok else "info",
        ))

        # ── T2.4: Depth-Appropriate LR Range (4 pts) ──────────────────────────
        # Model dalam membutuhkan LR lebih kecil untuk konvergen stabil.
        # Reference: LR_effective ≈ base_lr / sqrt(num_layers / 12)
        # Kita mengaudit apakah optimizer_type sesuai untuk depth ini.
        depth = cfg.num_layers

        if depth <= 12:
            # Shallow: Lion dan Adam sama bagusnya
            t24_pts = 4.0
            t24_det = f"✓ Shallow model (L={depth}): semua optimizer LR-compatible"
            t24_ok  = True
            t24_ins = ""
        elif depth <= 24:
            # Medium depth: Lion mulai butuh tune yang lebih hati-hati
            if cfg.optimizer_type == OptimizerType.LION:
                t24_pts = 2.5
                t24_det = (f"~ L={depth} dengan Lion: perlu LR ~10× lebih kecil dari Adam. "
                           f"Pastikan lr ≤ 1e-4  [2.5/4 pts]")
                t24_ok  = False
                t24_ins = "Gunakan Adam FP32 untuk model medium-depth untuk LR robustness"
            else:
                t24_pts = 4.0
                t24_det = f"✓ L={depth}: optimizer {cfg.optimizer_type.name} cocok untuk depth ini"
                t24_ok  = True
                t24_ins = ""
        else:
            # Deep model (>24 layers): butuh Adam FP32 atau Lion dengan sangat hati-hati
            if cfg.optimizer_type == OptimizerType.ADAM_FP32:
                t24_pts = 3.5
                t24_det = (f"✓ L={depth} (deep) dengan Adam FP32: robust, "
                           f"tapi perlu warmup panjang & LR schedule yang tepat  [3.5/4 pts]")
                t24_ok  = True
                t24_ins = "Rekomendasikan cosine decay dengan warmup ≥5000 steps"
            elif cfg.optimizer_type == OptimizerType.LION:
                t24_pts = 1.0
                t24_det = (f"✗ L={depth} (deep) dengan Lion: sangat fragile — "
                           f"sering diverge tanpa extensive grid search  [1.0/4 pts]")
                t24_ok  = False
                t24_ins = "Ganti ke Adam FP32 untuk model dengan L>24 layers"
            elif cfg.optimizer_type == OptimizerType.ADAM_8BIT:
                t24_pts = 2.0
                t24_det = (f"~ L={depth} dengan Adam 8-bit: quantized states dapat "
                           f"menyebabkan instability di deep model  [2.0/4 pts]")
                t24_ok  = False
                t24_ins = "Pertimbangkan Adam FP32 untuk training stability"
            else:
                t24_pts = 3.0
                t24_det = (f"✓ L={depth} dengan {cfg.optimizer_type.name}: acceptable  [3.0/4 pts]")
                t24_ok  = True
                t24_ins = ""

        results.append(TrainingCheck(
            name          = "T2.4: Depth-LR compatibility",
            dimension     = "T2",
            passed        = t24_ok,
            points_earned = t24_pts,
            points_max    = 4.0,
            detail        = t24_det,
            insight       = t24_ins,
            severity      = "critical" if t24_pts <= 1.0 else ("warning" if not t24_ok else "info"),
        ))

        # ── T2.5: Warmup Budget Adequacy (3 pts) ──────────────────────────────
        # Warmup yang cukup penting untuk model dalam dan optimizer non-standard.
        # Aturan empiris: warmup ≥ sqrt(total_steps) steps, minimum 1000.
        total_steps = self._steps_for_tokens(cfg, chin_tokens)
        warmup_needed = max(1000, math.sqrt(total_steps))
        warmup_pct    = warmup_needed / max(1, total_steps) * 100

        # Faktor yang mempengaruhi kebutuhan warmup:
        warmup_need_factor = 1.0
        if cfg.num_layers > 24:
            warmup_need_factor *= 1.5   # deep model butuh warmup lebih lama
        if cfg.optimizer_type == OptimizerType.LION:
            warmup_need_factor *= 2.0   # Lion sangat sensitif di awal training
        if cfg.optimizer_type == OptimizerType.ADAM_8BIT:
            warmup_need_factor *= 1.3

        adj_warmup = warmup_needed * warmup_need_factor
        adj_warmup_pct = adj_warmup / max(1, total_steps) * 100

        if adj_warmup_pct <= 5.0:
            # Warmup reasonable (<5% dari total steps)
            t25_pts = 3.0
            t25_ok  = True
            t25_det = (f"✓ Warmup≈{adj_warmup:.0f} steps ({adj_warmup_pct:.1f}% of "
                       f"total {total_steps:.0f}) — practical")
            t25_ins = ""
        elif adj_warmup_pct <= 15.0:
            t25_pts = 2.0
            t25_ok  = True
            t25_det = (f"~ Warmup≈{adj_warmup:.0f} steps ({adj_warmup_pct:.1f}%) — "
                       f"substantial but manageable  [2.0/3 pts]")
            t25_ins = "Training efficiency bisa ditingkatkan dengan batch_size yang lebih besar"
        else:
            t25_pts = 0.5
            t25_ok  = False
            t25_det = (f"✗ Warmup≈{adj_warmup:.0f} steps ({adj_warmup_pct:.1f}%) — "
                       f"terlalu besar dari total training budget  [0.5/3 pts]")
            t25_ins = ("Total steps terlalu sedikit untuk warmup yang memadai. "
                       "Naikkan batch_size ATAU ganti ke optimizer yang kurang sensitif")

        results.append(TrainingCheck(
            name          = "T2.5: Warmup budget adequacy",
            dimension     = "T2",
            passed        = t25_ok,
            points_earned = t25_pts,
            points_max    = 3.0,
            detail        = t25_det,
            insight       = t25_ins,
            severity      = "warning" if not t25_ok else "info",
        ))

        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  T3: TRAINING STABILITY  (20 pts)
    # ══════════════════════════════════════════════════════════════════════════
    #   T3.1  Mixed precision numerical stability    6 pts  (graduated)
    #   T3.2  Loss landscape smoothness              6 pts  (graduated)
    #   T3.3  Gradient checkpointing correctness     4 pts  (graduated)
    #   T3.4  MoE routing stability (if applicable)  4 pts  (or 4 pts non-MoE bonus)
    #                                               ──────
    #                                               20 pts

    def _dim_t3_stability(self, cfg: ArchConfig) -> List[TrainingCheck]:
        results = []

        # ── T3.1: Mixed Precision Stability (6 pts) ───────────────────────────
        mp_stab = self._mixed_precision_stability(cfg)
        t31_pts = round(6.0 * mp_stab, 2)
        t31_ok  = mp_stab >= 0.80

        stab_issues = []
        if not cfg.use_mixed_precision:
            stab_issues.append("FP32-only: sangat lambat, kehilangan throughput >2×")
        if cfg.optimizer_type == OptimizerType.ADAM_8BIT:
            stab_issues.append("Adam 8-bit: quantization noise di optimizer states")
        if cfg.optimizer_type == OptimizerType.ADAMW_BF16:
            stab_issues.append("AdamW BF16 states: potential underflow untuk gradient kecil")
        if cfg.num_layers > 36 and cfg.use_mixed_precision:
            stab_issues.append(f"Deep model (L={cfg.num_layers}) dengan BF16: overflow risk di final layers")

        results.append(TrainingCheck(
            name          = "T3.1: Mixed precision stability",
            dimension     = "T3",
            passed        = t31_ok,
            points_earned = t31_pts,
            points_max    = 6.0,
            detail        = (f"{'✓' if t31_ok else '~'} MP stability={mp_stab:.3f}  "
                             + (f"Issues: {'; '.join(stab_issues)}" if stab_issues
                                else "BF16+FP32 optimizer — excellent")
                             + f"  [{t31_pts:.1f}/6 pts]"),
            insight       = (_mp_recommendation(cfg) if not t31_ok else ""),
            severity      = "warning" if not t31_ok and mp_stab >= 0.60 else
                           ("critical" if not t31_ok else "info"),
        ))

        # ── T3.2: Loss Landscape Smoothness (6 pts) ───────────────────────────
        # Seberapa smooth loss surface arsitektur ini.
        # Faktor: FFN type, norm type, dropout, tied embeddings, head dim size.
        landscape_score = 1.0

        # FFN type: GeGLU/SwiGLU lebih smooth dari ReLU
        if cfg.ffn_type == FFNType.GEGLU:
            landscape_score *= 1.05   # sangat smooth
        elif cfg.ffn_type == FFNType.DENSE:
            pass   # SwiGLU, ok
        elif cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            landscape_score *= 0.85   # routing discontinuity
        elif cfg.ffn_type == FFNType.GATED:
            landscape_score *= 1.00

        # Head dim yang terlalu kecil → attention lebih noisy
        if cfg.head_dim < 32:
            landscape_score *= 0.85
        elif cfg.head_dim >= 64:
            landscape_score *= 1.02

        # Dropout 0: cleaner gradient signal, less stochastic noise
        if cfg.dropout == 0.0:
            landscape_score *= 1.03
        elif cfg.dropout > 0.1:
            landscape_score *= 0.92   # high dropout = noisy gradients

        # Tied embeddings: shared parameter → implicit regularization
        if cfg.tie_embeddings:
            landscape_score *= 1.02

        # Norm type
        if cfg.norm_type == NormType.RMSNORM:
            landscape_score *= 1.03
        elif cfg.norm_type == NormType.GROUPNORM:
            landscape_score *= 0.90

        # Cap at 1.0
        landscape_score = min(1.0, landscape_score)
        t32_pts = round(6.0 * landscape_score, 2)
        t32_ok  = landscape_score >= 0.80

        results.append(TrainingCheck(
            name          = "T3.2: Loss landscape smoothness",
            dimension     = "T3",
            passed        = t32_ok,
            points_earned = t32_pts,
            points_max    = 6.0,
            detail        = (f"{'✓' if t32_ok else '~'} Landscape smoothness={landscape_score:.3f} "
                             f"(FFN={cfg.ffn_type.name}, head_dim={cfg.head_dim}, "
                             f"dropout={cfg.dropout})  [{t32_pts:.1f}/6 pts]"),
            insight       = ("Pertimbangkan GeGLU FFN dan dropout=0.0 untuk smoother loss landscape"
                             if not t32_ok else ""),
            severity      = "warning" if not t32_ok else "info",
        ))

        # ── T3.3: Gradient Checkpointing Correctness (4 pts) ──────────────────
        # GC adalah trade-off: mengurangi VRAM tapi menambah compute (recompute fwd).
        # Yang penting: apakah GC diperlukan (fits_gpu), dan apakah konsekuensinya dipahami?
        uses_gc = cfg.use_gradient_checkpointing
        needs_gc = cfg.vram_usage_pct > 70.0   # biasanya diperlukan jika VRAM tight

        if not uses_gc and not needs_gc:
            # Ideal: tidak pakai GC karena tidak perlu
            t33_pts = 4.0
            t33_det = (f"✓ GC disabled — tidak diperlukan (VRAM={cfg.vram_usage_pct:.1f}%). "
                       f"Gradient computation optimal")
            t33_ok  = True
            t33_ins = ""
        elif uses_gc and needs_gc:
            # GC diperlukan dan aktif: correct tapi ada trade-off
            gc_recompute_pct = 33.0   # standard transformer: ~33% extra compute
            t33_pts = 3.0
            t33_det = (f"~ GC enabled (VRAM={cfg.vram_usage_pct:.1f}% — diperlukan). "
                       f"Trade-off: +{gc_recompute_pct:.0f}% compute, "
                       f"training time ↑  [3.0/4 pts]")
            t33_ok  = True
            t33_ins = "Jika VRAM memungkinkan, disable GC untuk mempercepat training"
        elif uses_gc and not needs_gc:
            # GC aktif padahal tidak diperlukan: membuang compute
            t33_pts = 1.5
            t33_det = (f"✗ GC enabled tapi VRAM={cfg.vram_usage_pct:.1f}% — tidak diperlukan! "
                       f"Membuang ~33% compute sia-sia  [1.5/4 pts]")
            t33_ok  = False
            t33_ins = "Disable gradient checkpointing — VRAM tersedia cukup"
        else:
            # Tidak pakai GC tapi VRAM sangat tight
            t33_pts = 2.0
            t33_det = (f"~ GC disabled tapi VRAM={cfg.vram_usage_pct:.1f}% tight — "
                       f"risiko OOM di edge case  [2.0/4 pts]")
            t33_ok  = False
            t33_ins = "Aktifkan GC sebagai safety net, atau kurangi batch_size"

        results.append(TrainingCheck(
            name          = "T3.3: Gradient checkpointing",
            dimension     = "T3",
            passed        = t33_ok,
            points_earned = t33_pts,
            points_max    = 4.0,
            detail        = t33_det,
            insight       = t33_ins,
            severity      = "warning" if not t33_ok else "info",
        ))

        # ── T3.4: MoE Routing Stability / Architecture Cohesion (4 pts) ───────
        moe_q = self._moe_routing_quality(cfg)

        if moe_q is not None:
            # MoE model: audit routing stability
            t34_pts = round(4.0 * moe_q, 2)
            t34_ok  = moe_q >= 0.75

            moe_issues = []
            if cfg.expert_capacity_factor < 1.25:
                moe_issues.append(f"capacity_factor={cfg.expert_capacity_factor:.2f} < 1.25 (token dropping)")
            if cfg.top_k_experts == 1:
                moe_issues.append("top_k=1 (sparse gradient — expert collapse risk)")
            if cfg.param_count / max(1, cfg.num_experts) < 10e6:
                moe_issues.append(f"params/expert={cfg.param_count/cfg.num_experts/1e6:.1f}M < 10M (under-specialized)")

            results.append(TrainingCheck(
                name          = "T3.4: MoE routing stability",
                dimension     = "T3",
                passed        = t34_ok,
                points_earned = t34_pts,
                points_max    = 4.0,
                detail        = (f"{'✓' if t34_ok else '~'} MoE routing quality={moe_q:.3f} "
                                 f"(E={cfg.num_experts}, top_k={cfg.top_k_experts}, "
                                 f"cap={cfg.expert_capacity_factor:.2f})  [{t34_pts:.1f}/4 pts]"
                                 + (f"  Issues: {'; '.join(moe_issues)}" if moe_issues else "")),
                insight       = ("Naikan capacity_factor ke ≥1.25 dan pastikan top_k≥2"
                                 if not t34_ok else ""),
                severity      = "warning" if not t34_ok else "info",
            ))
        else:
            # Non-MoE: berikan poin untuk simplicity bonus (non-MoE lebih mudah debug + train)
            t34_pts = 3.5   # Slight bonus tapi bukan full marks (MoE bisa lebih bagus jika benar)
            results.append(TrainingCheck(
                name          = "T3.4: Architecture cohesion (non-MoE)",
                dimension     = "T3",
                passed        = True,
                points_earned = t34_pts,
                points_max    = 4.0,
                detail        = (f"✓ Dense FFN ({cfg.ffn_type.name}): no routing instability, "
                                 f"deterministic training  [3.5/4 pts]"),
                insight       = "",
                severity      = "info",
            ))

        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  T4: SAMPLE EFFICIENCY  (15 pts)
    # ══════════════════════════════════════════════════════════════════════════
    #   T4.1  Batch gradient quality             6 pts  (graduated)
    #   T4.2  Embedding efficiency               4 pts  (graduated)
    #   T4.3  Sequence length utilization        5 pts  (graduated)
    #                                           ──────
    #                                           15 pts

    def _dim_t4_sample_efficiency(self, cfg: ArchConfig) -> List[TrainingCheck]:
        results = []

        # ── T4.1: Batch Gradient Quality (6 pts) ──────────────────────────────
        # Kualitas estimasi gradien dari batch yang tersedia.
        # Effective signal = sqrt(batch × seq / noise_floor)
        bgq   = self._batch_gradient_quality(cfg)
        t41_pts = round(6.0 * bgq, 2)
        t41_ok  = bgq >= 0.70

        eff_tok = cfg.batch_size * cfg.seq_len
        if eff_tok >= 8192:
            bgq_label = "excellent gradient signal"
        elif eff_tok >= 4096:
            bgq_label = "good gradient signal"
        elif eff_tok >= 2048:
            bgq_label = "moderate gradient signal"
        else:
            bgq_label = "noisy gradient signal — high variance updates"

        results.append(TrainingCheck(
            name          = "T4.1: Batch gradient quality",
            dimension     = "T4",
            passed        = t41_ok,
            points_earned = t41_pts,
            points_max    = 6.0,
            detail        = (f"{'✓' if t41_ok else '✗'} Batch gradient quality={bgq:.3f} "
                             f"({bgq_label}) | effective_tokens={eff_tok:,} "
                             f"(bs={cfg.batch_size}×seq={cfg.seq_len})  [{t41_pts:.1f}/6 pts]"),
            insight       = (f"Naikkan batch_size atau seq_len agar eff_tokens ≥ {GRAD_NOISE_THRESHOLD}"
                             if not t41_ok else ""),
            severity      = "warning" if eff_tok < 1024 else ("info" if t41_ok else "warning"),
        ))

        # ── T4.2: Embedding Efficiency (4 pts) ────────────────────────────────
        # Tied embeddings: input embedding = output projection
        # Ini menghemat params dan memberikan implicit regularization.
        # Tapi: untuk vocab_size >> hidden_dim, tied embeddings bisa bottleneck gradient.
        vocab_dim_ratio = cfg.vocab_size / max(1, cfg.hidden_dim)

        if cfg.tie_embeddings:
            if vocab_dim_ratio <= 50:
                # Optimal: vocab tidak terlalu besar relative to hidden_dim
                t42_pts = 4.0
                t42_det = (f"✓ Tied embeddings (vocab/hidden={vocab_dim_ratio:.1f}×): "
                           f"saves {2*cfg.vocab_size*cfg.hidden_dim/1e6:.1f}M params, "
                           f"implicit regularization")
                t42_ins = ""
                t42_ok  = True
            else:
                # Vocab sangat besar relative ke hidden_dim → gradient bottleneck
                t42_pts = 2.5
                t42_det = (f"~ Tied embeddings dengan vocab/hidden={vocab_dim_ratio:.1f}× "
                           f"(besar) — gradient sharing mungkin bottleneck  [2.5/4 pts]")
                t42_ins = (f"Pertimbangkan untied embeddings jika vocab_size >> hidden_dim "
                           f"({cfg.vocab_size} vs {cfg.hidden_dim})")
                t42_ok  = False
        else:
            # Untied: lebih fleksibel tapi lebih banyak params
            t42_pts = 2.0
            t42_det = (f"~ Untied embeddings: {2*cfg.vocab_size*cfg.hidden_dim/1e6:.1f}M extra params "
                       f"(vocab×hidden×2). Kurang efisien tapi lebih fleksibel  [2.0/4 pts]")
            t42_ins = "Aktifkan tie_embeddings untuk efisiensi parameter dan regularisasi gratis"
            t42_ok  = False

        results.append(TrainingCheck(
            name          = "T4.2: Embedding efficiency",
            dimension     = "T4",
            passed        = t42_ok,
            points_earned = t42_pts,
            points_max    = 4.0,
            detail        = t42_det,
            insight       = t42_ins,
            severity      = "info",
        ))

        # ── T4.3: Sequence Length Utilization (5 pts) ─────────────────────────
        # Apakah seq_len sesuai dengan kebutuhan arsitektur dan memory budget?
        # Terlalu panjang tanpa cukup layers → model tidak bisa model long-range
        # Terlalu pendek → tidak mempelajari long-context dependencies
        seq = cfg.seq_len
        depth = cfg.num_layers

        # Receptive field: dalam Transformer, setiap layer melihat seluruh seq.
        # Tapi untuk modelling yang efektif, depth menentukan kompresi informasi.
        # Rule: seq / (hidden_dim × 0.5) → bits of context per parameter
        # Optimal range: seq sekitar 2–16× dari batch-normalized processing window
        context_efficiency = min(1.0, (seq * depth) / (cfg.hidden_dim * 40))

        # Bonus untuk seq panjang dengan Flash Attention (FA handle memory)
        fa_bonus = 0.10 if cfg.use_flash_attn and seq >= 2048 else 0.0

        # Penalty untuk seq yang jauh melebihi apa yang attention dapat handle efektif
        if seq > 4096 and cfg.attn_type not in (AttentionType.SLIDE, AttentionType.HYBRID,
                                                  AttentionType.LINEAR, AttentionType.ALIBI):
            # Non-sliding attention dengan seq sangat panjang: quadratic attention cost
            # dan mungkin tidak efektif dipelajari
            seq_penalty = min(0.20, (seq - 4096) / 20000)
        else:
            seq_penalty = 0.0

        seq_score = min(1.0, context_efficiency + fa_bonus - seq_penalty)
        seq_score = max(0.0, seq_score)
        t43_pts   = round(5.0 * seq_score, 2)
        t43_ok    = seq_score >= 0.70

        results.append(TrainingCheck(
            name          = "T4.3: Sequence length utilization",
            dimension     = "T4",
            passed        = t43_ok,
            points_earned = t43_pts,
            points_max    = 5.0,
            detail        = (f"{'✓' if t43_ok else '~'} Seq utilization={seq_score:.3f} "
                             f"(seq={seq}, L={depth}, D={cfg.hidden_dim}, "
                             f"FA={'on' if cfg.use_flash_attn else 'off'})  [{t43_pts:.1f}/5 pts]"),
            insight       = (_seq_recommendation(cfg) if not t43_ok else ""),
            severity      = "info",
        ))

        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  T5: OPTIMIZER COMPATIBILITY  (15 pts)
    # ══════════════════════════════════════════════════════════════════════════
    #   T5.1  Optimizer–architecture fit          6 pts  (graduated)
    #   T5.2  Optimizer state quality             5 pts  (graduated)
    #   T5.3  Multi-GPU / sharding readiness      4 pts  (graduated)
    #                                            ──────
    #                                            15 pts

    def _dim_t5_optimizer_compat(self, cfg: ArchConfig) -> List[TrainingCheck]:
        results = []

        # ── T5.1: Optimizer–Architecture Fit (6 pts) ──────────────────────────
        # Seberapa cocok optimizer dengan karakteristik arsitektur.
        opt = cfg.optimizer_type
        depth = cfg.num_layers
        is_moe = cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)

        # Scoring matrix berdasarkan architecture profile
        # (optimizer, is_deep, is_moe) → score
        score_51 = 1.0
        detail_51_extra = []

        if opt == OptimizerType.ADAM_FP32:
            score_51 = 1.0   # universal, terbaik untuk sebagian besar kasus
            detail_51_extra.append("Adam FP32: gold standard, highly compatible")
        elif opt == OptimizerType.ZERO1:
            score_51 = 0.95  # ZeRO-1 optimal multi-GPU
            detail_51_extra.append("ZeRO-1: excellent for multi-GPU training")
        elif opt == OptimizerType.ZERO2:
            score_51 = 0.92
            detail_51_extra.append("ZeRO-2: very good for large models, multi-GPU")
        elif opt == OptimizerType.ZERO3:
            # ZeRO-3 sangat bagus untuk large model tapi overhead komunikasi tinggi
            if cfg.param_count > 1e9:
                score_51 = 0.90
                detail_51_extra.append("ZeRO-3: justified for >1B params, high comm overhead")
            else:
                score_51 = 0.70
                detail_51_extra.append("ZeRO-3: overkill for sub-1B model, comm overhead wasted")
        elif opt == OptimizerType.ADAMW_BF16:
            # BF16 states: sedikit kurang stabil, tapi ok untuk model tidak terlalu dalam
            if depth <= 24:
                score_51 = 0.88
                detail_51_extra.append("AdamW BF16: ok for shallow models")
            else:
                score_51 = 0.72
                detail_51_extra.append("AdamW BF16: risky for deep models (gradient underflow)")
        elif opt == OptimizerType.LION:
            # Lion: bagus untuk shallow, berisiko untuk deep
            if depth <= 12:
                score_51 = 0.92
                detail_51_extra.append("Lion: excellent for shallow models, efficient memory")
            elif depth <= 24:
                score_51 = 0.75
                detail_51_extra.append("Lion: workable but needs careful LR tuning")
            else:
                score_51 = 0.50
                detail_51_extra.append("Lion: fragile for deep models (L>24)")
        elif opt == OptimizerType.ADAM_8BIT:
            # 8-bit: hemat VRAM tapi precision loss
            if is_moe:
                score_51 = 0.60   # MoE + 8-bit: expert states ter-quantize agresif
                detail_51_extra.append("Adam 8-bit + MoE: quantized expert states — unstable")
            elif depth > 36:
                score_51 = 0.65
                detail_51_extra.append("Adam 8-bit + deep model: precision loss compounds")
            else:
                score_51 = 0.78
                detail_51_extra.append("Adam 8-bit: good VRAM savings, moderate precision loss")

        # Additional penalties/bonuses
        if is_moe and opt not in (OptimizerType.ADAM_FP32, OptimizerType.ZERO2, OptimizerType.ZERO3):
            score_51 *= 0.90
            detail_51_extra.append("MoE butuh optimizer dengan stable states untuk router")

        score_51 = min(1.0, max(0.0, score_51))
        t51_pts  = round(6.0 * score_51, 2)
        t51_ok   = score_51 >= 0.75

        results.append(TrainingCheck(
            name          = "T5.1: Optimizer-architecture fit",
            dimension     = "T5",
            passed        = t51_ok,
            points_earned = t51_pts,
            points_max    = 6.0,
            detail        = (f"{'✓' if t51_ok else '~'} Fit={score_51:.3f} "
                             f"({cfg.optimizer_type.value[:40]})  [{t51_pts:.1f}/6 pts]  "
                             f"| {' | '.join(detail_51_extra)}"),
            insight       = (_optimizer_recommendation(cfg) if not t51_ok else ""),
            severity      = "critical" if score_51 < 0.50 else ("warning" if not t51_ok else "info"),
        ))

        # ── T5.2: Optimizer State Quality (5 pts) ─────────────────────────────
        # Kualitas optimizer states berpengaruh langsung pada parameter update quality.
        opt_quality = 1.0
        opt_detail_parts = []

        if opt in (OptimizerType.ADAM_FP32, OptimizerType.ZERO1,
                   OptimizerType.ZERO2, OptimizerType.ZERO3):
            opt_quality = 1.0
            opt_detail_parts.append("FP32 states: full precision parameter updates")
        elif opt == OptimizerType.ADAMW_BF16:
            opt_quality = 0.85
            opt_detail_parts.append("BF16 states: ~3 decimal digit precision (vs FP32 7 digits)")
        elif opt == OptimizerType.LION:
            opt_quality = 0.88
            opt_detail_parts.append("Lion: single momentum, efficient but less smooth updates")
        elif opt == OptimizerType.ADAM_8BIT:
            opt_quality = 0.72
            opt_detail_parts.append("8-bit quantization: ~2 decimal digit precision")

        # Penalty: tidak menggunakan mixed precision dengan optimizer berkualitas tinggi
        if not cfg.use_mixed_precision and opt_quality >= 0.85:
            opt_quality *= 0.95   # FP32 semua — slower tapi bukan masalah kualitas

        t52_pts = round(5.0 * opt_quality, 2)
        t52_ok  = opt_quality >= 0.80

        # Hitung VRAM overhead optimizer
        opt_overhead = cfg.vram_optimizer_gb
        results.append(TrainingCheck(
            name          = "T5.2: Optimizer state quality",
            dimension     = "T5",
            passed        = t52_ok,
            points_earned = t52_pts,
            points_max    = 5.0,
            detail        = (f"{'✓' if t52_ok else '~'} State quality={opt_quality:.3f} "
                             f"(VRAM={opt_overhead:.2f}GB for optimizer)  [{t52_pts:.1f}/5 pts]  "
                             f"| {' | '.join(opt_detail_parts)}"),
            insight       = ("Upgrade ke Adam FP32 untuk state quality yang lebih baik"
                             if not t52_ok else ""),
            severity      = "warning" if not t52_ok else "info",
        ))

        # ── T5.3: Multi-GPU / Sharding Readiness (4 pts) ──────────────────────
        # Apakah optimizer dan arsitektur siap untuk scaling ke multi-GPU?
        gpu = self.gpu
        has_nvlink = gpu.nvlink_version > 0

        sharding_score = 1.0
        sharding_notes = []

        if opt in (OptimizerType.ZERO1, OptimizerType.ZERO2, OptimizerType.ZERO3):
            sharding_score = 1.0
            sharding_notes.append(f"ZeRO-{opt.name[-1]}: designed for multi-GPU, linear scaling")
        elif opt == OptimizerType.ADAM_FP32:
            sharding_score = 0.85   # bisa DDP tapi tidak seoptimal ZeRO
            sharding_notes.append("Adam FP32 + DDP: ok untuk multi-GPU, ZeRO lebih efisien")
        else:
            sharding_score = 0.65   # tidak optimal untuk multi-GPU scaling
            sharding_notes.append(f"{opt.name}: limited multi-GPU support")

        # NVLink bonus: transfer optimizer states lebih efisien
        if has_nvlink:
            sharding_score = min(1.0, sharding_score + 0.10)
            sharding_notes.append(f"NVLink v{gpu.nvlink_version}: efficient cross-GPU optimizer sync")

        # MoE additional consideration
        if is_moe and opt in (OptimizerType.ZERO2, OptimizerType.ZERO3):
            # ZeRO + MoE: expert sharding bekerja dengan baik
            sharding_score = min(1.0, sharding_score + 0.05)
            sharding_notes.append("ZeRO+MoE: expert parallelism compatible")
        elif is_moe:
            sharding_score *= 0.90
            sharding_notes.append("MoE tanpa ZeRO: expert routing state harus disinkronkan manual")

        t53_pts = round(4.0 * sharding_score, 2)
        t53_ok  = sharding_score >= 0.70

        results.append(TrainingCheck(
            name          = "T5.3: Multi-GPU / sharding readiness",
            dimension     = "T5",
            passed        = t53_ok,
            points_earned = t53_pts,
            points_max    = 4.0,
            detail        = (f"{'✓' if t53_ok else '~'} Sharding readiness={sharding_score:.3f}  "
                             f"[{t53_pts:.1f}/4 pts]  | {' | '.join(sharding_notes)}"),
            insight       = ("Pertimbangkan ZeRO-1/2 untuk scaling efisien ke multi-GPU"
                             if not t53_ok else ""),
            severity      = "info",
        ))

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _describe_training_regime(cfg: ArchConfig) -> str:
    """Menghasilkan deskripsi singkat karakteristik training arsitektur ini."""
    parts = []

    dw = cfg.num_layers / max(1.0, math.sqrt(cfg.hidden_dim))
    if dw > 0.55:
        parts.append("deep-narrow (high gradient depth)")
    elif dw < 0.15:
        parts.append("wide-shallow (high capacity per layer)")
    else:
        parts.append("balanced depth-width")

    if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
        parts.append(f"sparse-MoE (E={cfg.num_experts})")
    elif cfg.ffn_type == FFNType.GEGLU:
        parts.append("GeGLU-smooth")

    if cfg.optimizer_type == OptimizerType.LION:
        parts.append("momentum-only (Lion)")
    elif cfg.optimizer_type in (OptimizerType.ZERO2, OptimizerType.ZERO3):
        parts.append("ZeRO-sharded")

    if cfg.use_gradient_checkpointing:
        parts.append("memory-optimized (GC on)")

    eff_tok = cfg.batch_size * cfg.seq_len
    if eff_tok < 2048:
        parts.append("micro-batch (noisy grads)")
    elif eff_tok >= 8192:
        parts.append("large-batch (stable grads)")

    return " | ".join(parts) if parts else "standard"


def _lr_recommendation(cfg: ArchConfig) -> str:
    """Rekomendasi learning rate berdasarkan arsitektur."""
    depth = cfg.num_layers
    opt   = cfg.optimizer_type

    if opt == OptimizerType.LION:
        base_lr = 1e-4 / (depth / 12) ** 0.5
        return (f"Lion dengan L={depth}: gunakan lr≈{base_lr:.2e}, warmup ≥5000 steps, "
                f"cosine decay, weight_decay=0.1")
    elif opt == OptimizerType.ADAM_8BIT:
        base_lr = 3e-4 / (depth / 12) ** 0.4
        return (f"Adam 8-bit dengan L={depth}: lr≈{base_lr:.2e}, pertimbangkan "
                f"gradient clipping max_norm=1.0")
    else:
        base_lr = 3e-4 / math.sqrt(depth / 12)
        return (f"Rekomendasikan lr≈{base_lr:.2e} dengan cosine warmup {max(1000, int(math.sqrt(depth)*500))} steps")


def _mp_recommendation(cfg: ArchConfig) -> str:
    """Rekomendasi mixed precision."""
    if not cfg.use_mixed_precision:
        return "Aktifkan mixed precision (BF16) untuk throughput 2× lebih tinggi"
    if cfg.optimizer_type == OptimizerType.ADAM_8BIT:
        return "Pertimbangkan upgrade ke Adam FP32 untuk menghindari double quantization (BF16 activation + 8-bit states)"
    if cfg.optimizer_type == OptimizerType.ADAMW_BF16:
        return "Monitor gradient norm carefully — BF16 states dapat underflow untuk small gradients"
    return ""


def _optimizer_recommendation(cfg: ArchConfig) -> str:
    """Rekomendasi penggantian optimizer."""
    depth = cfg.num_layers
    is_moe = cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
    params = cfg.param_count

    if is_moe:
        return f"Untuk MoE: gunakan Adam FP32 (single GPU) atau ZeRO-2 (multi GPU)"
    if depth > 36:
        return f"Untuk deep model (L={depth}): Adam FP32 adalah pilihan terbaik"
    if params > 500e6:
        return "Untuk model >500M params: pertimbangkan ZeRO-1 atau ZeRO-2"
    return "Adam FP32 adalah pilihan universal yang paling aman"


def _seq_recommendation(cfg: ArchConfig) -> str:
    """Rekomendasi sequence length."""
    if cfg.seq_len > 4096 and cfg.attn_type not in (AttentionType.SLIDE, AttentionType.HYBRID):
        return (f"seq={cfg.seq_len} dengan non-sliding attention: ganti ke SLIDE atau HYBRID attention, "
                f"atau kurangi seq_len ke ≤4096")
    if cfg.seq_len * cfg.num_layers < cfg.hidden_dim * 10:
        return "Naikkan seq_len atau num_layers untuk konteks yang lebih kaya"
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED SCORE  (Public API untuk integrasi pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_training_aware_combined(
    quality_pct:   float,
    fitness_score: float,
    training_pct:  float,
    *,
    w_quality:  float = 0.25,
    w_fitness:  float = 0.40,
    w_training: float = 0.35,
) -> float:
    """
    Skor gabungan baru yang seimbang antara hardware-aware dan training-aware.

    Parameters
    ──────────
    quality_pct   : 0–100, dari ArcQualityScorer.score().pct
    fitness_score : 0–1,   dari generator._fitness_score()
    training_pct  : 0–100, dari TrainingAwareScorer.score().pct
    w_quality     : bobot dimensi kualitas (default 0.25)
    w_fitness     : bobot dimensi hardware/GPU (default 0.40)
    w_training    : bobot dimensi training (default 0.35)

    Returns
    ───────
    float: combined score 0.0–1.0
    """
    assert abs(w_quality + w_fitness + w_training - 1.0) < 1e-6, \
        "Bobot harus berjumlah 1.0"
    return round(
        w_quality  * (quality_pct  / 100.0) +
        w_fitness  * fitness_score           +
        w_training * (training_pct / 100.0),
        5,
    )


def compute_training_fitness_breakdown(
    cfg:         ArchConfig,
    quality_pct: float,
    gpu:         GPUSpec,
) -> TrainingFitnessBreakdown:
    """
    Hitung TrainingFitnessBreakdown lengkap untuk satu ArchConfig.
    Convenience function untuk integrasi pipeline.
    """
    scorer = TrainingAwareScorer(gpu)
    report = scorer.score(cfg)
    return TrainingFitnessBreakdown(
        arch_id       = cfg.arch_id,
        quality_pct   = quality_pct,
        fitness_score = cfg.fitness_score,
        training_pct  = report.pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_training_score_report(
    report: TrainingScoreReport,
    *,
    console=None,
    verbose: bool = True,
) -> None:
    """
    Cetak laporan training-aware scoring ke console (Rich atau plain text).

    Parameters
    ──────────
    report  : TrainingScoreReport dari TrainingAwareScorer.score()
    console : Rich Console object (optional)
    verbose : Jika True, tampilkan semua checks; jika False, hanya summary
    """
    _use_rich = console is not None

    def _print(msg: str, style: str = "") -> None:
        if _use_rich:
            console.print(msg, style=style) if not style else console.print(f"[{style}]{msg}[/{style}]")
        else:
            # Strip Rich markup untuk plain text
            import re
            clean = re.sub(r'\[/?[^\]]+\]', '', msg)
            print(clean)

    header = (f"  Training-Aware Score: {report.pct:.1f}%  "
              f"({report.total_score:.1f}/100)  —  {report.grade}")
    color  = report.grade_color

    if _use_rich:
        from rich.panel import Panel
        from rich.table import Table
        from rich import box

        # Header panel
        console.print(Panel(
            f"[{color}]{header}[/{color}]",
            title=f"[bold]Training Analysis — {report.arch_id}[/bold]",
            border_style="blue",
        ))

        # Dimension summary table
        table = Table(
            title="Dimension Breakdown",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Dim", style="cyan", width=4)
        table.add_column("Description", width=30)
        table.add_column("Score", justify="right", width=12)
        table.add_column("Pts", justify="right", width=10)

        dim_info = {
            "T1": ("Gradient Flow Health",      25),
            "T2": ("Convergence Dynamics",       25),
            "T3": ("Training Stability",         20),
            "T4": ("Sample Efficiency",          15),
            "T5": ("Optimizer Compatibility",    15),
        }
        for dim, (desc, max_pts) in dim_info.items():
            earned, _max = report.dimension_score(dim)
            pct = earned / max(_max, 0.001) * 100
            color_d = ("green" if pct >= 80 else "yellow" if pct >= 60 else "red")
            table.add_row(
                dim, desc,
                f"[{color_d}]{pct:.1f}%[/{color_d}]",
                f"{earned:.1f}/{max_pts}",
            )
        console.print(table)

        # Key metrics
        console.print(f"  [dim]Training Regime:    {report.training_regime}[/dim]")
        console.print(f"  [dim]Chinchilla Tokens:  {report.chinchilla_optimal_tokens:,.0f}M "
                      f"(Chinchilla-optimal)[/dim]")
        console.print(f"  [dim]Steps Estimate:     {report.estimated_steps_to_target:,.0f} "
                      f"optimizer steps[/dim]")
        console.print(f"  [dim]Gradient Risk:      {report.gradient_attenuation_risk.upper()}[/dim]")
        console.print(f"  [dim]LR Sensitivity:     {report.lr_sensitivity_class.upper()}[/dim]")

        # Critical issues
        if report.critical_issues:
            console.print("\n  [bold red]⚠ CRITICAL ISSUES:[/bold red]")
            for c in report.critical_issues:
                console.print(f"    [red]• {c.name}: {c.detail}[/red]")
                if c.insight:
                    console.print(f"      [dim]→ {c.insight}[/dim]")

        # Verbose: all checks
        if verbose:
            console.print("\n  [bold]All Checks:[/bold]")
            dim_colors = {"T1": "cyan", "T2": "green", "T3": "yellow",
                          "T4": "magenta", "T5": "blue"}
            for c in report.checks:
                dc = dim_colors.get(c.dimension, "white")
                pts_str = f"{c.points_earned:.1f}/{c.points_max:.0f}"
                console.print(
                    f"  [{dc}][{c.dimension}][/{dc}] {c.name:<40} "
                    f"[{'green' if c.passed else 'yellow'}]{pts_str}[/]  "
                    f"{c.detail[:80]}"
                )

    else:
        # Plain text fallback
        print("\n" + "═" * 78)
        print(f"  TRAINING-AWARE SCORE: {report.pct:.1f}%  ({report.grade})")
        print("═" * 78)
        print(f"  Arch:             {report.arch_id} — {report.arch_name[:50]}")
        print(f"  Training Regime:  {report.training_regime}")
        print(f"  Chinchilla Tok:   {report.chinchilla_optimal_tokens:,.0f}M tokens")
        print(f"  Steps Estimate:   {report.estimated_steps_to_target:,.0f} steps")
        print(f"  Gradient Risk:    {report.gradient_attenuation_risk.upper()}")
        print(f"  LR Sensitivity:   {report.lr_sensitivity_class.upper()}")
        print()

        dim_info = [
            ("T1", "Gradient Flow",       25),
            ("T2", "Convergence",         25),
            ("T3", "Stability",           20),
            ("T4", "Sample Efficiency",   15),
            ("T5", "Optimizer Compat",    15),
        ]
        for dim, desc, max_pts in dim_info:
            earned, _ = report.dimension_score(dim)
            pct = earned / max(max_pts, 0.001) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {dim} {desc:<20} [{bar}] {pct:5.1f}%  ({earned:.1f}/{max_pts})")

        if report.critical_issues:
            print("\n  ⚠ CRITICAL ISSUES:")
            for c in report.critical_issues:
                print(f"    • {c.name}: {c.detail[:70]}")

        if verbose:
            print("\n  All Checks:")
            for c in report.checks:
                status = "✓" if c.passed else ("~" if c.points_earned > 0 else "✗")
                print(f"    {status} [{c.dimension}] {c.name:<42} {c.points_earned:.1f}/{c.points_max:.0f}")
                if not c.passed and c.insight:
                    print(f"        → {c.insight[:70]}")

        print("═" * 78)


def print_combined_recommendation(
    cfg:          ArchConfig,
    quality_pct:  float,
    training_rpt: TrainingScoreReport,
    *,
    console=None,
) -> TrainingFitnessBreakdown:
    """
    Tampilkan rekomendasi gabungan dengan bobot baru (quality + fitness + training).
    Returns TrainingFitnessBreakdown untuk sorting/ranking.
    """
    breakdown = TrainingFitnessBreakdown(
        arch_id       = cfg.arch_id,
        quality_pct   = quality_pct,
        fitness_score = cfg.fitness_score,
        training_pct  = training_rpt.pct,
    )

    msg = (
        f"\n  ══ COMBINED SCORE (Training-Aware) ══\n"
        f"  {breakdown.summary()}\n"
        f"  Combined: {breakdown.combined:.5f}  →  {breakdown.verdict}\n"
    )

    if console is not None:
        color = ("bold green" if breakdown.combined >= 0.65
                 else "yellow" if breakdown.combined >= 0.45 else "red")
        console.print(f"[{color}]{msg}[/{color}]")
    else:
        print(msg)

    # Tampilkan refine hints jika ada
    hints = training_rpt.refine_hints()
    if hints:
        hint_header = "\n  Training Improvement Hints:"
        if console:
            console.print(f"[dim]{hint_header}[/dim]")
            for h in hints[:5]:   # max 5 hints
                console.print(f"  [dim]  → {h}[/dim]")
        else:
            print(hint_header)
            for h in hints[:5]:
                print(f"    → {h}")

    return breakdown


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION HOOKS  (untuk pipeline.py dan adaptive_refiner.py)
# ═══════════════════════════════════════════════════════════════════════════════

def training_refine_hints(cfg: ArchConfig, gpu: GPUSpec) -> List[Tuple[str, float]]:
    """
    Menghasilkan daftar perturbasi yang disarankan berdasarkan training score.
    Didesain untuk diintegrasikan ke dalam AdaptiveRefiner._perturbation_candidates().

    Returns: List of (hint_name, priority_weight)
    Hint name sesuai dengan kunci perturbasi di AdaptiveRefiner.
    """
    scorer  = TrainingAwareScorer(gpu)
    report  = scorer.score(cfg)
    hints: List[Tuple[str, float]] = []

    for c in report.failed_checks:
        dim = c.dimension
        pts_missing = c.points_max - c.points_earned

        # T1: Gradient flow
        if dim == "T1" and "T1.1" in c.name:
            # Depth-width ratio off: hint ke layer count atau hidden dim
            dw = cfg.num_layers / math.sqrt(max(1, cfg.hidden_dim))
            if dw > OPTIMAL_DW_RATIO_HI:
                hints.append(("decrease_layers", pts_missing * 0.5))
                hints.append(("increase_hidden_dim", pts_missing * 0.3))
            else:
                hints.append(("increase_layers", pts_missing * 0.4))

        if dim == "T1" and "T1.3" in c.name:
            hints.append(("switch_to_rmsnorm", pts_missing * 0.8))

        # T2: Convergence
        if dim == "T2" and "T2.2" in c.name:
            # LR sensitivity: depth terlalu besar untuk optimizer
            hints.append(("decrease_layers", pts_missing * 0.4))
            hints.append(("switch_optimizer_adam_fp32", pts_missing * 0.6))

        if dim == "T2" and "T2.4" in c.name:
            # Deep model dengan Lion → ganti optimizer
            if cfg.optimizer_type == OptimizerType.LION:
                hints.append(("switch_optimizer_adam_fp32", pts_missing * 1.0))

        # T3: Stability
        if dim == "T3" and "T3.1" in c.name:
            if not cfg.use_mixed_precision:
                hints.append(("enable_mixed_precision", pts_missing * 0.8))
            if cfg.optimizer_type == OptimizerType.ADAM_8BIT:
                hints.append(("switch_optimizer_adam_fp32", pts_missing * 0.6))

        if dim == "T3" and "T3.3" in c.name:
            if cfg.use_gradient_checkpointing and cfg.vram_usage_pct < 70:
                hints.append(("disable_gradient_checkpointing", pts_missing * 0.9))

        # T4: Sample efficiency
        if dim == "T4" and "T4.1" in c.name:
            # Noisy gradients: naikkan effective batch
            if cfg.batch_size < 8:
                hints.append(("increase_batch_size", pts_missing * 0.7))

        if dim == "T4" and "T4.2" in c.name:
            if not cfg.tie_embeddings:
                hints.append(("enable_tie_embeddings", pts_missing * 0.8))

        # T5: Optimizer compat
        if dim == "T5" and "T5.1" in c.name:
            hints.append(("switch_optimizer_compatible", pts_missing * 0.9))

    # Deduplicate dan sort by priority
    seen: set = set()
    unique_hints: List[Tuple[str, float]] = []
    for name, weight in sorted(hints, key=lambda x: -x[1]):
        if name not in seen:
            seen.add(name)
            unique_hints.append((name, weight))

    return unique_hints[:8]   # top 8 hints


def score_and_report(
    cfg: ArchConfig,
    gpu: GPUSpec,
    quality_pct: float = 0.0,
    *,
    console=None,
    verbose: bool = True,
) -> Tuple[TrainingScoreReport, Optional[TrainingFitnessBreakdown]]:
    """
    One-shot: score + print + return.
    Convenience untuk penggunaan standalone atau dari pipeline.

    Usage:
        report, breakdown = score_and_report(cfg, gpu, quality_pct=96.0, console=console)
        print(f"Training score: {report.pct:.1f}%")
        print(f"Combined: {breakdown.combined:.4f}")
    """
    scorer  = TrainingAwareScorer(gpu)
    report  = scorer.score(cfg)

    print_training_score_report(report, console=console, verbose=verbose)

    breakdown = None
    if quality_pct > 0:
        breakdown = print_combined_recommendation(
            cfg, quality_pct, report, console=console
        )

    return report, breakdown


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test standalone: jalankan python training_aware.py
    Akan men-score arsitektur contoh dari pipeline yang ada.
    """
    from hardware import GPU_DATABASE
    from generator import ArchitectureGenerator

    print("Training-Aware NAS Scorer — Standalone Test")
    print("=" * 60)

    # Pilih GPU
    gpu_name = "RTX 4090"
    gpu = GPU_DATABASE.get(gpu_name)
    if gpu is None:
        gpu = list(GPU_DATABASE.values())[0]
        print(f"GPU '{gpu_name}' tidak ditemukan, menggunakan: {gpu.name}")
    else:
        print(f"GPU: {gpu.name}")

    print()

    # Generate beberapa arsitektur
    gen   = ArchitectureGenerator(gpu, rng_seed=42)
    archs = gen.generate_all_families(n_per_family=1)

    scorer = TrainingAwareScorer(gpu)

    results = []
    for cfg in archs[:5]:
        report = scorer.score(cfg)
        results.append((cfg, report))
        print(f"\n{'─' * 60}")
        print(f"Arch: {cfg.arch_id}  {cfg.arch_family}")
        print(f"  Params: {cfg.param_count/1e6:.1f}M  |  "
              f"L={cfg.num_layers}  D={cfg.hidden_dim}")
        print(f"  Training Score: {report.pct:.1f}%  ({report.grade})")
        print(f"  Gradient Risk: {report.gradient_attenuation_risk.upper()}")
        print(f"  LR Sensitivity: {report.lr_sensitivity_class.upper()}")
        print(f"  Chinchilla tokens: {report.chinchilla_optimal_tokens:,.0f}M")
        print(f"  Regime: {report.training_regime}")

        # Tampilkan dimensi
        for dim in ["T1", "T2", "T3", "T4", "T5"]:
            earned, mx = report.dimension_score(dim)
            bar = "█" * int(earned / mx * 20) + "░" * (20 - int(earned / mx * 20))
            print(f"  {dim} [{bar}] {earned:.1f}/{mx:.0f}")

        # Combined score
        combined = compute_training_aware_combined(
            quality_pct   = 85.0,   # contoh
            fitness_score = cfg.fitness_score,
            training_pct  = report.pct,
        )
        print(f"  Combined (w/ 85% quality): {combined:.5f}")

        # Refine hints
        hints = training_refine_hints(cfg, gpu)
        if hints:
            print(f"  Top refine hints: {[h[0] for h in hints[:3]]}")

    # Ranking final
    print(f"\n{'═' * 60}")
    print("RANKING BY TRAINING SCORE:")
    results.sort(key=lambda x: x[1].pct, reverse=True)
    for i, (cfg, report) in enumerate(results, 1):
        combined = compute_training_aware_combined(
            quality_pct   = 85.0,
            fitness_score = cfg.fitness_score,
            training_pct  = report.pct,
        )
        print(f"  #{i}  {cfg.arch_id:<10} {cfg.arch_family:<18} "
              f"train={report.pct:.1f}%  "
              f"gpu_fit={cfg.fitness_score:.4f}  "
              f"combined={combined:.5f}  "
              f"({report.gradient_attenuation_risk} grad risk)")
