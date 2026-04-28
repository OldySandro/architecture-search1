from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

from arch_types import ArchConfig, FFNType, AttentionType, OptimizerType
from hardware import GPUSpec
from generator import ArchitectureGenerator, VRAM_LIMIT_PCT


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Result of a single audit check."""
    name:           str
    dimension:      str       # A / B / C / D / E
    passed:         bool
    points_earned:  float
    points_max:     float
    detail:         str       # human-readable pass/fail explanation
    fix_hint:       str = ""  # what the refiner will do to fix it

    @property
    def pct(self) -> float:
        return self.points_earned / max(0.001, self.points_max) * 100


@dataclass
class ScoreReport:
    """Full scoring report for one ArchConfig."""
    arch_id:      str
    arch_name:    str
    total_score:  float
    max_score:    float        # always 100.0
    pct:          float        # 0–100
    checks:       List[CheckResult] = field(default_factory=list)
    is_stable:    bool = False

    @property
    def grade(self) -> str:
        p = self.pct
        if p >= 100.0: return "S ★★★  PERFECT"
        if p >= 90:    return "A+ ★★★"
        if p >= 80:    return "A  ★★"
        if p >= 70:    return "B  ★"
        if p >= 60:    return "C"
        return                "F  ✗  UNSTABLE"

    @property
    def grade_color(self) -> str:
        p = self.pct
        if p >= 100.0: return "bold green"
        if p >= 90:    return "green"
        if p >= 80:    return "cyan"
        if p >= 70:    return "yellow"
        if p >= 60:    return "dark_orange"
        return               "red"

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def partial_checks(self) -> List[CheckResult]:
        """Checks that passed but earned < 100% of max points."""
        return [c for c in self.checks if c.points_earned < c.points_max]

    def by_dimension(self) -> Dict[str, List[CheckResult]]:
        out: Dict[str, List[CheckResult]] = {}
        for c in self.checks:
            out.setdefault(c.dimension, []).append(c)
        return out

    def dimension_score(self, dim: str) -> Tuple[float, float]:
        checks = self.by_dimension().get(dim, [])
        return sum(c.points_earned for c in checks), sum(c.points_max for c in checks)


@dataclass
class RefinementLog:
    """Full history of a refine() call on one ARC."""
    arch_id:          str
    arch_name:        str
    iterations:       int
    initial_pct:      float
    final_pct:        float
    initial_fitness:  float = 0.0
    final_fitness:    float = 0.0
    score_history:    List[float] = field(default_factory=list)
    fixes_applied:    List[str]   = field(default_factory=list)
    converged:        bool = False
    stagnated:        bool = False

    @property
    def improved_by(self) -> float:
        return round(self.final_pct - self.initial_pct, 2)

    @property
    def fitness_delta(self) -> float:
        return round(self.final_fitness - self.initial_fitness, 4)

    @property
    def status(self) -> str:
        if self.converged:   return "✓ STABLE (100%)"
        if self.stagnated:   return f"~ STAGNATED at {self.final_pct:.1f}%"
        return                      f"↑ IMPROVED → {self.final_pct:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  SCORER
# ══════════════════════════════════════════════════════════════════════════════

class ArcQualityScorer:
    """
    Mengaudit ArchConfig di 5 dimensi dan mengembalikan ScoreReport.

    FILOSOFI: Semua skor bersifat DETERMINISTIK dan berbasis ANALISIS NYATA
    dari parameter ARC yang dihasilkan, bukan referensi ke model yang sudah ada.

    Setiap check menggunakan partial credit kontinu — bukan binary pass/fail —
    sehingga ARC dengan nilai marginal mendapatkan skor yang tepat mencerminkan
    kualitasnya yang juga marginal.
    """

    def __init__(self, gpu: GPUSpec):
        self.gpu = gpu

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ridge_point(self) -> float:
        """Titik ridge hardware: FLOPs/byte yang sebenarnya (ECC-adjusted, thermal-derated)."""
        gpu = self.gpu
        peak_flops = gpu.bf16_tflops * 1e12 * gpu.thermal_factor
        eff_bw     = gpu.effective_memory_bw_gbps * 1e9 * gpu.thermal_factor
        return peak_flops / max(1.0, eff_bw)

    def _theoretical_min_ms(self, cfg: ArchConfig) -> float:
        """
        Waktu langkah minimum teoritis berdasarkan roofline model.
        Ini adalah batas bawah fisika: seberapa cepat GPU BISA menyelesaikan
        langkah ini jika bekerja pada efisiensi puncaknya.
        """
        gpu = self.gpu
        total_tokens = cfg.seq_len * cfg.batch_size
        total_flops  = (cfg.flops_per_token_fwd + cfg.flops_per_token_bwd) * total_tokens
        peak_flops   = gpu.bf16_tflops * 1e12 * gpu.thermal_factor
        # Gunakan mfu_typical_max sebagai ceiling efisiensi yang dapat dicapai
        return total_flops / max(1.0, peak_flops * gpu.mfu_typical_max) * 1000.0

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, cfg: ArchConfig) -> ScoreReport:
        checks: List[CheckResult] = []
        checks.extend(self._dim_a_structural(cfg))
        checks.extend(self._dim_b_memory(cfg))
        checks.extend(self._dim_c_flops(cfg))
        checks.extend(self._dim_d_hardware(cfg))
        checks.extend(self._dim_e_optflags(cfg))

        total   = sum(c.points_earned for c in checks)
        max_pts = sum(c.points_max    for c in checks)   # always 100

        return ScoreReport(
            arch_id     = cfg.arch_id,
            arch_name   = cfg.arch_name,
            total_score = round(total,   2),
            max_score   = round(max_pts, 2),
            pct         = round(total / max(0.001, max_pts) * 100, 2),
            checks      = checks,
            is_stable   = (total >= max_pts - 0.01),
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Dimension A: Structural Integrity  (30 pts)
    # ──────────────────────────────────────────────────────────────────────────
    # A1  num_heads × head_dim == hidden_dim              8 pts  (hard constraint)
    # A2  num_kv_heads ≤ num_heads                        4 pts  (hard constraint)
    # A3  num_heads % num_kv_heads == 0                   4 pts  (hard constraint)
    # A4  head_dim divisible by 32 (TC aligned)           6 pts  (graduated)
    # A5  hidden_dim divisible by 64                      4 pts  (graduated)
    # A6  FFN dim divisible by 128 (TC aligned)           4 pts  (graduated)
    #                                                    ──────
    #                                                    30 pts

    def _dim_a_structural(self, cfg: ArchConfig) -> List[CheckResult]:
        results = []

        # A1 — num_heads × head_dim == hidden_dim  (8 pts, hard constraint)
        product = cfg.num_heads * cfg.head_dim
        ok = (product == cfg.hidden_dim)
        results.append(CheckResult(
            name          = "A1: heads×head_dim==hidden_dim",
            dimension     = "A",
            passed        = ok,
            points_earned = 8.0 if ok else 0.0,
            points_max    = 8.0,
            detail        = (f"✓ {cfg.num_heads}×{cfg.head_dim}={product}={cfg.hidden_dim}"
                             if ok else
                             f"✗ {cfg.num_heads}×{cfg.head_dim}={product} ≠ {cfg.hidden_dim}  "
                             f"(delta={abs(product - cfg.hidden_dim)})"),
            fix_hint      = "Recompute head_dim=hidden_dim//num_heads, snap to 32-multiple",
        ))

        # A2 — num_kv_heads ≤ num_heads  (4 pts, hard constraint)
        ok = cfg.num_kv_heads <= cfg.num_heads
        results.append(CheckResult(
            name          = "A2: kv_heads ≤ num_heads",
            dimension     = "A",
            passed        = ok,
            points_earned = 4.0 if ok else 0.0,
            points_max    = 4.0,
            detail        = (f"✓ kv_heads={cfg.num_kv_heads} ≤ num_heads={cfg.num_heads}"
                             if ok else
                             f"✗ kv_heads={cfg.num_kv_heads} > num_heads={cfg.num_heads}"),
            fix_hint      = "Clip num_kv_heads = min(num_kv_heads, num_heads)",
        ))

        # A3 — num_heads % num_kv_heads == 0  (4 pts, hard constraint)
        ok = cfg.num_kv_heads > 0 and cfg.num_heads % cfg.num_kv_heads == 0
        results.append(CheckResult(
            name          = "A3: num_heads%kv_heads==0",
            dimension     = "A",
            passed        = ok,
            points_earned = 4.0 if ok else 0.0,
            points_max    = 4.0,
            detail        = (f"✓ {cfg.num_heads}%{cfg.num_kv_heads}=0"
                             if ok else
                             f"✗ {cfg.num_heads}%{cfg.num_kv_heads}="
                             f"{cfg.num_heads % max(1, cfg.num_kv_heads)} ≠ 0"),
            fix_hint      = "Snap kv_heads ke divisor valid terdekat dari num_heads",
        ))

        # A4 — head_dim TC alignment  (6 pts, GRADUATED)
        # Scoring bertingkat berdasarkan alignment ke tile size optimal GPU:
        #   % 128   → 6 pts  (ideal: muat penuh di Gen4 tensor core tile)
        #   % 64    → 5 pts  (baik: 2 tiles)
        #   % 32    → 3 pts  (cukup: minimum TC alignment)
        #   % 16    → 1 pt   (sangat marginal, hanya untuk arsitektur sangat kecil)
        #   else    → 0 pts  (tidak ter-align, performa TC rusak)
        hd = cfg.head_dim
        if hd > 0 and hd % 128 == 0:
            a4_pts, a4_detail = 6.0, f"✓ head_dim={hd} (128-aligned, optimal TC tile)"
        elif hd > 0 and hd % 64 == 0:
            a4_pts, a4_detail = 5.0, f"✓ head_dim={hd} (64-aligned, baik)"
        elif hd > 0 and hd % 32 == 0:
            a4_pts, a4_detail = 3.0, f"~ head_dim={hd} (32-aligned, minimum TC)  [-3 pts]"
        elif hd > 0 and hd % 16 == 0:
            a4_pts, a4_detail = 1.0, f"✗ head_dim={hd} (16-aligned, marginal)  [-5 pts]"
        else:
            a4_pts, a4_detail = 0.0, f"✗ head_dim={hd} (tidak ter-align)  [-6 pts]"
        results.append(CheckResult(
            name          = "A4: head_dim TC alignment",
            dimension     = "A",
            passed        = hd > 0 and hd % 32 == 0,
            points_earned = a4_pts,
            points_max    = 6.0,
            detail        = a4_detail,
            fix_hint      = "Round head_dim ke kelipatan 64 terdekat yang membagi hidden_dim",
        ))

        # A5 — hidden_dim alignment  (4 pts, GRADUATED)
        # % 256 → 4 pts, % 128 → 3 pts, % 64 → 2 pts, else → 0
        hid = cfg.hidden_dim
        if hid > 0 and hid % 256 == 0:
            a5_pts, a5_detail = 4.0, f"✓ hidden_dim={hid} (256-aligned, optimal)"
        elif hid > 0 and hid % 128 == 0:
            a5_pts, a5_detail = 3.0, f"~ hidden_dim={hid} (128-aligned)  [-1 pt]"
        elif hid > 0 and hid % 64 == 0:
            a5_pts, a5_detail = 2.0, f"~ hidden_dim={hid} (64-aligned, minimum)  [-2 pts]"
        else:
            a5_pts, a5_detail = 0.0, f"✗ hidden_dim={hid} (tidak ter-align)  [-4 pts]"
        results.append(CheckResult(
            name          = "A5: hidden_dim alignment",
            dimension     = "A",
            passed        = hid > 0 and hid % 64 == 0,
            points_earned = a5_pts,
            points_max    = 4.0,
            detail        = a5_detail,
            fix_hint      = "Round hidden_dim ke kelipatan 128 terdekat",
        ))

        # A6 — FFN dim TC alignment  (4 pts, GRADUATED)
        # ffn_dim = int(hidden_dim * ffn_multiplier)
        # Penting: operasi FFN (matmul besar) mendominasi FLOPs → TC alignment kritis
        # % 128 → 4 pts, % 64 → 2 pts, % 32 → 1 pt, else → 0
        ffn_dim = int(cfg.hidden_dim * cfg.ffn_multiplier)
        if ffn_dim > 0 and ffn_dim % 128 == 0:
            a6_pts, a6_detail = 4.0, (f"✓ ffn_dim={ffn_dim} (128-aligned, TC optimal)"
                                       f"  [{cfg.hidden_dim}×{cfg.ffn_multiplier:.1f}]")
        elif ffn_dim > 0 and ffn_dim % 64 == 0:
            a6_pts, a6_detail = 2.0, (f"~ ffn_dim={ffn_dim} (64-aligned, cukup)  [-2 pts]"
                                       f"  [{cfg.hidden_dim}×{cfg.ffn_multiplier:.1f}]")
        elif ffn_dim > 0 and ffn_dim % 32 == 0:
            a6_pts, a6_detail = 1.0, (f"✗ ffn_dim={ffn_dim} (32-aligned saja)  [-3 pts]"
                                       f"  [{cfg.hidden_dim}×{cfg.ffn_multiplier:.1f}]")
        else:
            a6_pts, a6_detail = 0.0, (f"✗ ffn_dim={ffn_dim} (tidak ter-align)  [-4 pts]"
                                       f"  [{cfg.hidden_dim}×{cfg.ffn_multiplier:.1f}]")
        results.append(CheckResult(
            name          = "A6: ffn_dim TC alignment",
            dimension     = "A",
            passed        = ffn_dim > 0 and ffn_dim % 64 == 0,
            points_earned = a6_pts,
            points_max    = 4.0,
            detail        = a6_detail,
            fix_hint      = "Sesuaikan ffn_multiplier agar int(hidden_dim × mult) % 128 == 0",
        ))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  Dimension B: Memory Consistency  (25 pts)
    # ──────────────────────────────────────────────────────────────────────────
    # B1  vram_total ≈ sum(parts) dalam 2%         5 pts  (partial continu)
    # B2  vram_usage_pct benar                     3 pts  (partial)
    # B3  fits_gpu konsisten                       2 pts  (biner)
    # B4  KV cache formula                         5 pts  (partial continu)
    # B5  Kualitas efisiensi VRAM  (BARU)          5 pts  (graduated quality)
    # B6  Kesehatan fraksi aktivasi  (BARU)        5 pts  (graduated quality)
    #                                             ──────
    #                                             25 pts

    def _dim_b_memory(self, cfg: ArchConfig) -> List[CheckResult]:
        results = []
        gpu = self.gpu

        # B1 — vram_total ≈ sum(parts) dalam 2%  (5 pts, partial continu)
        expected_total = round(
            cfg.vram_weights_gb + cfg.vram_activations_gb +
            cfg.vram_optimizer_gb + cfg.vram_kv_cache_gb +
            cfg.vram_fragmentation_gb, 3)
        rel_err = abs(cfg.vram_total_gb - expected_total) / max(0.001, expected_total)
        ok = rel_err < 0.02
        # Credit linier: 5 pts saat err=0%, 0 pts saat err≥2%
        b1_pts = round(5.0 * max(0.0, 1.0 - rel_err / 0.02), 2)
        results.append(CheckResult(
            name          = "B1: vram_total == sum(parts)",
            dimension     = "B",
            passed        = ok,
            points_earned = 5.0 if ok else b1_pts,
            points_max    = 5.0,
            detail        = (f"✓ total={cfg.vram_total_gb:.3f}≈{expected_total:.3f} GB"
                             f"  (Δ={rel_err*100:.2f}%)"
                             if ok else
                             f"✗ total={cfg.vram_total_gb:.3f} vs parts={expected_total:.3f} GB"
                             f"  (Δ={rel_err*100:.1f}%)  [{b1_pts:.1f}/5 pts]"),
            fix_hint      = "vram_total = w + a + o + kv + frag",
        ))

        # B2 — vram_usage_pct == vram_total/gpu.vram×100 (3 pts, partial)
        expected_pct = round(cfg.vram_total_gb / gpu.vram_gb * 100, 2)
        pct_err = abs(cfg.vram_usage_pct - expected_pct)
        ok = pct_err < 1.0
        b2_pts = round(3.0 * max(0.0, 1.0 - pct_err / 1.0), 2) if not ok else 3.0
        results.append(CheckResult(
            name          = "B2: vram_usage_pct benar",
            dimension     = "B",
            passed        = ok,
            points_earned = b2_pts,
            points_max    = 3.0,
            detail        = (f"✓ {cfg.vram_usage_pct:.2f}%≈{expected_pct:.2f}%"
                             if ok else
                             f"✗ pct={cfg.vram_usage_pct:.2f}% expected={expected_pct:.2f}%"
                             f"  (Δ={pct_err:.2f} pp)"),
            fix_hint      = "vram_usage_pct = vram_total_gb / gpu.vram_gb * 100",
        ))

        # B3 — fits_gpu konsisten dengan VRAM_LIMIT_PCT  (2 pts, biner)
        should_fit = cfg.vram_total_gb <= gpu.vram_gb * VRAM_LIMIT_PCT
        ok = cfg.fits_gpu == should_fit
        results.append(CheckResult(
            name          = "B3: fits_gpu konsisten",
            dimension     = "B",
            passed        = ok,
            points_earned = 2.0 if ok else 0.0,
            points_max    = 2.0,
            detail        = (f"✓ fits_gpu={cfg.fits_gpu}"
                             f"  ({cfg.vram_usage_pct:.1f}%≤{VRAM_LIMIT_PCT*100:.0f}%)"
                             if ok else
                             f"✗ fits_gpu={cfg.fits_gpu} tidak konsisten dengan"
                             f"  {cfg.vram_total_gb:.2f}/{gpu.vram_gb:.0f} GB"
                             f"  ({'≤' if should_fit else '>'}{VRAM_LIMIT_PCT*100:.0f}%)"),
            fix_hint      = f"fits_gpu = vram_total_gb ≤ gpu.vram_gb × {VRAM_LIMIT_PCT}",
        ))

        # B4 — KV cache formula  (5 pts, partial continu)
        expected_kv = (2 * cfg.num_kv_heads * cfg.head_dim * 2 *
                       cfg.num_layers * cfg.seq_len * cfg.batch_size) / 1e9
        rel_kv = abs(cfg.vram_kv_cache_gb - expected_kv) / max(0.0001, expected_kv)
        ok = rel_kv < 0.05
        b4_pts = round(5.0 * max(0.0, 1.0 - rel_kv / 0.05), 2)
        results.append(CheckResult(
            name          = "B4: kv_cache_gb formula",
            dimension     = "B",
            passed        = ok,
            points_earned = 5.0 if ok else b4_pts,
            points_max    = 5.0,
            detail        = (f"✓ kv={cfg.vram_kv_cache_gb:.5f}≈{expected_kv:.5f} GB"
                             if ok else
                             f"✗ kv={cfg.vram_kv_cache_gb:.5f} expected={expected_kv:.5f}"
                             f"  (Δ={rel_kv*100:.1f}%)  [{b4_pts:.1f}/5 pts]"),
            fix_hint      = "kv = 2×kv_heads×head_dim×2bytes×layers×seq×batch / 1e9",
        ))

        # B5 — Kualitas efisiensi VRAM  (5 pts, GRADUATED QUALITY — BARU)
        # Ini mengukur KUALITAS NYATA arsitektur, bukan hanya konsistensi formula.
        # Sweet spot pretraining: 55–80% VRAM usage (GPU terpakai optimal)
        # < 40%: VRAM sangat terbuang → parallelism dan throughput rendah
        # 40–55%: sub-optimal, batch size bisa ditingkatkan
        # 55–80%: IDEAL — komputasi efisien, margin cukup untuk gradient buffer
        # 80–88%: sedikit ketat, risiko OOM meningkat di step besar
        # > 88%: terlalu berbahaya untuk pretraining stabil
        u = cfg.vram_usage_pct
        if 55.0 <= u <= 80.0:
            b5_pts  = 5.0
            b5_pass = True
            b5_det  = f"✓ VRAM={u:.1f}% (55–80% sweet spot) — efisiensi optimal"
        elif 45.0 <= u < 55.0:
            # Ramping naik: 0→5 pts dari 45% ke 55%
            b5_pts  = round(5.0 * (u - 45.0) / 10.0, 2)
            b5_pass = False
            b5_det  = (f"~ VRAM={u:.1f}% (sub-optimal, batch size bisa dinaikkan)"
                       f"  [{b5_pts:.1f}/5 pts]")
        elif 80.0 < u <= 88.0:
            # Ramping turun: 5→0 pts dari 80% ke 88%
            b5_pts  = round(5.0 * (88.0 - u) / 8.0, 2)
            b5_pass = False
            b5_det  = (f"~ VRAM={u:.1f}% (terlalu ketat, risiko OOM)"
                       f"  [{b5_pts:.1f}/5 pts]")
        elif 35.0 <= u < 45.0:
            # Sangat sub-optimal
            b5_pts  = round(2.5 * (u - 35.0) / 10.0, 2)
            b5_pass = False
            b5_det  = (f"✗ VRAM={u:.1f}% (sangat rendah, VRAM terbuang banyak)"
                       f"  [{b5_pts:.1f}/5 pts]")
        else:
            b5_pts  = 0.0
            b5_pass = False
            b5_det  = (f"✗ VRAM={u:.1f}% ({'>88%, berbahaya OOM' if u > 88 else '<35%, sangat boros'})"
                       f"  [0/5 pts]")
        results.append(CheckResult(
            name          = "B5: kualitas efisiensi VRAM",
            dimension     = "B",
            passed        = b5_pass,
            points_earned = b5_pts,
            points_max    = 5.0,
            detail        = b5_det,
            fix_hint      = "Tuning batch_size untuk mencapai VRAM usage 55–80%",
        ))

        # B6 — Kesehatan fraksi aktivasi  (5 pts, GRADUATED QUALITY — BARU)
        # Untuk pretraining: aktivasi harus 15–50% dari total VRAM
        # Terlalu rendah  (<10%): weight-dominated → tidak memanfaatkan parallelism
        # Ideal (15–50%) : compute-activation balance yang sehat
        # Terlalu tinggi (>60%): risiko activation overflow, perlu gradient checkpointing
        total_vram = max(0.001, cfg.vram_total_gb)
        act_frac   = cfg.vram_activations_gb / total_vram
        if 0.15 <= act_frac <= 0.50:
            b6_pts  = 5.0
            b6_pass = True
            b6_det  = (f"✓ act_frac={act_frac*100:.1f}% (15–50% ideal pretraining)"
                       f"  [{cfg.vram_activations_gb:.3f}/{total_vram:.3f} GB]")
        elif 0.10 <= act_frac < 0.15:
            b6_pts  = round(5.0 * (act_frac - 0.10) / 0.05, 2)
            b6_pass = False
            b6_det  = (f"~ act_frac={act_frac*100:.1f}% (terlalu rendah, weight-dominated)"
                       f"  [{b6_pts:.1f}/5 pts]")
        elif 0.50 < act_frac <= 0.65:
            b6_pts  = round(5.0 * (0.65 - act_frac) / 0.15, 2)
            b6_pass = False
            b6_det  = (f"~ act_frac={act_frac*100:.1f}% (tinggi, perlu gradient checkpointing)"
                       f"  [{b6_pts:.1f}/5 pts]")
        elif 0.65 < act_frac <= 0.80:
            b6_pts  = round(2.0 * (0.80 - act_frac) / 0.15, 2)
            b6_pass = False
            b6_det  = (f"✗ act_frac={act_frac*100:.1f}% (kritis, activation overflow risk)"
                       f"  [{b6_pts:.1f}/5 pts]")
        else:
            b6_pts  = 0.0
            b6_pass = False
            b6_det  = (f"✗ act_frac={act_frac*100:.1f}% "
                       f"({'ekstrim tinggi' if act_frac > 0.80 else 'sangat rendah'})"
                       f"  [0/5 pts]")
        results.append(CheckResult(
            name          = "B6: fraksi aktivasi pretraining",
            dimension     = "B",
            passed        = b6_pass,
            points_earned = b6_pts,
            points_max    = 5.0,
            detail        = b6_det,
            fix_hint      = "Aktifkan gradient_checkpointing jika act_frac>0.5; naikkan batch_size jika <0.10",
        ))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  Dimension C: FLOPs Correctness  (20 pts)
    # ──────────────────────────────────────────────────────────────────────────
    # C1  fwd FLOPs vs aturan 2N (range KETAT)     7 pts  (partial continu)
    # C2  (attn+ffn)/fwd_total dalam [0.70,1.15]   7 pts  (partial continu)
    # C3  bwd/fwd ratio dalam [1.80,2.70]           6 pts  (KETAT: partial continu)
    #                                              ──────
    #                                              20 pts

    def _dim_c_flops(self, cfg: ArchConfig) -> List[CheckResult]:
        results = []

        # C1 — fwd FLOPs per token vs aturan 2N  (7 pts)
        # Aturan 2N: untuk transformer dense, fwd ≈ 2 × param_count per token.
        # Range KETAT yang mencerminkan pretraining nyata:
        #   [0.85, 2.5]×2N → 7 pts  (optimal: attn overhead dan FFN seimbang)
        #   [0.70, 0.85)   → partial ramping (FLOPs/param ratio sedikit rendah)
        #   (2.5, 3.5]     → partial ramping (FLOPs/param ratio sedikit tinggi)
        #   [0.50, 0.70)   → partial kecil (terlalu sedikit FLOPs)
        #   (3.5, 4.5]     → partial kecil (terlalu banyak FLOPs, attn-dominated)
        #   lainnya        → 0 pts
        approx_2n = 2.0 * max(1, cfg.param_count)
        ratio = cfg.flops_per_token_fwd / approx_2n if approx_2n > 0 else 0.0
        if 0.85 <= ratio <= 2.5:
            c1_pts = 7.0
        elif 0.70 <= ratio < 0.85:
            c1_pts = round(7.0 * (ratio - 0.70) / 0.15, 2)
        elif 2.5 < ratio <= 3.5:
            c1_pts = round(7.0 * (3.5 - ratio) / 1.0, 2)
        elif 0.50 <= ratio < 0.70:
            c1_pts = round(3.5 * (ratio - 0.50) / 0.20, 2)
        elif 3.5 < ratio <= 4.5:
            c1_pts = round(3.5 * (4.5 - ratio) / 1.0, 2)
        else:
            c1_pts = 0.0
        ok = 0.85 <= ratio <= 2.5
        results.append(CheckResult(
            name          = "C1: fwd_flops vs aturan 2N",
            dimension     = "C",
            passed        = ok,
            points_earned = round(c1_pts, 2),
            points_max    = 7.0,
            detail        = (f"✓ fwd={cfg.flops_per_token_fwd/1e9:.2f} GFLOP/tok"
                             f"  ratio={ratio:.3f}×2N  [ideal: 0.85–2.50]"
                             if ok else
                             f"✗ fwd={cfg.flops_per_token_fwd/1e9:.3f} GFLOP/tok"
                             f"  ratio={ratio:.3f}×2N  [di luar 0.85–2.50]"
                             f"  [{c1_pts:.1f}/7 pts]"),
            fix_hint      = "Recompute via _compute_flops() — cek formula attn/ffn",
        ))

        # C2 — (attn_fwd + ffn_fwd) / fwd_total dalam [0.70, 1.15]  (7 pts)
        component_sum = cfg.flops_attn_fwd + cfg.flops_ffn_fwd
        if cfg.flops_per_token_fwd > 0:
            comp_ratio = component_sum / cfg.flops_per_token_fwd
        else:
            comp_ratio = 0.0
        ok = 0.70 <= comp_ratio <= 1.15
        # Partial credit: full di [0.85,1.05], partial di tepi range
        if 0.85 <= comp_ratio <= 1.05:
            c2_pts = 7.0
        elif 0.70 <= comp_ratio < 0.85:
            c2_pts = round(7.0 * (comp_ratio - 0.70) / 0.15, 2)
        elif 1.05 < comp_ratio <= 1.15:
            c2_pts = round(7.0 * (1.15 - comp_ratio) / 0.10, 2)
        else:
            c2_pts = 0.0
        results.append(CheckResult(
            name          = "C2: (attn+ffn)/fwd dalam [0.70,1.15]",
            dimension     = "C",
            passed        = ok,
            points_earned = round(c2_pts, 2),
            points_max    = 7.0,
            detail        = (f"✓ (attn+ffn)/fwd={comp_ratio:.3f}"
                             if ok else
                             f"✗ (attn+ffn)/fwd={comp_ratio:.3f}"
                             f"  [attn={cfg.flops_attn_fwd/1e9:.2f}G"
                             f"  ffn={cfg.flops_ffn_fwd/1e9:.2f}G"
                             f"  fwd={cfg.flops_per_token_fwd/1e9:.2f}G]"
                             f"  [{c2_pts:.1f}/7 pts]"),
            fix_hint      = "attn+ffn harus 70–115% dari total fwd",
        ))

        # C3 — bwd/fwd ratio  (6 pts, range KETAT dengan partial continu)
        # Fisika backward pass LLM: bwd = attn×2.5 + ffn×2.0 (dari FIX-1 generator)
        # Range yang valid: [1.80, 2.70]
        # Range OPTIMAL pretraining: [1.90, 2.50] → full credit
        # Di luar range optimal tapi masih valid: partial credit
        if cfg.flops_per_token_fwd > 0:
            bwd_ratio = cfg.flops_per_token_bwd / cfg.flops_per_token_fwd
        else:
            bwd_ratio = 0.0
        if 1.90 <= bwd_ratio <= 2.50:
            c3_pts = 6.0
        elif 1.80 <= bwd_ratio < 1.90:
            c3_pts = round(6.0 * (bwd_ratio - 1.80) / 0.10, 2)
        elif 2.50 < bwd_ratio <= 2.70:
            c3_pts = round(6.0 * (2.70 - bwd_ratio) / 0.20, 2)
        elif 1.70 <= bwd_ratio < 1.80:
            c3_pts = round(3.0 * (bwd_ratio - 1.70) / 0.10, 2)
        else:
            c3_pts = 0.0
        ok = 1.80 <= bwd_ratio <= 2.70
        results.append(CheckResult(
            name          = "C3: bwd/fwd ratio [1.80–2.70]",
            dimension     = "C",
            passed        = ok,
            points_earned = round(c3_pts, 2),
            points_max    = 6.0,
            detail        = (f"✓ bwd/fwd={bwd_ratio:.3f}×  [optimal: 1.90–2.50]"
                             if ok else
                             f"✗ bwd/fwd={bwd_ratio:.3f}×  [di luar [1.80, 2.70]]"
                             f"  (bwd={cfg.flops_per_token_bwd/1e9:.2f}G"
                             f"  fwd={cfg.flops_per_token_fwd/1e9:.2f}G)"
                             f"  [{c3_pts:.1f}/6 pts]"),
            fix_hint      = "bwd = attn_fwd×2.5 + ffn_fwd×2.0 (dari FIX-1 generator.py)",
        ))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  Dimension D: Hardware Fitness  (15 pts)
    # ──────────────────────────────────────────────────────────────────────────
    # D1  Kualitas MFU vs range tipikal GPU         8 pts  (GRADUATED QUALITY)
    # D2  Step time vs minimum teoritis             4 pts  (GRADUATED QUALITY)
    # D3  Arithmetic intensity vs ridge point       3 pts  (GRADUATED QUALITY)
    #                                             ──────
    #                                             15 pts
    #
    # CATATAN KRITIS:
    #   Dimensi ini adalah YANG PALING SERING SALAH di versi sebelumnya.
    #   Versi lama memberi 5 pts hanya karena MFU > 0.01 (trivial).
    #   Versi ini menskor KUALITAS NYATA: MFU rendah = skor rendah.

    def _dim_d_hardware(self, cfg: ArchConfig) -> List[CheckResult]:
        results = []
        gpu = self.gpu

        # D1 — Kualitas MFU vs range tipikal GPU  (8 pts, GRADUATED QUALITY)
        #
        # MFU (Model FLOP Utilization) adalah metrik paling penting untuk
        # efisiensi pelatihan. Dikalibrasi dari MLPerf + Meta LLaMA papers.
        #
        # gpu.mfu_typical_min: MFU minimum yang masih bisa diterima (default 0.30)
        # gpu.mfu_typical_max: MFU maksimum yang dapat dicapai (default 0.55)
        #
        # Scoring:
        #   MFU ≥ mfu_typical_max              → 8 pts  (di atas tipikal, sangat baik)
        #   mfu_typical_min ≤ MFU < mfu_max   → 5.0–8.0 pts (linier dalam range)
        #   0.50×mfu_min ≤ MFU < mfu_min      → 0.0–5.0 pts (di bawah tipikal)
        #   0.20×mfu_min ≤ MFU < 0.50×mfu_min → 0.0–2.0 pts (sangat rendah)
        #   MFU < 0.20×mfu_min                → 0 pts (tidak dapat diterima)
        mfu     = cfg.mfu_estimate
        mfu_min = gpu.mfu_typical_min
        mfu_max = gpu.mfu_typical_max
        if mfu >= mfu_max:
            d1_pts  = 8.0
            d1_pass = True
            d1_det  = (f"✓ MFU={mfu:.4f} ≥ mfu_max={mfu_max:.2f}"
                       f"  (melebihi tipikal, sangat baik)")
        elif mfu >= mfu_min:
            # Linier: 5 pts di mfu_min, 8 pts di mfu_max
            ratio   = (mfu - mfu_min) / max(0.001, mfu_max - mfu_min)
            d1_pts  = round(5.0 + 3.0 * ratio, 2)
            d1_pass = True
            d1_det  = (f"✓ MFU={mfu:.4f}  [{mfu_min:.2f}–{mfu_max:.2f}]"
                       f"  [{d1_pts:.1f}/8 pts]  (dalam range tipikal {gpu.name})")
        elif mfu >= mfu_min * 0.50:
            # Di bawah minimum tipikal: 0→5 pts
            ratio   = (mfu - mfu_min * 0.50) / max(0.001, mfu_min * 0.50)
            d1_pts  = round(5.0 * ratio, 2)
            d1_pass = False
            d1_det  = (f"✗ MFU={mfu:.4f} < mfu_min={mfu_min:.2f}"
                       f"  (di bawah tipikal {gpu.name})"
                       f"  [{d1_pts:.1f}/8 pts]")
        elif mfu >= mfu_min * 0.20:
            # Sangat rendah: 0→2 pts
            ratio   = (mfu - mfu_min * 0.20) / max(0.001, mfu_min * 0.30)
            d1_pts  = round(2.0 * ratio, 2)
            d1_pass = False
            d1_det  = (f"✗ MFU={mfu:.4f}  (sangat rendah, arsitektur tidak efisien)"
                       f"  [{d1_pts:.1f}/8 pts]")
        else:
            d1_pts  = 0.0
            d1_pass = False
            d1_det  = (f"✗ MFU={mfu:.4f}  (tidak dapat diterima, < 20% dari mfu_min={mfu_min:.2f})"
                       f"  [0/8 pts]")
        results.append(CheckResult(
            name          = "D1: kualitas MFU vs range tipikal GPU",
            dimension     = "D",
            passed        = d1_pass,
            points_earned = d1_pts,
            points_max    = 8.0,
            detail        = d1_det,
            fix_hint      = "Recompute via _estimate_throughput() — cek roofline model dan TC alignment",
        ))

        # D2 — Step time vs minimum teoritis  (4 pts, GRADUATED QUALITY)
        #
        # Minimum teoritis = total_flops / (peak_flops × mfu_max)
        # Overhead ratio = actual_ms / theoretical_min_ms
        #
        # ≤ 1.5×: sangat efisien, dekat dengan roofline  → 4 pts
        # ≤ 2.0×: efisien, overhead wajar               → 3.0–4.0 pts
        # ≤ 3.0×: overhead sedang, ada ruang perbaikan  → 1.0–3.0 pts
        # ≤ 5.0×: overhead tinggi, ada masalah signifikan → 0–1.0 pts
        # > 5.0×: overhead tidak dapat diterima          → 0 pts
        if cfg.ms_per_step <= 0:
            d2_pts  = 0.0
            d2_pass = False
            d2_det  = "✗ ms_per_step ≤ 0 — tidak dihitung  [0/4 pts]"
        else:
            theor_ms = self._theoretical_min_ms(cfg)
            overhead = cfg.ms_per_step / max(0.001, theor_ms)
            if overhead <= 1.5:
                d2_pts = 4.0
                d2_det = (f"✓ ms={cfg.ms_per_step:.2f}ms  teoritis_min={theor_ms:.2f}ms"
                          f"  overhead={overhead:.2f}× (sangat efisien)")
            elif overhead <= 2.0:
                d2_pts = round(3.0 + (2.0 - overhead) / 0.5 * 1.0, 2)
                d2_det = (f"✓ ms={cfg.ms_per_step:.2f}ms  teoritis_min={theor_ms:.2f}ms"
                          f"  overhead={overhead:.2f}×  [{d2_pts:.1f}/4 pts]")
            elif overhead <= 3.0:
                d2_pts = round(1.0 + (3.0 - overhead) / 1.0 * 2.0, 2)
                d2_det = (f"~ ms={cfg.ms_per_step:.2f}ms  teoritis_min={theor_ms:.2f}ms"
                          f"  overhead={overhead:.2f}×  [{d2_pts:.1f}/4 pts]")
            elif overhead <= 5.0:
                d2_pts = round(1.0 * (5.0 - overhead) / 2.0, 2)
                d2_det = (f"✗ ms={cfg.ms_per_step:.2f}ms  teoritis_min={theor_ms:.2f}ms"
                          f"  overhead={overhead:.2f}× (terlalu tinggi)  [{d2_pts:.1f}/4 pts]")
            else:
                d2_pts = 0.0
                d2_det = (f"✗ ms={cfg.ms_per_step:.2f}ms  teoritis_min={theor_ms:.2f}ms"
                          f"  overhead={overhead:.2f}× (tidak dapat diterima)  [0/4 pts]")
            d2_pass = overhead <= 2.0
        results.append(CheckResult(
            name          = "D2: step time vs minimum teoritis",
            dimension     = "D",
            passed        = d2_pass,
            points_earned = round(d2_pts, 2),
            points_max    = 4.0,
            detail        = d2_det,
            fix_hint      = "Recompute via _estimate_throughput() — cek kernel fusion dan alignment",
        ))

        # D3 — Kualitas Arithmetic Intensity vs ridge point  (3 pts, GRADUATED)
        #
        # AI = FLOPs / bytes_transferred_per_token
        # Ridge point = peak_FLOPS / peak_BW (hardware constant, ECC-adjusted)
        #
        # AI ≥ ridge: compute-bound → tensor cores terpakai penuh  → 3 pts
        # 0.60×ridge ≤ AI < ridge: transisi, sebagian besar compute-bound  → 2–3 pts
        # 0.30×ridge ≤ AI < 0.60: memory-bandwidth-bound               → 1–2 pts
        # 0.10×ridge ≤ AI < 0.30: memory-latency-bound                 → 0–1 pt
        # AI < 0.10×ridge: extreme memory-latency-bound (sangat buruk)  → 0 pts
        ai    = cfg.arithmetic_intensity
        ridge = self._ridge_point()
        if ai <= 0:
            d3_pts = 0.0
            d3_det = "✗ arithmetic_intensity = 0 — tidak dihitung  [0/3 pts]"
            d3_pass = False
        else:
            ai_ratio = ai / max(0.001, ridge)
            if ai_ratio >= 1.0:
                d3_pts  = 3.0
                d3_pass = True
                d3_det  = (f"✓ AI={ai:.1f} ≥ ridge={ridge:.1f} FLOP/byte"
                           f"  (compute-bound, TC optimal)")
            elif ai_ratio >= 0.60:
                d3_pts  = round(2.0 + (ai_ratio - 0.60) / 0.40 * 1.0, 2)
                d3_pass = True
                d3_det  = (f"✓ AI={ai:.1f} vs ridge={ridge:.1f}"
                           f"  (ratio={ai_ratio:.2f}×, sebagian compute-bound)"
                           f"  [{d3_pts:.1f}/3 pts]")
            elif ai_ratio >= 0.30:
                d3_pts  = round(1.0 + (ai_ratio - 0.30) / 0.30 * 1.0, 2)
                d3_pass = False
                d3_det  = (f"~ AI={ai:.1f} vs ridge={ridge:.1f}"
                           f"  (ratio={ai_ratio:.2f}×, memory-bandwidth-bound)"
                           f"  [{d3_pts:.1f}/3 pts]")
            elif ai_ratio >= 0.10:
                d3_pts  = round(1.0 * (ai_ratio - 0.10) / 0.20, 2)
                d3_pass = False
                d3_det  = (f"✗ AI={ai:.1f} vs ridge={ridge:.1f}"
                           f"  (ratio={ai_ratio:.2f}×, memory-latency-bound)"
                           f"  [{d3_pts:.1f}/3 pts]")
            else:
                d3_pts  = 0.0
                d3_pass = False
                d3_det  = (f"✗ AI={ai:.1f} vs ridge={ridge:.1f}"
                           f"  (ratio={ai_ratio:.3f}×, extreme latency-bound)"
                           f"  [0/3 pts]")
        results.append(CheckResult(
            name          = "D3: arithmetic intensity vs ridge",
            dimension     = "D",
            passed        = d3_pass,
            points_earned = round(d3_pts, 2),
            points_max    = 3.0,
            detail        = d3_det,
            fix_hint      = "Naikkan batch_size atau hidden_dim untuk meningkatkan AI",
        ))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    #  Dimension E: Optimization Flags  (10 pts)
    # ──────────────────────────────────────────────────────────────────────────
    # E1  Flash attention untuk seq≥1024        3 pts
    # E2  Dropout=0 untuk pretraining           2 pts
    # E3  Mixed precision diaktifkan            3 pts
    # E4  param_count > 0                       2 pts
    #                                          ──────
    #                                          10 pts

    def _dim_e_optflags(self, cfg: ArchConfig) -> List[CheckResult]:
        results = []

        # E1 — Flash attention untuk seq_len ≥ 1024  (3 pts)
        long_seq = cfg.seq_len >= 1024
        ok = not (long_seq and not cfg.use_flash_attn)
        results.append(CheckResult(
            name          = "E1: flash_attn untuk seq≥1024",
            dimension     = "E",
            passed        = ok,
            points_earned = 3.0 if ok else 0.0,
            points_max    = 3.0,
            detail        = (f"✓ flash_attn={cfg.use_flash_attn}  seq={cfg.seq_len}"
                             if ok else
                             f"✗ seq={cfg.seq_len}≥1024 tapi use_flash_attn=False"
                             f"  (O(S²) HBM bottleneck)"),
            fix_hint      = "use_flash_attn=True menghilangkan O(S²) memory untuk seq≥1024",
        ))

        # E2 — Dropout == 0 untuk pretraining  (2 pts, partial untuk non-zero kecil)
        ok = cfg.dropout == 0.0
        # Dropout kecil (< 0.1) lebih bisa dimaklumi daripada besar
        if ok:
            e2_pts = 2.0
        elif cfg.dropout < 0.05:
            e2_pts = 1.0   # partial credit
        else:
            e2_pts = 0.0
        results.append(CheckResult(
            name          = "E2: dropout=0 (pretraining)",
            dimension     = "E",
            passed        = ok,
            points_earned = e2_pts,
            points_max    = 2.0,
            detail        = (f"✓ dropout={cfg.dropout}"
                             if ok else
                             f"~ dropout={cfg.dropout}"
                             f"  (pretraining sebaiknya 0.0)  [{e2_pts}/2 pts]"),
            fix_hint      = "dropout=0.0 untuk pretraining (tidak ada regularisasi default)",
        ))

        # E3 — Mixed precision diaktifkan  (3 pts)
        ok = cfg.use_mixed_precision
        results.append(CheckResult(
            name          = "E3: use_mixed_precision=True",
            dimension     = "E",
            passed        = ok,
            points_earned = 3.0 if ok else 0.0,
            points_max    = 3.0,
            detail        = ("✓ mixed precision aktif (BF16/FP16 tensor cores)"
                             if ok else
                             "✗ mixed precision tidak aktif — TC tidak terpakai penuh"),
            fix_hint      = "use_mixed_precision=True — wajib untuk BF16/FP16 tensor cores",
        ))

        # E4 — param_count > 0  (2 pts)
        ok = cfg.param_count > 0
        results.append(CheckResult(
            name          = "E4: param_count > 0",
            dimension     = "E",
            passed        = ok,
            points_earned = 2.0 if ok else 0.0,
            points_max    = 2.0,
            detail        = (f"✓ {cfg.param_count:,} params"
                             if ok else
                             "✗ param_count=0 — belum dihitung"),
            fix_hint      = "Jalankan _compute_params(cfg)",
        ))

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRETRAINING RULE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class _PretrainingRuleEngine:
    """
    Engine perbaikan arsitektur pretraining dengan aturan yang terus disempurnakan.

    Berbeda dari formula-fix (yang hanya mengkoreksi inkonsistensi matematis),
    rule engine ini menerapkan PERBAIKAN ARSITEKTUR yang meningkatkan kualitas
    pretraining secara nyata:

    R1 — Batch size optimization   : tuning ke sweet-spot VRAM 55–80%
    R2 — FFN dim TC alignment      : snap ffn_mult agar ffn_dim % 128 == 0
    R3 — head_dim TC alignment     : sesuaikan ke gpu.optimal_tile_size
    R4 — Gradient checkpointing    : aktifkan otomatis jika VRAM > 82%
    R5 — Attention type upgrade    : rekomendasikan SLIDE/HYBRID untuk seq≥8192
    R6 — KV heads GQA tuning       : optimal kv_heads untuk memory efficiency

    Setiap rule:
    - Punya kondisi penerapan (precondition)
    - Menghasilkan deskripsi perubahan atau string kosong jika tidak diterapkan
    - Dilacak: jika rule sudah diterapkan dan skor tidak meningkat, rule dinonaktifkan
    """

    def __init__(self, gpu: GPUSpec, gen: ArchitectureGenerator):
        self.gpu           = gpu
        self.gen           = gen
        self._retired: Set[str] = set()   # rule yang sudah dinonaktifkan karena stagnan

    def retire(self, rule_id: str) -> None:
        """Nonaktifkan rule yang tidak memberikan peningkatan."""
        self._retired.add(rule_id)

    def apply_all(self, cfg: ArchConfig, log: List[str]) -> int:
        """
        Terapkan semua rule yang aktif. Return jumlah perubahan yang diterapkan.
        """
        applied = 0
        for rule_id, rule_fn in [
            ("R1", self._r1_batch_size_optimization),
            ("R2", self._r2_ffn_tc_alignment),
            ("R3", self._r3_head_dim_tc_alignment),
            ("R4", self._r4_gradient_checkpointing),
            ("R5", self._r5_attention_type_upgrade),
            ("R6", self._r6_kv_heads_tuning),
        ]:
            if rule_id in self._retired:
                continue
            desc = rule_fn(cfg)
            if desc:
                log.append(f"[Rule {rule_id}] {desc}")
                applied += 1
        return applied

    def _r1_batch_size_optimization(self, cfg: ArchConfig) -> str:
        """
        R1: Tuning batch_size untuk mencapai sweet-spot VRAM 55–80%.
        Menggunakan binary search sederhana dengan re-derivasi memori.
        """
        gpu = self.gpu
        gen = self.gen
        current_pct = cfg.vram_usage_pct
        if 55.0 <= current_pct <= 80.0:
            return ""   # sudah ideal

        old_bs = cfg.batch_size
        best_bs = cfg.batch_size

        if current_pct < 55.0:
            # Coba naikkan batch_size (selama muat)
            for bs in [cfg.batch_size * 2, cfg.batch_size * 4, cfg.batch_size * 8]:
                cfg.batch_size = bs
                w_gb, a_gb, o_gb, kv_gb = gen._compute_memory(cfg)
                frag = gen._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb)
                total = w_gb + a_gb + o_gb + kv_gb + frag
                pct   = total / gpu.vram_gb * 100
                if 55.0 <= pct <= 80.0:
                    best_bs = bs
                    break
                elif pct < 80.0:
                    best_bs = bs   # masih muat, simpan kandidat

        elif current_pct > 82.0:
            # Coba turunkan batch_size
            for bs in [max(1, cfg.batch_size // 2), max(1, cfg.batch_size // 4)]:
                cfg.batch_size = bs
                w_gb, a_gb, o_gb, kv_gb = gen._compute_memory(cfg)
                frag = gen._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb)
                total = w_gb + a_gb + o_gb + kv_gb + frag
                pct   = total / gpu.vram_gb * 100
                if pct <= 80.0:
                    best_bs = bs
                    break

        cfg.batch_size = best_bs
        if best_bs != old_bs:
            return f"batch_size {old_bs}→{best_bs} (VRAM tuning: {current_pct:.1f}%→target 55–80%)"
        cfg.batch_size = old_bs   # kembalikan jika tidak ada perubahan
        return ""

    def _r2_ffn_tc_alignment(self, cfg: ArchConfig) -> str:
        """
        R2: Snap ffn_multiplier agar ffn_dim = int(hidden_dim × ffn_mult) % 128 == 0.
        Coba penyesuaian kecil ffn_mult (±0.25) yang mempertahankan rentang wajar.
        """
        ffn_dim = int(cfg.hidden_dim * cfg.ffn_multiplier)
        if ffn_dim % 128 == 0:
            return ""   # sudah aligned
        # Cari ffn_mult terdekat yang aligned ke 128
        base_mult = cfg.ffn_multiplier
        for delta in [0.0, 0.125, -0.125, 0.25, -0.25, 0.375, -0.375, 0.5, -0.5]:
            candidate = round(base_mult + delta, 3)
            if candidate < 1.5 or candidate > 6.5:
                continue
            candidate_ffn = int(cfg.hidden_dim * candidate)
            if candidate_ffn % 128 == 0 and candidate_ffn > 0:
                old_mult = cfg.ffn_multiplier
                old_dim  = ffn_dim
                cfg.ffn_multiplier = candidate
                return (f"ffn_multiplier {old_mult:.3f}→{candidate:.3f}"
                        f"  (ffn_dim {old_dim}→{candidate_ffn}, 128-aligned)")
        return ""

    def _r3_head_dim_tc_alignment(self, cfg: ArchConfig) -> str:
        """
        R3: Sesuaikan head_dim ke gpu.optimal_tile_size (16/32/64 tergantung GPU gen).
        Setelah penyesuaian, num_heads di-rederivasi dari hidden_dim // head_dim.
        """
        tile = self.gpu.optimal_tile_size
        if cfg.head_dim % tile == 0:
            return ""   # sudah aligned
        old_hd = cfg.head_dim
        # Floor ke kelipatan tile terdekat yang membagi hidden_dim
        new_hd = max(tile, (cfg.head_dim // tile) * tile)
        if cfg.hidden_dim % new_hd != 0:
            # Coba ceil
            new_hd = max(tile, ((cfg.head_dim + tile - 1) // tile) * tile)
            if cfg.hidden_dim % new_hd != 0:
                new_hd = tile  # fallback aman
        new_heads = cfg.hidden_dim // new_hd
        if new_heads < 4:
            return ""   # terlalu sedikit heads, abaikan
        cfg.head_dim  = new_hd
        cfg.num_heads = new_heads
        return (f"head_dim {old_hd}→{new_hd} (aligned ke tile={tile})"
                f"  num_heads→{new_heads}")

    def _r4_gradient_checkpointing(self, cfg: ArchConfig) -> str:
        """
        R4: Aktifkan gradient checkpointing otomatis jika VRAM usage > 82%.
        Ini adalah safety valve untuk arsitektur yang terlalu ketat.
        """
        if cfg.use_gradient_checkpointing:
            return ""   # sudah aktif
        if cfg.vram_usage_pct > 82.0:
            cfg.use_gradient_checkpointing = True
            return (f"gradient_checkpointing=True"
                    f"  (VRAM={cfg.vram_usage_pct:.1f}% > 82%, safety valve aktif)")
        return ""

    def _r5_attention_type_upgrade(self, cfg: ArchConfig) -> str:
        """
        R5: Untuk seq_len >= 8192, rekomendasikan SLIDE atau HYBRID
        untuk mengurangi O(S²) attention complexity.
        Hanya diterapkan jika MFU sangat rendah (< 0.5×mfu_min) dan seq panjang.
        """
        if cfg.seq_len < 8192:
            return ""
        if cfg.attn_type in (AttentionType.SLIDE, AttentionType.HYBRID, AttentionType.LINEAR):
            return ""
        if cfg.mfu_estimate >= self.gpu.mfu_typical_min * 0.5:
            return ""   # MFU masih acceptable, jangan ubah tipe attention
        old_type = cfg.attn_type
        cfg.attn_type = AttentionType.SLIDE
        cfg.window_size = min(cfg.seq_len, max(512, cfg.seq_len // 8))
        return (f"attn_type {old_type.name}→SLIDE"
                f"  (seq={cfg.seq_len}≥8192, window={cfg.window_size})"
                f"  (MFU={cfg.mfu_estimate:.4f} terlalu rendah untuk full attention)")

    def _r6_kv_heads_tuning(self, cfg: ArchConfig) -> str:
        """
        R6: Untuk GQA/MQA, optimasi kv_heads agar:
        - Mem KV cache tidak terlalu besar (< 15% total VRAM)
        - Rasio num_heads/kv_heads dalam [4, 16] untuk efisiensi GQA
        Hanya berlaku jika tipe attention mendukung GQA.
        """
        if cfg.attn_type not in (AttentionType.GQA, AttentionType.MQA):
            return ""
        if cfg.num_kv_heads == cfg.num_heads:
            return ""   # MHA mode, tidak relevan

        kv_frac = cfg.vram_kv_cache_gb / max(0.001, cfg.vram_total_gb)
        gqa_ratio = cfg.num_heads / max(1, cfg.num_kv_heads)

        old_kv = cfg.num_kv_heads
        if kv_frac > 0.15 or gqa_ratio < 4:
            # KV cache terlalu besar atau ratio terlalu rendah: kurangi kv_heads
            valid = [h for h in [1, 2, 4, 8]
                     if h <= cfg.num_heads // 4 and cfg.num_heads % h == 0]
            if valid:
                cfg.num_kv_heads = max(valid)
                if cfg.num_kv_heads != old_kv:
                    return (f"kv_heads {old_kv}→{cfg.num_kv_heads}"
                            f"  (GQA ratio {gqa_ratio:.1f}×→{cfg.num_heads/cfg.num_kv_heads:.1f}×"
                            f"  kv_frac={kv_frac*100:.1f}%)")
        elif gqa_ratio > 16 and cfg.num_kv_heads < cfg.num_heads // 4:
            # Ratio terlalu tinggi: tambah kv_heads sedikit
            valid = [h for h in [2, 4, 8, 16]
                     if h <= cfg.num_heads // 4 and cfg.num_heads % h == 0
                     and h > cfg.num_kv_heads]
            if valid:
                cfg.num_kv_heads = min(valid)
                if cfg.num_kv_heads != old_kv:
                    return (f"kv_heads {old_kv}→{cfg.num_kv_heads}"
                            f"  (GQA ratio {gqa_ratio:.1f}×→{cfg.num_heads/cfg.num_kv_heads:.1f}×"
                            f", menambah kapasitas attention)")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  REFINER
# ══════════════════════════════════════════════════════════════════════════════

class ArcRefiner:
    """
    Iteratif memperbaiki ArchConfig sampai quality score mencapai target_pct
    atau stagnasi terdeteksi.

    Algoritma dua fase per iterasi:
      Phase 1 — Formula corrections (per check yang gagal):
        Koreksi semua inkonsistensi matematis (head product, VRAM sum, FLOPs, dll)
      Phase 2 — Pretraining architectural improvements:
        Terapkan 6 rule perbaikan arsitektur nyata (batch tuning, TC alignment, dll)
      → Re-derive SEMUA derived fields via generator math engine
      → Re-score → cek konvergensi / stagnasi

    Stagnation detection:
      Jika skor tidak meningkat ≥ 0.1 pp selama 3 iterasi berurutan,
      refinement berhenti lebih awal.
    """

    def __init__(
        self,
        gpu:            GPUSpec,
        max_iterations: int   = 30,
        target_pct:     float = 100.0,
    ):
        self.gpu            = gpu
        self.max_iterations = max_iterations
        self.target_pct     = target_pct
        self._gen           = ArchitectureGenerator(gpu, rng_seed=42)
        self._scorer        = ArcQualityScorer(gpu)
        self._rules         = _PretrainingRuleEngine(gpu, self._gen)

    # ── Public API ─────────────────────────────────────────────────────────────

    def refine(self, cfg: ArchConfig) -> Tuple[ArchConfig, RefinementLog]:
        """Refine satu ARC. Kembalikan (improved_cfg, log)."""

        log = RefinementLog(
            arch_id         = cfg.arch_id,
            arch_name       = cfg.arch_name,
            iterations      = 0,
            initial_pct     = 0.0,
            final_pct       = 0.0,
            initial_fitness = cfg.fitness_score,
            final_fitness   = cfg.fitness_score,
        )

        # Reset rule engine untuk setiap ARC baru
        self._rules = _PretrainingRuleEngine(self.gpu, self._gen)

        # Skor awal
        init_report     = self._scorer.score(cfg)
        log.initial_pct = init_report.pct
        log.score_history.append(init_report.pct)

        if init_report.is_stable:
            log.final_pct = init_report.pct
            log.converged = True
            log.final_fitness = cfg.fitness_score
            return cfg, log

        prev_rule_score: Optional[float] = None   # untuk deteksi stagnasi per-rule

        for iteration in range(1, self.max_iterations + 1):
            log.iterations = iteration

            report = self._scorer.score(cfg)
            if report.pct >= self.target_pct:
                log.converged = True
                break

            # ── Phase 1: Formula corrections ───────────────────────────────
            phase1_fixes: List[str] = []
            for check in report.failed_checks:
                fix_desc = self._dispatch_fix(cfg, check)
                if fix_desc:
                    phase1_fixes.append(f"[iter{iteration}/P1] {fix_desc}")

            # ── Phase 2: Pretraining architectural improvements ─────────────
            # Re-derive dulu supaya rules punya data akurat
            if phase1_fixes:
                self._full_rederive(cfg)

            phase2_fixes: List[str] = []
            self._rules.apply_all(cfg, phase2_fixes)
            phase2_tagged = [f"[iter{iteration}/P2] {f}" for f in phase2_fixes]

            all_fixes = phase1_fixes + phase2_tagged
            log.fixes_applied.extend(all_fixes)

            # ── Re-derive setelah semua fixes ───────────────────────────────
            self._full_rederive(cfg)

            new_score = self._scorer.score(cfg).pct
            log.score_history.append(new_score)

            # Deteksi stagnasi rules: jika Phase 2 tidak membantu, nonaktifkan
            if phase2_fixes and prev_rule_score is not None:
                if new_score - prev_rule_score < 0.05:
                    # Rules tidak meningkatkan skor → track tapi tidak retire semua
                    pass
            if phase2_fixes:
                prev_rule_score = new_score

            # Stagnation: 3 iterasi terakhir tidak improve ≥ 0.1 pp
            if len(log.score_history) >= 4:
                recent_gain = log.score_history[-1] - log.score_history[-4]
                if recent_gain < 0.10:
                    log.stagnated = True
                    break

            if new_score >= self.target_pct:
                log.converged = True
                break

        final_report    = self._scorer.score(cfg)
        log.final_pct   = final_report.pct
        log.final_fitness = cfg.fitness_score
        log.converged   = log.converged or (final_report.pct >= self.target_pct)
        return cfg, log

    def refine_batch(
        self,
        archs:   List[ArchConfig],
        verbose: bool = True,
    ) -> Tuple[List[ArchConfig], List[RefinementLog]]:
        """Refine daftar ARCs. Kembalikan diurutkan berdasarkan fitness_score."""
        refined_archs: List[ArchConfig] = []
        all_logs:      List[RefinementLog] = []

        for cfg in archs:
            refined_cfg, log = self.refine(cfg)
            refined_archs.append(refined_cfg)
            all_logs.append(log)

        refined_archs.sort(key=lambda x: x.fitness_score, reverse=True)
        return refined_archs, all_logs

    # ── Fix Dispatcher ─────────────────────────────────────────────────────────

    def _dispatch_fix(self, cfg: ArchConfig, check: CheckResult) -> str:
        dispatch = {
            "A1": self._fix_a1_head_product,
            "A2": self._fix_a2_kv_heads_upper,
            "A3": self._fix_a3_kv_heads_divisor,
            "A4": self._fix_a4_head_dim_align,
            "A5": self._fix_a5_hidden_align,
            "A6": self._fix_a6_ffn_dim_align,
            "B1": self._fix_b1_vram_total,
            "B2": self._fix_b2_vram_pct,
            "B3": self._fix_b3_fits_gpu,
            "B4": self._fix_b4_kv_cache,
            "B5": self._fix_b5_vram_efficiency,
            "B6": self._fix_b6_act_fraction,
            "C1": self._fix_c_flops,
            "C2": self._fix_c_flops,
            "C3": self._fix_c_flops,
            "D1": self._fix_d_throughput,
            "D2": self._fix_d_throughput,
            "D3": self._fix_d_throughput,
            "E1": self._fix_e1_flash_attn,
            "E2": self._fix_e2_dropout,
            "E3": self._fix_e3_mixed_prec,
            "E4": self._fix_e4_params,
        }
        key = check.name[:2]
        fn  = dispatch.get(key)
        return fn(cfg) if fn else ""

    # ── Individual Fix Methods ─────────────────────────────────────────────────

    def _fix_a1_head_product(self, cfg: ArchConfig) -> str:
        """A1: Pastikan num_heads × head_dim == hidden_dim."""
        # hidden_dim adalah autoritas; rederivasi head_dim lalu num_heads
        new_hd = cfg.hidden_dim // max(1, cfg.num_heads)
        # Align head_dim ke 64 (memudahkan A4 sekaligus)
        new_hd = max(64, (new_hd // 64) * 64)
        if cfg.hidden_dim % new_hd != 0:
            new_hd = 64   # safe fallback
        new_heads = cfg.hidden_dim // new_hd
        if new_heads < 4:
            new_hd    = cfg.hidden_dim // 4
            new_heads = 4
        cfg.head_dim  = new_hd
        cfg.num_heads = new_heads
        return f"A1: head_dim={new_hd}, num_heads={new_heads} → {new_heads}×{new_hd}={cfg.hidden_dim}"

    def _fix_a2_kv_heads_upper(self, cfg: ArchConfig) -> str:
        """A2: num_kv_heads ≤ num_heads."""
        if cfg.num_kv_heads > cfg.num_heads:
            old = cfg.num_kv_heads
            cfg.num_kv_heads = cfg.num_heads
            return f"A2: kv_heads {old}→{cfg.num_heads}"
        return ""

    def _fix_a3_kv_heads_divisor(self, cfg: ArchConfig) -> str:
        """A3: num_heads % num_kv_heads == 0."""
        valid = [h for h in [1, 2, 4, 8, 16, 32, 64]
                 if h <= cfg.num_heads and cfg.num_heads % h == 0]
        if valid:
            target = cfg.num_kv_heads
            best   = min(valid, key=lambda h: abs(h - target))
            old    = cfg.num_kv_heads
            cfg.num_kv_heads = best
            return f"A3: kv_heads {old}→{best} (divisor valid terdekat)"
        return ""

    def _fix_a4_head_dim_align(self, cfg: ArchConfig) -> str:
        """A4: head_dim harus kelipatan 64 (lebih baik dari 32 untuk modern GPU)."""
        target_align = 64   # lebih ambisius dari sebelumnya (min 32)
        new_hd = max(target_align, (cfg.head_dim // target_align) * target_align)
        if new_hd != cfg.head_dim:
            old = cfg.head_dim
            # Cek hidden_dim divisibility
            if cfg.hidden_dim % new_hd != 0:
                # Coba 32 sebagai fallback
                new_hd = max(32, (cfg.head_dim // 32) * 32)
                if cfg.hidden_dim % new_hd != 0:
                    new_hd = 64
            cfg.head_dim  = new_hd
            cfg.num_heads = max(1, cfg.hidden_dim // new_hd)
            return f"A4: head_dim {old}→{new_hd}, num_heads→{cfg.num_heads}"
        return ""

    def _fix_a5_hidden_align(self, cfg: ArchConfig) -> str:
        """A5: hidden_dim harus kelipatan 128 (lebih ambisius dari 64)."""
        new_hd = max(128, ((cfg.hidden_dim + 127) // 128) * 128)
        if new_hd != cfg.hidden_dim:
            old = cfg.hidden_dim
            cfg.hidden_dim = new_hd
            return f"A5: hidden_dim {old}→{new_hd}"
        return ""

    def _fix_a6_ffn_dim_align(self, cfg: ArchConfig) -> str:
        """A6: ffn_dim = int(hidden_dim × ffn_mult) harus kelipatan 128."""
        ffn_dim = int(cfg.hidden_dim * cfg.ffn_multiplier)
        if ffn_dim % 128 == 0:
            return ""
        # Cari penyesuaian terkecil pada ffn_multiplier
        base = cfg.ffn_multiplier
        for delta in [0.0, 0.125, -0.125, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0]:
            cand = round(base + delta, 3)
            if cand < 1.5 or cand > 6.5:
                continue
            cand_ffn = int(cfg.hidden_dim * cand)
            if cand_ffn % 128 == 0 and cand_ffn > 0:
                old_mult = cfg.ffn_multiplier
                old_dim  = ffn_dim
                cfg.ffn_multiplier = cand
                return (f"A6: ffn_mult {old_mult:.3f}→{cand:.3f}"
                        f"  ffn_dim {old_dim}→{cand_ffn} (128-aligned)")
        return ""

    def _fix_b1_vram_total(self, cfg: ArchConfig) -> str:
        """B1: vram_total = sum of parts."""
        correct = round(
            cfg.vram_weights_gb + cfg.vram_activations_gb +
            cfg.vram_optimizer_gb + cfg.vram_kv_cache_gb +
            cfg.vram_fragmentation_gb, 3)
        old = cfg.vram_total_gb
        cfg.vram_total_gb = correct
        return f"B1: vram_total {old:.3f}→{correct:.3f} GB"

    def _fix_b2_vram_pct(self, cfg: ArchConfig) -> str:
        """B2: recompute vram_usage_pct."""
        correct = round(cfg.vram_total_gb / self.gpu.vram_gb * 100, 2)
        old = cfg.vram_usage_pct
        cfg.vram_usage_pct = correct
        return f"B2: vram_pct {old:.2f}%→{correct:.2f}%"

    def _fix_b3_fits_gpu(self, cfg: ArchConfig) -> str:
        """B3: fits_gpu konsisten dengan budget."""
        should_fit = cfg.vram_total_gb <= self.gpu.vram_gb * VRAM_LIMIT_PCT
        old = cfg.fits_gpu
        cfg.fits_gpu = should_fit
        return f"B3: fits_gpu {old}→{should_fit}"

    def _fix_b4_kv_cache(self, cfg: ArchConfig) -> str:
        """B4: recompute kv_cache dari first principles."""
        kv_gb = (2 * cfg.num_kv_heads * cfg.head_dim * 2 *
                 cfg.num_layers * cfg.seq_len * cfg.batch_size) / 1e9
        old = cfg.vram_kv_cache_gb
        cfg.vram_kv_cache_gb = round(kv_gb, 5)
        return f"B4: kv_cache {old:.5f}→{cfg.vram_kv_cache_gb:.5f} GB"

    def _fix_b5_vram_efficiency(self, cfg: ArchConfig) -> str:
        """
        B5: Perbaiki efisiensi VRAM dengan tuning batch_size.
        Ini adalah formula fix — re-derivasi memori dengan batch_size saat ini.
        Perbaikan arsitektur sesungguhnya ada di R1 (rule engine).
        """
        # Re-compute VRAM dengan parameter saat ini
        gen = self._gen
        gpu = self.gpu
        w_gb, a_gb, o_gb, kv_gb = gen._compute_memory(cfg)
        frag = gen._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb)
        total = w_gb + a_gb + o_gb + kv_gb + frag
        new_pct = total / gpu.vram_gb * 100

        old_pct = cfg.vram_usage_pct
        cfg.vram_weights_gb     = round(w_gb, 3)
        cfg.vram_activations_gb = round(a_gb, 3)
        cfg.vram_optimizer_gb   = round(o_gb, 3)
        cfg.vram_kv_cache_gb    = round(kv_gb, 5)
        cfg.vram_fragmentation_gb = round(frag, 3)
        cfg.vram_total_gb       = round(total, 3)
        cfg.vram_usage_pct      = round(new_pct, 2)
        cfg.fits_gpu            = total <= gpu.vram_gb * VRAM_LIMIT_PCT
        return f"B5: VRAM re-derived {old_pct:.1f}%→{new_pct:.1f}%"

    def _fix_b6_act_fraction(self, cfg: ArchConfig) -> str:
        """
        B6: Perbaiki fraksi aktivasi.
        Jika act_frac terlalu tinggi dan belum ada GC, aktifkan GC.
        Jika act_frac terlalu rendah, re-derive memori (batch_size sudah di-fix di R1).
        """
        total    = max(0.001, cfg.vram_total_gb)
        act_frac = cfg.vram_activations_gb / total
        if act_frac > 0.60 and not cfg.use_gradient_checkpointing:
            cfg.use_gradient_checkpointing = True
            return f"B6: gradient_checkpointing=True (act_frac={act_frac*100:.1f}%>60%)"
        # Untuk case lain, full rederive di _full_rederive akan menanganinya
        return ""

    def _fix_c_flops(self, cfg: ArchConfig) -> str:
        """C1-C3: Recompute semua FLOPs dari generator math engine."""
        fwd, bwd, attn_fwd, ffn_fwd = self._gen._compute_flops(cfg)
        cfg.flops_per_token_fwd  = round(fwd,      0)
        cfg.flops_per_token_bwd  = round(bwd,      0)
        cfg.flops_attn_fwd       = round(attn_fwd, 0)
        cfg.flops_ffn_fwd        = round(ffn_fwd,  0)
        cfg.arithmetic_intensity = round(
            self._gen._compute_arithmetic_intensity(cfg), 2)
        return (f"C: FLOPs recomputed — fwd={fwd/1e9:.2f}G"
                f"  bwd={bwd/1e9:.2f}G  AI={cfg.arithmetic_intensity:.1f}")

    def _fix_d_throughput(self, cfg: ArchConfig) -> str:
        """D1-D3: Recompute throughput, MFU, bottleneck."""
        thr = self._gen._estimate_throughput(cfg)
        cfg.tokens_per_sec_estimate = thr["tokens_per_sec"]
        cfg.mfu_estimate            = thr["mfu"]
        cfg.ms_per_step             = thr["ms_per_step"]
        cfg.bottleneck              = thr["bottleneck"]
        cfg.bottleneck_factors      = thr["bottleneck_factors"]
        cfg.compiler_speedup        = thr["compiler_speedup"]
        cfg.warp_divergence_pct     = thr["warp_divergence"]["warp_divergence_pct"]
        cfg.sm_occupancy            = thr["sm_occupancy"]
        return (f"D: throughput recomputed — MFU={cfg.mfu_estimate:.4f}"
                f"  ms/step={cfg.ms_per_step:.2f}"
                f"  tok/s={cfg.tokens_per_sec_estimate:,}")

    def _fix_e1_flash_attn(self, cfg: ArchConfig) -> str:
        """E1: Aktifkan flash attention untuk sequence panjang."""
        if cfg.seq_len >= 1024 and not cfg.use_flash_attn:
            cfg.use_flash_attn = True
            return f"E1: use_flash_attn=True (seq={cfg.seq_len})"
        return ""

    def _fix_e2_dropout(self, cfg: ArchConfig) -> str:
        """E2: Set dropout=0 untuk pretraining."""
        if cfg.dropout != 0.0:
            old = cfg.dropout
            cfg.dropout = 0.0
            return f"E2: dropout {old}→0.0"
        return ""

    def _fix_e3_mixed_prec(self, cfg: ArchConfig) -> str:
        """E3: Aktifkan mixed precision."""
        if not cfg.use_mixed_precision:
            cfg.use_mixed_precision = True
            return "E3: use_mixed_precision=True"
        return ""

    def _fix_e4_params(self, cfg: ArchConfig) -> str:
        """E4: Recompute param_count."""
        old = cfg.param_count
        cfg.param_count = self._gen._compute_params(cfg)
        return f"E4: param_count {old:,}→{cfg.param_count:,}"

    # ── Full Re-derivation ─────────────────────────────────────────────────────

    def _full_rederive(self, cfg: ArchConfig):
        """
        Setelah structural fixes, recompute SETIAP derived field sehingga
        memory / FLOPs / throughput / fitness semuanya mutually consistent.
        Ini adalah jantung dari sistem refinement.
        """
        gen = self._gen
        gpu = self.gpu

        # 1. Snap structural sanity (guard terhadap partial fixes)
        if cfg.num_heads > 0 and cfg.hidden_dim % cfg.num_heads != 0:
            cfg.num_heads = max(1, cfg.hidden_dim // max(64, cfg.head_dim))
        if cfg.num_kv_heads > cfg.num_heads:
            cfg.num_kv_heads = cfg.num_heads
        if cfg.num_kv_heads > 0 and cfg.num_heads % cfg.num_kv_heads != 0:
            valid = [h for h in [1, 2, 4, 8, 16, 32, 64]
                     if h <= cfg.num_heads and cfg.num_heads % h == 0]
            if valid:
                cfg.num_kv_heads = min(valid, key=lambda h: abs(h - cfg.num_kv_heads))

        # 2. Param count
        cfg.param_count = gen._compute_params(cfg)

        # 3. Memory breakdown
        w_gb, a_gb, o_gb, kv_gb = gen._compute_memory(cfg)
        frag_frac = gen._compute_fragmentation(cfg)
        frag_gb   = round(frag_frac * (w_gb + a_gb + o_gb + kv_gb), 3)
        total_gb  = round(w_gb + a_gb + o_gb + kv_gb + frag_gb, 3)

        cfg.vram_weights_gb       = round(w_gb,  3)
        cfg.vram_activations_gb   = round(a_gb,  3)
        cfg.vram_optimizer_gb     = round(o_gb,  3)
        cfg.vram_kv_cache_gb      = round(kv_gb, 5)
        cfg.vram_fragmentation_gb = frag_gb
        cfg.vram_total_gb         = total_gb
        cfg.vram_usage_pct        = round(total_gb / gpu.vram_gb * 100, 2)
        cfg.fits_gpu              = total_gb <= gpu.vram_gb * VRAM_LIMIT_PCT

        # 4. FLOPs
        fwd, bwd, attn_fwd, ffn_fwd = gen._compute_flops(cfg)
        cfg.flops_per_token_fwd  = round(fwd,      0)
        cfg.flops_per_token_bwd  = round(bwd,      0)
        cfg.flops_attn_fwd       = round(attn_fwd, 0)
        cfg.flops_ffn_fwd        = round(ffn_fwd,  0)
        cfg.arithmetic_intensity = round(gen._compute_arithmetic_intensity(cfg), 2)

        # 5. Throughput + bottleneck
        thr = gen._estimate_throughput(cfg)
        cfg.tokens_per_sec_estimate = thr["tokens_per_sec"]
        cfg.mfu_estimate            = thr["mfu"]
        cfg.ms_per_step             = thr["ms_per_step"]
        cfg.bottleneck              = thr["bottleneck"]
        cfg.bottleneck_factors      = thr["bottleneck_factors"]
        cfg.compiler_speedup        = thr["compiler_speedup"]
        cfg.warp_divergence_pct     = thr["warp_divergence"]["warp_divergence_pct"]
        cfg.sm_occupancy            = thr["sm_occupancy"]

        # 6. Final fitness score
        cfg.fitness_score = gen._fitness_score(cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  BALANCED COMBINED SCORE  (50% Hardware + 50% Training)
# ══════════════════════════════════════════════════════════════════════════════

def compute_combined_score_balanced(
    quality_pct:     float,
    hardware_score:  float,
    training_score:  float,
    *,
    hw_weight:       float = 0.50,
    train_weight:    float = 0.50,
    min_quality:     float = 70.0,
) -> float:
    """
    FORMULA BARU — Seimbang 50/50 antara hardware dan training.

    Args:
        quality_pct:    Skor konsistensi internal ArcQualityScorer [0–100]
        hardware_score: Skor dari HardwareNASEvaluator [0–1]  (7 dimensi GPU)
        training_score: Skor dari TrainingDynamicsEvaluator [0–1] (6 dimensi training)
        hw_weight:      Bobot hardware (default 0.50)
        train_weight:   Bobot training (default 0.50)
        min_quality:    Threshold minimum quality [default 70.0]

    Returns:
        combined score [0.0, 1.0]

    Rumus:
        quality_gate = (quality_pct - min_quality) / (100 - min_quality)  [0,1]
        quality_gate → multiplier penalti jika quality rendah (max 20% penalti)
        combined = hw_weight × hardware_score + train_weight × training_score
        combined *= (0.80 + 0.20 × quality_gate)   ← quality gate ringan

    Catatan:
        Quality score (refiner.py) hanya menjadi "gate" ringan karena ia
        mengukur konsistensi formula, BUKAN kualitas training nyata.
        Hardware dan training score masing-masing 50% → seimbang.
    """
    if quality_pct < min_quality:
        return 0.0

    q_gate = (quality_pct - min_quality) / max(0.001, 100.0 - min_quality)
    q_gate = float(max(0.0, min(1.0, q_gate)))

    hw  = float(max(0.0, min(1.0, hardware_score)))
    tr  = float(max(0.0, min(1.0, training_score)))

    raw = hw_weight * hw + train_weight * tr

    # Quality gate ringan: jika quality=100% → ×1.0, quality=min_quality → ×0.80
    gate_mult = 0.80 + 0.20 * q_gate

    return round(raw * gate_mult, 6)


def compute_combined_score_legacy(
    quality_pct:    float,
    fitness:        float,
    *,
    quality_weight: float = 0.35,
    fitness_weight: float = 0.65,
    min_quality:    float = 70.0,
) -> float:
    """
    Formula LAMA — 35% quality + 65% fitness.
    Dipertahankan untuk backward compatibility dengan pipeline.py lama.
    Direkomendasikan menggunakan compute_combined_score_balanced() sebagai gantinya.
    """
    if quality_pct < min_quality:
        return 0.0
    q_norm = (quality_pct - min_quality) / max(0.001, 100.0 - min_quality)
    q_norm = float(max(0.0, min(1.0, q_norm)))
    f_norm = float(max(0.0, min(1.0, fitness)))
    return round(quality_weight * q_norm + fitness_weight * f_norm, 6)


def compute_combined_score(
    quality_pct:    float,
    fitness:        float,
    *,
    quality_weight: float = 0.35,
    fitness_weight: float = 0.65,
    min_quality:    float = 70.0,
) -> float:
    """
    Alias compute_combined_score_legacy untuk backward compatibility.
    Dipakai pipeline.py lama / adaptive_refiner.py lama.
    """
    return compute_combined_score_legacy(
        quality_pct, fitness,
        quality_weight=quality_weight,
        fitness_weight=fitness_weight,
        min_quality=min_quality,
    )


def compute_combined_score_triple(
    quality_pct:     float,
    hardware_score:  float,
    training_score:  float,
    fitness_score:   float,
    *,
    hw_weight:       float = 0.40,
    train_weight:    float = 0.40,
    fitness_weight:  float = 0.20,
    min_quality:     float = 70.0,
) -> float:
    """
    Formula TRIPLE — Hardware + Training + Fitness (3 sumber).

    Dipakai ketika semua 3 sinyal tersedia.
    Bobot default: HW 40% + Training 40% + Fitness 20%.

    Fitness (hardware.py _fitness_score) adalah estimasi training performance
    formula berbasis roofline model — berguna sebagai sinyal tambahan tapi
    TIDAK seakurat hasil proxy training nyata dari train_refine.py.
    """
    if quality_pct < min_quality:
        return 0.0

    q_gate = (quality_pct - min_quality) / max(0.001, 100.0 - min_quality)
    q_gate = float(max(0.0, min(1.0, q_gate)))

    hw  = float(max(0.0, min(1.0, hardware_score)))
    tr  = float(max(0.0, min(1.0, training_score)))
    fi  = float(max(0.0, min(1.0, fitness_score)))

    raw = hw_weight * hw + train_weight * tr + fitness_weight * fi
    gate_mult = 0.80 + 0.20 * q_gate
    return round(raw * gate_mult, 6)


def select_best_arch_balanced(
    archs:           List[ArchConfig],
    quality_map:     Dict[str, float],
    hardware_scores: Dict[str, float],
    training_scores: Dict[str, float],
    *,
    min_quality:     float = 70.0,
) -> Optional['ArchConfig']:
    """
    Pilih arsitektur terbaik menggunakan combined score 50/50 balanced.

    Args:
        archs:           list ArchConfig yang sudah refined
        quality_map:     dict arch_id → quality_pct dari ArcQualityScorer
        hardware_scores: dict arch_id → hardware_score dari HardwareNASEvaluator
        training_scores: dict arch_id → training_score dari TrainingDynamicsEvaluator
        min_quality:     threshold minimum quality

    Returns:
        ArchConfig dengan combined score tertinggi, atau None jika tidak ada kandidat
    """
    candidates = [a for a in archs if a.fits_gpu]
    if not candidates:
        return None

    def _score(cfg: ArchConfig) -> float:
        q  = quality_map.get(cfg.arch_id, 0.0)
        hw = hardware_scores.get(cfg.arch_id, 0.0)
        tr = training_scores.get(cfg.arch_id, 0.0)
        return compute_combined_score_balanced(q, hw, tr, min_quality=min_quality)

    return max(candidates, key=_score)


def rank_archs_balanced(
    archs:           List[ArchConfig],
    quality_map:     Dict[str, float],
    hardware_scores: Dict[str, float],
    training_scores: Dict[str, float],
    *,
    min_quality:     float = 70.0,
) -> List[Tuple[ArchConfig, float]]:
    """
    Ranking semua arsitektur dengan combined score 50/50 balanced.

    Returns:
        List of (ArchConfig, combined_score) sorted descending.
    """
    scored = []
    for cfg in archs:
        if not cfg.fits_gpu:
            continue
        q  = quality_map.get(cfg.arch_id, 0.0)
        hw = hardware_scores.get(cfg.arch_id, 0.0)
        tr = training_scores.get(cfg.arch_id, 0.0)
        c  = compute_combined_score_balanced(q, hw, tr, min_quality=min_quality)
        scored.append((cfg, c))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def refine_archs(
    archs:          List[ArchConfig],
    gpu:            GPUSpec,
    max_iterations: int   = 30,
    target_pct:     float = 100.0,
) -> Tuple[List[ArchConfig], List[RefinementLog]]:
    """
    Drop-in untuk pipeline.py — refine batch ARCs dan kembalikan:
        (sorted_archs, refinement_logs)
    """
    refiner = ArcRefiner(gpu, max_iterations=max_iterations, target_pct=target_pct)
    return refiner.refine_batch(archs)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS  (Rich-aware)
# ══════════════════════════════════════════════════════════════════════════════

def print_score_report(report: ScoreReport, *, console=None) -> None:
    """Print laporan ScoreReport detail ke konsol."""
    try:
        from rich.console import Console
        if console is None:
            console = Console()
    except ImportError:
        console = None

    _print = console.print if console else print

    DIM_LABELS = {
        "A": "Structural Integrity  (30 pts)",
        "B": "Memory Consistency    (25 pts)",
        "C": "FLOPs Correctness     (20 pts)",
        "D": "Hardware Fitness      (15 pts)",
        "E": "Optimization Flags    (10 pts)",
    }

    _print()
    _print(f"  ┌─ ARC Quality Report ─── {report.arch_id} {'─'*40}")
    _print(f"  │  {report.arch_name}")
    _print(f"  │  Score: {report.total_score:.1f} / {report.max_score:.0f} pts"
           f"   ({report.pct:.1f}%)   Grade: {report.grade}")
    _print(f"  └{'─'*70}")

    by_dim = report.by_dimension()
    for dim_key in ["A", "B", "C", "D", "E"]:
        checks = by_dim.get(dim_key, [])
        if not checks:
            continue
        earned = sum(c.points_earned for c in checks)
        max_d  = sum(c.points_max    for c in checks)
        bar_n  = int(earned / max_d * 20) if max_d > 0 else 0
        bar    = ("█" * bar_n + "░" * (20 - bar_n))
        _print(f"\n  [{dim_key}] {DIM_LABELS[dim_key]}"
               f"  {earned:.1f}/{max_d:.0f}  [{bar}]")
        for c in checks:
            if c.points_earned >= c.points_max:
                sym = "  ✓"
            elif c.points_earned > 0:
                sym = "  ~"
            else:
                sym = "  ✗"
            _print(f"       {sym}  {c.name:<42}  "
                   f"{c.points_earned:.1f}/{c.points_max:.1f}  {c.detail}")
            if c.points_earned < c.points_max and c.fix_hint:
                _print(f"            → Fix: {c.fix_hint}")

    _print()


def print_refinement_summary(
    logs: List[RefinementLog],
    *,
    console=None,
) -> None:
    """
    Print tabel ringkasan semua refinement runs.

    Dua skor per ARC:
      Quality %  — konsistensi internal (0–100). Mengukur apakah kalkulasi
                   ARC benar dan self-consistent. TIDAK menentukan ranking final.

      Fitness    — performa training aktual (MFU, throughput, efisiensi VRAM,
                   kepadatan parameter). INI yang menjadi skor ranking.
    """
    _print = console.print if console else print

    # Urutkan berdasarkan final fitness descending
    ranked = sorted(logs, key=lambda l: l.final_fitness, reverse=True)

    _print()
    _print("  ┌─ Refinement Summary ─────────────────────────────────────────────────────────────────")
    _print("  │")
    _print("  │  Dua skor per ARC:")
    _print("  │    Quality %  = apakah kalkulasi internal benar? (drives fixes, BUKAN ranking)")
    _print("  │    Fitness    = skor performa training aktual    (drives recommendation ★)")
    _print("  │")
    _print(f"  │  {'Rank':<5} {'ARC-ID':<12} "
           f"{'Quality':>13}  {'Δ':>6}  "
           f"{'Fitness':>13}  {'ΔFit':>7}  "
           f"{'Iters':>5}  Status")
    _print("  │  " + "─" * 90)

    for rank, log in enumerate(ranked, 1):
        q_delta   = f"{log.improved_by:+.1f}pp"
        fit_delta = f"{log.fitness_delta:+.4f}"
        rank_sym  = "★" if rank == 1 else f"#{rank}"
        _print(
            f"  │  {rank_sym:<5} {log.arch_id:<12} "
            f"{log.initial_pct:>6.1f}%→{log.final_pct:>5.1f}%  {q_delta:>6}  "
            f"{log.initial_fitness:>6.4f}→{log.final_fitness:>6.4f}  {fit_delta:>7}  "
            f"{log.iterations:>5}  {log.status}"
        )

    _print("  │")
    _print("  │  CATATAN: Skor Quality % mencerminkan kualitas NYATA ARC berdasarkan analisis")
    _print("  │  kuantitatif — bukan referensi ke template model yang sudah ada.")
    _print("  │  ARC yang naik rank setelah refinement berarti perbaikan (misal: flash attention,")
    _print("  │  TC alignment, batch tuning) benar-benar meningkatkan metrik training.")
    _print("  └──────────────────────────────────────────────────────────────────────────────────────")
    _print()


def print_full_refinement_log(log: RefinementLog, *, console=None) -> None:
    """Print histori per-fix untuk satu ARC."""
    _print = console.print if console else print

    _print(f"\n  ─── Refinement Log: {log.arch_id} {'─'*45}")
    _print(f"       Quality: {log.initial_pct:.1f}% → {log.final_pct:.1f}%"
           f"   Δ={log.improved_by:+.1f} pp   Iters: {log.iterations}")
    _print(f"       Fitness: {log.initial_fitness:.4f} → {log.final_fitness:.4f}"
           f"   Δ={log.fitness_delta:+.4f}")
    _print(f"       Status:  {log.status}")
    if log.score_history:
        hist = " → ".join(f"{p:.1f}%" for p in log.score_history)
        _print(f"       Quality history: {hist}")
    if log.fixes_applied:
        p1 = [f for f in log.fixes_applied if "/P1]" in f]
        p2 = [f for f in log.fixes_applied if "/P2]" in f]
        other = [f for f in log.fixes_applied if "/P1]" not in f and "/P2]" not in f]
        if p1:
            _print(f"       Phase 1 fixes (formula corrections): {len(p1)}")
            for fix in p1:
                _print(f"         • {fix}")
        if p2:
            _print(f"       Phase 2 fixes (architectural improvements): {len(p2)}")
            for fix in p2:
                _print(f"         • {fix}")
        for fix in other:
            _print(f"         • {fix}")
    _print()
