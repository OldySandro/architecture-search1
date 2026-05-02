"""
fast.py — Auto NAS Pipeline (Hardware + Training) ·  PARALLEL EDITION
════════════════════════════════════════════════════════════════════════════════

Paralel:
  • Total ARC dibagi rata ke N_WORKERS worker (default 10)
  • Tiap worker menjalankan pipeline LENGKAP: Generate → HW-NAS
  • Training NAS dijalankan serial di main process (CUDA-safe)
  • Hasil semua worker digabung lalu diranking bersama (Stage 4-6 tetap serial)
  • Seed tiap worker = seed_base + worker_index  (reproduksi tetap bisa)

Output sama persis dengan pipeline.py:
  • Balanced ranking table dengan PRE→POST rank (↑↓=)
  • Final recommendation box: H1-H7 hardware + T1-T6 training + NAS-RL log
  • JSON export komprehensif: nas_journey, best_arch_detail, sub-scores

Scoring: balanced 50/50 (sama persis):
  combined = (0.5 × hw_score + 0.5 × train_score) × quality_gate
"""

from __future__ import annotations

import sys
import os
import json
import math
import time
import copy
import datetime
import random
import traceback
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Path setup ────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# ── Rich ──────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, \
                               TimeElapsedColumn, MofNCompleteColumn
    from rich         import box as rbox
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class _FallbackConsole:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 80)
    console = _FallbackConsole()

# ── Module imports ────────────────────────────────────────────────────────────
from hardware        import GPU_DATABASE, GPUSpec
from arch_types      import ArchConfig
from generator       import ArchitectureGenerator
from profiler        import TorchProfiler, TORCH, DEVICE
from refiner         import (
    ArcQualityScorer, ArcRefiner,
    compute_combined_score_balanced,
    compute_combined_score_triple,
    print_score_report,
)
from hardware_refine import (
    HardwareNASRefiner, HardwareNASEvaluator,
    HardwareAdaptiveLog,
    print_hardware_adaptive_summary,
    print_hardware_nas_result,
    print_hardware_adaptive_log,
)
from train_refine    import (
    TrainingNASRefiner, TrainingDynamicsEvaluator, ProxyTrainer,
    TrainingAdaptiveLog,
    print_training_adaptive_summary,
    print_training_nas_result,
    print_training_adaptive_log,
)
from metrics import MetricsReport

# ── Family map ────────────────────────────────────────────────────────────────
_FAMILIES = [
    ("CoT-Optimizer",  "Deep narrow — chain-of-thought, math, code"),
    ("Speed-Demon",    "Wide shallow — max tokens/sec throughput"),
    ("Balanced-Pro",   "Balanced depth/width — general-purpose"),
    ("MoE-Sparse",     "Mixture-of-Experts — sparse compute"),
    ("Long-Horizon",   "Extended context — long-range dependencies"),
    ("Nano-Efficient", "Ultra-small — edge/embedded devices"),
    ("Compute-Dense",  "High FLOP/byte — tensor cores maximal"),
]

_FAMILY_NAMES = [f for f, _ in _FAMILIES]

# ── GPU keyword map ───────────────────────────────────────────────────────────
_GPU_KEYWORDS: Dict[str, str] = {
    "t4": "T4",
    "v100-16": "V100-16GB", "v100 16": "V100-16GB",
    "v100-32": "V100-32GB", "v100 32": "V100-32GB",
    "v100":    "V100-32GB",
    "rtx3090": "RTX-3090", "rtx 3090": "RTX-3090", "3090": "RTX-3090",
    "a6000": "A6000", "rtxa6000": "A6000",
    "a100-40": "A100-40GB", "a100 40": "A100-40GB",
    "a100-80": "A100-80GB", "a100 80": "A100-80GB",
    "a100":    "A100-80GB",
    "rtx4090": "RTX-4090", "rtx 4090": "RTX-4090", "4090": "RTX-4090",
    "h100-pcie": "H100-PCIe", "h100 pcie": "H100-PCIe", "h100pcie": "H100-PCIe",
    "h100-sxm":  "H100-SXM",  "h100 sxm":  "H100-SXM",  "h100sxm":  "H100-SXM",
    "h100":      "H100-SXM",
    "h200":     "H200-SXM", "h200-sxm": "H200-SXM", "h200 sxm": "H200-SXM",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """Semua parameter pipeline dikumpulkan di sini sebelum eksekusi."""

    def __init__(self):
        self.gpu:           GPUSpec       = None
        self.families:      List[str]     = []       # [] = all
        self.total_archs:   int           = 1000     # Range
        self.n_per_family:  int           = 0        # dihitung dari total_archs
        self.seed:          int           = 42
        self.run_profiling: bool          = True
        self.device:        str           = "cpu"
        self.max_hw_iters:  int           = 25
        self.max_tr_iters:  int           = 25
        self.max_explore:   int           = 30
        self.output_file:   str           = "nas_results_custom.json"
        # ── Paralel ───────────────────────────────────────────────────────────
        self.n_workers:     int           = 10       # jumlah worker paralel

    def finalize(self):
        """Hitung derived fields setelah semua input dikumpulkan."""
        fam_count = len(self.families) if self.families else 7
        self.n_per_family = max(1, self.total_archs // fam_count)
        if TORCH:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS  (sama persis dengan pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def _enum_str(val) -> str:
    """Extract clean string dari enum atau string biasa."""
    if val is None:
        return "—"
    if hasattr(val, 'value'):
        return str(val.value)
    s = str(val)
    if '.' in s:
        s = s.split('.')[-1]
    return s


def _row(label: str, value: str, width: int = 88) -> str:
    """Format satu baris kotak rekomendasi dengan padding yang tepat."""
    content = f"  {label:<22} {value}"
    if len(content) > width - 2:
        content = content[:width - 5] + "..."
    return f"│{content:<{width}}│"


def _print_balanced_ranking_table(
    archs:           List[ArchConfig],
    quality_map:     Dict[str, float],
    hw_scores:       Dict[str, float],
    train_scores:    Dict[str, float],
    combined_map:    Dict[str, float],
    hw_log_map:      Dict[str, object],   # arch_id → HardwareAdaptiveLog (atau dict)
    train_log_map:   Dict[str, object],   # arch_id → TrainingAdaptiveLog (atau dict)
    pre_nas_rank:    Dict[str, int],
    *,
    console=None,
) -> None:
    """Print tabel ranking balanced komprehensif dengan perbandingan pre/post NAS."""
    _p = console.print if console else print

    _p()
    _p("  ┌─ Balanced 50/50 Combined Score Ranking ─────────────────────────────────────────────────────────────────────────")
    _p("  │")
    _p("  │  SISTEM PENILAIAN (seimbang):")
    _p("  │    Hardware Score  = 50%  → 7 dimensi GPU real (MFU·Throughput·VRAM·TC·SM·Compute·FA)")
    _p("  │    Training Score  = 50%  → 6 dimensi training real (Conv·Stab·Grad·GenGap·SampEff·OptComp)")
    _p("  │    Quality Gate    = ×(0.80–1.00)  → ArcQualityScorer konsistensi formula internal")
    _p("  │    combined = (0.50×hw + 0.50×ts) × (0.80 + 0.20×quality_gate)")
    _p("  │")
    _p("  │  PRE→POST = perubahan rank sebelum vs sesudah Dual NAS (↑naik ↓turun =sama)")
    _p("  │")

    header = (f"  │  {'Rank':>4}  {'PRE→POST':>8}  {'ARC-ID':<12}  "
              f"{'Quality':>7}  {'HW-Score':>8}  {'Train-Score':>11}  "
              f"{'Combined':>9}  {'VRAM':>6}  {'MFU':>6}  {'NaN':>4}  Status")
    _p(header)
    _p("  │  " + "─" * 120)

    for rank, cfg in enumerate(archs[:15], 1):
        aid    = cfg.arch_id
        q      = quality_map.get(aid, 0.0)
        hw     = hw_scores.get(aid, 0.0)
        tr     = train_scores.get(aid, 0.0)
        comb   = combined_map.get(aid, 0.0)
        hw_log = hw_log_map.get(aid)
        tr_log = train_log_map.get(aid)

        sym = "★" if rank == 1 else f"#{rank}"

        pre_r = pre_nas_rank.get(aid, rank)
        if pre_r > rank:
            rank_arrow = f"#{pre_r}→★" if rank == 1 else f"#{pre_r}↑#{rank}"
        elif pre_r < rank:
            rank_arrow = f"#{pre_r}↓#{rank}"
        else:
            rank_arrow = f"#{pre_r}→#{rank}"

        # NaN dari training log
        def _get(obj, attr, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        has_nan = (_get(tr_log, "nas_nan_count", 0) > 0)
        nan_str  = "NaN⚠" if has_nan else "ok"

        status_parts = []
        hw_acc = _get(hw_log, "perturbations_accepted", 0)
        tr_acc = _get(tr_log, "perturbations_accepted", 0)
        if hw_log and hw_acc > 0:
            status_parts.append(f"HW↑{hw_acc}")
        if tr_log and tr_acc > 0:
            status_parts.append(f"TR↑{tr_acc}")
        status = " ".join(status_parts) if status_parts else "—"

        fits_marker = " [OOM]" if not cfg.fits_gpu else ""
        _p(
            f"  │  {sym:>4}  {rank_arrow:>8}  {aid:<12}  "
            f"{q:>6.1f}%  {hw:>8.4f}  {tr:>11.4f}  "
            f"{comb:>9.5f}  {cfg.vram_usage_pct:>5.1f}%  {cfg.mfu_estimate:>6.4f}  "
            f"{nan_str:>4}  {status}{fits_marker}"
        )

    _p("  │")
    _p("  │  ★ = Top recommendation  ↑/↓ = naik/turun rank dari pre-NAS")
    _p("  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────")
    _p()


def stage_final_recommendation(
    best:            ArchConfig,
    quality_map:     Dict[str, float],
    hw_scores:       Dict[str, float],
    train_scores:    Dict[str, float],
    combined_map:    Dict[str, float],
    gpu:             GPUSpec,
    hw_logs:         list,
    train_logs:      list,
    archs_sorted:    List[ArchConfig],
    nas_results_map: Dict[str, object],
    pre_nas_rank:    Dict[str, int],
) -> None:
    """
    Print final recommendation box identik dengan pipeline.py.
    Menggunakan cached NASResult dari Stage Training NAS — tidak re-run proxy.
    """
    _p = console.print

    def _get(obj, attr, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    aid  = best.arch_id
    q    = quality_map.get(aid, 0.0)
    hw   = hw_scores.get(aid, 0.0)
    tr   = train_scores.get(aid, 0.0)
    comb = combined_map.get(aid, 0.0)

    hw_log    = next((l for l in hw_logs    if _get(l, "arch_id") == aid), None)
    train_log = next((l for l in train_logs if _get(l, "arch_id") == aid), None)

    # Hardware detail (cepat, tanpa proxy training)
    hw_eval   = HardwareNASEvaluator(gpu)
    hw_result = hw_eval.evaluate(best)

    # Cached NASResult dari Stage Training NAS
    nas_result = nas_results_map.get(aid)
    if nas_result is not None:
        tr_result = nas_result
        proxy_res = getattr(nas_result, "proxy_result", None)
    else:
        tr_result = None
        proxy_res = None

    pre_rank  = pre_nas_rank.get(aid, "?")
    rank_info = f"pre-NAS #{pre_rank} → balanced #1"

    W = 90
    S = "─" * W

    _p()
    _p(f"╭{S}╮")
    _p(f"│{'  🏆 FINAL RECOMMENDATION — BALANCED 50/50 NAS':^{W}}│")
    _p(f"├{S}┤")
    _p(_row("Top Pick:", f"{aid} — {best.arch_name[:52]}", W))
    _p(_row("Rank Journey:", rank_info, W))
    _p(f"├{S}┤")
    _p(f"│{'  COMBINED SCORE (50% Hardware + 50% Training)':^{W}}│")
    gate_m = 0.80 + 0.20 * max(0.0, min(1.0, (q - 70) / 30))
    _p(_row("Combined (50/50):", f"{comb:.5f}", W))
    _p(_row("  ← Hardware Score:", f"{hw:.4f}  ×50%  ({hw*100:.1f}%)", W))
    tr_grade = _get(tr_result, "grade", "")
    tr_suffix = f"  [{tr_grade[:18]}]" if tr_result and tr_grade else ""
    _p(_row("  ← Training Score:", f"{tr:.4f}  ×50%  ({tr*100:.1f}%)" + tr_suffix, W))
    _p(_row("  ← Quality Gate:", f"{q:.1f}%  (×{gate_m:.3f} multiplier)", W))
    _p(f"├{S}┤")
    _p(f"│{'  HARDWARE DIMENSIONS (7 dari GPUSpec)':^{W}}│")
    _p(_row("H1 MFU Utilization:",    f"{hw_result.pts_h1:>5.1f}/25  score={hw_result.mfu_score:.3f}  mfu={best.mfu_estimate:.4f}", W))
    _p(_row("H2 Throughput Eff:",     f"{hw_result.pts_h2:>5.1f}/20  score={hw_result.throughput_score:.3f}  tok/s={best.tokens_per_sec_estimate:,}", W))
    _p(_row("H3 VRAM Utilization:",   f"{hw_result.pts_h3:>5.1f}/15  score={hw_result.vram_efficiency:.3f}  {best.vram_usage_pct:.1f}% of {gpu.vram_gb:.0f}GB", W))
    _p(_row("H4 TC Alignment:",       f"{hw_result.pts_h4:>5.1f}/15  score={hw_result.tc_alignment:.3f}  tile={hw_result.tc_tile_size}", W))
    _p(_row("H5 SM Occupancy:",       f"{hw_result.pts_h5:>5.1f}/10  score={hw_result.sm_occupancy_score:.3f}  sm_occ={best.sm_occupancy:.3f}", W))
    _p(_row("H6 Compute Boundness:",  f"{hw_result.pts_h6:>5.1f}/10  score={hw_result.compute_bound_score:.3f}  AI={best.arithmetic_intensity:.1f} FLOP/B", W))
    _p(_row("H7 FA Tile Feasibility:",f"{hw_result.pts_h7:>5.1f}/5   score={hw_result.flash_attn_score:.3f}  FA={best.use_flash_attn}", W))
    _p(_row("Hardware Total:",        f"{hw_result.total_pts:.1f}/100  [{hw_result.grade[:30]}]", W))
    _p(f"├{S}┤")
    _p(f"│{'  TRAINING DIMENSIONS (6 dari proxy PyTorch)':^{W}}│")
    if tr_result and proxy_res:
        nan_lbl = "YES ⚠" if proxy_res.nan_detected else "no ✓"
        loss_i  = getattr(proxy_res, "loss_initial", 0.0)
        loss_f  = getattr(proxy_res, "loss_final", 0.0)
        gen_gap = getattr(proxy_res, "generalization_gap", 0.0)
        reduction = max(0, (loss_i - loss_f) / max(0.001, loss_i) * 100)
        _p(_row("T1 Convergence Rate:",   f"{tr_result.pts_t1:>5.1f}/22  score={tr_result.convergence_score:.3f}  reduction={reduction:.1f}%", W))
        _p(_row("T2 Training Stability:", f"{tr_result.pts_t2:>5.1f}/22  score={tr_result.stability_score:.3f}  NaN={nan_lbl}", W))
        _p(_row("T3 Gradient Health:",    f"{tr_result.pts_t3:>5.1f}/18  score={tr_result.gradient_health:.3f}  risk={tr_result.gradient_risk}", W))
        _p(_row("T4 Generalization Gap:", f"{tr_result.pts_t4:>5.1f}/15  score={tr_result.generalization_score:.3f}  gap={gen_gap:.4f}", W))
        _p(_row("T5 Sample Efficiency:",  f"{tr_result.pts_t5:>5.1f}/13  score={tr_result.sample_efficiency:.3f}  eff_batch={best.batch_size*best.seq_len}", W))
        _p(_row("T6 Optimizer Compat:",   f"{tr_result.pts_t6:>5.1f}/10  score={tr_result.optimizer_compat:.3f}  lr_sens={tr_result.lr_sensitivity}", W))
        _p(_row("Training Total:",        f"{tr_result.total_pts:.1f}/100  [{tr_result.grade[:30]}]", W))
        _p(_row("Training Regime:",       tr_result.regime[:55], W))
    else:
        _p(_row("Training Score:", f"{tr:.4f}  (cached dari Training NAS)", W))
    _p(f"├{S}┤")
    _p(f"│{'  ARCHITECTURE PARAMETERS':^{W}}│")
    ffn_str  = _enum_str(best.ffn_type)
    attn_str = _enum_str(best.attn_type)
    opt_str  = _enum_str(best.optimizer_type)
    norm_str = _enum_str(best.norm_type)
    _p(_row("Params / Layers / Hidden:", f"{best.param_count/1e6:.1f}M  L={best.num_layers}  D={best.hidden_dim}  H={best.num_heads}/{best.num_kv_heads}  hd={best.head_dim}", W))
    _p(_row("FFN / Attn / Norm:",        f"{ffn_str}  (×{best.ffn_multiplier:.2f})  |  {attn_str}  |  {norm_str}", W))
    _p(_row("Optimizer:",                f"{opt_str}  |  SeqLen={best.seq_len}  Batch={best.batch_size}", W))
    flags = (f"FA={best.use_flash_attn}  MixedPrec={best.use_mixed_precision}  "
             f"TieEmbed={best.tie_embeddings}  GradCkpt={best.use_gradient_checkpointing}  "
             f"Compile={best.use_torch_compile}")
    _p(_row("Flags:", flags, W))
    _p(_row("VRAM / Bottleneck / Step:", f"{best.vram_total_gb:.2f}GB ({best.vram_usage_pct:.1f}%)  {best.bottleneck}  {best.ms_per_step:.1f}ms/step", W))

    if hw_log:
        _p(f"├{S}┤")
        _p(f"│{'  HARDWARE NAS-RL SUMMARY':^{W}}│")
        hw_s  = _get(hw_log, "hw_score_start", 0.0)
        hw_e  = _get(hw_log, "hw_score_end", 0.0)
        hw_t  = _get(hw_log, "perturbation_tries", 0)
        hw_a  = _get(hw_log, "perturbations_accepted", 0)
        hw_tc = _get(hw_log, "tc_improvements", 0)
        _p(_row("HW Score journey:", f"{hw_s:.4f}→{hw_e:.4f}  tries={hw_t}  accepted={hw_a}  TC-improv={hw_tc}", W))

    if train_log:
        _p(f"├{S}┤")
        _p(f"│{'  TRAINING NAS-RL SUMMARY':^{W}}│")
        ts_s   = _get(train_log, "train_score_start", 0.0)
        ts_e   = _get(train_log, "train_score_end", 0.0)
        nas_ev = _get(train_log, "nas_evaluations", 0)
        nan_c  = _get(train_log, "nas_nan_count", 0)
        nas_ms = _get(train_log, "nas_training_ms_total", 0.0)
        pt     = _get(train_log, "perturbation_tries", 0)
        pa     = _get(train_log, "perturbations_accepted", 0)
        ci     = _get(train_log, "convergence_improvements", 0)
        si     = _get(train_log, "stability_improvements", 0)
        _p(_row("Train Score journey:", f"{ts_s:.4f}→{ts_e:.4f}  NAS-evals={nas_ev}  NaN={nan_c}  {nas_ms:.0f}ms", W))
        _p(_row("RL:", f"tries={pt}  accepted={pa}  conv-improv={ci}  stab-improv={si}", W))

    _p(f"╰{S}╯")
    _p()

    # Runner-up
    if len(archs_sorted) > 1:
        _p("  Runner-up architectures:")
        for i, cfg in enumerate(archs_sorted[1:4], 2):
            aid2   = cfg.arch_id
            pre_r2 = pre_nas_rank.get(aid2, "?")
            pre_info = f"(pre-NAS #{pre_r2})" if pre_r2 != "?" else ""
            _p(f"    #{i}  {aid2:<12}  {pre_info:<14}  "
               f"combined={combined_map.get(aid2, 0):.5f}  "
               f"hw={hw_scores.get(aid2, 0):.4f}  "
               f"train={train_scores.get(aid2, 0):.4f}  "
               f"quality={quality_map.get(aid2, 0):.1f}%  "
               f"params={cfg.param_count/1e6:.1f}M  VRAM={cfg.vram_usage_pct:.1f}%")
        _p()


# ══════════════════════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _show_banner():
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]  CUSTOM AUTO NAS PIPELINE  [PARALLEL EDITION][/bold cyan]\n"
            "[dim]  Hardware NAS 7D + Training NAS 6D  ·  Balanced 50/50  ·  Auto-Run[/dim]\n"
            "[dim]  Config-First · Parallel Workers · Full Pipeline Output[/dim]",
            border_style="cyan", padding=(1, 4)
        ))
    else:
        print("=" * 70)
        print("  CUSTOM AUTO NAS PIPELINE  [PARALLEL EDITION]")
        print("  Hardware NAS + Training NAS · Balanced 50/50 · Parallel Workers")
        print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  UI — GPU SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_gpu() -> GPUSpec:
    console.print("\n[bold yellow]Available GPUs (ketik keyword):[/bold yellow]\n")

    if RICH:
        t = Table(show_header=True, header_style="bold magenta",
                  box=rbox.ROUNDED, padding=(0, 1))
        t.add_column("#",       style="dim",    width=3)
        t.add_column("Keyword", style="cyan",   width=14)
        t.add_column("Name",    style="white",  width=28)
        t.add_column("VRAM",    style="green",  width=6)
        t.add_column("TFLOPS",  style="yellow", width=8)
        t.add_column("BW GB/s", style="blue",   width=8)
        t.add_column("Mem",     style="magenta",width=8)

        keys = list(GPU_DATABASE.keys())
        for i, k in enumerate(keys, 1):
            g = GPU_DATABASE[k]
            t.add_row(str(i), k, g.name,
                      f"{g.vram_gb:.0f}GB",
                      f"{g.bf16_tflops:.0f}",
                      f"{g.memory_bw_gbps:.0f}",
                      g.memory_type)
        console.print(t)
        console.print("[dim]  Keywords: T4, V100, V100-32, RTX3090, A6000, A100, A100-40, "
                      "A100-80, RTX4090, H100, H100-PCIe, H100-SXM, H200[/dim]\n")
    else:
        for k, g in GPU_DATABASE.items():
            print(f"  {k:<14} {g.name:<30} {g.vram_gb:.0f}GB  {g.bf16_tflops:.0f}TFLOPS")
        print()

    while True:
        try:
            raw = input("Hardware : ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

        for db_key in GPU_DATABASE:
            if raw.upper() == db_key.upper() or raw == db_key.lower():
                gpu = GPU_DATABASE[db_key]
                console.print(f"\n[bold green]✓ GPU: {gpu.name}[/bold green]\n")
                return gpu

        matched_key = _GPU_KEYWORDS.get(raw)
        if matched_key and matched_key in GPU_DATABASE:
            gpu = GPU_DATABASE[matched_key]
            console.print(f"\n[bold green]✓ GPU: {gpu.name}[/bold green]\n")
            return gpu

        matches = [k for k in GPU_DATABASE if raw in k.lower()]
        if len(matches) == 1:
            gpu = GPU_DATABASE[matches[0]]
            console.print(f"\n[bold green]✓ GPU: {gpu.name}[/bold green]\n")
            return gpu
        elif len(matches) > 1:
            console.print(f"[yellow]  Ambigu: {matches}. Pilih lebih spesifik.[/yellow]")
        else:
            console.print(f"[red]  Tidak ditemukan '{raw}'. Coba: T4, A100, H100, RTX4090[/red]")


# ══════════════════════════════════════════════════════════════════════════════
#  UI — AI TYPE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _select_ai_type() -> List[str]:
    console.print("[bold yellow]AI Type (pilih 1–7 atau 'all'):[/bold yellow]")

    if RICH:
        t = Table(show_header=True, header_style="bold magenta",
                  box=rbox.ROUNDED, padding=(0, 1))
        t.add_column("#",       style="dim",    width=3)
        t.add_column("Family",  style="cyan",   width=18)
        t.add_column("Description", style="dim")
        for i, (name, desc) in enumerate(_FAMILIES, 1):
            t.add_row(str(i), name, desc)
        console.print(t)
    else:
        for i, (name, desc) in enumerate(_FAMILIES, 1):
            print(f"  [{i}] {name:<18} {desc}")

    console.print("[dim]  Contoh: '1' atau '1,3,5' atau 'all' (Enter = all)[/dim]")

    while True:
        try:
            raw = input("Type AI  : ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return []

        if not raw or raw == "all":
            console.print("[bold green]✓ Type AI: ALL (7 families)[/bold green]\n")
            return []

        parts = [p.strip() for p in raw.replace(" ", "").split(",") if p.strip()]
        chosen, invalid = [], []
        for p in parts:
            try:
                idx = int(p) - 1
                if 0 <= idx < len(_FAMILIES):
                    chosen.append(_FAMILIES[idx][0])
                else:
                    invalid.append(p)
            except ValueError:
                matches = [f for f, _ in _FAMILIES if p in f.lower()]
                if matches:
                    chosen.extend(matches)
                else:
                    invalid.append(p)

        if invalid:
            console.print(f"[red]  Tidak valid: {invalid}. Masukkan angka 1–7.[/red]")
            continue

        chosen = list(dict.fromkeys(chosen))
        if not chosen:
            console.print("[yellow]  Tidak ada pilihan valid. Coba lagi.[/yellow]")
            continue

        console.print(f"[bold green]✓ Type AI: {', '.join(chosen)}[/bold green]\n")
        return chosen


# ══════════════════════════════════════════════════════════════════════════════
#  UI — RANGE
# ══════════════════════════════════════════════════════════════════════════════

def _select_range() -> int:
    console.print("[dim]  Berapa total ARC yang akan digenerate? (min 10, max 100000)[/dim]")
    while True:
        try:
            raw = input("Range    : ").strip()
        except (KeyboardInterrupt, EOFError):
            return 300

        if not raw:
            console.print("[yellow]  Default: 300[/yellow]\n")
            return 300

        try:
            val = int(raw.replace(",", "").replace(".", ""))
            if val < 10:
                console.print(f"[yellow]  Minimum 10. Set ke 10.[/yellow]\n")
                return 10
            if val > 100000:
                console.print(f"[yellow]  Maximum 100000. Set ke 100000.[/yellow]\n")
                return 100000
            console.print(f"[bold green]✓ Range: {val:,} architectures[/bold green]\n")
            return val
        except ValueError:
            console.print(f"[red]  Masukkan angka valid (contoh: 300, 1000, 5000)[/red]")


# ══════════════════════════════════════════════════════════════════════════════
#  UI — WORKERS
# ══════════════════════════════════════════════════════════════════════════════

def _select_workers(total_archs: int) -> int:
    cpu_count = os.cpu_count() or 4
    suggested = min(10, cpu_count, total_archs)
    console.print(
        f"[dim]  Jumlah worker paralel? "
        f"(CPU tersedia: {cpu_count}, disarankan: {suggested}, max: 32)[/dim]"
    )
    try:
        raw = input(f"Workers  : ").strip()
    except (KeyboardInterrupt, EOFError):
        return suggested

    if not raw:
        console.print(f"[bold green]✓ Workers: {suggested} (default)[/bold green]\n")
        return suggested

    try:
        val = int(raw)
        val = max(1, min(32, val))
        val = min(val, total_archs)
        console.print(f"[bold green]✓ Workers: {val}[/bold green]\n")
        return val
    except ValueError:
        console.print(f"[yellow]  Tidak valid. Default: {suggested}[/yellow]\n")
        return suggested


# ══════════════════════════════════════════════════════════════════════════════
#  UI — SEED
# ══════════════════════════════════════════════════════════════════════════════

def _select_seed() -> int:
    console.print("[dim]  Seed: '42' (fixed) atau 'auto' (random setiap run)[/dim]")
    try:
        raw = input("Seed     : ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return 42

    if not raw or raw == "42" or raw == "default":
        console.print("[bold green]✓ Seed: 42 (fixed)[/bold green]\n")
        return 42
    elif raw in ("auto", "random"):
        seed = random.randint(1, 99999)
        console.print(f"[bold green]✓ Seed: {seed} (auto-generated)[/bold green]\n")
        return seed
    else:
        try:
            seed = int(raw)
            console.print(f"[bold green]✓ Seed: {seed}[/bold green]\n")
            return seed
        except ValueError:
            console.print("[yellow]  Tidak valid. Default: 42[/yellow]\n")
            return 42


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def collect_config() -> PipelineConfig:
    """Kumpulkan semua konfigurasi sebelum pipeline dijalankan."""
    _show_banner()
    console.print("\n[bold cyan]━━ KONFIGURASI PIPELINE ━━[/bold cyan]\n")

    cfg = PipelineConfig()
    cfg.gpu          = _select_gpu()
    cfg.families     = _select_ai_type()
    cfg.total_archs  = _select_range()
    cfg.n_workers    = _select_workers(cfg.total_archs)
    cfg.seed         = _select_seed()
    cfg.run_profiling = True
    cfg.finalize()

    fam_str  = ", ".join(cfg.families) if cfg.families else "ALL (7 families)"
    prof_str = ("Yes (GPU tersedia)" if (TORCH and cfg.device == "cuda")
                else "Yes (analytical fallback)")

    batch_sizes = _compute_batch_sizes(cfg.total_archs, cfg.n_workers)

    if RICH:
        console.print(Panel(
            f"  [bold]GPU:[/bold]            {cfg.gpu.name}\n"
            f"  [bold]Type AI:[/bold]        {fam_str}\n"
            f"  [bold]Range:[/bold]          {cfg.total_archs:,} archs total "
            f"(~{cfg.n_per_family} per family)\n"
            f"  [bold]Workers:[/bold]        {cfg.n_workers} paralel  "
            f"(~{batch_sizes[0]}–{batch_sizes[-1]} ARC per worker)\n"
            f"  [bold]Seed:[/bold]           {cfg.seed} (base, +worker_idx per worker)\n"
            f"  [bold]Auto Profiling:[/bold] {prof_str}\n"
            f"  [bold]Auto NAS HW+TR:[/bold] Yes (Hardware 7D + Training 6D)\n"
            f"  [bold]Auto Save:[/bold]      Yes → {cfg.output_file}\n"
            f"  [bold]Device:[/bold]         {cfg.device}",
            title="[bold]Pipeline Configuration[/bold]",
            border_style="cyan", padding=(0, 2)
        ))
    else:
        print(f"\n  GPU          : {cfg.gpu.name}")
        print(f"  Type AI      : {fam_str}")
        print(f"  Range        : {cfg.total_archs:,} archs (~{cfg.n_per_family}/family)")
        print(f"  Workers      : {cfg.n_workers} paralel")
        print(f"  Seed         : {cfg.seed} (base)")
        print(f"  Auto NAS     : Yes")
        print(f"  Auto Save    : {cfg.output_file}")

    try:
        go = input("\n▶  START pipeline? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

    if go in ("n", "no"):
        console.print("[yellow]  Pipeline dibatalkan.[/yellow]")
        sys.exit(0)

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  PARALLEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_batch_sizes(total: int, n_workers: int) -> List[int]:
    """Hitung ukuran tiap batch. Sisa dibagi ke worker pertama."""
    base  = total // n_workers
    extra = total %  n_workers
    sizes = []
    for i in range(n_workers):
        sizes.append(base + (1 if i < extra else 0))
    return [s for s in sizes if s > 0]


def _hwlog_to_dict(log) -> dict:
    """Serialize HardwareAdaptiveLog → dict untuk IPC antar process."""
    def _get(obj, attr, default=None):
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
    return {
        "arch_id":                _get(log, "arch_id", ""),
        "hw_score_start":         _get(log, "hw_score_start", 0.0),
        "hw_score_end":           _get(log, "hw_score_end", 0.0),
        "perturbation_tries":     _get(log, "perturbation_tries", 0),
        "perturbations_accepted": _get(log, "perturbations_accepted", 0),
        "tc_improvements":        _get(log, "tc_improvements", 0),
        "status":                 _get(log, "status", ""),
    }


def _worker_pipeline(
    worker_idx:    int,
    n_archs:       int,
    gpu_key:       str,
    families:      List[str],
    seed:          int,
    device:        str,
    max_hw_iters:  int,
    max_tr_iters:  int,
    max_explore:   int,
) -> dict:
    """
    Satu unit kerja subprocess.
    Mengembalikan archs (serialized), hw_score_map, quality_map, hw_logs_serialized.
    """
    import sys, os
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

    from hardware        import GPU_DATABASE
    from arch_types      import ArchConfig
    from generator       import ArchitectureGenerator
    from refiner         import ArcQualityScorer, compute_combined_score_balanced
    from hardware_refine import HardwareNASRefiner

    gpu = GPU_DATABASE[gpu_key]

    # Subprocess wajib CPU — CUDA context tidak fork-safe
    device = "cpu"

    # ── Generate ──────────────────────────────────────────────────────────────
    gen          = ArchitectureGenerator(gpu, rng_seed=seed)
    all_families = list(ArchitectureGenerator.FAMILY_PROFILES.keys())
    selected     = [f for f in families if f in all_families] if families else all_families
    fam_count    = len(selected)
    n_per_fam    = max(1, n_archs // fam_count)
    remainder    = n_archs - n_per_fam * fam_count

    archs = []
    for fi, fam in enumerate(selected):
        count = n_per_fam + (1 if fi < remainder else 0)
        for _ in range(count):
            archs.append(gen.generate_one(fam))

    archs.sort(key=lambda x: (
        (0 if x.fits_gpu else 1),
        -(x.fitness_score * 0.6 + (1.0 - x.vram_usage_pct / 100.0) * 0.4),
    ))

    # ── Hardware NAS ──────────────────────────────────────────────────────────
    hw_refiner = HardwareNASRefiner(
        gpu,
        max_iterations    = max_hw_iters,
        target_pct        = 100.0,
        max_explore_iters = max_explore,
        rng_seed          = seed,
    )
    scorer = ArcQualityScorer(gpu)

    hw_archs      = []
    hw_score_map: dict = {}
    quality_map:  dict = {}
    hw_logs_raw:  list = []

    for a in archs:
        refined, alog = hw_refiner.refine(a)
        q = scorer.score(refined).pct
        quality_map[refined.arch_id]  = q
        hw_score_map[refined.arch_id] = alog.hw_score_end
        hw_archs.append(refined)
        hw_logs_raw.append(alog)

    # ── Serialize ──────────────────────────────────────────────────────────────
    def _arch_to_dict(a: ArchConfig) -> dict:
        def _ev(v):
            if hasattr(v, "value"): return v.value
            return str(v).split(".")[-1] if "." in str(v) else str(v)
        return {
            "_arch_obj": True,
            "arch_id":            a.arch_id,
            "arch_name":          getattr(a, "arch_name", ""),
            "arch_family":        a.arch_family,
            "vocab_size":         getattr(a, "vocab_size", 32000),
            "hidden_dim":         a.hidden_dim,
            "num_layers":         a.num_layers,
            "num_heads":          a.num_heads,
            "num_kv_heads":       a.num_kv_heads,
            "head_dim":           a.head_dim,
            "seq_len":            a.seq_len,
            "window_size":        getattr(a, "window_size", a.seq_len),
            "global_attn_layers": getattr(a, "global_attn_layers", 0),
            "batch_size":         a.batch_size,
            "attn_type":          _ev(a.attn_type),
            "ffn_type":           _ev(a.ffn_type),
            "norm_type":          _ev(getattr(a, "norm_type", "RMSNORM")),
            "pos_enc":            _ev(getattr(a, "pos_enc", "ROPE")),
            "optimizer_type":     _ev(a.optimizer_type),
            "ffn_multiplier":     getattr(a, "ffn_multiplier", 4.0),
            "num_experts":        getattr(a, "num_experts", 1),
            "top_k_experts":      getattr(a, "top_k_experts", 1),
            "expert_capacity_factor": getattr(a, "expert_capacity_factor", 1.25),
            "dropout":            getattr(a, "dropout", 0.0),
            "tie_embeddings":     getattr(a, "tie_embeddings", True),
            "use_flash_attn":     a.use_flash_attn,
            "use_gradient_checkpointing": getattr(a, "use_gradient_checkpointing", False),
            "use_mixed_precision":        getattr(a, "use_mixed_precision", True),
            "use_torch_compile":          getattr(a, "use_torch_compile", True),
            "param_count":        a.param_count,
            "vram_total_gb":      a.vram_total_gb,
            "vram_usage_pct":     a.vram_usage_pct,
            "vram_weights_gb":    getattr(a, "vram_weights_gb", 0.0),
            "vram_activations_gb":getattr(a, "vram_activations_gb", 0.0),
            "vram_optimizer_gb":  getattr(a, "vram_optimizer_gb", 0.0),
            "vram_kv_cache_gb":   getattr(a, "vram_kv_cache_gb", 0.0),
            "vram_fragmentation_gb": getattr(a, "vram_fragmentation_gb", 0.0),
            "mfu_estimate":       a.mfu_estimate,
            "tokens_per_sec_estimate": a.tokens_per_sec_estimate,
            "ms_per_step":        a.ms_per_step,
            "fits_gpu":           a.fits_gpu,
            "fitness_score":      a.fitness_score,
            "bottleneck":         a.bottleneck,
            "hardware_score":     getattr(a, "hardware_score", 0.0),
            "quality_score_pct":  getattr(a, "quality_score_pct", 0.0),
            "sm_occupancy":       getattr(a, "sm_occupancy", 0.0),
            "warp_divergence_pct":getattr(a, "warp_divergence_pct", 0.0),
            "compiler_speedup":   getattr(a, "compiler_speedup", 1.0),
            "arithmetic_intensity": getattr(a, "arithmetic_intensity", 0.0),
            "flops_per_token_fwd":  getattr(a, "flops_per_token_fwd", 0.0),
            "flops_per_token_bwd":  getattr(a, "flops_per_token_bwd", 0.0),
        }

    def _log_to_dict(log) -> dict:
        def _g(attr, default=None):
            if isinstance(log, dict): return log.get(attr, default)
            return getattr(log, attr, default)
        return {
            "arch_id":                _g("arch_id", ""),
            "hw_score_start":         _g("hw_score_start", 0.0),
            "hw_score_end":           _g("hw_score_end", 0.0),
            "perturbation_tries":     _g("perturbation_tries", 0),
            "perturbations_accepted": _g("perturbations_accepted", 0),
            "tc_improvements":        _g("tc_improvements", 0),
            "status":                 _g("status", ""),
        }

    return {
        "worker_idx":   worker_idx,
        "n_generated":  len(hw_archs),
        "archs":        [_arch_to_dict(a) for a in hw_archs],
        "hw_score_map": hw_score_map,
        "quality_map":  quality_map,
        "hw_logs":      [_log_to_dict(l) for l in hw_logs_raw],
    }


def _dict_to_arch(d: dict) -> ArchConfig:
    """Rekonstruksi ArchConfig dari dict hasil serialisasi worker."""
    from arch_types import (ArchConfig, FFNType, AttentionType,
                            OptimizerType, NormType, PosEncType)

    def _parse_enum(cls, val):
        try:
            return cls(val)
        except Exception:
            for m in cls:
                if m.name == val or m.value == val:
                    return m
            return list(cls)[0]

    a = ArchConfig.__new__(ArchConfig)

    a.arch_id            = d["arch_id"]
    a.arch_name          = d.get("arch_name", d["arch_id"])
    a.arch_family        = d["arch_family"]
    a.vocab_size         = d.get("vocab_size", 32000)
    a.hidden_dim         = d["hidden_dim"]
    a.num_layers         = d["num_layers"]
    a.num_heads          = d["num_heads"]
    a.num_kv_heads       = d["num_kv_heads"]
    a.head_dim           = d["head_dim"]
    a.seq_len            = d["seq_len"]
    a.window_size        = d.get("window_size", d["seq_len"])
    a.global_attn_layers = d.get("global_attn_layers", 0)
    a.batch_size         = d["batch_size"]
    a.attn_type      = _parse_enum(AttentionType, d["attn_type"])
    a.ffn_type       = _parse_enum(FFNType,        d["ffn_type"])
    a.norm_type      = _parse_enum(NormType,       d.get("norm_type", "RMSNORM"))
    a.pos_enc        = _parse_enum(PosEncType,     d.get("pos_enc", "ROPE"))
    a.optimizer_type = _parse_enum(OptimizerType,  d["optimizer_type"])
    a.ffn_multiplier         = d.get("ffn_multiplier", 4.0)
    a.num_experts            = d.get("num_experts", 1)
    a.top_k_experts          = d.get("top_k_experts", 1)
    a.expert_capacity_factor = d.get("expert_capacity_factor", 1.25)
    a.dropout                    = d.get("dropout", 0.0)
    a.tie_embeddings             = d.get("tie_embeddings", True)
    a.use_flash_attn             = d["use_flash_attn"]
    a.use_gradient_checkpointing = d.get("use_gradient_checkpointing", False)
    a.use_mixed_precision        = d.get("use_mixed_precision", True)
    a.use_torch_compile          = d.get("use_torch_compile", True)
    a.param_count             = d["param_count"]
    a.vram_total_gb           = d["vram_total_gb"]
    a.vram_usage_pct          = d["vram_usage_pct"]
    a.vram_weights_gb         = d.get("vram_weights_gb", 0.0)
    a.vram_activations_gb     = d.get("vram_activations_gb", 0.0)
    a.vram_optimizer_gb       = d.get("vram_optimizer_gb", 0.0)
    a.vram_kv_cache_gb        = d.get("vram_kv_cache_gb", 0.0)
    a.vram_fragmentation_gb   = d.get("vram_fragmentation_gb", 0.0)
    a.mfu_estimate            = d["mfu_estimate"]
    a.tokens_per_sec_estimate = d["tokens_per_sec_estimate"]
    a.ms_per_step             = d["ms_per_step"]
    a.fits_gpu                = d["fits_gpu"]
    a.fitness_score           = d["fitness_score"]
    a.bottleneck              = d["bottleneck"]
    a.hardware_score        = d.get("hardware_score", 0.0)
    a.quality_score_pct     = d.get("quality_score_pct", 0.0)
    a.sm_occupancy          = d.get("sm_occupancy", 0.0)
    a.warp_divergence_pct   = d.get("warp_divergence_pct", 0.0)
    a.compiler_speedup      = d.get("compiler_speedup", 1.0)
    a.arithmetic_intensity  = d.get("arithmetic_intensity", 0.0)
    a.flops_per_token_fwd   = d.get("flops_per_token_fwd", 0.0)
    a.flops_per_token_bwd   = d.get("flops_per_token_bwd", 0.0)

    return a


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1–2: PARALLEL GENERATE + HW-NAS
# ══════════════════════════════════════════════════════════════════════════════

def stage_parallel(cfg: PipelineConfig):
    """
    Jalankan Stage 1 (Generate) + Stage 2 (HW-NAS) secara paralel.
    Returns: all_archs, hw_scores, quality_map, hw_logs (as dicts)
    """
    console.rule("[bold cyan]  Stage 1–2 — Parallel Generate + HW-NAS  ")

    batch_sizes = _compute_batch_sizes(cfg.total_archs, cfg.n_workers)
    n_actual_workers = len(batch_sizes)

    gpu_key = None
    for k, g in GPU_DATABASE.items():
        if g.name == cfg.gpu.name:
            gpu_key = k
            break
    if gpu_key is None:
        gpu_key = list(GPU_DATABASE.keys())[0]

    if RICH:
        console.print(
            f"[cyan]  Total ARC: [bold]{cfg.total_archs:,}[/bold]  "
            f"Worker: [bold]{n_actual_workers}[/bold]  "
            f"Batch: ~[bold]{batch_sizes[0]}[/bold] ARC/worker  "
            f"GPU: [bold]{cfg.gpu.name}[/bold][/cyan]\n"
        )
        console.print(
            f"[dim]  Tiap worker: Generate → HW-NAS 7D  "
            f"(Training NAS di main process setelah ini)[/dim]\n"
        )
    else:
        print(f"  Total ARC: {cfg.total_archs}  Workers: {n_actual_workers}  "
              f"Batch: ~{batch_sizes[0]}/worker")

    t0 = time.perf_counter()

    all_archs:     List[ArchConfig]  = []
    all_hw_scores: Dict[str, float]  = {}
    all_quality:   Dict[str, float]  = {}
    all_hw_logs:   List[dict]        = []   # serialized HW logs dari semua worker

    worker_results = {}
    worker_errors  = {}

    with ProcessPoolExecutor(max_workers=n_actual_workers) as executor:
        futures = {}
        for idx, n_archs in enumerate(batch_sizes):
            future = executor.submit(
                _worker_pipeline,
                idx, n_archs, gpu_key,
                cfg.families, cfg.seed + idx, cfg.device,
                cfg.max_hw_iters, cfg.max_tr_iters, cfg.max_explore,
            )
            futures[future] = idx

        done_count = 0
        if RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                console=console,
            ) as prog:
                task = prog.add_task(
                    "[cyan]Parallel workers running...", total=n_actual_workers
                )
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        worker_results[idx] = result
                        n = result["n_generated"]
                        prog.update(task, advance=1,
                                    description=f"[cyan]Worker-{idx} done ({n} ARC)")
                    except Exception as e:
                        worker_errors[idx] = traceback.format_exc()
                        prog.update(task, advance=1,
                                    description=f"[red]Worker-{idx} FAILED: {e}")
        else:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    worker_results[idx] = result
                    done_count += 1
                    print(f"  Worker-{idx} selesai  ({result['n_generated']} ARC)  "
                          f"[{done_count}/{n_actual_workers}]")
                except Exception as e:
                    worker_errors[idx] = traceback.format_exc()
                    done_count += 1
                    print(f"  Worker-{idx} GAGAL: {e}  [{done_count}/{n_actual_workers}]")

    if worker_errors:
        console.print(f"\n[red]  ⚠ {len(worker_errors)} worker gagal:[/red]")
        for idx, err in worker_errors.items():
            console.print(f"[red]    Worker-{idx}:[/red]")
            for line in err.strip().splitlines()[-5:]:
                console.print(f"[red dim]      {line}[/red dim]")

    for idx in sorted(worker_results.keys()):
        res = worker_results[idx]
        for d in res["archs"]:
            try:
                a = _dict_to_arch(d)
                all_archs.append(a)
            except Exception as e:
                console.print(f"[yellow]  ⚠ Gagal rekonstruksi arch dari worker-{idx}: {e}[/yellow]")

        all_hw_scores.update(res["hw_score_map"])
        all_quality.update(res["quality_map"])
        all_hw_logs.extend(res.get("hw_logs", []))

    elapsed = time.perf_counter() - t0
    n_ok = len(worker_results)

    console.print(
        f"\n[bold cyan]✓ Parallel Stage 1–2 selesai — "
        f"{len(all_archs):,} ARC dari {n_ok}/{n_actual_workers} worker  "
        f"({elapsed:.1f}s)[/bold cyan]\n"
    )

    if not all_archs:
        raise RuntimeError("Semua worker gagal — tidak ada arsitektur yang berhasil dibuat.")

    return all_archs, all_hw_scores, all_quality, all_hw_logs


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: TRAINING NAS (SERIAL — MAIN PROCESS, CUDA-SAFE)
# ══════════════════════════════════════════════════════════════════════════════

def stage_training_nas(
    archs:    List[ArchConfig],
    gpu:      GPUSpec,
    hw_scores: Dict[str, float],
    *,
    max_iterations:    int = 25,
    target_pct:        float = 100.0,
    max_explore_iters: int = 30,
    seed:              int = 42,
    device:            str = "cpu",
) -> Tuple[List[ArchConfig], Dict[str, float], List, Dict[str, object]]:
    """
    Stage 3: Training NAS + RL — serial di main process (CUDA-safe).

    Returns:
        (refined_archs, train_map, train_logs, nas_results_map)
    """
    console.rule("[bold magenta]  Stage 3 — Training NAS + RL (Serial, Main Process)  ")
    console.print(
        f"[dim]  Dimensi: T1-Conv(22)+T2-Stab(22)+T3-Grad(18)+T4-GenGap(15)+T5-SampEff(13)+T6-OptComp(10)  |  "
        f"Proxy: 50 steps real training · NaN → ts=0 · RL iterate until ts≥0.75[/dim]\n"
    )

    refiner = TrainingNASRefiner(
        gpu,
        max_iterations    = max_iterations,
        target_pct        = target_pct,
        max_explore_iters = max_explore_iters,
        rng_seed          = seed,
        device            = device,
    )

    refined_archs:   List[ArchConfig] = []
    train_map:       Dict[str, float] = {}
    train_logs:      list             = []
    nas_results_map: Dict[str, object] = {}

    total = len(archs)
    t0    = time.perf_counter()
    total_nas_ms = 0.0

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[magenta]Training NAS + RL...", total=total)
            for cfg_arc in archs:
                prog.update(task, description=f"[magenta]TRAIN-NAS {cfg_arc.arch_id}…")
                hw      = hw_scores.get(cfg_arc.arch_id, 0.0)
                refined, alog = refiner.refine(cfg_arc, hw_score=hw)
                train_map[refined.arch_id]       = alog.train_score_end
                total_nas_ms                    += alog.nas_training_ms_total
                final_nas = refiner._evaluate_cached(refined)
                nas_results_map[refined.arch_id] = final_nas
                refined_archs.append(refined)
                train_logs.append(alog)
                prog.advance(task)
    else:
        for i, cfg_arc in enumerate(archs, 1):
            hw      = hw_scores.get(cfg_arc.arch_id, 0.0)
            refined, alog = refiner.refine(cfg_arc, hw_score=hw)
            train_map[refined.arch_id]       = alog.train_score_end
            total_nas_ms                    += alog.nas_training_ms_total
            final_nas = refiner._evaluate_cached(refined)
            nas_results_map[refined.arch_id] = final_nas
            refined_archs.append(refined)
            train_logs.append(alog)
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{total}] {cfg_arc.arch_id}  "
                  f"ts={alog.train_score_start:.4f}→{alog.train_score_end:.4f}"
                  f"  NAS={alog.nas_evaluations}evals/{alog.nas_training_ms_total:.0f}ms"
                  f"  NaN={alog.nas_nan_count}"
                  f"  acc={alog.perturbations_accepted}/{alog.perturbation_tries}"
                  f"  {elapsed:.1f}s  {alog.status}")

    elapsed_total = time.perf_counter() - t0

    # Print summary sama dengan pipeline.py
    print_training_adaptive_summary(train_logs, train_map, console=console)

    n_improved  = sum(1 for l in train_logs if l.perturbations_accepted > 0)
    n_optimal   = sum(1 for l in train_logs if l.train_score_end >= 0.75)
    n_nan       = sum(l.nas_nan_count for l in train_logs)
    ts_mean     = sum(train_map.values()) / max(1, len(train_map))
    total_evals = sum(l.nas_evaluations for l in train_logs)

    console.print(
        f"[magenta]✓ Training NAS complete — "
        f"{n_improved}/{total} improved  "
        f"{n_optimal}/{total} optimal(ts≥0.75)  "
        f"avg ts={ts_mean:.4f}  "
        f"total NAS evals={total_evals}  "
        f"total proxy ms={total_nas_ms:.0f}ms  "
        f"NaN detected={n_nan}  "
        f"elapsed={elapsed_total:.1f}s[/magenta]\n"
    )

    refined_archs.sort(key=lambda a: train_map.get(a.arch_id, 0.0), reverse=True)
    return refined_archs, train_map, train_logs, nas_results_map


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: BALANCED SCORING + FINAL RANKING  (output identik pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def stage_balanced_scoring(
    archs:         List[ArchConfig],
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    hw_logs:       list,
    train_logs:    list,
    gpu:           GPUSpec,
    pre_nas_rank:  Dict[str, int],
) -> Tuple[List[ArchConfig], Dict[str, float], ArchConfig]:

    console.rule("[bold green]  Stage 4 — Balanced 50/50 Scoring & Final Ranking  ")

    combined_map: Dict[str, float] = {}
    for a in archs:
        q  = quality_map.get(a.arch_id, 0.0)
        hw = hw_scores.get(a.arch_id, 0.0)
        tr = train_scores.get(a.arch_id, 0.0)
        combined_map[a.arch_id] = compute_combined_score_balanced(q, hw, tr)

    archs_sorted = sorted(
        archs,
        key=lambda a: combined_map.get(a.arch_id, 0.0),
        reverse=True,
    )

    # Build log maps (dicts atau obyek — keduanya didukung oleh _print_balanced_ranking_table)
    hw_log_map    = {}
    for l in hw_logs:
        aid = l.get("arch_id") if isinstance(l, dict) else getattr(l, "arch_id", None)
        if aid:
            hw_log_map[aid] = l
    train_log_map = {}
    for l in train_logs:
        aid = l.get("arch_id") if isinstance(l, dict) else getattr(l, "arch_id", None)
        if aid:
            train_log_map[aid] = l

    # Print ranking table dengan pre/post NAS arrows
    _print_balanced_ranking_table(
        archs_sorted, quality_map, hw_scores, train_scores,
        combined_map, hw_log_map, train_log_map, pre_nas_rank, console=console,
    )

    # Pilih best
    best = None
    MIN_COMBINED_THRESHOLD = 0.75

    tier1 = [
        a for a in archs_sorted
        if a.fits_gpu and combined_map.get(a.arch_id, 0.0) >= MIN_COMBINED_THRESHOLD
    ]
    tier2 = [a for a in archs_sorted if a.fits_gpu]

    if tier1:
        best = tier1[0]
        tier_label = "combined≥0.75"
    elif tier2:
        best = tier2[0]
        tier_label = "fits_gpu only"
    else:
        best = archs_sorted[0] if archs_sorted else None
        tier_label = "⚠ fallback"

    if best is None:
        console.print("[red]  ❌ Tidak ada arc valid![/red]")
        return archs_sorted, combined_map, None

    pre_rank = pre_nas_rank.get(best.arch_id, "?")
    console.print(
        f"[bold green]★ Top Pick: {best.arch_id}  "
        f"[dim](pre-NAS rank: #{pre_rank} → post-NAS: #1)[/dim]  "
        f"combined={combined_map.get(best.arch_id, 0):.5f}  "
        f"hw={hw_scores.get(best.arch_id, 0):.4f}  "
        f"train={train_scores.get(best.arch_id, 0):.4f}  "
        f"quality={quality_map.get(best.arch_id, 0):.1f}%  "
        f"[{tier_label}][/bold green]\n"
    )

    return archs_sorted, combined_map, best


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5: PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def stage_profile_top(best: ArchConfig, cfg: PipelineConfig) -> dict:
    console.rule("[bold yellow]  Stage 5 — Profiling Best Architecture  ")
    profiler = TorchProfiler(best, cfg.gpu)
    if cfg.run_profiling and TORCH and cfg.device == "cuda":
        console.print(f"[yellow]  Running torch.profiler on {best.arch_id}...[/yellow]")
        result = profiler.run()
    else:
        console.print(f"[dim]  Analytical profiling (no CUDA): {best.arch_id}[/dim]")
        result = profiler._analytical_fallback()
    console.print(f"[green]  ✓ Profiling done: MFU={result.get('est_mfu', best.mfu_estimate):.4f}[/green]\n")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6: JSON EXPORT (identik pipeline.py — export_results_json_onlytype)
# ══════════════════════════════════════════════════════════════════════════════

def stage_export(
    archs_sorted:    List[ArchConfig],
    quality_map:     Dict[str, float],
    hw_scores:       Dict[str, float],
    train_scores:    Dict[str, float],
    combined_map:    Dict[str, float],
    best:            ArchConfig,
    gpu:             GPUSpec,
    hw_logs:         list,
    train_logs:      list,
    nas_results_map: Dict[str, object],
    pre_nas_rank:    Dict[str, int],
    cfg:             PipelineConfig,
) -> str:
    """
    Export komprehensif identik dengan pipeline.py export_results_json_onlytype:
      • pipeline_summary, scoring_schema, nas_journey
      • best_arch_detail (sub-scores T1–T6 dari cached NASResult)
      • ranked_architectures lengkap
    """
    console.rule("[bold dim]  Stage 6 — Auto Save Results  ")

    def _clean(val) -> str:
        if val is None: return "—"
        if hasattr(val, "value"): return str(val.value)
        s = str(val)
        return s.split(".")[-1] if "." in s else s

    def _get(obj, attr, default=None):
        if obj is None: return default
        if isinstance(obj, dict): return obj.get(attr, default)
        return getattr(obj, attr, default)

    # ── Pipeline summary ──────────────────────────────────────────────────────
    all_combined = [combined_map.get(a.arch_id, 0.0) for a in archs_sorted]
    all_hw       = [hw_scores.get(a.arch_id, 0.0)    for a in archs_sorted]
    all_tr       = [train_scores.get(a.arch_id, 0.0) for a in archs_sorted]
    all_q        = [quality_map.get(a.arch_id, 0.0)  for a in archs_sorted]
    n            = max(1, len(archs_sorted))

    fits_gpu_archs = [a for a in archs_sorted if a.fits_gpu]
    families_seen  = list(dict.fromkeys(a.arch_family for a in archs_sorted))

    pipeline_summary = {
        "total_architectures": len(archs_sorted),
        "fits_gpu_count":      len(fits_gpu_archs),
        "families_evaluated":  families_seen,
        "best_arch_id":        best.arch_id if best else "—",
        "best_arch_name":      best.arch_name if best else "—",
        "best_combined_score": round(combined_map.get(best.arch_id, 0.0), 5) if best else 0.0,
        "best_hardware_score": round(hw_scores.get(best.arch_id, 0.0), 4)    if best else 0.0,
        "best_training_score": round(train_scores.get(best.arch_id, 0.0), 4) if best else 0.0,
        "best_quality_pct":    round(quality_map.get(best.arch_id, 0.0), 2)  if best else 0.0,
        "avg_combined_score":  round(sum(all_combined) / n, 5),
        "avg_hardware_score":  round(sum(all_hw) / n, 4),
        "avg_training_score":  round(sum(all_tr) / n, 4),
        "avg_quality_pct":     round(sum(all_q) / n, 2),
        "top_family":          (archs_sorted[0].arch_family if archs_sorted else "—"),
        "n_workers":           cfg.n_workers,
        "parallel":            True,
    }

    # ── Scoring schema ────────────────────────────────────────────────────────
    scoring_schema = {
        "formula":            "combined = (0.50 × hw_score + 0.50 × train_score) × quality_gate",
        "quality_gate_range": "0.80–1.00",
        "quality_gate_formula": "0.80 + 0.20 × max(0, (quality_pct - 70) / 30)",
        "hardware_weight":    0.50,
        "training_weight":    0.50,
        "hardware_dimensions": {
            "H1_MFU_Utilization":   "25 pts — seberapa mendekati peak GPU throughput",
            "H2_Throughput_Eff":    "20 pts — tokens/sec vs GPU ceiling",
            "H3_VRAM_Utilization":  "15 pts — penggunaan budget VRAM efisien",
            "H4_TC_Alignment":      "15 pts — Tensor Core tile alignment per GPU gen",
            "H5_SM_Occupancy":      "10 pts — Streaming Multiprocessor occupancy",
            "H6_Compute_Boundness": "10 pts — arithmetic intensity vs ridge point",
            "H7_FA_Tile_Feasibility": "5 pts — FlashAttention SMEM constraint",
        },
        "training_dimensions": {
            "T1_Convergence_Rate":   "22 pts — loss turun seberapa cepat",
            "T2_Training_Stability": "22 pts — loss variance, NaN detection",
            "T3_Gradient_Health":    "18 pts — grad norm, vanishing/exploding",
            "T4_Generalization_Gap": "15 pts — train vs val loss di proxy",
            "T5_Sample_Efficiency":  "13 pts — Chinchilla, noise, tied embed",
            "T6_Optimizer_Compat":   "10 pts — optimizer vs depth compat",
        },
        "vs_old_system": "LAMA: 35%×quality + 65%×fitness (bias GPU, training diabaikan)",
    }

    # ── NAS journey per-arc ───────────────────────────────────────────────────
    hw_log_map    = {_get(l, "arch_id"): l for l in hw_logs    if _get(l, "arch_id")}
    train_log_map = {_get(l, "arch_id"): l for l in train_logs if _get(l, "arch_id")}

    nas_journey = []
    for post_rank, cfg_arc in enumerate(archs_sorted, 1):
        aid      = cfg_arc.arch_id
        pre_rank = pre_nas_rank.get(aid, post_rank)
        hw_log   = hw_log_map.get(aid)
        tr_log   = train_log_map.get(aid)

        rank_change = pre_rank - post_rank
        if rank_change > 0:
            rank_direction = "↑ naik"
        elif rank_change < 0:
            rank_direction = "↓ turun"
        else:
            rank_direction = "= sama"

        entry = {
            "arch_id":        aid,
            "pre_nas_rank":   pre_rank,
            "post_nas_rank":  post_rank,
            "rank_change":    rank_change,
            "rank_direction": rank_direction,
            "combined_score": round(combined_map.get(aid, 0.0), 5),
            "hardware_score": round(hw_scores.get(aid, 0.0), 4),
            "training_score": round(train_scores.get(aid, 0.0), 4),
            "quality_pct":    round(quality_map.get(aid, 0.0), 2),
        }
        if hw_log:
            entry["hw_rl"] = {
                "hw_score_start":        round(_get(hw_log, "hw_score_start", 0.0), 4),
                "hw_score_end":          round(_get(hw_log, "hw_score_end", 0.0), 4),
                "hw_score_delta":        round(_get(hw_log, "hw_score_end", 0.0) - _get(hw_log, "hw_score_start", 0.0), 4),
                "perturbation_tries":    _get(hw_log, "perturbation_tries", 0),
                "perturbations_accepted":_get(hw_log, "perturbations_accepted", 0),
                "tc_improvements":       _get(hw_log, "tc_improvements", 0),
                "status":                _get(hw_log, "status", ""),
            }
        if tr_log:
            entry["train_rl"] = {
                "train_score_start":     round(_get(tr_log, "train_score_start", 0.0), 4),
                "train_score_end":       round(_get(tr_log, "train_score_end", 0.0), 4),
                "train_score_delta":     round(_get(tr_log, "train_score_end", 0.0) - _get(tr_log, "train_score_start", 0.0), 4),
                "nas_evaluations":       _get(tr_log, "nas_evaluations", 0),
                "nas_training_ms":       round(_get(tr_log, "nas_training_ms_total", 0.0), 1),
                "nan_detected_count":    _get(tr_log, "nas_nan_count", 0),
                "perturbation_tries":    _get(tr_log, "perturbation_tries", 0),
                "perturbations_accepted":_get(tr_log, "perturbations_accepted", 0),
                "convergence_improvements": _get(tr_log, "convergence_improvements", 0),
                "stability_improvements":   _get(tr_log, "stability_improvements", 0),
                "status":                _get(tr_log, "status", ""),
            }
        nas_journey.append(entry)

    # ── Best arch detail ──────────────────────────────────────────────────────
    best_detail: dict = {}
    if best:
        aid    = best.arch_id
        nas_res = nas_results_map.get(aid)
        q       = quality_map.get(aid, 0.0)
        hw      = hw_scores.get(aid, 0.0)
        tr      = train_scores.get(aid, 0.0)
        comb    = combined_map.get(aid, 0.0)
        gate_m  = 0.80 + 0.20 * max(0.0, min(1.0, (q - 70.0) / 30.0))

        best_detail = {
            "arch_id":    aid,
            "arch_name":  best.arch_name,
            "arch_family":best.arch_family,
            "scores": {
                "combined_score":    round(comb, 5),
                "hardware_score":    round(hw, 4),
                "training_score":    round(tr, 4),
                "quality_pct":       round(q, 2),
                "quality_gate_mult": round(gate_m, 4),
            },
            "architecture_params": {
                "param_count_M":  round(best.param_count / 1e6, 2),
                "num_layers":     best.num_layers,
                "hidden_dim":     best.hidden_dim,
                "num_heads":      best.num_heads,
                "num_kv_heads":   best.num_kv_heads,
                "head_dim":       best.head_dim,
                "ffn_multiplier": best.ffn_multiplier,
                "seq_len":        best.seq_len,
                "batch_size":     best.batch_size,
                "ffn_type":       _clean(best.ffn_type),
                "attn_type":      _clean(best.attn_type),
                "norm_type":      _clean(best.norm_type),
                "optimizer_type": _clean(best.optimizer_type),
                "pos_enc":        _clean(best.pos_enc),
            },
            "hardware_profile": {
                "vram_total_gb":     round(best.vram_total_gb, 3),
                "vram_usage_pct":    round(best.vram_usage_pct, 2),
                "mfu_estimate":      round(best.mfu_estimate, 4),
                "tokens_per_sec":    best.tokens_per_sec_estimate,
                "ms_per_step":       round(best.ms_per_step, 2),
                "bottleneck":        best.bottleneck,
                "sm_occupancy":      round(best.sm_occupancy, 4),
                "arithmetic_intensity": round(best.arithmetic_intensity, 2),
                "fits_gpu":          best.fits_gpu,
            },
            "training_flags": {
                "use_flash_attn":             best.use_flash_attn,
                "use_mixed_precision":        best.use_mixed_precision,
                "use_gradient_checkpointing": best.use_gradient_checkpointing,
                "use_torch_compile":          best.use_torch_compile,
                "tie_embeddings":             best.tie_embeddings,
                "dropout":                    best.dropout,
            },
        }

        if nas_res is not None:
            pr = getattr(nas_res, "proxy_result", None)
            best_detail["training_sub_scores"] = {
                "pts_t1_convergence":  round(_get(nas_res, "pts_t1", 0.0), 2),
                "pts_t2_stability":    round(_get(nas_res, "pts_t2", 0.0), 2),
                "pts_t3_gradient":     round(_get(nas_res, "pts_t3", 0.0), 2),
                "pts_t4_gen_gap":      round(_get(nas_res, "pts_t4", 0.0), 2),
                "pts_t5_sample_eff":   round(_get(nas_res, "pts_t5", 0.0), 2),
                "pts_t6_opt_compat":   round(_get(nas_res, "pts_t6", 0.0), 2),
                "total_pts":           round(_get(nas_res, "total_pts", 0.0), 2),
                "grade":               _get(nas_res, "grade", "—"),
                "regime":              _get(nas_res, "regime", "—"),
                "gradient_risk":       str(_get(nas_res, "gradient_risk", "—")),
                "lr_sensitivity":      str(_get(nas_res, "lr_sensitivity", "—")),
            }
            if pr is not None:
                best_detail["proxy_training"] = {
                    "loss_initial":       round(_get(pr, "loss_initial", 0.0), 6),
                    "loss_final":         round(_get(pr, "loss_final", 0.0), 6),
                    "generalization_gap": round(_get(pr, "generalization_gap", 0.0), 6),
                    "nan_detected":       _get(pr, "nan_detected", False),
                }

    # ── Ranked architectures ──────────────────────────────────────────────────
    from dataclasses import asdict as _asdict

    ranked_architectures = []
    for rank, a in enumerate(archs_sorted, 1):
        aid = a.arch_id
        try:
            arch_dict = _asdict(a)
        except Exception:
            arch_dict = {k: getattr(a, k, None) for k in a.__dataclass_fields__}

        for key in ("attn_type", "ffn_type", "optimizer_type", "norm_type", "pos_enc"):
            if key in arch_dict:
                arch_dict[key] = _clean(arch_dict[key])

        arch_dict.update({
            "rank":              rank,
            "pre_nas_rank":      pre_nas_rank.get(aid, rank),
            "quality_score_pct": round(quality_map.get(aid, 0.0), 2),
            "hardware_score":    round(hw_scores.get(aid, 0.0), 4),
            "training_score":    round(train_scores.get(aid, 0.0), 4),
            "combined_score":    round(combined_map.get(aid, 0.0), 5),
        })
        ranked_architectures.append(arch_dict)

    # ── Assemble ──────────────────────────────────────────────────────────────
    result = {
        "export_type":          "only_type",
        "generated_at":         datetime.datetime.now().isoformat(),
        "gpu":                  gpu.name,
        "pipeline_mode":        "parallel_nas",
        "scoring_system":       "balanced_50_50",
        "pipeline_summary":     pipeline_summary,
        "scoring_schema":       scoring_schema,
        "nas_journey":          nas_journey,
        "best_arch_detail":     best_detail,
        "ranked_architectures": ranked_architectures,
    }

    filepath = cfg.output_file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[green]✓ Results saved → {filepath}[/green]")
        console.print(f"[dim]  {len(archs_sorted)} architectures  |  "
                      f"Best: {best.arch_id}  |  "
                      f"combined={combined_map.get(best.arch_id, 0):.5f}[/dim]\n")
    except Exception as e:
        console.print(f"[yellow]  ⚠ Save failed: {e}[/yellow]")

    return filepath


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    # ── Collect config ────────────────────────────────────────────────────────
    cfg = collect_config()

    # ── Stage 1–2: Parallel (Generate + HW-NAS) ──────────────────────────────
    all_archs, hw_scores, quality_map, hw_logs = stage_parallel(cfg)

    if not all_archs:
        console.print("[red]  Tidak ada arsitektur berhasil digenerate.[/red]")
        return

    # Pre-NAS rank: berdasarkan urutan fitness_score sebelum Training NAS
    # (post HW-NAS, pre Training-NAS — closest equivalent di parallel mode)
    pre_sorted    = sorted(all_archs, key=lambda a: a.fitness_score, reverse=True)
    pre_nas_rank  = {a.arch_id: i for i, a in enumerate(pre_sorted, 1)}

    # ── Stage 3: Training NAS (Serial, Main Process, CUDA-safe) ──────────────
    all_archs, train_scores, train_logs, nas_results_map = stage_training_nas(
        all_archs, cfg.gpu, hw_scores,
        max_iterations    = cfg.max_tr_iters,
        max_explore_iters = cfg.max_explore,
        seed              = cfg.seed,
        device            = cfg.device,
    )

    # ── Stage 4: Balanced scoring + ranking identik pipeline.py ──────────────
    archs_sorted, combined_map, best = stage_balanced_scoring(
        all_archs, quality_map, hw_scores, train_scores,
        hw_logs, train_logs, cfg.gpu, pre_nas_rank,
    )

    if best is None:
        console.print("[red]  Pipeline berhenti: tidak ada arc valid.[/red]")
        return

    # ── Detail NAS-RL logs (opsional) ─────────────────────────────────────────
    log_prompt = input("\n▶  Show detailed NAS-RL logs? [y/N]: ").strip().lower()
    if log_prompt in ("y", "yes"):
        console.rule("[bold]  Hardware NAS-RL Detail Logs  ")
        for alog in hw_logs[:5]:
            # hw_logs adalah list of dict dari worker
            console.print(
                f"  ─── HW Log: {alog.get('arch_id', '?')} ───  "
                f"hw={alog.get('hw_score_start', 0):.4f}→{alog.get('hw_score_end', 0):.4f}  "
                f"acc={alog.get('perturbations_accepted', 0)}/{alog.get('perturbation_tries', 0)}"
            )

        console.rule("[bold]  Training NAS-RL Detail Logs  ")
        for alog in train_logs[:5]:
            print_training_adaptive_log(alog, console=console)

    # ── Stage 5: Profile best arch ────────────────────────────────────────────
    pr = stage_profile_top(best, cfg)

    # ── Stage 6 (Final Recommendation) — identik pipeline.py ─────────────────
    console.rule("[bold]  Final Recommendation  ")
    stage_final_recommendation(
        best, quality_map, hw_scores, train_scores, combined_map,
        cfg.gpu, hw_logs, train_logs, archs_sorted,
        nas_results_map, pre_nas_rank,
    )

    # ── Stage 7: Export JSON komprehensif identik pipeline.py ─────────────────
    export_prompt = input("▶  Export results to JSON? [Y/n]: ").strip().lower()
    if export_prompt not in ("n", "no"):
        stage_export(
            archs_sorted, quality_map, hw_scores, train_scores, combined_map,
            best, cfg.gpu, hw_logs, train_logs, nas_results_map, pre_nas_rank, cfg,
        )

    elapsed_total = time.perf_counter() - t_start
    console.print(
        f"[dim]  Total pipeline time: {elapsed_total:.1f}s  |  "
        f"{len(archs_sorted):,} archs evaluated  |  "
        f"{cfg.n_workers} parallel workers[/dim]\n"
    )


if __name__ == "__main__":
    main()
