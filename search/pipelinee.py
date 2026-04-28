"""
pipeline.py — Dual NAS Pipeline (Hardware NAS + Training NAS)
═══════════════════════════════════════════════════════════════════════════════

Pipeline stages:
  Stage 1  : Banner
  Stage 2  : GPU selection
  Stage 3  : AI type (family) selection
  Stage 4  : Run options
  Stage 5  : Generate raw architectures
  Stage 6  : Pre-refinement summary
  Stage 7A : Hardware NAS + RL  (hardware_refine.py)
             → 7 dimensi GPU: MFU, Throughput, VRAM, TC-Align,
               SM-Occ, Compute-Bound, FA-Tile
  Stage 7B : Training NAS + RL  (train_refine.py)
             → 6 dimensi training: Convergence, Stability, Gradient,
               Generalization, SampleEff, OptimizerCompat
             → WAJIB proxy training real PyTorch per-arc
             → RL iterate hingga training_score optimal
  Stage 7C : Balanced scoring 50/50 Hardware + Training
  Stage 8  : Post-refinement ranking
  Stage 9  : Detailed profiling (top N)
  Stage 10 : Final ranking + JSON export + recommendation

Sistem Penilaian Baru (50/50 Seimbang):
  combined = 50% × hardware_score + 50% × training_score
  × quality_gate(80–100%) dari ArcQualityScorer

  LAMA (bias GPU):  combined = 35% quality + 65% fitness
  BARU (seimbang): combined = 50% hardware + 50% training

Catatan:
  hardware_score  = 7 dimensi dari GPUSpec real (hardware_refine.py)
  training_score  = 6 dimensi dari proxy training nyata (train_refine.py)
  quality_pct     = konsistensi internal formula (refiner.py)
  fitness_score   = estimasi training formula (generator.py) — hanya info
"""

import sys
import os
import json
import time
from typing import Dict, List, Optional, Tuple

# ── Path setup — tambahkan direktori multi-type ke sys.path ──────────────────
_BASE_DIR = "/content/architecture-search1/search/multi-type"
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

# ── Internal module imports ───────────────────────────────────────────────────
from hardware   import GPU_DATABASE, GPUSpec
from generator  import ArchitectureGenerator
from profiler   import TorchProfiler, TORCH, DEVICE
from metrics    import MetricsReport
from arch_types import ArchConfig

# Scorer & base refiner
from refiner import (
    ArcQualityScorer,
    ArcRefiner,
    print_score_report,
    compute_combined_score_balanced,
    compute_combined_score_triple,
    select_best_arch_balanced,
    rank_archs_balanced,
)

# Hardware NAS engine
from hardware_refine import (
    HardwareNASRefiner,
    HardwareNASEvaluator,
    HardwareAdaptiveLog,
    print_hardware_nas_result,
    print_hardware_adaptive_summary,
    print_hardware_adaptive_log,
    compute_hardware_score,
)

# Training NAS engine
from train_refine import (
    TrainingNASRefiner,
    TrainingDynamicsEvaluator,
    ProxyTrainer,
    TrainingAdaptiveLog,
    print_training_nas_result,
    print_training_adaptive_summary,
    print_training_adaptive_log,
)

# Combination NAS engine
from combination_nas import (
    CombinationSpec,
    CombinationNASRefiner,
    CombinationNASEvaluator,
    ask_combination_type,
    run_combination_pipeline,
    print_combination_result,
    print_combination_summary,
)

# Combination RL Refiner (dedicated RL untuk mode combination)
try:
    from combination_refiner import (
        CombinationRefiner,
        CombinationRLConfig,
        CombinationRLLog,
        print_rl_log      as print_combination_rl_log,
        print_rl_summary  as print_combination_rl_summary,
    )
    _COMBO_REFINER_AVAILABLE = True
except ImportError:
    _COMBO_REFINER_AVAILABLE = False

# UI
from ui import (
    RICH, console,
    display_banner, select_gpu, ask_ai_type, ask_run_options,
    ask_ai_type_local, _ALL_FAMILIES_ORDERED,
    ask_combination_mode,
    print_arch_summary, print_detailed_report, print_ranking,
)

# ── Optional Rich progress bar ────────────────────────────────────────────────
if RICH:
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, MofNCompleteColumn,
    )
    from rich.table  import Table
    from rich.panel  import Panel
    from rich.text   import Text


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGE 5: GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def stage_generate(gpu: GPUSpec, families, n_per_family: int, seed: int):
    """
    Stage 5: Generate architecture candidates.
    Returns list ArchConfig sorted by fitness_score (descending).
    """
    gen          = ArchitectureGenerator(gpu, rng_seed=seed)
    all_families = list(ArchitectureGenerator.FAMILY_PROFILES.keys())
    selected     = ([f for f in families if f in all_families]
                    if families else all_families)

    console.rule("[bold cyan]  Stage 5 — Generating Architectures  ")

    total = len(selected) * n_per_family

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task  = prog.add_task("[cyan]Generating...", total=total)
            archs = []
            for fam in selected:
                for _ in range(n_per_family):
                    cfg = gen.generate_one(fam)
                    archs.append(cfg)
                    prog.update(task, advance=1,
                                description=f"[cyan]{fam[:20]} {cfg.arch_id}")
    else:
        archs = []
        for fam in selected:
            for _ in range(n_per_family):
                archs.append(gen.generate_one(fam))

    archs.sort(key=lambda x: x.fitness_score, reverse=True)
    console.print(f"\n[green]✓ Generated {len(archs)} architectures "
                  f"across {len(selected)} families[/green]")
    return archs


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGE 7A: HARDWARE NAS + RL
# ══════════════════════════════════════════════════════════════════════════════

def stage_hardware_nas(
    archs:             List[ArchConfig],
    gpu:               GPUSpec,
    *,
    max_iterations:    int   = 25,
    target_pct:        float = 100.0,
    max_explore_iters: int   = 30,
    seed:              Optional[int] = None,
) -> Tuple[List[ArchConfig], List[HardwareAdaptiveLog], Dict[str, float], Dict[str, float]]:
    """
    Stage 7A: Hardware NAS + RL Refinement.

    Evaluasi 7 dimensi hardware dari GPUSpec:
      H1 MFU Utilization (25 pts)    — seberapa mendekati peak GPU throughput
      H2 Throughput Efficiency (20)  — tokens/sec vs GPU ceiling
      H3 VRAM Utilization (15)       — penggunaan budget VRAM efisien
      H4 TC Alignment (15)           — Tensor Core tile alignment per GPU gen
      H5 SM Occupancy (10)           — Streaming Multiprocessor occupancy
      H6 Compute Boundness (10)      — arithmetic intensity vs ridge point
      H7 FA Tile Feasibility (5)     — FlashAttention SMEM constraint

    RL Actions: ALIGN_HIDDEN, ALIGN_HEAD_DIM, ALIGN_FFN, INCR/DECR_BATCH,
                ENABLE_FA, ENABLE_COMPILE, DISABLE_GC, OPT_EFFICIENT

    Returns:
        (refined_archs, hw_logs, hw_score_map, quality_map)
    """
    console.rule("[bold blue]  Stage 7A — Hardware NAS + RL  ")
    console.print(
        f"[dim]GPU: {gpu.name}  |  "
        f"Dimensi: H1-MFU(25)+H2-Thru(20)+H3-VRAM(15)+H4-TC(15)+H5-SM(10)+H6-Comp(10)+H7-FA(5)  |  "
        f"RL Actions: ALIGN_HIDDEN/HEAD/FFN, BATCH, FA, COMPILE, GC, OPT[/dim]\n"
    )

    refiner  = HardwareNASRefiner(
        gpu,
        max_iterations    = max_iterations,
        target_pct        = target_pct,
        max_explore_iters = max_explore_iters,
        rng_seed          = seed,
    )
    scorer = ArcQualityScorer(gpu)

    refined_archs: List[ArchConfig]       = []
    hw_logs:       List[HardwareAdaptiveLog] = []
    hw_score_map:  Dict[str, float]       = {}
    quality_map:   Dict[str, float]       = {}

    total = len(archs)
    t0    = time.perf_counter()

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[blue]Hardware NAS + RL...", total=total)
            for cfg in archs:
                prog.update(task, description=f"[blue]HW-NAS {cfg.arch_id}…")
                refined, alog = refiner.refine(cfg)

                # Quality score
                q = scorer.score(refined).pct
                quality_map[refined.arch_id]  = q
                hw_score_map[refined.arch_id] = alog.hw_score_end

                refined_archs.append(refined)
                hw_logs.append(alog)
                prog.advance(task)
    else:
        for i, cfg in enumerate(archs, 1):
            refined, alog = refiner.refine(cfg)
            q = scorer.score(refined).pct
            quality_map[refined.arch_id]  = q
            hw_score_map[refined.arch_id] = alog.hw_score_end
            refined_archs.append(refined)
            hw_logs.append(alog)
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{total}] {cfg.arch_id}  "
                  f"hw={alog.hw_score_start:.4f}→{alog.hw_score_end:.4f}"
                  f"  q={q:.1f}%"
                  f"  acc={alog.perturbations_accepted}/{alog.perturbation_tries}"
                  f"  {elapsed:.1f}s  {alog.status}")

    elapsed_total = time.perf_counter() - t0

    # Print summary
    print_hardware_adaptive_summary(hw_logs, hw_score_map, console=console)

    n_improved = sum(1 for l in hw_logs if l.perturbations_accepted > 0)
    hw_mean    = sum(hw_score_map.values()) / max(1, len(hw_score_map))
    console.print(
        f"[blue]✓ Hardware NAS complete — {n_improved}/{total} improved  "
        f"avg hw_score={hw_mean:.4f}  elapsed={elapsed_total:.1f}s[/blue]\n"
    )

    # Sort by hw_score descending
    refined_archs.sort(
        key=lambda a: hw_score_map.get(a.arch_id, 0.0),
        reverse=True,
    )
    return refined_archs, hw_logs, hw_score_map, quality_map


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGE 7B: TRAINING NAS + RL
# ══════════════════════════════════════════════════════════════════════════════

def stage_training_nas(
    archs:             List[ArchConfig],
    gpu:               GPUSpec,
    hw_scores:         Dict[str, float],
    *,
    max_iterations:    int   = 25,
    target_pct:        float = 100.0,
    max_explore_iters: int   = 30,
    seed:              Optional[int] = None,
    device:            str   = "cpu",
) -> Tuple[List[ArchConfig], List[TrainingAdaptiveLog], Dict[str, float]]:
    """
    Stage 7B: Training NAS + RL dengan real PyTorch proxy.

    WAJIB mengevaluasi setiap arc melalui proxy training:
      T1  Convergence Rate      (22 pts) — loss turun seberapa cepat
      T2  Training Stability    (22 pts) — loss variance, NaN detection
      T3  Gradient Health       (18 pts) — grad norm, vanishing/exploding
      T4  Generalization Gap    (15 pts) — train vs val loss di proxy
      T5  Sample Efficiency     (13 pts) — Chinchilla, noise, tied embed
      T6  Optimizer Compatibility(10 pts)— optimizer vs depth compat

    RL Actions:
      FIX_DEPTH_WIDTH, SWITCH_RMSNORM, INCR_BATCH_TRAIN, SWITCH_OPT_STABLE,
      ENABLE_MIXED_PREC, TIE_EMBEDDINGS, ADJUST_FFN_MULT, DISABLE_DROPOUT,
      FIX_OPTIMIZER_DEPTH

    RL terus iterate sampai training_score ≥ 0.75 atau max_explore tercapai.

    Returns:
        (refined_archs, train_logs, training_score_map, nas_results_map)
    """
    console.rule("[bold magenta]  Stage 7B — Training NAS + RL (Real PyTorch Proxy)  ")
    console.print(
        f"[dim]Dimensi: T1-Conv(22)+T2-Stab(22)+T3-Grad(18)+T4-GenGap(15)+T5-SampEff(13)+T6-OptComp(10)  |  "
        f"Proxy: {50} steps real training · NaN → ts=0 · RL iterate until ts≥0.75[/dim]\n"
    )

    refiner = TrainingNASRefiner(
        gpu,
        max_iterations     = max_iterations,
        target_pct         = target_pct,
        max_explore_iters  = max_explore_iters,
        rng_seed           = seed,
        device             = device,
    )

    refined_archs:  List[ArchConfig]          = []
    train_logs:     List[TrainingAdaptiveLog]  = []
    train_map:      Dict[str, float]           = {}
    nas_results_map: Dict[str, "TrainingNASResult"] = {}   # ← cache hasil NAS

    total = len(archs)
    t0    = time.perf_counter()
    total_nas_ms = 0.0

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[magenta]Training NAS + RL...", total=total)
            for cfg in archs:
                prog.update(task,
                            description=f"[magenta]TRAIN-NAS {cfg.arch_id}…")
                hw = hw_scores.get(cfg.arch_id, 0.0)
                refined, alog = refiner.refine(cfg, hw_score=hw)

                train_map[refined.arch_id]       = alog.train_score_end
                total_nas_ms                    += alog.nas_training_ms_total
                # Simpan NASResult final untuk dipakai stage_final_recommendation
                final_nas = refiner._evaluate_cached(refined)
                nas_results_map[refined.arch_id] = final_nas
                refined_archs.append(refined)
                train_logs.append(alog)
                prog.advance(task)
    else:
        for i, cfg in enumerate(archs, 1):
            hw      = hw_scores.get(cfg.arch_id, 0.0)
            refined, alog = refiner.refine(cfg, hw_score=hw)
            train_map[refined.arch_id]       = alog.train_score_end
            total_nas_ms                    += alog.nas_training_ms_total
            final_nas = refiner._evaluate_cached(refined)
            nas_results_map[refined.arch_id] = final_nas
            refined_archs.append(refined)
            train_logs.append(alog)
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{total}] {cfg.arch_id}  "
                  f"ts={alog.train_score_start:.4f}→{alog.train_score_end:.4f}"
                  f"  NAS={alog.nas_evaluations}evals/{alog.nas_training_ms_total:.0f}ms"
                  f"  NaN={alog.nas_nan_count}"
                  f"  acc={alog.perturbations_accepted}/{alog.perturbation_tries}"
                  f"  {elapsed:.1f}s  {alog.status}")

    elapsed_total = time.perf_counter() - t0

    # Print summary
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

    # Sort by training_score descending
    refined_archs.sort(
        key=lambda a: train_map.get(a.arch_id, 0.0),
        reverse=True,
    )
    return refined_archs, train_logs, train_map, nas_results_map


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGE 7C: BALANCED SCORING 50/50
# ══════════════════════════════════════════════════════════════════════════════

def stage_balanced_scoring(
    archs:          List[ArchConfig],
    quality_map:    Dict[str, float],
    hw_scores:      Dict[str, float],
    train_scores:   Dict[str, float],
    hw_logs:        List[HardwareAdaptiveLog],
    train_logs:     List[TrainingAdaptiveLog],
    gpu:            GPUSpec,
    pre_nas_archs:  List[ArchConfig],         # ← list asli sebelum NAS untuk ranking
) -> Tuple[List[ArchConfig], Dict[str, float], ArchConfig]:
    """
    Stage 7C: Compute balanced combined score dan ranking final.

    combined = quality_gate × (50% hw_score + 50% train_score)
    quality_gate = 0.80 + 0.20 × (quality - 70%) / 30%

    Returns:
        (sorted_archs, combined_map, best_arch)
    """
    console.rule("[bold green]  Stage 7C — Balanced 50/50 Scoring & Final Ranking  ")

    # Pre-NAS rank map: arch_id → rank (1-based) berdasarkan fitness_score
    pre_nas_rank: Dict[str, int] = {
        a.arch_id: i for i, a in enumerate(pre_nas_archs, 1)
    }

    combined_map: Dict[str, float] = {}
    hw_log_map    = {l.arch_id: l for l in hw_logs}
    train_log_map = {l.arch_id: l for l in train_logs}

    for cfg in archs:
        q  = quality_map.get(cfg.arch_id, 0.0)
        hw = hw_scores.get(cfg.arch_id, 0.0)
        tr = train_scores.get(cfg.arch_id, 0.0)
        combined_map[cfg.arch_id] = compute_combined_score_balanced(q, hw, tr)

    # Sort by combined descending
    archs_sorted = sorted(
        archs,
        key=lambda a: combined_map.get(a.arch_id, 0.0),
        reverse=True,
    )

    # Print balanced ranking table (dengan pre-NAS rank comparison)
    _print_balanced_ranking_table(
        archs_sorted, quality_map, hw_scores, train_scores,
        combined_map, hw_log_map, train_log_map, pre_nas_rank, console=console,
    )

    best = next((a for a in archs_sorted if a.fits_gpu), None)
    if best is None:
        best = archs_sorted[0] if archs_sorted else archs[0]

    pre_rank = pre_nas_rank.get(best.arch_id, "?")
    post_rank = 1
    console.print(
        f"[bold green]★ Top Pick: {best.arch_id}  "
        f"[dim](pre-NAS rank: #{pre_rank} → post-NAS: #1)[/dim]  "
        f"combined={combined_map.get(best.arch_id, 0):.5f}  "
        f"hw={hw_scores.get(best.arch_id, 0):.4f}  "
        f"train={train_scores.get(best.arch_id, 0):.4f}  "
        f"quality={quality_map.get(best.arch_id, 0):.1f}%[/bold green]\n"
    )

    return archs_sorted, combined_map, best


def _print_balanced_ranking_table(
    archs:           List[ArchConfig],
    quality_map:     Dict[str, float],
    hw_scores:       Dict[str, float],
    train_scores:    Dict[str, float],
    combined_map:    Dict[str, float],
    hw_log_map:      Dict[str, HardwareAdaptiveLog],
    train_log_map:   Dict[str, TrainingAdaptiveLog],
    pre_nas_rank:    Dict[str, int],          # ← rank sebelum NAS
    *,
    console=None,
) -> None:
    """Print tabel ranking balanced komprehensif dengan perbandingan pre/post NAS."""
    _p = console.print if console else print

    _p()
    _p("  ┌─ Balanced 50/50 Combined Score Ranking ─────────────────────────────────────────────────────────────────────────")
    _p("  │")
    _p("  │  SISTEM PENILAIAN BARU (seimbang):")
    _p("  │    Hardware Score  = 50%  → 7 dimensi GPU real (MFU·Throughput·VRAM·TC·SM·Compute·FA)")
    _p("  │    Training Score  = 50%  → 6 dimensi training real (Conv·Stab·Grad·GenGap·SampEff·OptComp)")
    _p("  │    Quality Gate    = ×(0.80–1.00)  → ArcQualityScorer konsistensi formula internal")
    _p("  │    combined = (0.50×hw + 0.50×ts) × (0.80 + 0.20×quality_gate)")
    _p("  │")
    _p("  │  vs SISTEM LAMA: combined = 0.35×quality + 0.65×fitness  [bias GPU, training diabaikan]")
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

        # Pre-NAS rank comparison
        pre_r = pre_nas_rank.get(aid, rank)
        if pre_r > rank:
            rank_arrow = f"#{pre_r}→★" if rank == 1 else f"#{pre_r}↑#{rank}"
        elif pre_r < rank:
            rank_arrow = f"#{pre_r}↓#{rank}"
        else:
            rank_arrow = f"#{pre_r}→#{rank}"

        # NaN detection dari training log
        has_nan  = (tr_log.nas_nan_count > 0) if tr_log else False
        nan_str  = "NaN⚠" if has_nan else "ok"

        # Status — HW dan Training improvements
        status_parts = []
        if hw_log and hw_log.perturbations_accepted > 0:
            status_parts.append(f"HW↑{hw_log.perturbations_accepted}")
        if tr_log and tr_log.perturbations_accepted > 0:
            status_parts.append(f"TR↑{tr_log.perturbations_accepted}")
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


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGE 9: PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def stage_profile(cfg: ArchConfig, gpu: GPUSpec, run_torch: bool) -> dict:
    """Stage 9: Profile single architecture."""
    profiler = TorchProfiler(cfg, gpu)
    if run_torch:
        console.print(f"\n[yellow]Running torch.profiler on {cfg.arch_id}...[/yellow]")
        return profiler.run()
    else:
        return profiler._analytical_fallback()


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGE 10: FULL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def stage_report(
    cfg:           ArchConfig,
    gpu:           GPUSpec,
    pr:            dict,
    n_gpus:        int,
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    combined_map:  Dict[str, float],
    hw_evaluator:  HardwareNASEvaluator,
    train_evaluator: TrainingDynamicsEvaluator,
    proxy_trainer: ProxyTrainer,
) -> MetricsReport:
    """
    Stage 10+11: Build MetricsReport, print detailed output,
    tampilkan quality scorecard + hardware NAS + training NAS.
    """
    report = MetricsReport(cfg, gpu, pr)
    print_detailed_report(cfg, gpu, report, pr, n_gpus)

    # Quality scorecard
    scorer       = ArcQualityScorer(gpu)
    score_report = scorer.score(cfg)
    print_score_report(score_report, console=console)

    # Hardware NAS detail
    hw_result = hw_evaluator.evaluate(cfg)
    print_hardware_nas_result(hw_result, console=console)

    # Training NAS detail (bisa lama, tapi berguna untuk top arch)
    console.print(f"[dim]  Running Training NAS evaluation for {cfg.arch_id}...[/dim]")
    proxy_result  = proxy_trainer.train(cfg)
    train_result  = train_evaluator.evaluate(cfg, proxy_result)
    print_training_nas_result(train_result, console=console)

    # Combined score breakdown
    q  = quality_map.get(cfg.arch_id, score_report.pct)
    hw = hw_scores.get(cfg.arch_id, hw_result.hardware_score)
    tr = train_scores.get(cfg.arch_id, train_result.training_score)
    c  = combined_map.get(cfg.arch_id, compute_combined_score_balanced(q, hw, tr))

    console.print()
    console.print(
        f"  ┌─ Combined Score Breakdown ─── {cfg.arch_id} {'─'*30}"
    )
    console.print(f"  │  Quality Score    : {q:.1f}%   (gate mult: {0.80 + 0.20*(q-70)/30:.3f})")
    console.print(f"  │  Hardware Score   : {hw:.4f}  (50% bobot)")
    console.print(f"  │  Training Score   : {tr:.4f}  (50% bobot)")
    console.print(f"  │  Combined (50/50) : [bold]{c:.5f}[/bold]")

    # Triple score jika fitness tersedia
    triple = compute_combined_score_triple(q, hw, tr, cfg.fitness_score)
    console.print(f"  │  Triple (HW40+TR40+Fit20): {triple:.5f}  [dim](legacy fitness={cfg.fitness_score:.4f})[/dim]")
    console.print(f"  └{'─'*65}")
    console.print()

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: enum → clean string
# ══════════════════════════════════════════════════════════════════════════════

def _enum_str(val) -> str:
    """Extract clean string dari enum atau string biasa."""
    if val is None:
        return "—"
    if hasattr(val, 'value'):
        return str(val.value)
    s = str(val)
    # Strip prefix seperti "FFNType.GEGLU" → "GEGLU", "AttentionType.GQA" → "GQA"
    if '.' in s:
        s = s.split('.')[-1]
    return s


def _row(label: str, value: str, width: int = 88) -> str:
    """Format satu baris kotak rekomendasi dengan padding yang tepat."""
    content = f"  {label:<22} {value}"
    # Truncate jika terlalu panjang, lalu pad ke width
    if len(content) > width - 2:
        content = content[:width - 5] + "..."
    return f"│{content:<{width}}│"


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL RECOMMENDATION DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def stage_final_recommendation(
    best:            ArchConfig,
    quality_map:     Dict[str, float],
    hw_scores:       Dict[str, float],
    train_scores:    Dict[str, float],
    combined_map:    Dict[str, float],
    gpu:             GPUSpec,
    hw_logs:         List[HardwareAdaptiveLog],
    train_logs:      List[TrainingAdaptiveLog],
    archs_sorted:    List[ArchConfig],
    nas_results_map: Dict[str, "TrainingNASResult"],   # ← cached NASResult
    pre_nas_rank:    Dict[str, int],                   # ← pre-NAS rank
) -> None:
    """
    Print final recommendation box — tidak re-run proxy training.
    Menggunakan cached NASResult dari Stage 7B.
    """
    _p = console.print

    aid  = best.arch_id
    q    = quality_map.get(aid, 0.0)
    hw   = hw_scores.get(aid, 0.0)
    tr   = train_scores.get(aid, 0.0)
    comb = combined_map.get(aid, 0.0)

    hw_log    = next((l for l in hw_logs    if l.arch_id == aid), None)
    train_log = next((l for l in train_logs if l.arch_id == aid), None)

    # Evaluasi hardware detail (cepat, tidak ada training)
    hw_eval   = HardwareNASEvaluator(gpu)
    hw_result = hw_eval.evaluate(best)

    # Gunakan cached NASResult dari Stage 7B — TIDAK re-run proxy training
    nas_result = nas_results_map.get(aid)
    if nas_result is not None:
        tr_result  = nas_result
        proxy_res  = nas_result.proxy_result
    else:
        # Fallback jika cache kosong (misal: NAS tidak dijalankan)
        proxy_res  = None
        tr_result  = None

    # Pre-NAS rank info
    pre_rank   = pre_nas_rank.get(aid, "?")
    rank_info  = f"pre-NAS #{pre_rank} → balanced #1"

    W = 90   # lebar kotak
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
    _p(_row("  ← Training Score:", f"{tr:.4f}  ×50%  ({tr*100:.1f}%)"
            + (f"  [{tr_result.grade[:18]}]" if tr_result else ""), W))
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
        _p(_row("T1 Convergence Rate:",   f"{tr_result.pts_t1:>5.1f}/22  score={tr_result.convergence_score:.3f}  reduction={max(0,(proxy_res.loss_initial-proxy_res.loss_final)/max(0.001,proxy_res.loss_initial)*100):.1f}%", W))
        _p(_row("T2 Training Stability:", f"{tr_result.pts_t2:>5.1f}/22  score={tr_result.stability_score:.3f}  NaN={nan_lbl}", W))
        _p(_row("T3 Gradient Health:",    f"{tr_result.pts_t3:>5.1f}/18  score={tr_result.gradient_health:.3f}  risk={tr_result.gradient_risk}", W))
        _p(_row("T4 Generalization Gap:", f"{tr_result.pts_t4:>5.1f}/15  score={tr_result.generalization_score:.3f}  gap={proxy_res.generalization_gap:.4f}", W))
        _p(_row("T5 Sample Efficiency:",  f"{tr_result.pts_t5:>5.1f}/13  score={tr_result.sample_efficiency:.3f}  eff_batch={best.batch_size*best.seq_len}", W))
        _p(_row("T6 Optimizer Compat:",   f"{tr_result.pts_t6:>5.1f}/10  score={tr_result.optimizer_compat:.3f}  lr_sens={tr_result.lr_sensitivity}", W))
        _p(_row("Training Total:",        f"{tr_result.total_pts:.1f}/100  [{tr_result.grade[:30]}]", W))
        _p(_row("Training Regime:",       tr_result.regime[:55], W))
    else:
        _p(_row("Training Score:", f"{tr:.4f}  (cached dari Stage 7B)", W))
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
        _p(_row("HW Score journey:", f"{hw_log.hw_score_start:.4f}→{hw_log.hw_score_end:.4f}  tries={hw_log.perturbation_tries}  accepted={hw_log.perturbations_accepted}  TC-improv={hw_log.tc_improvements}", W))

    if train_log:
        _p(f"├{S}┤")
        _p(f"│{'  TRAINING NAS-RL SUMMARY':^{W}}│")
        _p(_row("Train Score journey:", f"{train_log.train_score_start:.4f}→{train_log.train_score_end:.4f}  NAS-evals={train_log.nas_evaluations}  NaN={train_log.nas_nan_count}  {train_log.nas_training_ms_total:.0f}ms", W))
        _p(_row("RL:", f"tries={train_log.perturbation_tries}  accepted={train_log.perturbations_accepted}  conv-improv={train_log.convergence_improvements}  stab-improv={train_log.stability_improvements}", W))

    _p(f"╰{S}╯")
    _p()

    # Runner-up
    if len(archs_sorted) > 1:
        _p("  Runner-up architectures:")
        for i, cfg in enumerate(archs_sorted[1:4], 2):
            aid2     = cfg.arch_id
            pre_r2   = pre_nas_rank.get(aid2, "?")
            pre_info = f"(pre-NAS #{pre_r2})" if pre_r2 != "?" else ""
            _p(f"    #{i}  {aid2:<12}  {pre_info:<14}  "
               f"combined={combined_map.get(aid2, 0):.5f}  "
               f"hw={hw_scores.get(aid2, 0):.4f}  "
               f"train={train_scores.get(aid2, 0):.4f}  "
               f"quality={quality_map.get(aid2, 0):.1f}%  "
               f"params={cfg.param_count/1e6:.1f}M  VRAM={cfg.vram_usage_pct:.1f}%")
        _p()


# ══════════════════════════════════════════════════════════════════════════════
#  JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_results_json(
    archs_sorted:  List[ArchConfig],
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    combined_map:  Dict[str, float],
    best:          ArchConfig,
    gpu:           GPUSpec,
    *,
    filepath:      str = "nas_results.json",
) -> None:
    """Export hasil NAS ke JSON yang komprehensif."""
    from dataclasses import asdict
    import datetime

    results = {
        "generated_at": datetime.datetime.now().isoformat(),
        "gpu": gpu.name,
        "scoring_system": "balanced_50_50",
        "weights": {
            "hardware_score": 0.50,
            "training_score": 0.50,
            "quality_gate":   "0.80-1.00 multiplier",
        },
        "best_arch": best.arch_id,
        "architectures": [],
    }

    for cfg in archs_sorted:
        aid  = cfg.arch_id
        try:
            arch_dict = asdict(cfg)
        except Exception:
            arch_dict = {k: getattr(cfg, k, None) for k in cfg.__dataclass_fields__}

        arch_dict.update({
            "quality_score_pct": round(quality_map.get(aid, 0.0), 2),
            "hardware_score":    round(hw_scores.get(aid, 0.0), 4),
            "training_score":    round(train_scores.get(aid, 0.0), 4),
            "combined_score":    round(combined_map.get(aid, 0.0), 5),
        })
        results["architectures"].append(arch_dict)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[dim]  ✓ Results exported to {filepath}[/dim]")
    except Exception as e:
        console.print(f"[yellow]  ⚠ JSON export failed: {e}[/yellow]")


def _run_combination_branch(
    combo_specs:  List["CombinationSpec"],
    gpu:          GPUSpec,
    *,
    seed:         Optional[int],
    device_str:   str,
) -> None:
    """
    Branch khusus untuk mode kombinasi.

    Filosofi:
      User telah memilih 1 CombinationSpec (N-way blend dari beberapa type).
      Pipeline ini menjalankan RL combination refiner untuk menemukan
      1 arsitektur terkuat dari blend tersebut.

      TIDAK ada multiple-combination output — tujuannya satu: temukan
      arsitektur terbaik yang menggabungkan kekuatan semua type yang dipilih.

    Alur:
      1. Konfirmasi parameter RL (explore steps, candidates, device)
      2. Jalankan CombinationRefiner (RL 16-aksi, anti-bias reward)
         atau fallback ke CombinationNASRefiner (legacy) jika tidak tersedia
      3. Tampilkan detail report: 5 dimensi combo score + RL log
      4. Final recommendation satu arsitektur terkuat
      5. Optional JSON export
    """
    console.rule("[bold cyan]  Combination Architecture NAS Pipeline  ")
    console.print(
        f"[dim]  Combination Score = 33% combo(C1-C5)"
        f" + 34% hardware + 33% training  |  "
        f"RL menyempurnakan ratio/strategy/struktur secara otomatis[/dim]\n"
    )

    # ── Konfirmasi device ─────────────────────────────────────────────────────
    device_inp = input(
        "▶  PyTorch proxy device [cpu/cuda] (default=cpu): "
    ).strip().lower() or "cpu"
    if device_inp not in ("cpu", "cuda"):
        device_inp = "cpu"

    # ── Konfirmasi RL steps ───────────────────────────────────────────────────
    explore_str = input(
        "▶  Max RL exploration steps (default=40): "
    ).strip()
    max_explore = int(explore_str) if explore_str.isdigit() else 40

    # ── Konfirmasi candidates awal ────────────────────────────────────────────
    cand_str = input(
        "▶  Candidates awal per blend (default=3): "
    ).strip()
    n_cands = int(cand_str) if cand_str.isdigit() else 3

    console.print()
    console.print(f"  [dim]Spec: {combo_specs[0].label}  |  "
                   f"Families: {' + '.join(combo_specs[0].families)}  |  "
                   f"Strategy awal: {combo_specs[0].strategy}[/dim]")
    console.print()

    # ── Jalankan RL refinement ────────────────────────────────────────────────
    best_cfg:    ArchConfig         = None
    best_spec:   "CombinationSpec"  = None
    combo_res_final = None
    score_map:   Dict               = {}
    rl_log_obj                      = None   # CombinationRLLog jika tersedia
    alog                            = None   # CombinationAdaptiveLog (legacy compat)

    if _COMBO_REFINER_AVAILABLE:
        # ── Mode baru: CombinationRefiner (RL 16-aksi, anti-bias) ────────────
        console.rule("[bold cyan]  RL Combination Refiner (16-aksi, anti-bias reward)  ")

        rl_cfg = CombinationRLConfig(
            max_explore_iters = max_explore,
            n_candidates      = n_cands,
            proxy_device      = device_inp,
        )
        rl_refiner = CombinationRefiner(gpu, cfg=rl_cfg, seed=seed)

        t0 = time.perf_counter()
        best_cfg, best_spec, rl_log_obj = rl_refiner.refine_to_best(
            combo_specs, n_candidates=n_cands
        )
        elapsed = time.perf_counter() - t0

        console.print(
            f"[green]  ✓ RL selesai: {elapsed:.1f}s  |  "
            f"combined={rl_log_obj.combined_end:.5f}  |  "
            f"{rl_log_obj.perturbations_accepted}/{rl_log_obj.perturbation_tries} accepted  |  "
            f"{rl_log_obj.status}[/green]\n"
        )

        score_map = {
            "combo":    rl_log_obj.combo_score_end,
            "hw":       rl_log_obj.hw_score_end,
            "train":    rl_log_obj.train_score_end,
            "combined": rl_log_obj.combined_end,
            "quality":  rl_log_obj.quality_end,
        }

    else:
        # ── Fallback: CombinationNASRefiner (legacy) ──────────────────────────
        console.print("[yellow]  ℹ combination_refiner.py tidak tersedia — menggunakan legacy refiner[/yellow]")
        combo_archs, final_specs, combo_logs, score_maps_all = run_combination_pipeline(
            combo_specs, gpu,
            max_iterations    = 25,
            max_explore_iters = max_explore,
            n_candidates      = n_cands,
            seed              = seed,
            device            = device_inp,
            console           = console,
            use_rl_refiner    = False,
        )
        if not combo_archs:
            console.print("[red]  ✗ Tidak ada arsitektur kombinasi yang valid.[/red]")
            return
        best_cfg   = combo_archs[0]
        best_spec  = final_specs[0]
        alog       = combo_logs[0]
        score_map  = score_maps_all.get(best_cfg.arch_id, {})

    if best_cfg is None:
        console.print("[red]  ✗ Refinement gagal menghasilkan arsitektur valid.[/red]")
        return

    # ── Evaluasi combo final (untuk sub-skor C1–C5) ───────────────────────────
    combo_evaluator  = CombinationNASEvaluator(gpu)
    combo_res_final  = combo_evaluator.evaluate(best_cfg, best_spec)

    # ── Detail RL log ─────────────────────────────────────────────────────────
    show_detail = input("\n▶  Tampilkan detail RL log? [Y/n]: ").strip().lower()
    if show_detail not in ("n", "no"):
        if rl_log_obj is not None and _COMBO_REFINER_AVAILABLE:
            print_combination_rl_log(rl_log_obj, console=console)
        elif alog is not None:
            # Legacy compat: print manual
            console.print(f"  ─── Combination NAS-RL Log: {alog.arch_id} ───")
            console.print(f"       Combo Score:  {alog.combo_score_start:.4f} → {alog.combo_score_end:.4f}")
            console.print(f"       HW Score:     {alog.hw_score_start:.4f} → {alog.hw_score_end:.4f}")
            console.print(f"       Train Score:  {alog.train_score_start:.4f} → {alog.train_score_end:.4f}")
            console.print(f"       Combined:     {alog.combined_start:.5f} → {alog.combined_end:.5f}")
            console.print(f"       RL: tries={alog.perturbation_tries} accepted={alog.perturbations_accepted} "
                           f"ratio_adj={alog.ratio_adjustments} strat_sw={alog.strategy_switches}")
            if alog.improvement_events:
                console.print(f"       Improvements ({len(alog.improvement_events)}):")
                for ev in alog.improvement_events[:6]:
                    console.print(f"         ↑ {ev}")
            if alog.warnings:
                for w in alog.warnings[:3]:
                    console.print(f"       ⚠ {w}")
            console.print()

    # ── Final recommendation box ──────────────────────────────────────────────
    console.rule("[bold cyan]  ★ Combination Final Recommendation  ")

    W = 90
    S = "─" * W

    # Bangun label families + ratio
    fam_ratio_str = " + ".join(
        f"{f} ({r:.0%})"
        for f, r in zip(best_spec.families, best_spec.ratios)
    )
    if len(fam_ratio_str) > W - 20:
        fam_ratio_str = " + ".join(
            f"{f.split('-')[0]}:{int(r*100)}%"
            for f, r in zip(best_spec.families, best_spec.ratios)
        )

    console.print()
    console.print(f"╭{S}╮")
    console.print(f"│{'  🔀 BEST COMBINATION — BALANCED NAS + RL':^{W}}│")
    console.print(f"├{S}┤")
    console.print(f"│  ARC: {best_cfg.arch_id:<10}  {best_cfg.arch_name[:58]:<58}  │")
    console.print(f"│  Blend: {fam_ratio_str[:W-10]:<{W-10}}  │")
    console.print(f"│  Strategy: {best_spec.strategy:<14}  Compat: {best_spec.compatibility:<18}  "
                   f"Synergy: {best_spec.synergy_mult:.3f}  │")
    console.print(f"├{S}┤")
    console.print(f"│  {'COMBINED SCORE (33% combo + 34% hardware + 33% training)':^{W}}│")
    combined_val = score_map.get("combined", combo_res_final.combination_score)
    combo_val    = score_map.get("combo",    combo_res_final.combination_score)
    hw_val       = score_map.get("hw",       0.0)
    train_val    = score_map.get("train",    0.0)
    quality_val  = score_map.get("quality",  0.0)
    console.print(f"│  Combined Score  : {combined_val:.5f}                              │")
    console.print(f"│  Combo Score     : {combo_val:.4f}  ×33%  "
                   f"(C1={combo_res_final.pts_c1:.1f}/25 C2={combo_res_final.pts_c2:.1f}/20 "
                   f"C3={combo_res_final.pts_c3:.1f}/20 C4={combo_res_final.pts_c4:.1f}/20 "
                   f"C5={combo_res_final.pts_c5:.1f}/15)  │")
    console.print(f"│  Hardware Score  : {hw_val:.4f}  ×34%                              │")
    console.print(f"│  Training Score  : {train_val:.4f}  ×33%                              │")
    console.print(f"│  Quality Gate    : {quality_val:.1f}%                                  │")
    total_pts = (combo_res_final.pts_c1 + combo_res_final.pts_c2 +
                  combo_res_final.pts_c3 + combo_res_final.pts_c4 +
                  combo_res_final.pts_c5)
    grade = combo_res_final.grade
    console.print(f"│  Combo Total: {total_pts:.1f}/100  Grade: {grade[:40]:<40}  │")
    console.print(f"├{S}┤")
    console.print(f"│  {'ARCHITECTURE PARAMETERS':^{W}}│")
    ffn_s  = (best_cfg.ffn_type.value if hasattr(best_cfg.ffn_type, 'value')
               else str(best_cfg.ffn_type).split('.')[-1])
    attn_s = (best_cfg.attn_type.value if hasattr(best_cfg.attn_type, 'value')
               else str(best_cfg.attn_type).split('.')[-1])
    opt_s  = (best_cfg.optimizer_type.value if hasattr(best_cfg.optimizer_type, 'value')
               else str(best_cfg.optimizer_type).split('.')[-1])
    console.print(f"│  Params: {best_cfg.param_count/1e6:.1f}M  L={best_cfg.num_layers}  "
                   f"D={best_cfg.hidden_dim}  H={best_cfg.num_heads}/{best_cfg.num_kv_heads}  "
                   f"hd={best_cfg.head_dim}  │")
    console.print(f"│  FFN: {ffn_s:<12}×{best_cfg.ffn_multiplier:.2f}  "
                   f"Attn: {attn_s:<15}  Seq={best_cfg.seq_len}  Batch={best_cfg.batch_size}  │")
    console.print(f"│  Optim: {opt_s:<25}  FA={best_cfg.use_flash_attn}  "
                   f"MixedPrec={best_cfg.use_mixed_precision}  │")
    console.print(f"│  VRAM: {best_cfg.vram_total_gb:.2f}GB ({best_cfg.vram_usage_pct:.1f}%)  "
                   f"MFU={best_cfg.mfu_estimate:.4f}  "
                   f"{best_cfg.bottleneck}  {best_cfg.ms_per_step:.1f}ms/step  │")

    # Tampilkan synergy info dari database
    if best_spec.n_families == 2:
        key  = (min(best_spec.families[0], best_spec.families[1]),
                max(best_spec.families[0], best_spec.families[1]))
        from combination_nas import _SYNERGY_NORMALIZED
        info = _SYNERGY_NORMALIZED.get(key, {})
        rat  = info.get("rationale", "")
        if rat:
            console.print(f"│  Rationale: {rat[:W-14]:<{W-14}}  │")

    if combo_res_final.warnings:
        console.print(f"├{S}┤")
        for w in combo_res_final.warnings[:2]:
            console.print(f"│  ⚠ {w[:W-6]:<{W-6}}  │")

    # RL summary singkat
    if rl_log_obj is not None:
        console.print(f"├{S}┤")
        console.print(f"│  {'RL SUMMARY':^{W}}│")
        console.print(
            f"│  Tries: {rl_log_obj.perturbation_tries}  "
            f"Accepted: {rl_log_obj.perturbations_accepted}  "
            f"Accept-rate: {rl_log_obj.accept_rate:.1%}  "
            f"Burst: {rl_log_obj.burst_explore_count}  "
            f"Proxy-evals: {rl_log_obj.proxy_eval_count}  │"
        )
        console.print(
            f"│  Ratio-adj: {rl_log_obj.ratio_adjustments}  "
            f"Strategy-sw: {rl_log_obj.strategy_switches}  "
            f"Structural: {rl_log_obj.structural_changes}  "
            f"Replay: {rl_log_obj.replay_updates}  │"
        )
        console.print(f"│  Status: {rl_log_obj.status[:W-12]:<{W-12}}  │")

    console.print(f"╰{S}╯")
    console.print()

    # ── JSON export ───────────────────────────────────────────────────────────
    export_prompt = input("▶  Export combination results to JSON? [Y/n]: ").strip().lower()
    if export_prompt not in ("n", "no"):
        import json
        import datetime
        from dataclasses import asdict

        results = {
            "generated_at":  datetime.datetime.now().isoformat(),
            "gpu":           gpu.name,
            "mode":          "combination",
            "rl_engine":     "CombinationRefiner" if _COMBO_REFINER_AVAILABLE
                              else "CombinationNASRefiner (legacy)",
            "best_arch":     best_cfg.arch_id,
            "combination": {
                "families":     best_spec.families,
                "ratios":       best_spec.ratios,
                "strategy":     best_spec.strategy,
                "compatibility": best_spec.compatibility,
                "synergy_mult": best_spec.synergy_mult,
            },
            "scores": {
                "combo_score":    round(combo_val, 4),
                "hardware_score": round(hw_val, 4),
                "training_score": round(train_val, 4),
                "combined_score": round(combined_val, 5),
                "quality_pct":    round(quality_val, 2),
                "c1_coherence":   round(combo_res_final.pts_c1, 2),
                "c2_balance":     round(combo_res_final.pts_c2, 2),
                "c3_synergy":     round(combo_res_final.pts_c3, 2),
                "c4_hw_compat":   round(combo_res_final.pts_c4, 2),
                "c5_train_syn":   round(combo_res_final.pts_c5, 2),
            },
        }

        if rl_log_obj is not None:
            results["rl_log"] = {
                "perturbation_tries":     rl_log_obj.perturbation_tries,
                "perturbations_accepted": rl_log_obj.perturbations_accepted,
                "accept_rate":            rl_log_obj.accept_rate,
                "ratio_adjustments":      rl_log_obj.ratio_adjustments,
                "strategy_switches":      rl_log_obj.strategy_switches,
                "structural_changes":     rl_log_obj.structural_changes,
                "proxy_eval_count":       rl_log_obj.proxy_eval_count,
                "proxy_nan_count":        rl_log_obj.proxy_nan_count,
                "proxy_ms_total":         round(rl_log_obj.proxy_ms_total, 1),
                "improvement_events":     rl_log_obj.improvement_events,
            }

        try:
            arch_dict = asdict(best_cfg)
        except Exception:
            arch_dict = {k: getattr(best_cfg, k, None)
                         for k in best_cfg.__dataclass_fields__}
        results["architecture"] = arch_dict

        filepath = "combination_result.json"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[dim]  ✓ Exported to {filepath}[/dim]")
        except Exception as e:
            console.print(f"[yellow]  ⚠ Export failed: {e}[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Stage 1: Banner
    display_banner()

    # Stage 2: GPU selection
    gpu = select_gpu()

    # Stage 3: Tanya mode — "Combination type or only type (Y/n):"
    want_combo = ask_combination_mode()

    if want_combo:
        # ── COMBINATION BRANCH ────────────────────────────────────────────────
        # Step 1: User pilih 2–4 AI type yang mau di-blend
        ai_families = ask_ai_type_local(combination_mode=True)

        opts         = ask_run_options()
        seed         = opts["seed"]

        # Step 2: Tampilkan semua kombinasi valid dari type yang dipilih,
        #         user pilih 1 → sistem hasilkan 1 arsitektur terkuat
        selected_fams = list(ai_families) if ai_families else [
            f for f, _ in _ALL_FAMILIES_ORDERED
        ]
        combo_specs = ask_combination_type(selected_families=selected_fams)

        if not combo_specs:
            console.print(
                "\n[yellow]  Tidak ada kombinasi dipilih — pipeline berhenti.[/yellow]"
            )
            return

        # Step 3: Jalankan combination NAS + RL → 1 arsitektur terkuat
        _run_combination_branch(
            combo_specs, gpu,
            seed       = seed,
            device_str = "cpu",
        )
        return
        # ── END COMBINATION BRANCH ────────────────────────────────────────────

    # ── SINGLE / NORMAL NAS BRANCH ────────────────────────────────────────────
    # Interface 1: ask_ai_type (Balanced / Fast / Light / Smart / dll)
    ai_families  = ask_ai_type()

    opts         = ask_run_options()
    n_per_family = opts["n_per_family"]
    run_torch    = opts["run_torch"]
    n_gpus       = opts["n_gpus"]
    seed         = opts["seed"]

    archs = stage_generate(gpu, ai_families, n_per_family, seed)
    console.rule("[bold]  Stage 6 — Pre-NAS Ranking  ")
    print_arch_summary(archs, gpu)

        # ── Init semua state variables sebelum if/else (fix scope bug) ───────────
    quality_map:     Dict[str, float] = {}
    hw_scores:       Dict[str, float] = {}
    train_scores:    Dict[str, float] = {}
    combined_map:    Dict[str, float] = {}
    nas_results_map: Dict[str, object] = {}
    hw_logs:         List[HardwareAdaptiveLog]  = []
    train_logs:      List[TrainingAdaptiveLog]  = []
    archs_sorted:    List[ArchConfig]           = list(archs)
    best:            ArchConfig                 = archs[0] if archs else None

    # pre-NAS rank: posisi sebelum NAS berdasarkan fitness_score
    pre_nas_rank: Dict[str, int] = {a.arch_id: i for i, a in enumerate(archs, 1)}

    # ── NAS prompt ────────────────────────────────────────────────────────────
    console.print()
    run_nas = input(
        "▶  Run Dual NAS (Hardware NAS 7D + Training NAS 6D, 50/50 seimbang)? [Y/n]: "
    ).strip().lower()

    if run_nas in ("n", "no"):
        # Fallback: hanya quality refiner lama
        from refiner import ArcRefiner
        refiner_base = ArcRefiner(gpu)
        refined_list, base_logs = refiner_base.refine_batch(archs)
        scorer_q = ArcQualityScorer(gpu)
        for a in refined_list:
            q = scorer_q.score(a).pct
            quality_map[a.arch_id]  = q
            hw_scores[a.arch_id]    = a.fitness_score
            train_scores[a.arch_id] = 0.0
            combined_map[a.arch_id] = q / 100.0

        archs_sorted = refined_list
        best         = archs_sorted[0] if archs_sorted else archs[0]

    else:
        # ── NAS config ─────────────────────────────────────────────────────
        target_str = input(
            "▶  Target quality score (default=100): "
        ).strip()
        target_pct = float(target_str) if target_str.replace(".", "").isdigit() else 100.0
        target_pct = max(50.0, min(100.0, target_pct))

        max_iter_str = input(
            "▶  Max iterations per ARC quality refine (default=25): "
        ).strip()
        max_iter = int(max_iter_str) if max_iter_str.isdigit() else 25

        explore_str = input(
            "▶  Max RL exploration steps per ARC (default=30): "
        ).strip()
        max_explore = int(explore_str) if explore_str.isdigit() else 30

        # ── Stage 7A: Hardware NAS ─────────────────────────────────────────
        hw_archs, hw_logs, hw_scores, quality_map = stage_hardware_nas(
            archs, gpu,
            max_iterations    = max_iter,
            target_pct        = target_pct,
            max_explore_iters = max_explore,
            seed              = seed,
        )

        # ── Stage 7B: Training NAS ─────────────────────────────────────────
        device_str = input(
            "▶  PyTorch proxy device [cpu/cuda] (default=cpu): "
        ).strip().lower() or "cpu"
        if device_str not in ("cpu", "cuda"):
            device_str = "cpu"

        train_archs, train_logs, train_scores, nas_results_map = stage_training_nas(
            hw_archs, gpu,
            hw_scores         = hw_scores,
            max_iterations    = max_iter,
            target_pct        = target_pct,
            max_explore_iters = max_explore,
            seed              = seed,
            device            = device_str,
        )

        # ── Stage 7C: Balanced Scoring ────────────────────────────────────
        archs_sorted, combined_map, best = stage_balanced_scoring(
            train_archs, quality_map, hw_scores, train_scores,
            hw_logs, train_logs, gpu, archs,          # ← archs = pre_nas_archs
        )

        # Update pre_nas_rank ke archs final yang mungkin berbeda setelah refinement
        for a in archs:
            if a.arch_id not in pre_nas_rank:
                pre_nas_rank[a.arch_id] = len(archs)

        # Offer per-ARC detail logs
        log_prompt = input(
            "\n▶  Show detailed NAS-RL logs? [y/N]: "
        ).strip().lower()
        if log_prompt in ("y", "yes"):
            console.rule("[bold]  Hardware NAS-RL Detail Logs  ")
            for alog in hw_logs[:5]:
                print_hardware_adaptive_log(alog, console=console)

            console.rule("[bold]  Training NAS-RL Detail Logs  ")
            for alog in train_logs[:5]:
                print_training_adaptive_log(alog, console=console)

    # ── Stage 8+: Detailed profiling ──────────────────────────────────────
    detail_prompt = input(
        "\n▶  Show full detailed report for top ARCs? [Y/n]: "
    ).strip().lower()

    hw_evaluator    = HardwareNASEvaluator(gpu)
    train_evaluator = TrainingDynamicsEvaluator(gpu)
    proxy_trainer   = ProxyTrainer()

    if detail_prompt not in ("n", "no"):
        top_n_str = input(
            "▶  How many top architectures to detail? [default=3]: "
        ).strip()
        top_n     = int(top_n_str) if top_n_str.isdigit() else 3
        top_archs = [a for a in archs_sorted if a.fits_gpu][:top_n]

        for cfg in top_archs:
            pr = stage_profile(cfg, gpu, run_torch)
            stage_report(
                cfg, gpu, pr, n_gpus,
                quality_map, hw_scores, train_scores, combined_map,
                hw_evaluator, train_evaluator, proxy_trainer,
            )

    # ── Stage 12: Final recommendation (SATU recommendation saja) ─────────
    console.rule("[bold]  Final Recommendation  ")
    if best:
        stage_final_recommendation(
            best, quality_map, hw_scores, train_scores, combined_map,
            gpu, hw_logs, train_logs, archs_sorted,
            nas_results_map, pre_nas_rank,
        )

    # JSON export
    export_prompt = input(
        "▶  Export results to JSON? [Y/n]: "
    ).strip().lower()
    if export_prompt not in ("n", "no"):
        export_results_json(
            archs_sorted, quality_map, hw_scores, train_scores, combined_map,
            best, gpu,
        )

    # ── TIDAK ada print_ranking lagi — rekomendasi hanya satu ─────────────
    # Legacy print_ranking dihapus karena menyebabkan DOUBLE recommendation
    # (Bug #1 yang dilaporkan). Gunakan stage_final_recommendation sebagai
    # satu-satunya output rekomendasi.


# ══════════════════════════════════════════════════════════════════════════════
#  JSON EXPORT — ONLY TYPE (single-family NAS, struktur baru)
# ══════════════════════════════════════════════════════════════════════════════

def export_results_json_onlytype(
    archs_sorted:    List[ArchConfig],
    quality_map:     Dict[str, float],
    hw_scores:       Dict[str, float],
    train_scores:    Dict[str, float],
    combined_map:    Dict[str, float],
    best:            ArchConfig,
    gpu:             GPUSpec,
    hw_logs:         List[HardwareAdaptiveLog],
    train_logs:      List[TrainingAdaptiveLog],
    nas_results_map: Dict[str, object],
    pre_nas_rank:    Dict[str, int],
    *,
    filepath: str = "nas_results_onlytype.json",
) -> None:
    """
    Export hasil single-type NAS ke JSON dengan struktur baru yang lebih
    terstruktur dan komprehensif dibandingkan export_results_json().

    Struktur baru:
      • export_type: "only_type"   — pembeda dari multi-type
      • pipeline_summary           — ringkasan pipeline (total arc, best, avg skor)
      • scoring_schema             — penjelasan formula dan dimensi NAS
      • nas_journey                — perjalanan RL per-arc (pre→post rank, improv)
      • best_arch_detail           — breakdown lengkap best arch (hw+train sub-skor)
      • ranked_architectures       — daftar lengkap dengan parameter + semua skor
    """
    import datetime
    from dataclasses import asdict

    # ── Helper untuk parse enum string ───────────────────────────────────────
    def _clean(val) -> str:
        if val is None:
            return "—"
        if hasattr(val, "value"):
            return str(val.value)
        s = str(val)
        return s.split(".")[-1] if "." in s else s

    # ── Pipeline summary ──────────────────────────────────────────────────────
    all_combined = [combined_map.get(a.arch_id, 0.0) for a in archs_sorted]
    all_hw       = [hw_scores.get(a.arch_id, 0.0)    for a in archs_sorted]
    all_tr       = [train_scores.get(a.arch_id, 0.0) for a in archs_sorted]
    all_q        = [quality_map.get(a.arch_id, 0.0)  for a in archs_sorted]
    n            = max(1, len(archs_sorted))

    fits_gpu_archs  = [a for a in archs_sorted if a.fits_gpu]
    families_seen   = list(dict.fromkeys(a.arch_family for a in archs_sorted))

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
            "T1_Convergence_Rate":     "22 pts — loss turun seberapa cepat",
            "T2_Training_Stability":   "22 pts — loss variance, NaN detection",
            "T3_Gradient_Health":      "18 pts — grad norm, vanishing/exploding",
            "T4_Generalization_Gap":   "15 pts — train vs val loss di proxy",
            "T5_Sample_Efficiency":    "13 pts — Chinchilla, noise, tied embed",
            "T6_Optimizer_Compat":     "10 pts — optimizer vs depth compat",
        },
        "vs_old_system": "LAMA: 35%×quality + 65%×fitness (bias GPU, training diabaikan)",
    }

    # ── NAS journey per-arc ───────────────────────────────────────────────────
    hw_log_map    = {l.arch_id: l for l in hw_logs}
    train_log_map = {l.arch_id: l for l in train_logs}

    nas_journey = []
    for post_rank, cfg in enumerate(archs_sorted, 1):
        aid      = cfg.arch_id
        pre_rank = pre_nas_rank.get(aid, post_rank)
        hw_log   = hw_log_map.get(aid)
        tr_log   = train_log_map.get(aid)

        rank_change = pre_rank - post_rank   # positif = naik rank
        if rank_change > 0:
            rank_direction = "↑ naik"
        elif rank_change < 0:
            rank_direction = "↓ turun"
        else:
            rank_direction = "= sama"

        journey_entry = {
            "arch_id":            aid,
            "pre_nas_rank":       pre_rank,
            "post_nas_rank":      post_rank,
            "rank_change":        rank_change,
            "rank_direction":     rank_direction,
            "combined_score":     round(combined_map.get(aid, 0.0), 5),
            "hardware_score":     round(hw_scores.get(aid, 0.0), 4),
            "training_score":     round(train_scores.get(aid, 0.0), 4),
            "quality_pct":        round(quality_map.get(aid, 0.0), 2),
        }
        if hw_log:
            journey_entry["hw_rl"] = {
                "hw_score_start":        round(hw_log.hw_score_start, 4),
                "hw_score_end":          round(hw_log.hw_score_end, 4),
                "hw_score_delta":        round(hw_log.hw_score_end - hw_log.hw_score_start, 4),
                "perturbation_tries":    hw_log.perturbation_tries,
                "perturbations_accepted":hw_log.perturbations_accepted,
                "tc_improvements":       hw_log.tc_improvements,
                "status":                hw_log.status,
            }
        if tr_log:
            journey_entry["train_rl"] = {
                "train_score_start":     round(tr_log.train_score_start, 4),
                "train_score_end":       round(tr_log.train_score_end, 4),
                "train_score_delta":     round(tr_log.train_score_end - tr_log.train_score_start, 4),
                "nas_evaluations":       tr_log.nas_evaluations,
                "nas_training_ms":       round(tr_log.nas_training_ms_total, 1),
                "nan_detected_count":    tr_log.nas_nan_count,
                "perturbation_tries":    tr_log.perturbation_tries,
                "perturbations_accepted":tr_log.perturbations_accepted,
                "convergence_improvements": tr_log.convergence_improvements,
                "stability_improvements":   tr_log.stability_improvements,
                "status":                tr_log.status,
            }
        nas_journey.append(journey_entry)

    # ── Best arch detail ──────────────────────────────────────────────────────
    best_detail: dict = {}
    if best:
        aid      = best.arch_id
        nas_res  = nas_results_map.get(aid)
        hw_log   = hw_log_map.get(aid)
        tr_log   = train_log_map.get(aid)
        q        = quality_map.get(aid, 0.0)
        hw       = hw_scores.get(aid, 0.0)
        tr       = train_scores.get(aid, 0.0)
        comb     = combined_map.get(aid, 0.0)
        gate_m   = 0.80 + 0.20 * max(0.0, min(1.0, (q - 70.0) / 30.0))

        best_detail = {
            "arch_id":    aid,
            "arch_name":  best.arch_name,
            "arch_family":best.arch_family,
            "scores": {
                "combined_score":     round(comb, 5),
                "hardware_score":     round(hw, 4),
                "training_score":     round(tr, 4),
                "quality_pct":        round(q, 2),
                "quality_gate_mult":  round(gate_m, 4),
            },
            "architecture_params": {
                "param_count_M":   round(best.param_count / 1e6, 2),
                "num_layers":      best.num_layers,
                "hidden_dim":      best.hidden_dim,
                "num_heads":       best.num_heads,
                "num_kv_heads":    best.num_kv_heads,
                "head_dim":        best.head_dim,
                "ffn_multiplier":  best.ffn_multiplier,
                "seq_len":         best.seq_len,
                "batch_size":      best.batch_size,
                "ffn_type":        _clean(best.ffn_type),
                "attn_type":       _clean(best.attn_type),
                "norm_type":       _clean(best.norm_type),
                "optimizer_type":  _clean(best.optimizer_type),
                "pos_enc":         _clean(best.pos_enc),
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
                "use_flash_attn":              best.use_flash_attn,
                "use_mixed_precision":         best.use_mixed_precision,
                "use_gradient_checkpointing":  best.use_gradient_checkpointing,
                "use_torch_compile":           best.use_torch_compile,
                "tie_embeddings":              best.tie_embeddings,
                "dropout":                     best.dropout,
            },
        }

        # Tambahkan sub-score NAS dari cached result jika tersedia
        if nas_res is not None:
            pr = getattr(nas_res, "proxy_result", None)
            best_detail["training_sub_scores"] = {
                "pts_t1_convergence":   round(getattr(nas_res, "pts_t1", 0.0), 2),
                "pts_t2_stability":     round(getattr(nas_res, "pts_t2", 0.0), 2),
                "pts_t3_gradient":      round(getattr(nas_res, "pts_t3", 0.0), 2),
                "pts_t4_gen_gap":       round(getattr(nas_res, "pts_t4", 0.0), 2),
                "pts_t5_sample_eff":    round(getattr(nas_res, "pts_t5", 0.0), 2),
                "pts_t6_opt_compat":    round(getattr(nas_res, "pts_t6", 0.0), 2),
                "total_pts":            round(getattr(nas_res, "total_pts", 0.0), 2),
                "grade":                getattr(nas_res, "grade", "—"),
                "regime":               getattr(nas_res, "regime", "—"),
                "gradient_risk":        str(getattr(nas_res, "gradient_risk", "—")),
                "lr_sensitivity":       str(getattr(nas_res, "lr_sensitivity", "—")),
            }
            if pr is not None:
                best_detail["proxy_training"] = {
                    "loss_initial":       round(getattr(pr, "loss_initial", 0.0), 6),
                    "loss_final":         round(getattr(pr, "loss_final", 0.0), 6),
                    "generalization_gap": round(getattr(pr, "generalization_gap", 0.0), 6),
                    "nan_detected":       getattr(pr, "nan_detected", False),
                }

    # ── Ranked architectures (full list) ─────────────────────────────────────
    ranked_architectures = []
    for rank, cfg in enumerate(archs_sorted, 1):
        aid = cfg.arch_id
        try:
            arch_dict = asdict(cfg)
        except Exception:
            arch_dict = {k: getattr(cfg, k, None) for k in cfg.__dataclass_fields__}

        # Bersihkan enum fields menjadi string
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

    # ── Assemble final JSON ───────────────────────────────────────────────────
    result = {
        "export_type":           "only_type",
        "generated_at":          datetime.datetime.now().isoformat(),
        "gpu":                   gpu.name,
        "pipeline_mode":         "single_type_nas",
        "scoring_system":        "balanced_50_50",
        "pipeline_summary":      pipeline_summary,
        "scoring_schema":        scoring_schema,
        "nas_journey":           nas_journey,
        "best_arch_detail":      best_detail,
        "ranked_architectures":  ranked_architectures,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[dim]  ✓ Only-type results exported → {filepath}[/dim]")
    except Exception as e:
        console.print(f"[yellow]  ⚠ Only-type JSON export failed: {e}[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
#  JSON EXPORT — MULTI TYPE (combination NAS, struktur baru)
# ══════════════════════════════════════════════════════════════════════════════

def export_results_json_multitype(
    best_cfg:       ArchConfig,
    best_spec:      "CombinationSpec",
    combo_res:      object,         # CombinationNASResult — sub-skor C1–C5
    score_map:      Dict[str, float],
    rl_log_obj:     Optional[object],    # CombinationRLLog (bisa None)
    gpu:            GPUSpec,
    *,
    filepath: str = "combination_result_multitype.json",
) -> None:
    """
    Export hasil combination/multi-type NAS ke JSON dengan struktur baru yang
    lebih terstruktur dan berbeda dari export combination inline di pipeline.

    Struktur baru:
      • export_type: "multi_type"     — pembeda dari only-type
      • blend_info                    — detail families, ratios, strategy, synergy
      • scoring_formula               — penjelasan formula combination scoring
      • combination_sub_scores        — C1–C5 dengan breakdown per dimensi
      • rl_optimization               — log RL lengkap: tries, accept_rate, events
      • best_arch_detail              — parameter arsitektur terbaik + profil HW
      • compatibility_analysis        — analisis kompatibilitas dan synergy antar family
    """
    import datetime
    from dataclasses import asdict

    def _clean(val) -> str:
        if val is None:
            return "—"
        if hasattr(val, "value"):
            return str(val.value)
        s = str(val)
        return s.split(".")[-1] if "." in s else s

    # ── Blend info ────────────────────────────────────────────────────────────
    families    = best_spec.families if best_spec else []
    ratios      = best_spec.ratios   if best_spec else []
    strategy    = best_spec.strategy if best_spec else "WEIGHTED"
    compat      = best_spec.compatibility if best_spec else "—"
    synergy_m   = best_spec.synergy_mult  if best_spec else 1.0
    n_families  = best_spec.n_families    if best_spec else len(families)

    family_ratio_pairs = [
        {"family": f, "ratio": round(r, 4), "ratio_pct": f"{r:.0%}"}
        for f, r in zip(families, ratios)
    ]

    blend_info = {
        "n_families":         n_families,
        "families":           families,
        "ratios":             ratios,
        "family_ratio_pairs": family_ratio_pairs,
        "strategy":           strategy,
        "compatibility":      compat,
        "synergy_multiplier": round(synergy_m, 4),
        "blend_label":        " + ".join(
            f"{f}({r:.0%})" for f, r in zip(families, ratios)
        ),
    }

    # ── Scoring formula ───────────────────────────────────────────────────────
    scoring_formula = {
        "formula":            "combined = 33%×combo_score + 34%×hardware_score + 33%×training_score",
        "combo_weight":       0.33,
        "hardware_weight":    0.34,
        "training_weight":    0.33,
        "combo_dimensions": {
            "C1_Coherence":     "25 pts — konsistensi internal arsitektur hybrid",
            "C2_Balance":       "20 pts — keseimbangan kontribusi antar family",
            "C3_Synergy":       "20 pts — sinergi struktural antar komponen",
            "C4_HW_Compat":     "20 pts — kompatibilitas hardware kombinasi",
            "C5_Train_Synergy": "15 pts — sinergi training dynamics",
        },
        "vs_only_type": "Multi-type: 33%×combo + 34%×HW + 33%×train (kombinasi memiliki dimensi C1–C5 tambahan)",
    }

    # ── Combination sub-scores ────────────────────────────────────────────────
    combo_val   = score_map.get("combo",    getattr(combo_res, "combination_score", 0.0))
    hw_val      = score_map.get("hw",       0.0)
    train_val   = score_map.get("train",    0.0)
    combined_val= score_map.get("combined", 0.0)
    quality_val = score_map.get("quality",  0.0)

    pts_c1 = getattr(combo_res, "pts_c1", 0.0)
    pts_c2 = getattr(combo_res, "pts_c2", 0.0)
    pts_c3 = getattr(combo_res, "pts_c3", 0.0)
    pts_c4 = getattr(combo_res, "pts_c4", 0.0)
    pts_c5 = getattr(combo_res, "pts_c5", 0.0)
    total_combo_pts = pts_c1 + pts_c2 + pts_c3 + pts_c4 + pts_c5

    combination_sub_scores = {
        "combo_score":          round(combo_val, 4),
        "hardware_score":       round(hw_val, 4),
        "training_score":       round(train_val, 4),
        "combined_score":       round(combined_val, 5),
        "quality_pct":          round(quality_val, 2),
        "combo_pts_breakdown": {
            "C1_coherence":     round(pts_c1, 2),
            "C2_balance":       round(pts_c2, 2),
            "C3_synergy":       round(pts_c3, 2),
            "C4_hw_compat":     round(pts_c4, 2),
            "C5_train_synergy": round(pts_c5, 2),
            "total_combo_pts":  round(total_combo_pts, 2),
            "max_possible":     100,
        },
        "grade":    getattr(combo_res, "grade", "—"),
        "warnings": getattr(combo_res, "warnings", []),
    }

    # ── RL optimization log ───────────────────────────────────────────────────
    rl_optimization: dict = {
        "engine": "CombinationRefiner" if rl_log_obj is not None else "legacy/fallback",
        "available": rl_log_obj is not None,
    }
    if rl_log_obj is not None:
        rl_optimization.update({
            "perturbation_tries":     rl_log_obj.perturbation_tries,
            "perturbations_accepted": rl_log_obj.perturbations_accepted,
            "accept_rate":            round(rl_log_obj.accept_rate, 4),
            "ratio_adjustments":      rl_log_obj.ratio_adjustments,
            "strategy_switches":      rl_log_obj.strategy_switches,
            "structural_changes":     rl_log_obj.structural_changes,
            "burst_explore_count":    rl_log_obj.burst_explore_count,
            "replay_updates":         rl_log_obj.replay_updates,
            "proxy_eval_count":       rl_log_obj.proxy_eval_count,
            "proxy_nan_count":        rl_log_obj.proxy_nan_count,
            "proxy_ms_total":         round(rl_log_obj.proxy_ms_total, 1),
            "score_journey": {
                "combo_start":    round(getattr(rl_log_obj, "combo_score_start", 0.0), 4),
                "combo_end":      round(rl_log_obj.combo_score_end, 4),
                "hw_start":       round(getattr(rl_log_obj, "hw_score_start", 0.0), 4),
                "hw_end":         round(rl_log_obj.hw_score_end, 4),
                "train_start":    round(getattr(rl_log_obj, "train_score_start", 0.0), 4),
                "train_end":      round(rl_log_obj.train_score_end, 4),
                "combined_start": round(getattr(rl_log_obj, "combined_start", 0.0), 5),
                "combined_end":   round(rl_log_obj.combined_end, 5),
                "quality_end":    round(rl_log_obj.quality_end, 2),
            },
            "key_improvement_events":  rl_log_obj.improvement_events,
            "status":                  rl_log_obj.status,
        })

    # ── Best arch detail ──────────────────────────────────────────────────────
    try:
        arch_dict = asdict(best_cfg)
    except Exception:
        arch_dict = {k: getattr(best_cfg, k, None) for k in best_cfg.__dataclass_fields__}

    for key in ("attn_type", "ffn_type", "optimizer_type", "norm_type", "pos_enc"):
        if key in arch_dict:
            arch_dict[key] = _clean(arch_dict[key])

    best_arch_detail = {
        "arch_id":    best_cfg.arch_id,
        "arch_name":  best_cfg.arch_name,
        "arch_family":best_cfg.arch_family,
        "architecture_params": {
            "param_count_M":  round(best_cfg.param_count / 1e6, 2),
            "num_layers":     best_cfg.num_layers,
            "hidden_dim":     best_cfg.hidden_dim,
            "num_heads":      best_cfg.num_heads,
            "num_kv_heads":   best_cfg.num_kv_heads,
            "head_dim":       best_cfg.head_dim,
            "ffn_multiplier": best_cfg.ffn_multiplier,
            "seq_len":        best_cfg.seq_len,
            "batch_size":     best_cfg.batch_size,
            "ffn_type":       _clean(best_cfg.ffn_type),
            "attn_type":      _clean(best_cfg.attn_type),
            "norm_type":      _clean(best_cfg.norm_type),
            "optimizer_type": _clean(best_cfg.optimizer_type),
            "pos_enc":        _clean(best_cfg.pos_enc),
        },
        "hardware_profile": {
            "vram_total_gb":   round(best_cfg.vram_total_gb, 3),
            "vram_usage_pct":  round(best_cfg.vram_usage_pct, 2),
            "mfu_estimate":    round(best_cfg.mfu_estimate, 4),
            "tokens_per_sec":  best_cfg.tokens_per_sec_estimate,
            "ms_per_step":     round(best_cfg.ms_per_step, 2),
            "bottleneck":      best_cfg.bottleneck,
            "sm_occupancy":    round(best_cfg.sm_occupancy, 4),
            "fits_gpu":        best_cfg.fits_gpu,
        },
        "training_flags": {
            "use_flash_attn":             best_cfg.use_flash_attn,
            "use_mixed_precision":        best_cfg.use_mixed_precision,
            "use_gradient_checkpointing": best_cfg.use_gradient_checkpointing,
            "use_torch_compile":          best_cfg.use_torch_compile,
            "tie_embeddings":             best_cfg.tie_embeddings,
            "dropout":                    best_cfg.dropout,
        },
        "full_arch_dict": arch_dict,
    }

    # ── Compatibility analysis ────────────────────────────────────────────────
    compatibility_analysis = {
        "compatibility_tier":    compat,
        "synergy_multiplier":    round(synergy_m, 4),
        "n_way_complexity":      n_families,
        "strategy_used":         strategy,
        "is_strongly_valid":     (compat == "STRONGLY_VALID"),
        "risk_level": (
            "LOW"      if compat == "STRONGLY_VALID" else
            "MEDIUM"   if compat == "COMPATIBLE" else
            "HIGH"     if compat == "MARGINAL" else "VERY HIGH"
        ),
        "strategy_description": {
            "WEIGHTED":    "Semua layer menggunakan weighted blend — paling stabil",
            "STAGED":      "Setiap stage layer menggunakan family berbeda",
            "INTERLEAVED": "Setiap layer bergantian antar family — paling kompleks",
        }.get(strategy, "—"),
    }

    # ── Assemble final JSON ───────────────────────────────────────────────────
    result = {
        "export_type":              "multi_type",
        "generated_at":             datetime.datetime.now().isoformat(),
        "gpu":                      gpu.name,
        "pipeline_mode":            "combination_nas",
        "blend_info":               blend_info,
        "scoring_formula":          scoring_formula,
        "combination_sub_scores":   combination_sub_scores,
        "rl_optimization":          rl_optimization,
        "best_arch_detail":         best_arch_detail,
        "compatibility_analysis":   compatibility_analysis,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[dim]  ✓ Multi-type results exported → {filepath}[/dim]")
    except Exception as e:
        console.print(f"[yellow]  ⚠ Multi-type JSON export failed: {e}[/yellow]")


if __name__ == "__main__":
    main()
