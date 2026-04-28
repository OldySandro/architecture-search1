"""
custom_pipeline.py — Auto NAS Pipeline (Hardware + Training)
════════════════════════════════════════════════════════════════════════════════

Cara jalannya BERBEDA dari pipeline.py:
  • Config-first   : Semua parameter dikonfigurasi di awal (tidak ada prompt di tengah jalan)
  • Auto-run       : Setelah START, pipeline berjalan penuh tanpa interupsi
  • Batch-friendly : Bisa di-script, tidak butuh interaction manual
  • Clean JSON     : Hanya field penting yang disimpan (no noise)

UI:
  Hardware → keyword match (T4, A100, H100, RTX4090, ...)
  Type AI  → 1–7 pilih family
  Range    → jumlah total ARC (1 000 – 100 000)
  Profiling→ Y (auto real profiling jika GPU tersedia)
  Seed     → 42 (fixed) atau auto-random
  NAS      → Hardware NAS 7D + Training NAS 6D, auto run
  Save     → auto save clean JSON

Scoring: balanced 50/50 (same as pipeline.py)
  combined = (0.5 × hw_score + 0.5 × train_score) × quality_gate

JSON clean output (no noise):
  generated_at, gpu, scoring_system, weights, best_arch, architectures[]
  Per arch: arch_id, arch_family, param_count_M, num_layers, hidden_dim,
            hardware_score, training_score, combined_score, quality_pct,
            vram_usage_pct, mfu_estimate, tokens_per_sec, fits_gpu
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
from typing import Dict, List, Optional, Tuple

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
)
from hardware_refine import (
    HardwareNASRefiner, HardwareNASEvaluator,
    HardwareAdaptiveLog,
    print_hardware_adaptive_summary,
)
from train_refine    import (
    TrainingNASRefiner, TrainingDynamicsEvaluator, ProxyTrainer,
    TrainingAdaptiveLog,
    print_training_adaptive_summary,
)

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
    # T4
    "t4": "T4",
    # V100
    "v100-16": "V100-16GB", "v100 16": "V100-16GB",
    "v100-32": "V100-32GB", "v100 32": "V100-32GB",
    "v100":    "V100-32GB",  # default ke 32GB
    # RTX 3090
    "rtx3090": "RTX-3090", "rtx 3090": "RTX-3090", "3090": "RTX-3090",
    # A6000
    "a6000": "A6000", "rtxa6000": "A6000",
    # A100
    "a100-40": "A100-40GB", "a100 40": "A100-40GB",
    "a100-80": "A100-80GB", "a100 80": "A100-80GB",
    "a100":    "A100-80GB",  # default ke 80GB
    # RTX 4090
    "rtx4090": "RTX-4090", "rtx 4090": "RTX-4090", "4090": "RTX-4090",
    # H100
    "h100-pcie": "H100-PCIe", "h100 pcie": "H100-PCIe", "h100pcie": "H100-PCIe",
    "h100-sxm":  "H100-SXM",  "h100 sxm":  "H100-SXM",  "h100sxm":  "H100-SXM",
    "h100":      "H100-SXM",   # default ke SXM
    # H200
    "h200":     "H200-SXM", "h200-sxm": "H200-SXM", "h200 sxm": "H200-SXM",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """Semua parameter pipeline dikumpulkan di sini sebelum eksekusi."""

    def __init__(self):
        self.gpu:          GPUSpec       = None
        self.families:     List[str]     = []      # [] = all
        self.total_archs:  int           = 300    # Range
        self.n_per_family: int           = 0       # dihitung dari total_archs
        self.seed:         int           = 42
        self.run_profiling: bool         = True    # auto real profiling
        self.device:       str           = "cpu"   # untuk proxy training
        self.max_hw_iters: int           = 25
        self.max_tr_iters: int           = 25
        self.max_explore:  int           = 30
        self.output_file:  str           = "nas_results_custom.json"

    def finalize(self):
        """Hitung derived fields setelah semua input dikumpulkan."""
        fam_count = len(self.families) if self.families else 7
        self.n_per_family = max(1, self.total_archs // fam_count)
        # Auto-detect device
        if TORCH:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"


# ══════════════════════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _show_banner():
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]  CUSTOM AUTO NAS PIPELINE[/bold cyan]\n"
            "[dim]  Hardware NAS 7D + Training NAS 6D  ·  Balanced 50/50  ·  Auto-Run[/dim]\n"
            "[dim]  Config-First · No Mid-Run Interruptions · Clean JSON Output[/dim]",
            border_style="cyan", padding=(1, 4)
        ))
    else:
        print("=" * 70)
        print("  CUSTOM AUTO NAS PIPELINE")
        print("  Hardware NAS + Training NAS · Balanced 50/50 · Auto-Run")
        print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  UI — GPU SELECTION (keyword)
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

        # Coba exact key match dulu
        for db_key in GPU_DATABASE:
            if raw.upper() == db_key.upper() or raw == db_key.lower():
                gpu = GPU_DATABASE[db_key]
                console.print(f"\n[bold green]✓ GPU: {gpu.name}[/bold green]\n")
                return gpu

        # Coba keyword map
        matched_key = _GPU_KEYWORDS.get(raw)
        if matched_key and matched_key in GPU_DATABASE:
            gpu = GPU_DATABASE[matched_key]
            console.print(f"\n[bold green]✓ GPU: {gpu.name}[/bold green]\n")
            return gpu

        # Coba partial match
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
            raw = input("Type AI : ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return []

        if not raw or raw == "all" or raw == "7":
            console.print("[bold green]✓ Type AI: ALL (7 families)[/bold green]\n")
            return []

        parts = [p.strip() for p in raw.replace(" ", "").split(",") if p.strip()]
        chosen = []
        invalid = []
        for p in parts:
            try:
                idx = int(p) - 1
                if 0 <= idx < len(_FAMILIES):
                    chosen.append(_FAMILIES[idx][0])
                else:
                    invalid.append(p)
            except ValueError:
                # Coba nama langsung
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
    console.print("[dim]  Berapa total ARC yang akan digenerate? (min 1000, max 100000)[/dim]")
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
            if val < 300:
                console.print(f"[yellow]  Minimum 300. Set ke 300.[/yellow]\n")
                return 300
            if val > 100000:
                console.print(f"[yellow]  Maximum 100000. Set ke 100000.[/yellow]\n")
                return 100000
            console.print(f"[bold green]✓ Range: {val:,} architectures[/bold green]\n")
            return val
        except ValueError:
            console.print(f"[red]  Masukkan angka valid (contoh: 1000, 5000, 10000)[/red]")


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
    elif raw == "auto" or raw == "random":
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
    cfg.gpu      = _select_gpu()
    cfg.families = _select_ai_type()
    cfg.total_archs = _select_range()
    cfg.seed     = _select_seed()

    # Auto profiling — selalu Y, hanya aktif jika GPU tersedia
    cfg.run_profiling = True

    cfg.finalize()

    # Tampilkan summary konfigurasi
    fam_str = ", ".join(cfg.families) if cfg.families else "ALL (7 families)"
    prof_str = "Yes (GPU tersedia)" if (TORCH and cfg.device == "cuda") else "Yes (analytical fallback)"

    if RICH:
        console.print(Panel(
            f"  [bold]GPU:[/bold]           {cfg.gpu.name}\n"
            f"  [bold]Type AI:[/bold]       {fam_str}\n"
            f"  [bold]Range:[/bold]         {cfg.total_archs:,} archs total "
            f"(~{cfg.n_per_family} per family)\n"
            f"  [bold]Seed:[/bold]          {cfg.seed}\n"
            f"  [bold]Auto Profiling:[/bold] {prof_str}\n"
            f"  [bold]Auto NAS HW+TR:[/bold] Yes (Hardware 7D + Training 6D)\n"
            f"  [bold]Auto Save:[/bold]     Yes → {cfg.output_file}\n"
            f"  [bold]Device:[/bold]        {cfg.device}",
            title="[bold]Pipeline Configuration[/bold]",
            border_style="cyan", padding=(0, 2)
        ))
    else:
        print(f"\n  GPU          : {cfg.gpu.name}")
        print(f"  Type AI      : {fam_str}")
        print(f"  Range        : {cfg.total_archs:,} archs (~{cfg.n_per_family}/family)")
        print(f"  Seed         : {cfg.seed}")
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
#  STAGE 1: GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def stage_generate(cfg: PipelineConfig) -> List[ArchConfig]:
    gen          = ArchitectureGenerator(cfg.gpu, rng_seed=cfg.seed)
    all_families = list(ArchitectureGenerator.FAMILY_PROFILES.keys())
    selected     = ([f for f in cfg.families if f in all_families]
                    if cfg.families else all_families)

    console.rule("[bold cyan]  Stage 1 — Generating Architectures  ")
    total = len(selected) * cfg.n_per_family

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task  = prog.add_task("[cyan]Generating...", total=total)
            archs = []
            for fam in selected:
                for _ in range(cfg.n_per_family):
                    a = gen.generate_one(fam)
                    archs.append(a)
                    prog.update(task, advance=1,
                                description=f"[cyan]{fam[:20]} {a.arch_id}")
    else:
        archs = []
        for fam in selected:
            for i in range(cfg.n_per_family):
                archs.append(gen.generate_one(fam))

    archs.sort(key=lambda x: x.fitness_score, reverse=True)
    console.print(f"\n[green]✓ Generated {len(archs)} architectures "
                  f"across {len(selected)} families[/green]\n")
    return archs


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: HARDWARE NAS
# ══════════════════════════════════════════════════════════════════════════════

def stage_hardware_nas(
    archs: List[ArchConfig],
    cfg:   PipelineConfig,
) -> Tuple[List[ArchConfig], List[HardwareAdaptiveLog], Dict[str, float], Dict[str, float]]:

    console.rule("[bold blue]  Stage 2 — Hardware NAS + RL  ")
    console.print(
        f"[dim]  GPU: {cfg.gpu.name}  |  "
        "7 dimensi: H1-MFU(25)+H2-Thru(20)+H3-VRAM(15)+H4-TC(15)"
        "+H5-SM(10)+H6-Comp(10)+H7-FA(5)  |  "
        "RL Actions: ALIGN_HIDDEN/HEAD/FFN, BATCH, FA, COMPILE, GC, OPT[/dim]\n"
    )

    refiner = HardwareNASRefiner(
        cfg.gpu,
        max_iterations    = cfg.max_hw_iters,
        target_pct        = 100.0,
        max_explore_iters = cfg.max_explore,
        rng_seed          = cfg.seed,
    )
    scorer = ArcQualityScorer(cfg.gpu)

    refined_archs: List[ArchConfig]          = []
    hw_logs:       List[HardwareAdaptiveLog]  = []
    hw_score_map:  Dict[str, float]           = {}
    quality_map:   Dict[str, float]           = {}

    total = len(archs)
    t0    = time.perf_counter()

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[blue]Hardware NAS...", total=total)
            for a in archs:
                prog.update(task, description=f"[blue]HW-NAS {a.arch_id}…")
                refined, alog = refiner.refine(a)
                q = scorer.score(refined).pct
                quality_map[refined.arch_id]  = q
                hw_score_map[refined.arch_id] = alog.hw_score_end
                refined_archs.append(refined)
                hw_logs.append(alog)
                prog.advance(task)
    else:
        for i, a in enumerate(archs, 1):
            refined, alog = refiner.refine(a)
            q = scorer.score(refined).pct
            quality_map[refined.arch_id]  = q
            hw_score_map[refined.arch_id] = alog.hw_score_end
            refined_archs.append(refined)
            hw_logs.append(alog)
            if i % max(1, total // 10) == 0:
                print(f"  HW-NAS [{i}/{total}] hw={alog.hw_score_end:.3f} q={q:.1f}%")

    elapsed = time.perf_counter() - t0
    n_improved = sum(1 for l in hw_logs if l.perturbations_accepted > 0)
    hw_mean    = sum(hw_score_map.values()) / max(1, len(hw_score_map))
    console.print(
        f"[blue]✓ Hardware NAS — {n_improved}/{total} improved  "
        f"avg hw={hw_mean:.4f}  {elapsed:.1f}s[/blue]\n"
    )

    print_hardware_adaptive_summary(hw_logs, hw_score_map, console=console)

    refined_archs.sort(key=lambda a: hw_score_map.get(a.arch_id, 0.0), reverse=True)
    return refined_archs, hw_logs, hw_score_map, quality_map


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: TRAINING NAS
# ══════════════════════════════════════════════════════════════════════════════

def stage_training_nas(
    archs:     List[ArchConfig],
    cfg:       PipelineConfig,
    hw_scores: Dict[str, float],
) -> Tuple[List[ArchConfig], List[TrainingAdaptiveLog], Dict[str, float], Dict]:

    console.rule("[bold magenta]  Stage 3 — Training NAS + RL  ")
    console.print(
        "[dim]  6 dimensi: T1-Conv(22)+T2-Stab(22)+T3-Grad(18)"
        "+T4-GenGap(15)+T5-SampEff(13)+T6-OptComp(10)  |  "
        f"Proxy: {50} steps real PyTorch  |  Device: {cfg.device}[/dim]\n"
    )

    refiner = TrainingNASRefiner(
        cfg.gpu,
        max_iterations    = cfg.max_tr_iters,
        target_pct        = 100.0,
        max_explore_iters = cfg.max_explore,
        rng_seed          = cfg.seed,
        device            = cfg.device,
    )

    refined_archs:   List[ArchConfig]          = []
    train_logs:      List[TrainingAdaptiveLog]  = []
    train_map:       Dict[str, float]           = {}
    nas_results_map: Dict[str, object]          = {}

    total = len(archs)
    t0    = time.perf_counter()

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[magenta]Training NAS...", total=total)
            for a in archs:
                prog.update(task, description=f"[magenta]TR-NAS {a.arch_id}…")
                hw = hw_scores.get(a.arch_id, 0.0)
                refined, alog = refiner.refine(a, hw_score=hw)
                train_map[refined.arch_id] = alog.train_score_end
                final_nas = refiner._evaluate_cached(refined)
                nas_results_map[refined.arch_id] = final_nas
                refined_archs.append(refined)
                train_logs.append(alog)
                prog.advance(task)
    else:
        for i, a in enumerate(archs, 1):
            hw = hw_scores.get(a.arch_id, 0.0)
            refined, alog = refiner.refine(a, hw_score=hw)
            train_map[refined.arch_id] = alog.train_score_end
            final_nas = refiner._evaluate_cached(refined)
            nas_results_map[refined.arch_id] = final_nas
            refined_archs.append(refined)
            train_logs.append(alog)
            if i % max(1, total // 10) == 0:
                print(f"  TR-NAS [{i}/{total}] ts={alog.train_score_end:.3f} NaN={alog.nas_nan_count}")

    elapsed = time.perf_counter() - t0
    n_improved = sum(1 for l in train_logs if l.perturbations_accepted > 0)
    ts_mean    = sum(train_map.values()) / max(1, len(train_map))
    console.print(
        f"[magenta]✓ Training NAS — {n_improved}/{total} improved  "
        f"avg ts={ts_mean:.4f}  {elapsed:.1f}s[/magenta]\n"
    )

    print_training_adaptive_summary(train_logs, train_map, console=console)

    refined_archs.sort(key=lambda a: train_map.get(a.arch_id, 0.0), reverse=True)
    return refined_archs, train_logs, train_map, nas_results_map


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: BALANCED SCORING + FINAL RANKING
# ══════════════════════════════════════════════════════════════════════════════

def stage_balanced_scoring(
    archs:        List[ArchConfig],
    quality_map:  Dict[str, float],
    hw_scores:    Dict[str, float],
    train_scores: Dict[str, float],
) -> Tuple[List[ArchConfig], Dict[str, float], ArchConfig]:

    console.rule("[bold green]  Stage 4 — Balanced 50/50 Scoring  ")

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

    best = next((a for a in archs_sorted if a.fits_gpu), None)
    if best is None:
        best = archs_sorted[0] if archs_sorted else archs[0]

    # Print top-10 ranking
    if RICH:
        t = Table(title="Top-10 Final Ranking (Balanced 50/50)",
                  show_header=True, header_style="bold green",
                  box=rbox.SIMPLE_HEAD, padding=(0, 1))
        t.add_column("Rank", style="dim",     width=5)
        t.add_column("ARC-ID",  style="cyan",    width=10)
        t.add_column("Family",  style="yellow",  width=16)
        t.add_column("Params",  style="white",   width=7)
        t.add_column("Quality", style="green",   width=8)
        t.add_column("HW",      style="blue",    width=8)
        t.add_column("Train",   style="magenta", width=8)
        t.add_column("Combined",style="bold",    width=9)
        t.add_column("VRAM%",   style="cyan",    width=6)

        for rank, a in enumerate(archs_sorted[:10], 1):
            sym = "★" if rank == 1 else f"#{rank}"
            p   = (f"{a.param_count/1e6:.0f}M" if a.param_count < 1e9
                   else f"{a.param_count/1e9:.2f}B")
            t.add_row(
                sym, a.arch_id, a.arch_family[:16], p,
                f"{quality_map.get(a.arch_id, 0):.1f}%",
                f"{hw_scores.get(a.arch_id, 0):.4f}",
                f"{train_scores.get(a.arch_id, 0):.4f}",
                f"{combined_map.get(a.arch_id, 0):.5f}",
                f"{a.vram_usage_pct:.1f}%",
            )
        console.print(t)
    else:
        print(f"\n  {'Rank':<5} {'ARC-ID':<10} {'Combined':<10} {'HW':<8} {'Train':<8}")
        print("  " + "─" * 50)
        for rank, a in enumerate(archs_sorted[:10], 1):
            sym = "★" if rank == 1 else f"#{rank}"
            print(f"  {sym:<5} {a.arch_id:<10} "
                  f"{combined_map.get(a.arch_id, 0):.5f}  "
                  f"{hw_scores.get(a.arch_id, 0):.4f}  "
                  f"{train_scores.get(a.arch_id, 0):.4f}")

    console.print(
        f"\n[bold green]★ Best: {best.arch_id}  "
        f"combined={combined_map.get(best.arch_id, 0):.5f}  "
        f"hw={hw_scores.get(best.arch_id, 0):.4f}  "
        f"train={train_scores.get(best.arch_id, 0):.4f}  "
        f"quality={quality_map.get(best.arch_id, 0):.1f}%[/bold green]\n"
    )

    return archs_sorted, combined_map, best


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5: PROFILING (top arch)
# ══════════════════════════════════════════════════════════════════════════════

def stage_profile_top(
    best:     ArchConfig,
    cfg:      PipelineConfig,
) -> dict:
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
#  STAGE 6: CLEAN JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _clean_arch_entry(
    a:             ArchConfig,
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    combined_map:  Dict[str, float],
) -> dict:
    """Hanya field penting — no noise."""

    def _enum_str(val) -> str:
        if val is None: return "—"
        if hasattr(val, "value"): return str(val.value)
        s = str(val)
        return s.split(".")[-1] if "." in s else s

    return {
        "arch_id":        a.arch_id,
        "arch_family":    a.arch_family,
        "param_count_M":  round(a.param_count / 1e6, 2),
        "num_layers":     a.num_layers,
        "hidden_dim":     a.hidden_dim,
        "num_heads":      a.num_heads,
        "num_kv_heads":   a.num_kv_heads,
        "head_dim":       a.head_dim,
        "ffn_type":       _enum_str(a.ffn_type),
        "attn_type":      _enum_str(a.attn_type),
        "seq_len":        a.seq_len,
        "batch_size":     a.batch_size,
        "hardware_score": round(hw_scores.get(a.arch_id, 0.0), 4),
        "training_score": round(train_scores.get(a.arch_id, 0.0), 4),
        "combined_score": round(combined_map.get(a.arch_id, 0.0), 5),
        "quality_pct":    round(quality_map.get(a.arch_id, 0.0), 2),
        "vram_total_gb":  round(a.vram_total_gb, 3),
        "vram_usage_pct": round(a.vram_usage_pct, 2),
        "mfu_estimate":   round(a.mfu_estimate, 4),
        "tokens_per_sec": a.tokens_per_sec_estimate,
        "ms_per_step":    round(a.ms_per_step, 2),
        "fits_gpu":       a.fits_gpu,
        "bottleneck":     a.bottleneck,
        "use_flash_attn": a.use_flash_attn,
        "optimizer_type": _enum_str(a.optimizer_type),
    }


def stage_export(
    archs_sorted:  List[ArchConfig],
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    combined_map:  Dict[str, float],
    best:          ArchConfig,
    cfg:           PipelineConfig,
) -> str:

    console.rule("[bold dim]  Stage 6 — Auto Save Results  ")

    result = {
        "generated_at": datetime.datetime.now().isoformat(),
        "gpu":          cfg.gpu.name,
        "scoring_system": "balanced_50_50",
        "weights": {
            "hardware_score": 0.5,
            "training_score": 0.5,
            "quality_gate":   "0.80-1.00 multiplier",
        },
        "pipeline": {
            "total_generated":   len(archs_sorted),
            "families_used":     cfg.families if cfg.families else "ALL",
            "seed":              cfg.seed,
            "n_per_family":      cfg.n_per_family,
        },
        "best_arch": best.arch_id,
        "architectures": [
            _clean_arch_entry(a, quality_map, hw_scores, train_scores, combined_map)
            for a in archs_sorted
        ],
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
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _print_final_summary(
    best:          ArchConfig,
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    combined_map:  Dict[str, float],
    archs_sorted:  List[ArchConfig],
    output_file:   str,
    elapsed_total: float,
    cfg:           PipelineConfig,
):
    def _enum(v):
        if hasattr(v, "value"): return v.value
        return str(v).split(".")[-1]

    aid   = best.arch_id
    q     = quality_map.get(aid, 0.0)
    hw    = hw_scores.get(aid, 0.0)
    tr    = train_scores.get(aid, 0.0)
    comb  = combined_map.get(aid, 0.0)
    p_str = (f"{best.param_count/1e9:.3f}B" if best.param_count >= 1e9
             else f"{best.param_count/1e6:.0f}M")

    console.rule("[bold green]  ★ FINAL RECOMMENDATION  ")

    if RICH:
        W = 80
        console.print(f"╭{'─'*W}╮")
        console.print(f"│{'  🏆 CUSTOM AUTO NAS — FINAL RESULT':^{W}}│")
        console.print(f"├{'─'*W}┤")
        console.print(f"│  Best Arch    : {aid}  |  {best.arch_family:<30}│")
        console.print(f"│  Combined     : {comb:.5f}  "
                      f"(HW={hw:.4f} ×50%  +  Train={tr:.4f} ×50%)  │")
        console.print(f"│  Quality Gate : {q:.1f}%  "
                      f"(×{0.80 + 0.20*max(0,(q-70)/30):.3f} multiplier)          │")
        console.print(f"├{'─'*W}┤")
        console.print(f"│  Params  : {p_str:<8}  "
                      f"Layers={best.num_layers}  "
                      f"Hidden={best.hidden_dim}  "
                      f"Heads={best.num_heads}/{best.num_kv_heads}  "
                      f"HeadDim={best.head_dim}   │")
        console.print(f"│  FFN     : {_enum(best.ffn_type):<14}  "
                      f"Attn: {_enum(best.attn_type):<20}  "
                      f"Seq={best.seq_len}   │")
        console.print(f"│  VRAM    : {best.vram_total_gb:.2f}GB ({best.vram_usage_pct:.1f}%)  "
                      f"MFU={best.mfu_estimate:.4f}  "
                      f"Tok/s={best.tokens_per_sec_estimate:,}            │")
        console.print(f"│  Bottleneck: {best.bottleneck:<52}│")
        console.print(f"├{'─'*W}┤")
        console.print(f"│  Total run time : {elapsed_total:.1f}s  "
                      f"  Total archs : {len(archs_sorted):,}  "
                      f"  Saved → {output_file:<14}│")
        console.print(f"╰{'─'*W}╯")
    else:
        print("\n" + "═" * 60)
        print(f"  BEST: {aid}  ({best.arch_family})")
        print(f"  Combined={comb:.5f}  HW={hw:.4f}  Train={tr:.4f}  Q={q:.1f}%")
        print(f"  Params={p_str}  VRAM={best.vram_usage_pct:.1f}%  MFU={best.mfu_estimate:.4f}")
        print(f"  Saved → {output_file}")
        print("=" * 60)

    # Runner-up
    if len(archs_sorted) > 1:
        console.print("\n  [dim]Runner-up:[/dim]")
        for i, a in enumerate(archs_sorted[1:4], 2):
            console.print(
                f"  [dim]  #{i}  {a.arch_id:<10}  "
                f"combined={combined_map.get(a.arch_id, 0):.5f}  "
                f"hw={hw_scores.get(a.arch_id, 0):.4f}  "
                f"train={train_scores.get(a.arch_id, 0):.4f}[/dim]"
            )
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    # ── Collect config ────────────────────────────────────────────────────────
    cfg = collect_config()

    # ── Stage 1: Generate ────────────────────────────────────────────────────
    archs = stage_generate(cfg)

    if not archs:
        console.print("[red]  Tidak ada arsitektur berhasil digenerate.[/red]")
        return

    # Pre-NAS rank untuk referensi
    pre_nas_rank = {a.arch_id: i for i, a in enumerate(archs, 1)}

    # ── Stage 2: Hardware NAS ─────────────────────────────────────────────────
    hw_archs, hw_logs, hw_scores, quality_map = stage_hardware_nas(archs, cfg)

    # ── Stage 3: Training NAS ─────────────────────────────────────────────────
    tr_archs, tr_logs, train_scores, nas_results_map = stage_training_nas(
        hw_archs, cfg, hw_scores
    )

    # ── Stage 4: Balanced scoring ─────────────────────────────────────────────
    archs_sorted, combined_map, best = stage_balanced_scoring(
        tr_archs, quality_map, hw_scores, train_scores
    )

    # ── Stage 5: Profile best arch ────────────────────────────────────────────
    pr = stage_profile_top(best, cfg)

    # ── Stage 6: Save clean JSON ──────────────────────────────────────────────
    output_file = stage_export(
        archs_sorted, quality_map, hw_scores, train_scores, combined_map, best, cfg
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    _print_final_summary(
        best, quality_map, hw_scores, train_scores, combined_map,
        archs_sorted, output_file, elapsed_total, cfg,
    )


if __name__ == "__main__":
    main()
