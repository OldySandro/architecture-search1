"""
pipeline.py — Auto NAS Pipeline (Hardware + Training) ·  PARALLEL EDITION
════════════════════════════════════════════════════════════════════════════════

Paralel:
  • Stage 1–2 (Generate + HW-NAS) dibagi ke N_WORKERS worker
  • Stage 3 (Training NAS) berjalan SERIAL di main process — CUDA-safe
    (torch + CUDA TIDAK fork-safe; TrainingNASRefiner butuh state PyTorch penuh)
  • Hasil semua worker digabung lalu diranking bersama (Stage 4-6 tetap serial)

FIX: Training NAS dipindah dari subprocess ke main process untuk menghindari:
  1. torch.cuda tidak fork-safe (CUDA context corrupt di child)
  2. ProxyTrainer membutuhkan state PyTorch yang tidak bisa di-pickle
  3. Deadlock multiprocessing saat CUDA dipakai di fork

Cara jalannya:
  • Config-first   : Semua parameter dikonfigurasi di awal
  • Auto-run       : Setelah START, pipeline berjalan penuh tanpa interupsi
  • Batch-friendly : Bisa di-script
  • Clean JSON     : Semua field penting disimpan (format lengkap)

Output JSON per arsitektur:
  → Format lengkap sesuai contoh (arch_id, arch_name, semua VRAM, FLOPS,
    bottleneck_factors, quality_score_pct, hardware_score, training_score,
    combined_score, semua enum sebagai ClassName.MemberName)

Scoring: balanced 50/50:
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
from arch_types      import (ArchConfig, AttentionType, FFNType, OptimizerType,
                              NormType, PosEncType)
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
#  UI — HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _show_banner():
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]  CUSTOM AUTO NAS PIPELINE  [PARALLEL EDITION][/bold cyan]\n"
            "[dim]  Hardware NAS 7D + Training NAS 6D  ·  Balanced 50/50  ·  Auto-Run[/dim]\n"
            "[dim]  Stage 1-2 Parallel · Stage 3 Serial (CUDA-safe) · Clean JSON Output[/dim]",
            border_style="cyan", padding=(1, 4)
        ))
    else:
        print("=" * 70)
        print("  CUSTOM AUTO NAS PIPELINE  [PARALLEL EDITION]")
        print("  Hardware NAS + Training NAS · Balanced 50/50 · Stage 1-2 Parallel")
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
        f"[dim]  Jumlah worker paralel (Stage 1-2: Generate+HW-NAS)? "
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
            f"  [bold]Workers:[/bold]        {cfg.n_workers} paralel Stage 1-2  "
            f"(~{batch_sizes[0]}–{batch_sizes[-1]} ARC per worker)\n"
            f"  [bold]Seed:[/bold]           {cfg.seed} (base, +worker_idx per worker)\n"
            f"  [bold]Auto Profiling:[/bold] {prof_str}\n"
            f"  [bold]HW-NAS:[/bold]         Parallel (subprocess, CUDA-free)\n"
            f"  [bold]Training-NAS:[/bold]   Serial main process (CUDA-safe)\n"
            f"  [bold]Auto Save:[/bold]      Yes → {cfg.output_file}\n"
            f"  [bold]Device:[/bold]         {cfg.device}",
            title="[bold]Pipeline Configuration[/bold]",
            border_style="cyan", padding=(0, 2)
        ))
    else:
        print(f"\n  GPU          : {cfg.gpu.name}")
        print(f"  Type AI      : {fam_str}")
        print(f"  Range        : {cfg.total_archs:,} archs (~{cfg.n_per_family}/family)")
        print(f"  Workers      : {cfg.n_workers} (Stage 1-2 parallel)")
        print(f"  Seed         : {cfg.seed} (base)")
        print(f"  HW-NAS       : Parallel (subprocess)")
        print(f"  Training-NAS : Serial main process (CUDA-safe)")
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


# ── WORKER: hanya Generate + HW-NAS (tanpa torch/CUDA — subprocess-safe) ─────

def _worker_hw_pipeline(
    worker_idx:    int,
    n_archs:       int,
    gpu_key:       str,
    families:      List[str],
    seed:          int,
    max_hw_iters:  int,
    max_explore:   int,
) -> dict:
    """
    Worker subprocess: Generate + HW-NAS saja.

    TIDAK menjalankan Training NAS karena:
      - torch.cuda tidak fork-safe (CUDA context corrupt di child process)
      - ProxyTrainer membutuhkan state PyTorch yang tidak bisa di-pickle
      - Training NAS dijalankan di main process secara serial setelah stage ini

    Returns dict berisi archs + hw_score_map + quality_map (semua serializable).
    """
    import sys, os
    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

    from hardware        import GPU_DATABASE
    from arch_types      import ArchConfig, FFNType, AttentionType, OptimizerType, NormType, PosEncType
    from generator       import ArchitectureGenerator
    from refiner         import ArcQualityScorer
    from hardware_refine import HardwareNASRefiner

    gpu = GPU_DATABASE[gpu_key]

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

    archs.sort(key=lambda x: x.fitness_score, reverse=True)

    # ── Hardware NAS ──────────────────────────────────────────────────────────
    hw_refiner = HardwareNASRefiner(
        gpu,
        max_iterations    = max_hw_iters,
        target_pct        = 100.0,
        max_explore_iters = max_explore,
        rng_seed          = seed,
    )
    scorer = ArcQualityScorer(gpu)

    hw_archs:     list = []
    hw_score_map: dict = {}
    quality_map:  dict = {}

    for a in archs:
        refined, alog = hw_refiner.refine(a)
        q = scorer.score(refined).pct
        quality_map[refined.arch_id]  = q
        hw_score_map[refined.arch_id] = alog.hw_score_end
        hw_archs.append(refined)

    # ── Serialisasi LENGKAP semua field ArchConfig ────────────────────────────
    def _arch_to_dict(a: ArchConfig) -> dict:
        def _ev(v):
            # Serialize enum sebagai "ClassName.MEMBER" untuk deserialisasi mudah
            if hasattr(v, "name"):
                return f"{type(v).__name__}.{v.name}"
            return str(v)

        return {
            "_arch_obj":              True,
            # Identity
            "arch_id":                a.arch_id,
            "arch_name":              getattr(a, "arch_name", ""),
            "arch_family":            a.arch_family,
            # Core dims
            "vocab_size":             getattr(a, "vocab_size", 32000),
            "hidden_dim":             a.hidden_dim,
            "num_layers":             a.num_layers,
            "seq_len":                a.seq_len,
            "batch_size":             a.batch_size,
            # Attention
            "attn_type":              _ev(a.attn_type),
            "num_heads":              a.num_heads,
            "num_kv_heads":           a.num_kv_heads,
            "head_dim":               a.head_dim,
            "window_size":            getattr(a, "window_size", 512),
            "global_attn_layers":     getattr(a, "global_attn_layers", 4),
            # FFN
            "ffn_type":               _ev(a.ffn_type),
            "ffn_multiplier":         getattr(a, "ffn_multiplier", 4.0),
            "num_experts":            getattr(a, "num_experts", 1),
            "top_k_experts":          getattr(a, "top_k_experts", 1),
            "expert_capacity_factor": getattr(a, "expert_capacity_factor", 1.25),
            # Optimizer / misc
            "optimizer_type":         _ev(a.optimizer_type),
            "norm_type":              _ev(getattr(a, "norm_type", NormType.RMSNORM)),
            "pos_enc":                _ev(getattr(a, "pos_enc", PosEncType.ROPE)),
            "tie_embeddings":         getattr(a, "tie_embeddings", True),
            "use_flash_attn":         a.use_flash_attn,
            "use_gradient_checkpointing": getattr(a, "use_gradient_checkpointing", False),
            "use_mixed_precision":    getattr(a, "use_mixed_precision", True),
            "use_torch_compile":      getattr(a, "use_torch_compile", False),
            "dropout":                getattr(a, "dropout", 0.0),
            # Params & VRAM
            "param_count":            a.param_count,
            "vram_weights_gb":        getattr(a, "vram_weights_gb", 0.0),
            "vram_activations_gb":    getattr(a, "vram_activations_gb", 0.0),
            "vram_optimizer_gb":      getattr(a, "vram_optimizer_gb", 0.0),
            "vram_kv_cache_gb":       getattr(a, "vram_kv_cache_gb", 0.0),
            "vram_fragmentation_gb":  getattr(a, "vram_fragmentation_gb", 0.0),
            "vram_total_gb":          a.vram_total_gb,
            "vram_usage_pct":         a.vram_usage_pct,
            # FLOPS
            "flops_per_token_fwd":    getattr(a, "flops_per_token_fwd", 0.0),
            "flops_per_token_bwd":    getattr(a, "flops_per_token_bwd", 0.0),
            "flops_attn_fwd":         getattr(a, "flops_attn_fwd", 0.0),
            "flops_ffn_fwd":          getattr(a, "flops_ffn_fwd", 0.0),
            "arithmetic_intensity":   getattr(a, "arithmetic_intensity", 0.0),
            # Profiling
            "tokens_per_sec_estimate": a.tokens_per_sec_estimate,
            "mfu_estimate":           a.mfu_estimate,
            "ms_per_step":            a.ms_per_step,
            "bottleneck":             a.bottleneck,
            "bottleneck_factors":     getattr(a, "bottleneck_factors", {}),
            "fits_gpu":               a.fits_gpu,
            "fitness_score":          a.fitness_score,
            "compiler_speedup":       getattr(a, "compiler_speedup", 1.0),
            "warp_divergence_pct":    getattr(a, "warp_divergence_pct", 0.0),
            "sm_occupancy":           getattr(a, "sm_occupancy", 0.0),
        }

    return {
        "worker_idx":    worker_idx,
        "n_generated":   len(hw_archs),
        "archs":         [_arch_to_dict(a) for a in hw_archs],
        "hw_score_map":  hw_score_map,
        "quality_map":   quality_map,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DESERIALISASI ArchConfig dari dict worker
# ══════════════════════════════════════════════════════════════════════════════

def _dict_to_arch(d: dict) -> ArchConfig:
    """
    Rekonstruksi ArchConfig LENGKAP dari dict hasil serialisasi worker.
    Mendukung format enum 'ClassName.MEMBER' maupun value string.
    """
    def _parse_enum(cls, val, default=None):
        if val is None:
            return default if default is not None else list(cls)[0]
        # Format "ClassName.MEMBER" → ambil bagian nama member
        if isinstance(val, str) and "." in val:
            member_name = val.split(".")[-1]
            try:
                return cls[member_name]
            except KeyError:
                pass
        # Coba by value
        try:
            return cls(val)
        except Exception:
            pass
        # Coba by name langsung
        for m in cls:
            if m.name == val or m.value == val:
                return m
        return default if default is not None else list(cls)[0]

    a = ArchConfig()

    # Identity
    a.arch_id           = d["arch_id"]
    a.arch_name         = d.get("arch_name", "")
    a.arch_family       = d["arch_family"]
    # Core dims
    a.vocab_size        = d.get("vocab_size", 32000)
    a.hidden_dim        = d["hidden_dim"]
    a.num_layers        = d["num_layers"]
    a.seq_len           = d["seq_len"]
    a.batch_size        = d["batch_size"]
    # Attention
    a.attn_type         = _parse_enum(AttentionType, d.get("attn_type"),
                                      AttentionType.GQA)
    a.num_heads         = d["num_heads"]
    a.num_kv_heads      = d["num_kv_heads"]
    a.head_dim          = d["head_dim"]
    a.window_size       = d.get("window_size", 512)
    a.global_attn_layers = d.get("global_attn_layers", 4)
    # FFN
    a.ffn_type          = _parse_enum(FFNType, d.get("ffn_type"),
                                      FFNType.DENSE)
    a.ffn_multiplier    = d.get("ffn_multiplier", 4.0)
    a.num_experts       = d.get("num_experts", 1)
    a.top_k_experts     = d.get("top_k_experts", 1)
    a.expert_capacity_factor = d.get("expert_capacity_factor", 1.25)
    # Optimizer / misc
    a.optimizer_type    = _parse_enum(OptimizerType, d.get("optimizer_type"),
                                      OptimizerType.ADAM_FP32)
    a.norm_type         = _parse_enum(NormType, d.get("norm_type"),
                                      NormType.RMSNORM)
    a.pos_enc           = _parse_enum(PosEncType, d.get("pos_enc"),
                                      PosEncType.ROPE)
    a.tie_embeddings    = d.get("tie_embeddings", True)
    a.use_flash_attn    = d.get("use_flash_attn", True)
    a.use_gradient_checkpointing = d.get("use_gradient_checkpointing", False)
    a.use_mixed_precision = d.get("use_mixed_precision", True)
    a.use_torch_compile = d.get("use_torch_compile", False)
    a.dropout           = d.get("dropout", 0.0)
    # Params & VRAM
    a.param_count           = d["param_count"]
    a.vram_weights_gb       = d.get("vram_weights_gb", 0.0)
    a.vram_activations_gb   = d.get("vram_activations_gb", 0.0)
    a.vram_optimizer_gb     = d.get("vram_optimizer_gb", 0.0)
    a.vram_kv_cache_gb      = d.get("vram_kv_cache_gb", 0.0)
    a.vram_fragmentation_gb = d.get("vram_fragmentation_gb", 0.0)
    a.vram_total_gb         = d["vram_total_gb"]
    a.vram_usage_pct        = d["vram_usage_pct"]
    # FLOPS
    a.flops_per_token_fwd   = d.get("flops_per_token_fwd", 0.0)
    a.flops_per_token_bwd   = d.get("flops_per_token_bwd", 0.0)
    a.flops_attn_fwd        = d.get("flops_attn_fwd", 0.0)
    a.flops_ffn_fwd         = d.get("flops_ffn_fwd", 0.0)
    a.arithmetic_intensity  = d.get("arithmetic_intensity", 0.0)
    # Profiling
    a.tokens_per_sec_estimate = d["tokens_per_sec_estimate"]
    a.mfu_estimate          = d["mfu_estimate"]
    a.ms_per_step           = d["ms_per_step"]
    a.bottleneck            = d["bottleneck"]
    a.bottleneck_factors    = d.get("bottleneck_factors", {})
    a.fits_gpu              = d["fits_gpu"]
    a.fitness_score         = d["fitness_score"]
    a.compiler_speedup      = d.get("compiler_speedup", 1.0)
    a.warp_divergence_pct   = d.get("warp_divergence_pct", 0.0)
    a.sm_occupancy          = d.get("sm_occupancy", 0.0)

    return a


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1–2: PARALLEL GENERATE + HW-NAS
#  STAGE 3:   SERIAL TRAINING NAS (main process, CUDA-safe)
# ══════════════════════════════════════════════════════════════════════════════

def stage_parallel(cfg: PipelineConfig):
    """
    Stage 1–2 (Generate + HW-NAS) berjalan PARALEL di subprocess.
    Stage 3 (Training NAS) berjalan SERIAL di main process.

    Alasan pemisahan:
      - HW-NAS hanya butuh numpy/math → aman di fork
      - TrainingNASRefiner pakai torch.nn + CUDA → TIDAK aman di fork
        (CUDA context corrupt, ProxyTrainer tidak picklable)
    """
    # ── Stage 1-2: Parallel Generate + HW-NAS ─────────────────────────────────
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
            f"HW-Workers: [bold]{n_actual_workers}[/bold]  "
            f"Batch: ~[bold]{batch_sizes[0]}[/bold] ARC/worker  "
            f"GPU: [bold]{cfg.gpu.name}[/bold][/cyan]\n"
        )
        console.print(
            "[dim]  Subprocess: Generate → HW-NAS 7D  "
            "(Training NAS di main process setelah ini)[/dim]\n"
        )
    else:
        print(f"  Total ARC: {cfg.total_archs}  HW-Workers: {n_actual_workers}  "
              f"Batch: ~{batch_sizes[0]}/worker")

    t0 = time.perf_counter()

    hw_archs_dict:   List[dict]       = []   # dict mentah dari worker
    all_hw_scores:   Dict[str, float] = {}
    all_quality:     Dict[str, float] = {}
    worker_results   = {}
    worker_errors    = {}

    with ProcessPoolExecutor(max_workers=n_actual_workers) as executor:
        futures = {}
        for idx, n_archs in enumerate(batch_sizes):
            future = executor.submit(
                _worker_hw_pipeline,
                idx,
                n_archs,
                gpu_key,
                cfg.families,
                cfg.seed + idx,
                cfg.max_hw_iters,
                cfg.max_explore,
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
                    "[cyan]HW-NAS workers running...", total=n_actual_workers
                )
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        worker_results[idx] = result
                        n = result["n_generated"]
                        prog.update(
                            task, advance=1,
                            description=f"[cyan]Worker-{idx} done ({n} ARC)"
                        )
                    except Exception as e:
                        worker_errors[idx] = traceback.format_exc()
                        prog.update(
                            task, advance=1,
                            description=f"[red]Worker-{idx} FAILED: {e}"
                        )
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

    # Laporan error worker
    if worker_errors:
        console.print(f"\n[red]  ⚠ {len(worker_errors)} worker gagal:[/red]")
        for idx, err in worker_errors.items():
            console.print(f"[red]    Worker-{idx}:[/red]")
            for line in err.strip().splitlines()[-5:]:
                console.print(f"[red dim]      {line}[/red dim]")

    # Merge hasil worker (urut berdasarkan worker_idx)
    all_archs: List[ArchConfig] = []
    for idx in sorted(worker_results.keys()):
        res = worker_results[idx]
        for d in res["archs"]:
            try:
                a = _dict_to_arch(d)
                all_archs.append(a)
            except Exception as e:
                console.print(
                    f"[yellow]  ⚠ Gagal rekonstruksi arch dari worker-{idx}: {e}[/yellow]"
                )
        all_hw_scores.update(res["hw_score_map"])
        all_quality.update(res["quality_map"])

    elapsed_hw = time.perf_counter() - t0
    console.print(
        f"\n[bold cyan]✓ Stage 1–2 selesai — "
        f"{len(all_archs):,} ARC dari {len(worker_results)}/{n_actual_workers} worker  "
        f"({elapsed_hw:.1f}s)[/bold cyan]\n"
    )

    if not all_archs:
        raise RuntimeError(
            "Semua HW-worker gagal — tidak ada arsitektur yang berhasil dibuat."
        )

    # ── Stage 3: Training NAS — SERIAL di main process ────────────────────────
    console.rule("[bold magenta]  Stage 3 — Training NAS (Serial, Main Process)  ")
    console.print(
        "[dim]  Training NAS berjalan di main process untuk menghindari:\n"
        "  • torch.cuda tidak fork-safe (CUDA context corrupt di subprocess)\n"
        "  • ProxyTrainer butuh full PyTorch state yang tidak bisa di-pickle[/dim]\n"
    )

    tr_refiner = TrainingNASRefiner(
        cfg.gpu,
        max_iterations    = cfg.max_tr_iters,
        target_pct        = 100.0,
        max_explore_iters = cfg.max_explore,
        rng_seed          = cfg.seed,
        device            = cfg.device,
    )

    all_train:     Dict[str, float] = {}
    tr_archs:      List[ArchConfig] = []
    tr_logs:       List             = []

    t_tr = time.perf_counter()

    if RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console,
        ) as prog:
            task = prog.add_task(
                "[magenta]Training NAS...", total=len(all_archs)
            )
            for i, a in enumerate(all_archs):
                hw = all_hw_scores.get(a.arch_id, 0.0)
                try:
                    refined, alog = tr_refiner.refine(a, hw_score=hw)
                    all_train[refined.arch_id] = alog.train_score_end
                    tr_archs.append(refined)
                    tr_logs.append(alog)
                    prog.update(
                        task, advance=1,
                        description=(
                            f"[magenta]Training NAS [{i+1}/{len(all_archs)}]  "
                            f"{refined.arch_id}  "
                            f"train={alog.train_score_end:.4f}"
                        )
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]  ⚠ Training NAS gagal untuk {a.arch_id}: {e}[/yellow]"
                    )
                    # Fallback: gunakan arch tanpa training refinement
                    all_train[a.arch_id] = 0.0
                    tr_archs.append(a)
    else:
        for i, a in enumerate(all_archs):
            hw = all_hw_scores.get(a.arch_id, 0.0)
            try:
                refined, alog = tr_refiner.refine(a, hw_score=hw)
                all_train[refined.arch_id] = alog.train_score_end
                tr_archs.append(refined)
                tr_logs.append(alog)
                print(f"  [{i+1}/{len(all_archs)}] {refined.arch_id}  "
                      f"train={alog.train_score_end:.4f}")
            except Exception as e:
                print(f"  ⚠ Training NAS gagal: {a.arch_id}: {e}")
                all_train[a.arch_id] = 0.0
                tr_archs.append(a)

    elapsed_tr = time.perf_counter() - t_tr
    console.print(
        f"\n[bold magenta]✓ Stage 3 selesai — "
        f"{len(tr_archs)} ARC training-refined  ({elapsed_tr:.1f}s)[/bold magenta]\n"
    )

    # Print Training NAS summary jika ada log
    if tr_logs:
        try:
            print_training_adaptive_summary(tr_logs, all_train, console=console)
        except Exception:
            pass

    return tr_archs, all_hw_scores, all_quality, all_train


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

    if RICH:
        t = Table(title="Top-10 Final Ranking (Balanced 50/50)",
                  show_header=True, header_style="bold green",
                  box=rbox.SIMPLE_HEAD, padding=(0, 1))
        t.add_column("Rank",    style="dim",     width=5)
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
    console.print(
        f"[green]  ✓ Profiling done: "
        f"MFU={result.get('est_mfu', best.mfu_estimate):.4f}[/green]\n"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6: CLEAN JSON EXPORT (format lengkap)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_arch_entry(
    a:             ArchConfig,
    quality_map:   Dict[str, float],
    hw_scores:     Dict[str, float],
    train_scores:  Dict[str, float],
    combined_map:  Dict[str, float],
) -> dict:
    """
    Buat satu entry JSON lengkap untuk sebuah ArchConfig.

    Format enum: "ClassName.MemberName" (misal "AttentionType.GQA")
    sesuai contoh output yang diinginkan.
    """
    def _enum_str(val) -> str:
        if val is None:
            return "—"
        # str(enum_member) → "ClassName.MEMBER_NAME"
        if hasattr(val, "name"):
            return f"{type(val).__name__}.{val.name}"
        return str(val)

    return {
        # ── Identity ────────────────────────────────────────────────────────
        "arch_id":                a.arch_id,
        "arch_name":              getattr(a, "arch_name", ""),
        "arch_family":            a.arch_family,
        # ── Core dims ───────────────────────────────────────────────────────
        "vocab_size":             getattr(a, "vocab_size", 32000),
        "hidden_dim":             a.hidden_dim,
        "num_layers":             a.num_layers,
        "seq_len":                a.seq_len,
        "batch_size":             a.batch_size,
        # ── Attention ───────────────────────────────────────────────────────
        "attn_type":              _enum_str(a.attn_type),
        "num_heads":              a.num_heads,
        "num_kv_heads":           a.num_kv_heads,
        "head_dim":               a.head_dim,
        "window_size":            getattr(a, "window_size", 2048),
        "global_attn_layers":     getattr(a, "global_attn_layers", 4),
        # ── FFN ─────────────────────────────────────────────────────────────
        "ffn_type":               _enum_str(a.ffn_type),
        "ffn_multiplier":         round(getattr(a, "ffn_multiplier", 4.0), 4),
        "num_experts":            getattr(a, "num_experts", 1),
        "top_k_experts":          getattr(a, "top_k_experts", 1),
        "expert_capacity_factor": getattr(a, "expert_capacity_factor", 1.25),
        # ── Optimizer / training opts ────────────────────────────────────────
        "optimizer_type":         _enum_str(a.optimizer_type),
        "norm_type":              _enum_str(getattr(a, "norm_type", NormType.RMSNORM)),
        "pos_enc":                _enum_str(getattr(a, "pos_enc", PosEncType.ROPE)),
        "tie_embeddings":         getattr(a, "tie_embeddings", True),
        "use_flash_attn":         a.use_flash_attn,
        "use_gradient_checkpointing": getattr(a, "use_gradient_checkpointing", False),
        "use_mixed_precision":    getattr(a, "use_mixed_precision", True),
        "use_torch_compile":      getattr(a, "use_torch_compile", False),
        "dropout":                getattr(a, "dropout", 0.0),
        # ── Params & VRAM ────────────────────────────────────────────────────
        "param_count":            a.param_count,
        "vram_weights_gb":        round(getattr(a, "vram_weights_gb", 0.0), 3),
        "vram_activations_gb":    round(getattr(a, "vram_activations_gb", 0.0), 3),
        "vram_optimizer_gb":      round(getattr(a, "vram_optimizer_gb", 0.0), 3),
        "vram_kv_cache_gb":       round(getattr(a, "vram_kv_cache_gb", 0.0), 5),
        "vram_fragmentation_gb":  round(getattr(a, "vram_fragmentation_gb", 0.0), 3),
        "vram_total_gb":          round(a.vram_total_gb, 3),
        "vram_usage_pct":         round(a.vram_usage_pct, 1),
        # ── FLOPS ────────────────────────────────────────────────────────────
        "flops_per_token_fwd":    getattr(a, "flops_per_token_fwd", 0.0),
        "flops_per_token_bwd":    getattr(a, "flops_per_token_bwd", 0.0),
        "flops_attn_fwd":         getattr(a, "flops_attn_fwd", 0.0),
        "flops_ffn_fwd":          getattr(a, "flops_ffn_fwd", 0.0),
        "arithmetic_intensity":   round(getattr(a, "arithmetic_intensity", 0.0), 2),
        # ── Profiling ────────────────────────────────────────────────────────
        "tokens_per_sec_estimate": a.tokens_per_sec_estimate,
        "mfu_estimate":           round(a.mfu_estimate, 4),
        "ms_per_step":            round(a.ms_per_step, 2),
        "bottleneck":             a.bottleneck,
        "bottleneck_factors":     getattr(a, "bottleneck_factors", {}),
        "fits_gpu":               a.fits_gpu,
        "fitness_score":          round(a.fitness_score, 4),
        "compiler_speedup":       round(getattr(a, "compiler_speedup", 1.0), 2),
        "warp_divergence_pct":    round(getattr(a, "warp_divergence_pct", 0.0), 1),
        "sm_occupancy":           round(getattr(a, "sm_occupancy", 0.0), 2),
        # ── Scores ──────────────────────────────────────────────────────────
        "quality_score_pct":      round(quality_map.get(a.arch_id, 0.0), 2),
        "hardware_score":         round(hw_scores.get(a.arch_id, 0.0), 4),
        "training_score":         round(train_scores.get(a.arch_id, 0.0), 4),
        "combined_score":         round(combined_map.get(a.arch_id, 0.0), 5),
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
            "n_workers":         cfg.n_workers,
            "parallel":          True,
            "training_nas_mode": "serial_main_process",
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
        console.print(
            f"[dim]  {len(archs_sorted)} architectures  |  "
            f"Best: {best.arch_id}  |  "
            f"combined={combined_map.get(best.arch_id, 0):.5f}[/dim]\n"
        )
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
        if hasattr(v, "name"): return v.name
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
                      f"  HW-Workers : {cfg.n_workers}  "
                      f"  Total archs : {len(archs_sorted):,}  "
                      f"  Saved → {output_file:<10}│")
        console.print(f"╰{'─'*W}╯")
    else:
        print("\n" + "═" * 60)
        print(f"  BEST: {aid}  ({best.arch_family})")
        print(f"  Combined={comb:.5f}  HW={hw:.4f}  Train={tr:.4f}  Q={q:.1f}%")
        print(f"  Params={p_str}  VRAM={best.vram_usage_pct:.1f}%  MFU={best.mfu_estimate:.4f}")
        print(f"  HW-Workers={cfg.n_workers}  Total={len(archs_sorted):,}  Saved → {output_file}")
        print("=" * 60)

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

    # ── Stage 1–2 (Parallel HW) + Stage 3 (Serial Training NAS) ──────────────
    all_archs, hw_scores, quality_map, train_scores = stage_parallel(cfg)

    if not all_archs:
        console.print("[red]  Tidak ada arsitektur berhasil digenerate.[/red]")
        return

    # ── Stage 4: Balanced scoring ─────────────────────────────────────────────
    archs_sorted, combined_map, best = stage_balanced_scoring(
        all_archs, quality_map, hw_scores, train_scores
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
