import sys
from typing import Dict, List, Optional, Any

from hardware import GPU_DATABASE, GPUSpec, gpu_pretraining_summary
from arch_types import ArchConfig

# ── Rich Terminal UI ──────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich         import box
    from rich.text    import Text
    RICH    = True
    console = Console()
except ImportError:
    RICH = False
    box  = None

    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw):  print("─" * 80)

    console = Console()


# ─── Banner ───────────────────────────────────────────────────────────────────

def display_banner():
    if not RICH:
        print("=" * 88)
        print("  ARCHITECTURE Search — GPU-Aware LLM Architecture Explorer")
        print("  11 GPU · 7 AI Families · Adaptive Scoring · ECC-Aware · Latency-Model")
        print("=" * 88)
        return
    console.print(Panel.fit(
        "[bold cyan]ARCHITECTURE Search — GPU-Aware LLM Architecture Explorer[/bold cyan]\n"
        "[dim]11 GPU · 7 AI Families · Physics-Comprehensive · ECC/Latency/NVLink-Aware[/dim]\n"
        "[dim]Adaptive Scoring · Continuous Self-Refinement · Combined Quality+Fitness Ranking[/dim]",
        border_style="cyan", padding=(1, 4)
    ))


# ─── GPU Selection ────────────────────────────────────────────────────────────

def select_gpu() -> GPUSpec:
    console.print("\n[bold yellow]Available GPUs:[/bold yellow]\n")
    keys = list(GPU_DATABASE.keys())

    if RICH:
        t = Table(show_header=True, header_style="bold magenta",
                  box=box.ROUNDED, padding=(0, 1))
        t.add_column("#",           style="dim",      width=3)
        t.add_column("Key",         style="cyan",     width=12)
        t.add_column("Name",        style="white",    width=28)
        t.add_column("VRAM",        style="green",    width=8)
        t.add_column("BF16 TFLOPS", style="yellow",   width=11)
        t.add_column("BW GB/s",     style="blue",     width=9)
        t.add_column("Mem Type",    style="magenta",  width=10)
        t.add_column("Lat ns",      style="cyan",     width=7)
        t.add_column("ECC",         style="dim",      width=5)
        t.add_column("L2 MB",       style="green",    width=7)
        t.add_column("NVLink",      style="yellow",   width=14)
        t.add_column("TDP-S W",     style="red",      width=7)
        t.add_column("Arch",        style="dim",      width=20)
        for i, k in enumerate(keys, 1):
            g = GPU_DATABASE[k]
            nvlink_str = (f"v{g.nvlink_version} {g.nvlink_bw_gbps:.0f}GB/s"
                          if g.has_nvlink else "PCIe only")
            ecc_str    = "ON" if g.ecc_mode == "full_in_band" else "opt"
            t.add_row(
                str(i), k, g.name,
                f"{g.vram_gb:.0f}GB",
                f"{g.bf16_tflops:.0f}",
                f"{g.memory_bw_gbps:.0f}",
                g.memory_type,
                f"{g.memory_latency_ns:.0f}",
                ecc_str,
                f"{g.l2_cache_mb:.0f}",
                nvlink_str,
                f"{g.tdp_sustained_w}",
                g.arch[:20],
            )
        console.print(t)
    else:
        hdr = (f"{'#':<3}{'Key':<14}{'VRAM':<7}{'BF16':<7}{'BW':<7}"
               f"{'MemType':<9}{'Lat':<6}{'L2':<6}{'NVLink':<16}{'Arch'}")
        print(hdr)
        print("─" * 100)
        for i, k in enumerate(keys, 1):
            g = GPU_DATABASE[k]
            nvlink_s = f"v{g.nvlink_version} {g.nvlink_bw_gbps:.0f}GB/s" if g.has_nvlink else "PCIe"
            print(f"{i:<3}{k:<14}{g.vram_gb:<7.0f}{g.bf16_tflops:<7.0f}"
                  f"{g.memory_bw_gbps:<7.0f}{g.memory_type:<9}"
                  f"{g.memory_latency_ns:<6.0f}{g.l2_cache_mb:<6.0f}"
                  f"{nvlink_s:<16}{g.arch}")

    while True:
        try:
            choice = input("\n▶  Enter GPU key or number (e.g. 'H100-SXM' or '10'): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(keys):
                    gpu = GPU_DATABASE[keys[idx]]
                    break
            elif choice in GPU_DATABASE:
                gpu = GPU_DATABASE[choice]
                break
            else:
                console.print("[red]Invalid selection. Try again.[/red]")
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    summary = gpu_pretraining_summary(gpu)
    if RICH:
        console.print(f"\n[bold green]✓ Selected: {gpu.name}[/bold green]")
        console.print(Panel(
            f"  [bold]Compute:[/bold]  BF16 {gpu.bf16_tflops:.0f} TFLOPS  "
            f"(sustained {gpu.peak_sustained_tflops:.0f} TFLOPS, "
            f"throttle {summary['thermal_throttle_pct']:.1f}%)\n"
            f"  [bold]Memory:[/bold]   {gpu.vram_gb} GB {gpu.memory_type}  "
            f"| BW {gpu.memory_bw_gbps:.0f} GB/s  "
            f"| Latency {gpu.memory_latency_ns:.0f} ns  "
            f"| ECC {gpu.ecc_mode}\n"
            f"  [bold]Cache:[/bold]    L2 {gpu.l2_cache_mb:.0f} MB  "
            f"| L2-BW {gpu.l2_bandwidth_tbps:.0f} TB/s  "
            f"| SMEM-max {gpu.shared_mem_max_kb:.0f} KB\n"
            f"  [bold]Interconnect:[/bold] {summary['nvlink']}"
            f"  | PCIe {gpu.pcie_version} ×{gpu.pcie_lanes}\n"
            f"  [bold]Ridge Point:[/bold] {summary['ridge_flop_per_byte']:.1f} FLOP/B  "
            f"| SM {gpu.sm_count}  | TC-gen {gpu.tensor_core_gen}  "
            f"| Warps/SM {gpu.max_warps_per_sm}\n"
            f"  [bold]MFU Range:[/bold]   {summary['mfu_range']}  "
            f"| Power {summary['power_eff_tflops_w']} TFLOPS/W",
            title="[bold]GPU Profile[/bold]", border_style="green", padding=(0, 2)
        ))
    else:
        print(f"\n✓ Selected: {gpu.name}")
        print(f"  BF16 {gpu.bf16_tflops:.0f} TFLOPS | BW {gpu.memory_bw_gbps:.0f} GB/s | "
              f"VRAM {gpu.vram_gb}GB {gpu.memory_type} | Lat {gpu.memory_latency_ns:.0f}ns | "
              f"Ridge {summary['ridge_flop_per_byte']:.1f} FLOP/B | {summary['nvlink']}")
    return gpu


# ─── AI Architecture Type Selector ───────────────────────────────────────────

def ask_ai_type() -> List[str]:
    """7 AI families dengan deskripsi lengkap."""
    AI_TYPE_MAP = {
        "1": ("Balanced",     ["Balanced-Pro"]),
        "2": ("Fast",         ["Speed-Demon"]),
        "3": ("Light",        ["Nano-Efficient"]),
        "4": ("Smart",        ["CoT-Optimizer", "Compute-Dense"]),
        "5": ("Long-context", ["Long-Horizon"]),
        "6": ("MoE",          ["MoE-Sparse"]),
        "7": ("All",          []),
    }

    if RICH:
        t = Table(show_header=True, header_style="bold magenta",
                  box=box.ROUNDED, padding=(0, 1))
        t.add_column("#",        style="dim",    width=3)
        t.add_column("Type",     style="cyan",   width=13)
        t.add_column("Families", style="yellow", width=30)
        t.add_column("Best For", style="dim")
        descriptions = {
            "1": ("Balanced-Pro",                  "General-purpose, good throughput+quality balance"),
            "2": ("Speed-Demon",                   "Max tok/s, rapid prototyping, limited VRAM"),
            "3": ("Nano-Efficient",                "Ultra-small, embedded/edge, VRAM-constrained"),
            "4": ("CoT-Optimizer + Compute-Dense", "Deep reasoning, math/code, large GPU exploit"),
            "5": ("Long-Horizon",                  "Long docs, extended context (4K–16K seq)"),
            "6": ("MoE-Sparse",                    "Massive capacity, sparse compute, ZeRO-2/3"),
            "7": ("ALL  (7 families)",             "Full design space exploration"),
        }
        console.print("\n[bold yellow]Select AI Architecture Type:[/bold yellow]\n")
        for k, (label, _) in AI_TYPE_MAP.items():
            fam_str, desc = descriptions[k]
            t.add_row(k, label, fam_str, desc)
        console.print(t)
    else:
        print("\nSelect AI Architecture Type:")
        for k, (label, fams) in AI_TYPE_MAP.items():
            print(f"  [{k}] {label:14s} {', '.join(fams) if fams else 'ALL FAMILIES'}")

    while True:
        try:
            raw = input("\n▶  Enter type number (e.g. '1'): ").strip()
            if not raw:
                raw = "7"

            # Tolak jika ada koma — hanya satu pilihan diizinkan
            if "," in raw:
                console.print("[red]  ✗ Hanya satu pilihan yang diizinkan di mode ini. "
                              "Gunakan Combination type untuk multi-family.[/red]")
                continue

            part = raw.strip()
            if part not in AI_TYPE_MAP:
                console.print(f"[red]  ✗ Pilihan '{part}' tidak valid. Masukkan angka 1–7.[/red]")
                continue

            label, fams = AI_TYPE_MAP[part]
            if not fams:
                console.print("\n[bold green]✓ Using all 7 architecture families[/bold green]")
                return []
            else:
                names = ", ".join(fams)
                console.print(f"\n[bold green]✓ Selected: {names}[/bold green]")
                return list(fams)

        except (KeyboardInterrupt, EOFError):
            return []



# ─── AI Architecture Family Selection (7 families lengkap) ───────────────────

_ALL_FAMILIES_ORDERED = [
    ("CoT-Optimizer",  "Deep narrow — chain-of-thought reasoning, math, code"),
    ("Speed-Demon",    "Wide shallow — max tokens/sec, fast inference"),
    ("Balanced-Pro",   "Balanced depth/width — general-purpose pretraining"),
    ("MoE-Sparse",     "Mixture-of-Experts — sparse capacity, low active params"),
    ("Long-Horizon",   "Extended context — long dependencies, hybrid attention"),
    ("Nano-Efficient", "Ultra-small — max quality per VRAM byte, edge devices"),
    ("Compute-Dense",  "Compute-heavy — high arithmetic intensity, dense FLOP"),
]


def ask_ai_type_local(combination_mode: bool = True) -> Optional[List[str]]:
    """
    Menampilkan SEMUA 7 AI family dan menerima multi-pilihan.
    Semua 7 family termasuk Compute-Dense ditampilkan.

    combination_mode=True (default untuk combo branch):
      - Wajib ≥ 2 family (single tidak diterima)
      - Input kosong tidak fallback ke "all", harus isi ulang

    Returns:
        List[str] keluarga yang dipilih, atau None (= semua keluarga, hanya non-combo).
    """
    console.print()
    console.rule("[bold cyan]  AI Architecture Family Selection  ")
    console.print()

    if combination_mode:
        console.print("[bold red] ( ! ) Hybrid gagal bukan karena salah implementasi, tapi tidak ada solusi stabil di space itu.[/bold red]")
        console.print()
        console.print(
            "  [bold yellow]Mode: COMBINATION[/bold yellow] — "
            "pilih [bold]minimal 2[/bold] family yang akan di-blend (comma-separated):\n"
        )
    else:
        console.print("  Pilih satu atau beberapa family (comma-separated), atau Enter untuk semua:\n")

    for i, (name, desc) in enumerate(_ALL_FAMILIES_ORDERED, 1):
        console.print(f"  [{i}]  [bold]{name:<18}[/bold]  {desc}")

    MAX_TYPES = 4
    MIN_TYPES = 2 if combination_mode else 1

    while True:
        try:
            console.print()
            hint = f"min {MIN_TYPES}, " if combination_mode else ""
            raw = input(
                f"▶  Enter type number(s), comma-separated, {hint}maks {MAX_TYPES}"
                f" (contoh: '2,3' atau '1,3,5'): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return None

        # ── Kosong ────────────────────────────────────────────────────────────
        if not raw:
            if combination_mode:
                console.print(
                    f"[red]  ✗ Mode combination wajib memilih minimal {MIN_TYPES} family. "
                    f"Masukkan nomor, contoh: '1,4' atau '2,3,5'[/red]"
                )
                continue
            else:
                console.print("\n[green]✓ Selected families: ALL (7 families)[/green]")
                return None   # None = semua

        parts = [p for p in raw.replace(" ", "").split(",") if p]

        # ── Tolak jika lebih dari MAX_TYPES ───────────────────────────────────
        if len(parts) > MAX_TYPES:
            console.print(
                f"[red]  ✗ Maks {MAX_TYPES} type yang bisa dipilih sekaligus "
                f"(kamu memasukkan {len(parts)}). Coba lagi.[/red]"
            )
            continue

        # ── Tolak angka tunggal di combination mode ────────────────────────
        if combination_mode and len(parts) < MIN_TYPES:
            console.print(
                f"[red]  ✗ Mode combination butuh minimal {MIN_TYPES} family berbeda — "
                f"kamu hanya memilih {len(parts)}. Contoh: '1,4' atau '2,3,5'[/red]"
            )
            continue

        # ── Validasi tiap nomor ────────────────────────────────────────────────
        chosen = []
        invalid = []
        for part in parts:
            try:
                idx = int(part) - 1
                if 0 <= idx < len(_ALL_FAMILIES_ORDERED):
                    chosen.append(_ALL_FAMILIES_ORDERED[idx][0])
                else:
                    invalid.append(part)
            except ValueError:
                invalid.append(part)

        if invalid:
            console.print(
                f"[red]  ✗ Nomor tidak valid: {', '.join(invalid)}. "
                f"Masukkan angka 1–{len(_ALL_FAMILIES_ORDERED)}.[/red]"
            )
            continue

        # ── Deduplicate & cek ulang minimum ───────────────────────────────────
        chosen = list(dict.fromkeys(chosen))

        if combination_mode and len(chosen) < MIN_TYPES:
            console.print(
                f"[red]  ✗ Setelah deduplikasi hanya {len(chosen)} family unik — "
                f"butuh minimal {MIN_TYPES}. Coba lagi dengan nomor berbeda.[/red]"
            )
            continue

        if not chosen:
            console.print("[yellow]  ⚠ Tidak ada pilihan valid — coba lagi.[/yellow]")
            continue

        console.print(f"\n[green]✓ Selected families: {', '.join(chosen)}[/green]")
        return chosen


# ─── Combination Mode Prompt ──────────────────────────────────────────────────

def ask_combination_mode() -> bool:
    """
    Tanya user apakah ingin mode kombinasi AI (N-way family blend) atau single-family.

    Returns:
        True  → user ingin kombinasi (combination_nas.py yang handle selanjutnya)
        False → mode single family biasa
    """
    console.print()
    try:
        ans = input(
            "▶  Combination type or only type (Y/n): "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans not in ("n", "no")


# ─── Run Options ─────────────────────────────────────────────────────────────

def ask_run_options() -> Dict:
    from profiler import TORCH, DEVICE
    opts = {}

    # ── Architectures per family — wajib isi, loop sampai ada ────────────────
    MAX_PER_FAM = 10
    while True:
        try:
            n_str = input(
                f"\n▶  Architectures per family to generate? [wajib isi, max={MAX_PER_FAM}]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            opts["n_per_family"] = 3
            break
        if not n_str:
            console.print(
                f"[red]  ✗ Wajib isi jumlah arsitektur. Masukkan angka 1–{MAX_PER_FAM}.[/red]"
            )
            continue
        try:
            n_val = int(n_str)
            if n_val < 1 or n_val > MAX_PER_FAM:
                console.print(f"[red]  ✗ Angka harus 1–{MAX_PER_FAM}.[/red]")
                continue
            opts["n_per_family"] = n_val
            break
        except ValueError:
            console.print(f"[red]  ✗ Masukkan angka valid 1–{MAX_PER_FAM}.[/red]")

    # ── Torch profiling ────────────────────────────────────────────────────────
    try:
        run_torch = input("▶  Run real torch profiling? (needs GPU) [y/N]: ").strip().lower()
        opts["run_torch"] = run_torch in ("y", "yes") and TORCH and DEVICE == "cuda"
    except (EOFError, KeyboardInterrupt):
        opts["run_torch"] = False

    # ── Multi-GPU ──────────────────────────────────────────────────────────────
    try:
        n4_str = input("▶  Multi-GPU estimate for N GPUs? [default=4]: ").strip()
        opts["n_gpus"] = max(1, int(n4_str) if n4_str.isdigit() else 4)
    except (EOFError, KeyboardInterrupt):
        opts["n_gpus"] = 4

    # ── Random seed ────────────────────────────────────────────────────────────
    try:
        seed_str = input("▶  Random seed (Enter for random): ").strip()
        opts["seed"] = int(seed_str) if seed_str.isdigit() else None
    except (EOFError, KeyboardInterrupt):
        opts["seed"] = None

    return opts


# ─── Architecture Summary Table ───────────────────────────────────────────────

def print_arch_summary(archs: List[ArchConfig], gpu: GPUSpec,
                        quality_map: Optional[Dict[str, float]] = None):
    """
    Tabel ringkasan arsitektur.
    Jika quality_map disediakan, kolom 'Score' menampilkan combined_score
    (quality × 0.35 + fitness × 0.65) dan arsitektur diurutkan by combined.

    Untuk combination architecture (arch_family mengandung '+'):
      - Kolom 'Blend' menampilkan profil hybrid (dominan family + karakteristik kunci)
      - Warna khusus untuk hybrid rows
      - Tidak ada dua arsitektur yang terlihat identik karena Params/L×D/Attn unik per blend
    """
    try:
        from adaptive_refiner import compute_combined_score
        _have_combined = quality_map is not None
    except ImportError:
        _have_combined = False

    def _combined(a: ArchConfig) -> float:
        if not _have_combined:
            return a.fitness_score
        q = quality_map.get(a.arch_id, 0.0) if quality_map else 0.0
        return compute_combined_score(q, a.fitness_score)

    def _hybrid_profile(cfg: ArchConfig) -> str:
        """Buat label profil hybrid berdasarkan nilai nyata cfg, bukan template."""
        fam = cfg.arch_family
        if "+" not in fam:
            return ""
        families = [f.strip() for f in fam.split("+")]
        traits = []
        # Layer density
        if cfg.num_layers >= 32:
            traits.append("deep")
        elif cfg.num_layers <= 10:
            traits.append("shallow")
        else:
            traits.append("mid")
        # Width
        if cfg.hidden_dim >= 2048:
            traits.append(f"wide-D{cfg.hidden_dim}")
        elif cfg.hidden_dim <= 512:
            traits.append(f"narrow-D{cfg.hidden_dim}")
        else:
            traits.append(f"D{cfg.hidden_dim}")
        # FFN type indicator
        ffn_name = cfg.ffn_type.name if hasattr(cfg.ffn_type, "name") else str(cfg.ffn_type)
        if "MOE" in ffn_name.upper():
            traits.append(f"MoE×{cfg.num_experts}")
        elif "GEGLU" in ffn_name.upper():
            traits.append("GeGLU")
        # Seq
        if cfg.seq_len >= 4096:
            traits.append(f"ctx{cfg.seq_len//1024}k")
        return "/".join(traits[:3])

    is_combo_mode = any("+" in cfg.arch_family for cfg in archs)

    if RICH:
        t = Table(title=f"[bold]Generated Architectures — {gpu.name}[/bold]",
                  show_header=True, header_style="bold cyan",
                  box=box.SIMPLE_HEAD, padding=(0, 1), show_lines=True)
        t.add_column("ID",        style="dim",        width=9)
        t.add_column("Family",    style="cyan",       width=16 if is_combo_mode else 14)
        t.add_column("Params",    style="yellow",     width=8)
        t.add_column("L×D",       style="white",      width=10)
        t.add_column("Attn",      style="green",      width=7)
        t.add_column("FFN",       style="green",      width=9)
        t.add_column("Seq",       style="blue",       width=6)
        t.add_column("VRAM GB",   style="magenta",    width=10)
        t.add_column("Status",    style="bold",       width=12)
        t.add_column("Tok/s",     style="yellow",     width=10)
        t.add_column("MFU%",      style="cyan",       width=6)
        t.add_column("AI FLOP/B", style="blue",       width=9)
        t.add_column("WarpDiv%",  style="red",        width=8)
        if is_combo_mode:
            t.add_column("Blend Profile", style="magenta", width=18)
        if _have_combined:
            t.add_column("Quality",   style="green",  width=8)
            t.add_column("Combined",  style="bold green", width=9)
        else:
            t.add_column("Fitness",   style="bold green", width=8)

        for cfg in archs:
            p  = (f"{cfg.param_count/1e6:.0f}M" if cfg.param_count < 1e9
                  else f"{cfg.param_count/1e9:.2f}B")
            vp = cfg.vram_usage_pct
            is_hybrid = "+" in cfg.arch_family
            if cfg.fits_gpu:
                vc     = "yellow" if vp >= 70 else "green"
                status = "[green]✓ ACCEPTED[/green]"
            else:
                vc     = "red"
                status = "[red]✗ REJECTED[/red]"

            fam_display = cfg.arch_family[:16] if is_combo_mode else cfg.arch_family[:14]
            # Hybrid families tampil dengan warna berbeda
            if is_hybrid:
                fam_display = f"[bold magenta]{fam_display}[/bold magenta]"

            row = [
                cfg.arch_id, fam_display, p,
                f"L{cfg.num_layers}×D{cfg.hidden_dim}",
                cfg.attn_type.name, cfg.ffn_type.name[:9],
                str(cfg.seq_len),
                f"[{vc}]{cfg.vram_total_gb:.1f} ({vp:.1f}%)[/{vc}]",
                status,
                f"{cfg.tokens_per_sec_estimate:,}",
                f"{cfg.mfu_estimate*100:.1f}",
                f"{cfg.arithmetic_intensity:.1f}",
                f"{cfg.warp_divergence_pct:.1f}",
            ]
            if is_combo_mode:
                row.append(_hybrid_profile(cfg) if is_hybrid else "—")
            if _have_combined:
                q = quality_map.get(cfg.arch_id, 0.0) if quality_map else 0.0
                c = compute_combined_score(q, cfg.fitness_score)
                row.append(f"{q:.1f}%")
                row.append(f"{c:.4f}")
            else:
                row.append(f"{cfg.fitness_score:.3f}")
            t.add_row(*row)
        console.print(t)
    else:
        print(f"\n{'─'*158}")
        print(f"{'ID':<9}{'Family':<16}{'Params':<9}{'L×D':<11}{'Attn':<8}{'FFN':<10}"
              f"{'VRAM GB (%)':<16}{'Status':<12}{'Tok/s':<11}{'MFU%':<7}"
              f"{'AI FLOP/B':<10}{'WarpDiv%':<9}"
              + ("{'Quality':<10}{'Combined'}" if _have_combined else "{'Fitness'}"))
        print('─' * 158)
        for cfg in archs:
            p      = (f"{cfg.param_count/1e6:.0f}M" if cfg.param_count < 1e9
                      else f"{cfg.param_count/1e9:.2f}B")
            status = "✓ ACCEPTED" if cfg.fits_gpu else "✗ REJECTED"
            vram_s = f"{cfg.vram_total_gb:.1f} ({cfg.vram_usage_pct:.1f}%)"
            score_s = ""
            if _have_combined:
                q = quality_map.get(cfg.arch_id, 0.0) if quality_map else 0.0
                c = compute_combined_score(q, cfg.fitness_score)
                score_s = f"{q:<10.1f}{c:.4f}"
            else:
                score_s = f"{cfg.fitness_score:.3f}"
            print(f"{cfg.arch_id:<9}{cfg.arch_family[:15]:<16}{p:<9}"
                  f"L{cfg.num_layers}×D{cfg.hidden_dim:<8}{cfg.attn_type.name:<8}"
                  f"{cfg.ffn_type.name[:9]:<10}{vram_s:<16}{status:<12}"
                  f"{cfg.tokens_per_sec_estimate:<11,}{cfg.mfu_estimate*100:<7.1f}"
                  f"{cfg.arithmetic_intensity:<10.1f}{cfg.warp_divergence_pct:<9.1f}"
                  f"{score_s}")


# ─── Helper: Rich Table Builder ───────────────────────────────────────────────

def _make_table(title: str, rows: Dict,
                k_style="cyan", v_style="yellow") -> Optional[Any]:
    if not RICH:
        return None
    t = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD,
              show_header=True, header_style="bold")
    t.add_column("Metric", style=k_style)
    t.add_column("Value",  style=v_style, justify="right")
    for k, v in rows.items():
        t.add_row(str(k), str(v))
    return t


# ─── Detailed Report ──────────────────────────────────────────────────────────

def print_detailed_report(cfg: ArchConfig, gpu: GPUSpec,
                           report, pr: Dict, n_gpus: int):
    """Print full profiling report untuk satu arsitektur."""
    from metrics import MetricsReport  # local import to avoid circular

    if RICH:
        console.rule(f"[bold cyan]  {cfg.arch_id}  ─  {cfg.arch_family}  ─  {gpu.name}  ")
    else:
        print(f"\n{'═'*80}\n  {cfg.arch_id}  ─  {cfg.arch_family}\n{'═'*80}")

    if RICH:
        lines = [
            f"[bold]Architecture:[/bold] {cfg.arch_name}",
            f"[bold]Params:[/bold] {cfg.param_count/1e9:.3f}B  |  "
            f"[bold]Layers:[/bold] {cfg.num_layers}  |  "
            f"[bold]Hidden:[/bold] {cfg.hidden_dim}  |  "
            f"[bold]Heads:[/bold] {cfg.num_heads} (KV:{cfg.num_kv_heads})  |  "
            f"[bold]HeadDim:[/bold] {cfg.head_dim}",
            f"[bold]Attention:[/bold] {cfg.attn_type.value}  |  "
            f"[bold]FFN:[/bold] {cfg.ffn_type.value}",
            f"[bold]Seq:[/bold] {cfg.seq_len}  |  "
            f"[bold]Batch:[/bold] {cfg.batch_size}  |  "
            f"[bold]PosEnc:[/bold] {cfg.pos_enc.value}",
            f"[bold]GradCkpt:[/bold] {'Yes' if cfg.use_gradient_checkpointing else 'No'}  |  "
            f"[bold]FlashAttn:[/bold] {'Yes' if cfg.use_flash_attn else 'No'}  |  "
            f"[bold]Compile:[/bold] {'Yes' if cfg.use_torch_compile else 'No'}  |  "
            f"[bold]Optimizer:[/bold] {cfg.optimizer_type.name}",
            f"[bold]GPU:[/bold] {gpu.name}  |  "
            f"[bold]Mem:[/bold] {gpu.memory_type} {gpu.vram_gb}GB  |  "
            f"[bold]Lat:[/bold] {gpu.memory_latency_ns:.0f}ns  |  "
            f"[bold]ECC:[/bold] {gpu.ecc_mode}  |  "
            f"[bold]TC-gen:[/bold] {gpu.tensor_core_gen}  |  "
            f"[bold]SMEM-max:[/bold] {gpu.shared_mem_max_kb:.0f}KB",
        ]
        console.print(Panel("\n".join(lines), title="Config",
                            border_style="blue", padding=(0, 2)))

    if pr and RICH:
        ct = Table(title="Profiler Results", box=box.MINIMAL_DOUBLE_HEAD)
        ct.add_column("Key",   style="cyan")
        ct.add_column("Value", style="yellow", justify="right")
        for k, v in pr.items():
            ct.add_row(k, str(v))
        console.print(ct)

    bt = cfg.bottleneck_factors
    if bt and RICH:
        bft = Table(title="Bottleneck Factor Analysis", box=box.MINIMAL_DOUBLE_HEAD)
        bft.add_column("Factor", style="cyan")
        bft.add_column("Value",  style="yellow", justify="right")
        for k, v in bt.items():
            bft.add_row(k.replace("_", " ").title(), str(v))
        console.print(bft)

    if RICH:
        console.print(_make_table("Memory Breakdown", report.memory_breakdown()))
        console.print(_make_table("CUDA Kernel Occupancy & Warp Analysis",
                                   report.kernel_occupancy()))
        console.print(_make_table("Cache Efficiency & HBM Bandwidth",
                                   report.cache_efficiency()))
        console.print(_make_table("Compute FLOPs Breakdown",
                                   report.compute_breakdown()))
        console.print(_make_table("Compiler Effects (torch.compile / Triton)",
                                   report.compiler_effects()))
        console.print(_make_table("Optimizer State Analysis",
                                   report.optimizer_detail()))

    recomp = report.recomputation_cost()
    if recomp and RICH:
        console.print(_make_table("Activation Recomputation (GC)", recomp))

    moe = report.moe_routing()
    if moe and RICH:
        console.print(_make_table("MoE Routing Analysis", moe))

    if RICH:
        console.print(_make_table("Thermal Throttling & Runtime Noise",
                                   report.thermal_and_noise()))

    seq_scale = report.sequence_scaling_estimate()
    if RICH and seq_scale:
        sst = Table(title="Sequence Length Scaling (tok/s)",
                    box=box.MINIMAL_DOUBLE_HEAD)
        sst.add_column("Seq Len",     style="cyan",    justify="right")
        sst.add_column("Tok/s",       style="yellow",  justify="right")
        sst.add_column("vs Baseline", style="magenta", justify="right")
        base = seq_scale.get(cfg.seq_len, 1) or 1
        for s, tps in seq_scale.items():
            sst.add_row(f"{s:,}", f"{tps:,}", f"{tps/base:.2f}×")
        console.print(sst)

    bs_scale = report.batch_scaling_estimate()
    if RICH and bs_scale:
        bst = Table(title="Batch Size Scaling — Amdahl Effect (tok/s)",
                    box=box.MINIMAL_DOUBLE_HEAD)
        bst.add_column("Batch", style="cyan",   justify="right")
        bst.add_column("Tok/s", style="yellow", justify="right")
        for bs, tps in bs_scale.items():
            bst.add_row(str(bs), f"{tps:,}")
        console.print(bst)

    mgpu = report.multi_gpu_estimate(n_gpus)
    if RICH:
        console.print(_make_table(
            f"Multi-GPU Scaling ({n_gpus}× {gpu.name})", mgpu))

    kv_curve = report.kv_cache_growth_curve()
    if RICH and kv_curve:
        kvt = Table(title="KV Cache Growth (GB vs Token Count)",
                    box=box.MINIMAL_DOUBLE_HEAD)
        kvt.add_column("Tokens", style="cyan",   justify="right")
        kvt.add_column("KV GB",  style="yellow", justify="right")
        for tokens, gb in kv_curve.items():
            kvt.add_row(f"{tokens:,}", f"{gb:.5f}")
        console.print(kvt)


# ─── Final Ranking ────────────────────────────────────────────────────────────

def print_ranking(
    archs:       List[ArchConfig],
    gpu:         GPUSpec,
    quality_map: Optional[Dict[str, float]] = None,
):
    """
    Tampilkan ranking final dan panel REKOMENDASI.

    PERUBAHAN v4:
      Rekomendasi dipilih berdasarkan combined_score TERTINGGI
      (bukan sekadar arsitektur pertama dalam list).

      combined = 35% × quality_norm + 65% × fitness
        quality_norm = (quality_pct - 70) / 30  → [0,1]

      Jika quality_map tidak disediakan (mode non-adaptive),
      fallback ke perilaku v3: fits[0] berdasarkan fitness_score.
    """
    import json
    from dataclasses import asdict

    # Import combined score util
    try:
        from adaptive_refiner import compute_combined_score, select_best_arch
        _have_combined = quality_map is not None
    except ImportError:
        _have_combined = False

    fits   = [a for a in archs if a.fits_gpu]
    no_fit = [a for a in archs if not a.fits_gpu]
    vlimit = 80

    console.rule(f"[bold green]  FINAL RANKING — ACCEPTED (VRAM ≤ {vlimit}%)  ")
    if fits:
        print_arch_summary(fits, gpu, quality_map)
    else:
        console.print(f"[yellow]  Tidak ada arsitektur yang lolos VRAM ≤ {vlimit}%.[/yellow]")

    if no_fit:
        console.rule(f"[bold red]  REJECTED — VRAM > {vlimit}% ({gpu.vram_gb * vlimit / 100:.1f} GB)  ")
        print_arch_summary(no_fit, gpu, quality_map)

    if fits:
        # ── Pilih rekomendasi terbaik ─────────────────────────────────────────
        if _have_combined:
            # v4: pilih berdasarkan combined_score tertinggi
            best = select_best_arch(fits, quality_map)
            if best is None:
                best = fits[0]
        else:
            # Fallback v3: posisi pertama (sudah sorted by fitness)
            best = fits[0]

        # Hitung skor untuk ditampilkan
        best_quality  = quality_map.get(best.arch_id, None) if quality_map else None
        best_combined = (compute_combined_score(best_quality, best.fitness_score)
                         if _have_combined and best_quality is not None
                         else None)

        p_str = (f"{best.param_count/1e9:.3f}B" if best.param_count >= 1e9
                 else f"{best.param_count/1e6:.0f}M")

        fa_tile = gpu.flash_attn_tile_feasibility(best.head_dim)
        fa_info = (f"Optimal ({fa_tile*100:.0f}%)" if fa_tile >= 0.9
                   else f"Reduced ({fa_tile*100:.0f}% of optimal)")

        # Susun teks skor untuk panel
        score_lines = (
            f"  • [cyan]Quality Score:[/cyan]   {best_quality:.1f}% (internal consistency)\n"
            f"  • [cyan]Fitness Score:[/cyan]   {best.fitness_score:.4f} (training performance)\n"
            f"  • [cyan]Combined Score:[/cyan]  {best_combined:.5f} "
            f"[dim](35% quality + 65% fitness → basis rekomendasi)[/dim]\n"
            if (_have_combined and best_quality is not None and best_combined is not None)
            else
            f"  • [cyan]Fitness Score:[/cyan]   {best.fitness_score:.4f}\n"
        )

        # Tambahkan note jika best bukan fits[0]
        note = ""
        if _have_combined and best.arch_id != fits[0].arch_id:
            runner_quality = quality_map.get(fits[0].arch_id, 0.0) if quality_map else 0.0
            runner_combined = compute_combined_score(runner_quality, fits[0].fitness_score)
            note = (
                f"\n  [dim]ⓘ  {fits[0].arch_id} memiliki fitness lebih tinggi "
                f"({fits[0].fitness_score:.4f}) tapi combined lebih rendah "
                f"({runner_combined:.5f}) karena quality lebih rendah "
                f"({runner_quality:.1f}%).[/dim]"
            )

        if RICH:
            console.print(Panel(
                f"[bold green]🏆 Top Pick: {best.arch_id} — {best.arch_family}[/bold green]\n\n"
                f"{score_lines}"
                f"  • [cyan]Parameters:[/cyan]     {p_str}\n"
                f"  • [cyan]Architecture:[/cyan]   L{best.num_layers}×D{best.hidden_dim} "
                f"| {best.attn_type.value}\n"
                f"  • [cyan]FFN:[/cyan]            {best.ffn_type.value}\n"
                f"  • [cyan]Optimizer:[/cyan]      {best.optimizer_type.value}\n"
                f"  • [cyan]VRAM Usage:[/cyan]     {best.vram_total_gb:.1f} GB "
                f"({best.vram_usage_pct:.1f}% of {gpu.vram_gb}GB — limit 80%)\n"
                f"  • [cyan]Throughput:[/cyan]     ~{best.tokens_per_sec_estimate:,} tokens/sec\n"
                f"  • [cyan]MFU:[/cyan]            {best.mfu_estimate*100:.1f}%\n"
                f"  • [cyan]Step Time:[/cyan]      {best.ms_per_step:.1f}ms\n"
                f"  • [cyan]Bottleneck:[/cyan]     {best.bottleneck}\n"
                f"  • [cyan]Arith. Int.:[/cyan]    {best.arithmetic_intensity:.1f} FLOP/byte\n"
                f"  • [cyan]Warp Div.:[/cyan]      {best.warp_divergence_pct:.1f}%\n"
                f"  • [cyan]SM Occupancy:[/cyan]   {best.sm_occupancy*100:.1f}%\n"
                f"  • [cyan]FA Tile:[/cyan]        {fa_info} "
                f"(SMEM {gpu.shared_mem_max_kb:.0f}KB, head_dim={best.head_dim})\n"
                f"  • [cyan]Memory:[/cyan]         {gpu.memory_type} "
                f"| {gpu.memory_latency_ns:.0f}ns latency "
                f"| ECC {gpu.ecc_mode}\n"
                f"  • [cyan]Compiler Spd:[/cyan]   {best.compiler_speedup:.3f}×"
                f"{note}",
                title="[bold]RECOMMENDATION[/bold]",
                border_style="green", padding=(1, 3)
            ))
        else:
            print(f"\n🏆 Top Pick: {best.arch_id} — {best.arch_family}")
            if best_quality is not None:
                print(f"  Quality: {best_quality:.1f}%  "
                      f"Fitness: {best.fitness_score:.4f}  "
                      f"Combined: {best_combined:.5f}")
            print(f"  Params: {p_str}  VRAM: {best.vram_total_gb:.1f}GB "
                  f"({best.vram_usage_pct:.1f}%)  Tok/s: {best.tokens_per_sec_estimate:,}")

    # Save JSON results
    def _serialize(v):
        if hasattr(v, 'value'):
            return v.value
        return v

    output = {
        "gpu": {k: _serialize(v)
                for k, v in asdict(gpu).items()
                if not callable(v) and not k.startswith('_')},
        "quality_map": quality_map or {},
        "architectures": [
            {k: _serialize(v) for k, v in asdict(a).items()}
            for a in archs
        ]
    }
    fname = f"arc_final_{gpu.name.replace(' ', '_')}.json"
    try:
        with open(fname, "w") as f:
            json.dump(output, f, indent=2)
        console.print(f"\n[dim]Results saved → {fname}[/dim]")
    except Exception as e:
        console.print(f"\n[dim yellow]Warning: could not save JSON — {e}[/dim yellow]")
