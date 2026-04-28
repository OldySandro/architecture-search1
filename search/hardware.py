from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
#  Konstanta Fisika
# ─────────────────────────────────────────────────────────────────────────────
_WARP_SIZE   = 32   # NVIDIA: selalu 32 thread per warp
_BF16_BYTES  = 2
_FP32_BYTES  = 4


@dataclass
class GPUSpec:
    """
    Spesifikasi hardware GPU komprehensif untuk simulasi pretraining.

    ╔════════════════════════════════════════════════════════════════════╗
    ║ FILOSOFI NILAI: perilaku NYATA di lapangan, bukan marketing spec  ║
    ║  • Clock       = nilai SUSTAINED saat beban pretraining penuh     ║
    ║  • MFU range   = dikalibrasi dari MLPerf + paper published        ║
    ║  • Memory BW   = setelah ECC overhead dan access pattern training ║
    ║  • Thermal fac = sustained_clock / boost_clock  (diukur)          ║
    ╚════════════════════════════════════════════════════════════════════╝
    """

    # ── [1] Identitas & Fabrikasi ─────────────────────────────────────────────
    name:              str
    arch:              str
    vendor:            str   = "NVIDIA"
    process_node_nm:   int   = 7       # node TSMC/Samsung (nm)
    transistors_b:     float = 0.0     # transistor (miliar)
    die_size_mm2:      float = 0.0     # luas die

    # ── [2] Unit Komputasi ────────────────────────────────────────────────────
    cuda_cores:        int   = 0       # CUDA core / shader processor
    sm_count:          int   = 0       # Streaming Multiprocessor
    tensor_core_gen:   int   = 0       # generasi TC: 1=Volta, 2=Turing,
                                       #               3=Ampere, 4=Hopper
    warp_size:         int   = 32      # selalu 32 untuk NVIDIA
    max_warps_per_sm:  int   = 64      # max warp aktif per SM

    # ── [3] Throughput Komputasi ──────────────────────────────────────────────
    fp16_tflops:           float = 0.0  # FP16 TC (dense, no sparsity)
    bf16_tflops:           float = 0.0  # BF16 TC — utama untuk pretraining
    fp32_tflops:           float = 0.0  # FP32 CUDA core
    int8_tops:             float = 0.0  # INT8 TOPS
    sparsity_bf16_tflops:  float = 0.0  # 2:4 structured sparsity (≈2× BF16)

    # ── [4] Clock Frequencies ─────────────────────────────────────────────────
    frequency_base_mhz:      float = 0.0  # base clock
    frequency_boost_mhz:     float = 0.0  # boost (peak, burst singkat)
    frequency_sustained_mhz: float = 0.0  # NYATA saat full pretraining load

    # ── [5] Subsistem Memori ──────────────────────────────────────────────────
    vram_gb:           float = 0.0
    memory_type:       str   = ""     # "HBM2","HBM2e","HBM3","HBM3e","GDDR6","GDDR6X"
    memory_channels:   int   = 1      # HBM stacks atau GDDR channels
    memory_bus_width:  int   = 256    # bus width (bit)
    memory_clock_mhz:  float = 0.0    # frekuensi memori
    memory_bw_gbps:    float = 0.0    # peak theoretical BW (GB/s)

    # Latency akses DRAM — kritis untuk KV cache access pattern:
    # HBM2/2e: ~80-120 ns   (stacked die, dekat chip, bus width besar)
    # HBM3/3e: ~70-80 ns    (lebih cepat, generasi terbaru)
    # GDDR6:   ~400-500 ns  (off-die controller, latency tinggi)
    # GDDR6X:  ~380-440 ns  (PAM4 → sedikit lebih baik dari GDDR6)
    memory_latency_ns: float = 300.0

    # ECC — mempengaruhi bandwidth efektif:
    # "full_in_band"   : HBM selalu ON, ~2-3% overhead BW
    # "optional_secded": GDDR6/6X optional; default OFF di sebagian besar cloud
    # "none"           : tidak ada ECC
    ecc_mode:          str   = "none"
    ecc_bw_overhead:   float = 0.0    # fraksi BW dikonsumsi ECC (0.0 – 0.06)

    # Efisiensi BW berdasarkan pola akses (training workload):
    # Streaming (weights, activations): bus penuh, prefetch efektif
    # Random (KV cache, MoE expert dispatch): cache miss → efisiensi turun
    # HBM secara fisik >efisien karena bus width 4096-bit vs 256-384 bit GDDR
    hbm_efficiency_streaming: float = 0.88
    hbm_efficiency_random:    float = 0.62

    # ── [6] Cache Hierarchy ───────────────────────────────────────────────────
    l2_cache_mb:       float = 0.0    # L2 cache total (shared semua SM)
    l1_cache_kb:       float = 128.0  # L1 per SM
    shared_mem_kb:     float = 96.0   # shared mem per SM (default config)
    shared_mem_max_kb: float = 0.0    # max configurable (penting untuk FlashAttn!)

    # L2 → SM bandwidth — menentukan FlashAttention tile feasibility:
    # H100-SXM: 33 TB/s   A100: 19.5 TB/s   V100: 9.7 TB/s   T4: 4.0 TB/s
    l2_bandwidth_tbps: float = 0.0

    # ── [7] Interconnect Multi-GPU ────────────────────────────────────────────
    nvlink_version:         int   = 0     # 0=tidak ada, 1-4=versi NVLink
    nvlink_links:           int   = 0     # jumlah link fisik
    nvlink_bw_gbps:         float = 0.0   # NVLink bidirectional BW (GB/s)
    nvlink_latency_us:      float = 0.0   # latency per hop (µs)
    nvlink_congestion_onset: float = 1.0  # fraksi BW threshold congestion
    pcie_version:           str   = "4.0"
    pcie_lanes:             int   = 16
    pcie_bw_gbps:           float = 16.0  # PCIe unidirectional (GB/s)

    # ── [8] Daya & Thermal ────────────────────────────────────────────────────
    max_power_w:     int   = 0      # TDP burst (marketing)
    tdp_sustained_w: int   = 0      # daya nyata saat pretraining sustained

    # thermal_factor = frequency_sustained / frequency_boost
    # Mencerminkan % performa yang hilang akibat throttle termal/daya
    # T4=0.86, V100=0.90, RTX-3090=0.87, A100=0.93, H100=0.95, H200=0.96
    thermal_factor:              float = 0.92
    power_efficiency_tflops_per_w: float = 0.0  # BF16 TFLOPS/W (dihitung)

    # ── [9] Kernel & Scheduler Overhead ──────────────────────────────────────
    # Overhead per kernel launch (turun drastis Turing→Hopper):
    # Turing: 8-10 µs   Volta: 5-6 µs   Ampere: 3.5-6 µs
    # Ada: 4-5 µs        Hopper: 2-3 µs
    kernel_launch_us:     float = 5.0
    mem_alloc_overhead:   float = 0.03   # fraksi overhead CUDA malloc
    typical_sm_occupancy: float = 0.70   # occupancy SM typical saat training

    # ── [10] Kalibrasi Pretraining Real-World ─────────────────────────────────
    # Dikalibrasi dari:
    #   MLPerf Training v3.0/v3.1, Megatron-LM (Narayanan et al.),
    #   Meta LLaMA-2 Tech Report, PaLM papers, Chinchilla training,
    #   nanoGPT/lit-llama single GPU benchmarks
    mfu_typical_min:      float = 0.30   # MFU min (misconfigured/small model)
    mfu_typical_max:      float = 0.55   # MFU max (optimal: FlashAttn + large bs)

    # FlashAttention speedup vs naive (tergantung SMEM size dan L2 BW):
    # T4 (64KB SMEM): 2.2×     A100 (164KB): 3.0×     H100 (228KB): 3.5-3.8×
    flash_attn_speedup:   float = 2.8

    # Variasi run-to-run:
    # Sumber: clock jitter, GC pauses, NCCL variance, OS scheduling
    runtime_variance_pct: float = 2.5

    # Fraksi waktu step yang terbuang menunggu dataloader:
    # GPU lebih cepat → relatif lebih terpengaruh oleh IO lambat
    dataloader_stall_frac: float = 0.03

    # ── [11] Multi-Instance GPU ───────────────────────────────────────────────
    mig_max_instances: int = 0    # 0=tidak didukung; A100/H100: 7

    # ── [12] Derived (dihitung __post_init__) ─────────────────────────────────
    flops_per_byte: float = 0.0   # ridge point hardware (FLOPs/byte)

    # ─────────────────────────────────────────────────────────────────────────
    def __post_init__(self):
        """Hitung derived fields dan validasi konsistensi parameter."""

        # Ridge point — konstanta hardware murni (Roofline Model)
        if self.memory_bw_gbps > 0:
            self.flops_per_byte = (self.bf16_tflops * 1e12) / (self.memory_bw_gbps * 1e9)
        else:
            self.flops_per_byte = 0.0

        # Sustained clock dari thermal_factor jika belum diset
        if self.frequency_sustained_mhz == 0.0 and self.frequency_boost_mhz > 0:
            self.frequency_sustained_mhz = self.frequency_boost_mhz * self.thermal_factor

        # Konsistensi thermal_factor dengan clock yang diukur
        if self.frequency_sustained_mhz > 0 and self.frequency_boost_mhz > 0:
            measured = self.frequency_sustained_mhz / self.frequency_boost_mhz
            if abs(measured - self.thermal_factor) > 0.02:
                self.thermal_factor = min(self.thermal_factor, measured)

        # Sparsity default: 2× BF16 untuk Ampere+ (gen ≥ 3)
        if self.sparsity_bf16_tflops == 0.0 and self.tensor_core_gen >= 3:
            self.sparsity_bf16_tflops = self.bf16_tflops * 2.0

        # Shared mem max
        if self.shared_mem_max_kb == 0.0:
            self.shared_mem_max_kb = self.shared_mem_kb

        # Power efficiency
        if self.power_efficiency_tflops_per_w == 0.0 and self.tdp_sustained_w > 0:
            self.power_efficiency_tflops_per_w = round(
                (self.bf16_tflops * self.thermal_factor) / self.tdp_sustained_w, 3)

        # Max warps per SM per arsitektur (penting untuk occupancy calculation)
        if self.max_warps_per_sm == 64:
            if "Turing" in self.arch:
                self.max_warps_per_sm = 32   # Turing: max 32 warps/SM
            elif "Ada" in self.arch:
                self.max_warps_per_sm = 48   # Ada Lovelace: 48 warps/SM

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def effective_memory_bw_gbps(self) -> float:
        """BW efektif setelah ECC overhead."""
        return self.memory_bw_gbps * (1.0 - self.ecc_bw_overhead)

    @property
    def has_nvlink(self) -> bool:
        return self.nvlink_bw_gbps > 0

    @property
    def is_hbm(self) -> bool:
        return "HBM" in self.memory_type

    @property
    def peak_sustained_tflops(self) -> float:
        """BF16 TFLOPS sustained (setelah thermal throttle)."""
        return self.bf16_tflops * self.thermal_factor

    @property
    def optimal_tile_size(self) -> int:
        """Tile size optimal Tensor Core per generasi."""
        return {1: 16, 2: 16, 3: 32, 4: 64}.get(self.tensor_core_gen, 16)

    def flash_attn_tile_feasibility(self, head_dim: int, block_size: int = 64) -> float:
        """
        Seberapa besar tile FlashAttention yang bisa digunakan.
        Return 1.0 jika tile penuh, < 1.0 jika harus dikecilkan.

        Rumus: FA SMEM per block = 2 × head_dim × block_size × 2 byte (BF16)
        """
        fa_smem_kb = (2 * head_dim * block_size * _BF16_BYTES) / 1024
        if fa_smem_kb <= self.shared_mem_max_kb:
            return 1.0
        feasible = max(16, int(self.shared_mem_max_kb * 1024 / (2 * head_dim * _BF16_BYTES)))
        return min(1.0, feasible / block_size)


# ═════════════════════════════════════════════════════════════════════════════
#  GPU DATABASE  —  11 GPU, Real-World Pretraining Parameters
#  Urutan: entry-level → flagship
# ═════════════════════════════════════════════════════════════════════════════

GPU_DATABASE: Dict[str, GPUSpec] = {

    # ─────────────────────────────────────────────────────────────────────────
    # [1] NVIDIA T4  —  Turing TU104, PCIe, Low-Power Cloud GPU
    # ─────────────────────────────────────────────────────────────────────────
    # Paling umum di cloud VM (GCP g4dn, AWS g4dn). Dirancang inferensi.
    # TDP 70W: sangat terbatas → clock throttle agresif saat training.
    # GDDR6 (bukan HBM): latency tinggi, BW terbatas.
    # Shared mem max hanya 64 KB → FA tile sangat kecil.
    # Real: GPT-2 117M bs=8 → ~20k tok/s (MFU ~21%), LLaMA 7B → ~320 tok/s
    "T4": GPUSpec(
        name="NVIDIA T4", arch="Turing (TU104)",
        vendor="NVIDIA", process_node_nm=12, transistors_b=13.6, die_size_mm2=545.0,

        cuda_cores=2560, sm_count=40, tensor_core_gen=2,
        warp_size=32, max_warps_per_sm=32,  # Turing: max 32 warps/SM

        fp16_tflops=65.0, bf16_tflops=65.0,   # Turing TC: FP16/INT8 saja
        fp32_tflops=8.1,  int8_tops=130.0,
        sparsity_bf16_tflops=0.0,              # Turing tidak punya 2:4 sparsity

        frequency_base_mhz=585.0,
        frequency_boost_mhz=1590.0,
        frequency_sustained_mhz=1365.0,        # diukur: ~1350-1380 MHz training

        vram_gb=16.0, memory_type="GDDR6",
        memory_channels=4,      # 4 GDDR6 channel
        memory_bus_width=256,   # 256-bit
        memory_clock_mhz=6251.0,
        memory_bw_gbps=300.0,

        memory_latency_ns=450.0,   # GDDR6: ~420-480 ns
        ecc_mode="optional_secded",
        ecc_bw_overhead=0.00,      # default OFF (cloud inference config)

        l2_cache_mb=4.0,
        l1_cache_kb=64.0,          # Turing: 64 KB (lebih kecil dari Ampere)
        shared_mem_kb=64.0,
        shared_mem_max_kb=64.0,    # KRITIS: FA tile sangat terbatas!
        l2_bandwidth_tbps=4.0,     # Turing: ~4 TB/s L2

        nvlink_version=0, nvlink_links=0, nvlink_bw_gbps=0.0,
        nvlink_latency_us=0.0, nvlink_congestion_onset=1.0,
        pcie_version="3.0", pcie_lanes=16, pcie_bw_gbps=16.0,

        max_power_w=70, tdp_sustained_w=68,
        thermal_factor=0.86,       # 1365/1590 = 0.859 → 14% hilang karena throttle

        kernel_launch_us=9.0,      # Turing: 8-10 µs overhead
        mem_alloc_overhead=0.04,
        typical_sm_occupancy=0.62, # rendah: 32 warps max/SM

        mfu_typical_min=0.18,
        mfu_typical_max=0.26,      # dikalibrasi nanoGPT T4 benchmarks

        flash_attn_speedup=2.2,    # tile sangat kecil (64 KB SMEM)
        runtime_variance_pct=3.8,  # clock throttle variabel
        dataloader_stall_frac=0.05,
        mig_max_instances=0,

        hbm_efficiency_streaming=0.80,  # GDDR6 streaming
        hbm_efficiency_random=0.50,     # GDDR6 random: latency bottleneck
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [2] NVIDIA V100 16GB  —  Volta GV100, SXM2, HBM2
    # ─────────────────────────────────────────────────────────────────────────
    # GPU datacenter pertama dengan Tensor Core (FP16 only, belum BF16).
    # SXM2: NVLink 2.0 → 300 GB/s bidirectional.
    # HBM2 900 GB/s — 3× T4 → kunci untuk training pre-2020.
    # Real: Megatron-LM GPT-3 scale → MFU 26-38%.
    "V100-16GB": GPUSpec(
        name="NVIDIA V100 SXM2 16GB", arch="Volta (GV100)",
        vendor="NVIDIA", process_node_nm=12, transistors_b=21.1, die_size_mm2=815.0,

        cuda_cores=5120, sm_count=80, tensor_core_gen=1,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=112.0, bf16_tflops=112.0,  # Volta TC: FP16 saja
        fp32_tflops=14.0,  int8_tops=0.0,       # TIDAK ADA INT8 TC di Volta
        sparsity_bf16_tflops=0.0,

        frequency_base_mhz=1230.0,
        frequency_boost_mhz=1530.0,
        frequency_sustained_mhz=1370.0,         # ~1350-1400 MHz

        vram_gb=16.0, memory_type="HBM2",
        memory_channels=4,        # 4 HBM2 stack
        memory_bus_width=4096,    # 4096-bit
        memory_clock_mhz=877.0,
        memory_bw_gbps=900.0,

        memory_latency_ns=100.0,   # HBM2: ~80-120 ns
        ecc_mode="full_in_band",   # HBM: ECC selalu aktif
        ecc_bw_overhead=0.025,

        l2_cache_mb=6.0,
        l1_cache_kb=128.0,         # Volta: 128 KB unified L1
        shared_mem_kb=96.0,
        shared_mem_max_kb=96.0,
        l2_bandwidth_tbps=9.7,

        nvlink_version=2, nvlink_links=6,
        nvlink_bw_gbps=300.0,
        nvlink_latency_us=1.0,
        nvlink_congestion_onset=0.75,
        pcie_version="3.0", pcie_lanes=16, pcie_bw_gbps=16.0,

        max_power_w=300, tdp_sustained_w=275,
        thermal_factor=0.90,       # 1370/1530 = 0.895 ≈ 0.90

        kernel_launch_us=5.5,
        mem_alloc_overhead=0.035,
        typical_sm_occupancy=0.68,

        mfu_typical_min=0.26, mfu_typical_max=0.38,

        flash_attn_speedup=2.2,
        runtime_variance_pct=2.2,
        dataloader_stall_frac=0.03,
        mig_max_instances=0,

        hbm_efficiency_streaming=0.88,
        hbm_efficiency_random=0.65,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [3] NVIDIA V100 32GB  —  Volta GV100, SXM2, HBM2
    # ─────────────────────────────────────────────────────────────────────────
    # Identik compute dengan V100-16GB. VRAM 2× → model lebih besar.
    "V100-32GB": GPUSpec(
        name="NVIDIA V100 SXM2 32GB", arch="Volta (GV100)",
        vendor="NVIDIA", process_node_nm=12, transistors_b=21.1, die_size_mm2=815.0,

        cuda_cores=5120, sm_count=80, tensor_core_gen=1,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=112.0, bf16_tflops=112.0,
        fp32_tflops=14.0, int8_tops=0.0,
        sparsity_bf16_tflops=0.0,

        frequency_base_mhz=1230.0,
        frequency_boost_mhz=1530.0,
        frequency_sustained_mhz=1375.0,

        vram_gb=32.0, memory_type="HBM2",
        memory_channels=8,         # 8 half-stack untuk 32 GB
        memory_bus_width=4096,
        memory_clock_mhz=877.0,
        memory_bw_gbps=900.0,

        memory_latency_ns=105.0,   # 8 half-stack → sedikit lebih tinggi
        ecc_mode="full_in_band",
        ecc_bw_overhead=0.025,

        l2_cache_mb=6.0,
        l1_cache_kb=128.0,
        shared_mem_kb=96.0,
        shared_mem_max_kb=96.0,
        l2_bandwidth_tbps=9.7,

        nvlink_version=2, nvlink_links=6,
        nvlink_bw_gbps=300.0,
        nvlink_latency_us=1.0,
        nvlink_congestion_onset=0.75,
        pcie_version="3.0", pcie_lanes=16, pcie_bw_gbps=16.0,

        max_power_w=300, tdp_sustained_w=275,
        thermal_factor=0.90,

        kernel_launch_us=5.5,
        mem_alloc_overhead=0.032,
        typical_sm_occupancy=0.68,

        mfu_typical_min=0.28, mfu_typical_max=0.40,  # sedikit lebih baik (batch lebih besar)

        flash_attn_speedup=2.2,
        runtime_variance_pct=2.0,
        dataloader_stall_frac=0.028,
        mig_max_instances=0,

        hbm_efficiency_streaming=0.88,
        hbm_efficiency_random=0.65,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [4] NVIDIA RTX 3090  —  Ampere GA102, PCIe, Consumer
    # ─────────────────────────────────────────────────────────────────────────
    # Consumer paling kuat era Ampere. 24GB GDDR6X, 3rd gen TC (BF16 native!).
    # Populer researcher karena VRAM besar, harga relatif terjangkau.
    # Clock throttle signifikan (350W TDP consumer).
    # Real: GPT-2 117M bs=4 → 25-30k tok/s (MFU ~26-30%).
    "RTX-3090": GPUSpec(
        name="NVIDIA RTX 3090", arch="Ampere (GA102)",
        vendor="NVIDIA", process_node_nm=8, transistors_b=28.3, die_size_mm2=628.4,

        cuda_cores=10496, sm_count=82, tensor_core_gen=3,
        warp_size=32, max_warps_per_sm=48,     # GA102 consumer: 48 warps/SM

        fp16_tflops=71.0, bf16_tflops=71.0,
        fp32_tflops=35.6, int8_tops=142.0,
        sparsity_bf16_tflops=142.0,

        frequency_base_mhz=1395.0,
        frequency_boost_mhz=1695.0,
        frequency_sustained_mhz=1505.0,       # ~1450-1550 MHz saat training

        vram_gb=24.0, memory_type="GDDR6X",
        memory_channels=12,
        memory_bus_width=384,
        memory_clock_mhz=9751.0,
        memory_bw_gbps=936.0,

        memory_latency_ns=420.0,   # GDDR6X: ~400-440 ns
        ecc_mode="optional_secded",
        ecc_bw_overhead=0.00,

        l2_cache_mb=6.0,
        l1_cache_kb=128.0,
        shared_mem_kb=100.0,
        shared_mem_max_kb=100.0,
        l2_bandwidth_tbps=12.0,

        nvlink_version=0, nvlink_links=0, nvlink_bw_gbps=0.0,
        nvlink_latency_us=0.0, nvlink_congestion_onset=1.0,
        pcie_version="4.0", pcie_lanes=16, pcie_bw_gbps=32.0,

        max_power_w=350, tdp_sustained_w=300,
        thermal_factor=0.87,       # 1505/1695 = 0.888 → power throttle juga

        kernel_launch_us=6.5,
        mem_alloc_overhead=0.04,
        typical_sm_occupancy=0.65,

        mfu_typical_min=0.25, mfu_typical_max=0.42,

        flash_attn_speedup=2.6,    # 100 KB SMEM → tile lebih besar dari T4
        runtime_variance_pct=3.0,
        dataloader_stall_frac=0.05,
        mig_max_instances=0,

        hbm_efficiency_streaming=0.85,
        hbm_efficiency_random=0.58,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [5] NVIDIA RTX A6000  —  Ampere GA102, PCIe, Professional Workstation
    # ─────────────────────────────────────────────────────────────────────────
    # Professional Ampere workstation. 48 GB GDDR6 (bukan GDDR6X!).
    # BW lebih rendah dari RTX-3090 (768 vs 936 GB/s) tapi VRAM 2×.
    # NVLink Bridge: P2P 2-GPU, 112 GB/s bidirectional.
    # Designed untuk sustained workload → thermal lebih stabil dari consumer.
    "A6000": GPUSpec(
        name="NVIDIA RTX A6000", arch="Ampere (GA102)",
        vendor="NVIDIA", process_node_nm=8, transistors_b=28.3, die_size_mm2=628.4,

        cuda_cores=10752, sm_count=84, tensor_core_gen=3,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=77.4, bf16_tflops=77.4,
        fp32_tflops=38.7, int8_tops=154.8,
        sparsity_bf16_tflops=154.8,

        frequency_base_mhz=1200.0,
        frequency_boost_mhz=1800.0,
        frequency_sustained_mhz=1590.0,       # professional: lebih stabil

        vram_gb=48.0, memory_type="GDDR6",    # GDDR6 bukan GDDR6X!
        memory_channels=12,
        memory_bus_width=384,
        memory_clock_mhz=8000.0,
        memory_bw_gbps=768.0,                  # 384-bit × 16 GT/s ÷ 8

        memory_latency_ns=440.0,   # GDDR6: ~420-460 ns
        ecc_mode="optional_secded",
        ecc_bw_overhead=0.00,

        l2_cache_mb=6.0,
        l1_cache_kb=128.0,
        shared_mem_kb=100.0,
        shared_mem_max_kb=100.0,
        l2_bandwidth_tbps=12.0,

        # NVLink Bridge: P2P 2-GPU (bukan NVLink fabric seperti A100)
        nvlink_version=3, nvlink_links=1,
        nvlink_bw_gbps=112.0,
        nvlink_latency_us=1.5,
        nvlink_congestion_onset=0.80,
        pcie_version="4.0", pcie_lanes=16, pcie_bw_gbps=32.0,

        max_power_w=300, tdp_sustained_w=270,
        thermal_factor=0.89,       # 1590/1800 = 0.883 ≈ 0.89

        kernel_launch_us=6.0,
        mem_alloc_overhead=0.035,
        typical_sm_occupancy=0.66,

        mfu_typical_min=0.24, mfu_typical_max=0.38,

        flash_attn_speedup=2.6,
        runtime_variance_pct=2.8,
        dataloader_stall_frac=0.04,
        mig_max_instances=0,

        hbm_efficiency_streaming=0.84,
        hbm_efficiency_random=0.60,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [6] NVIDIA A100 40GB  —  Ampere GA100, SXM4, HBM2e
    # ─────────────────────────────────────────────────────────────────────────
    # BF16 TC native pertama kali! HBM2e 1555 GB/s, NVLink 3.0 600 GB/s.
    # MIG support: 7 instance per GPU.
    # Gold standard LLM training 2021-2023.
    # Real: MLPerf GPT-3 175B 1024 GPU → MFU ~46%; LLaMA 7B single → ~38%.
    "A100-40GB": GPUSpec(
        name="NVIDIA A100 SXM4 40GB", arch="Ampere (GA100)",
        vendor="NVIDIA", process_node_nm=7, transistors_b=54.2, die_size_mm2=826.0,

        cuda_cores=6912, sm_count=108, tensor_core_gen=3,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=312.0, bf16_tflops=312.0,
        fp32_tflops=19.5, int8_tops=624.0,
        sparsity_bf16_tflops=624.0,

        frequency_base_mhz=1095.0,
        frequency_boost_mhz=1410.0,
        frequency_sustained_mhz=1305.0,       # ~1290-1320 MHz

        vram_gb=40.0, memory_type="HBM2e",
        memory_channels=5,
        memory_bus_width=5120,
        memory_clock_mhz=1215.0,
        memory_bw_gbps=1555.0,

        memory_latency_ns=85.0,    # HBM2e: ~80-90 ns
        ecc_mode="full_in_band",
        ecc_bw_overhead=0.025,

        l2_cache_mb=40.0,          # A100: 40 MB L2 (sangat besar!)
        l1_cache_kb=192.0,
        shared_mem_kb=164.0,
        shared_mem_max_kb=164.0,
        l2_bandwidth_tbps=19.5,

        nvlink_version=3, nvlink_links=12,
        nvlink_bw_gbps=600.0,
        nvlink_latency_us=0.6,
        nvlink_congestion_onset=0.72,
        pcie_version="4.0", pcie_lanes=16, pcie_bw_gbps=64.0,

        max_power_w=400, tdp_sustained_w=370,
        thermal_factor=0.93,       # SXM4 cooling excellent: 1305/1410 = 0.926

        kernel_launch_us=3.5,
        mem_alloc_overhead=0.025,
        typical_sm_occupancy=0.75,

        mfu_typical_min=0.34, mfu_typical_max=0.52,

        flash_attn_speedup=3.0,    # 164 KB SMEM → tile optimal
        runtime_variance_pct=1.8,
        dataloader_stall_frac=0.025,
        mig_max_instances=7,

        hbm_efficiency_streaming=0.90,
        hbm_efficiency_random=0.68,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [7] NVIDIA A100 80GB  —  Ampere GA100, SXM4, HBM2e
    # ─────────────────────────────────────────────────────────────────────────
    # Compute identik dengan A100-40GB, VRAM 2×, BW lebih tinggi (2000 GB/s).
    # Digunakan: LLaMA-2 70B, GPT-4 training (spekulasi), BLOOM, PaLM.
    "A100-80GB": GPUSpec(
        name="NVIDIA A100 SXM4 80GB", arch="Ampere (GA100)",
        vendor="NVIDIA", process_node_nm=7, transistors_b=54.2, die_size_mm2=826.0,

        cuda_cores=6912, sm_count=108, tensor_core_gen=3,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=312.0, bf16_tflops=312.0,
        fp32_tflops=19.5, int8_tops=624.0,
        sparsity_bf16_tflops=624.0,

        frequency_base_mhz=1095.0,
        frequency_boost_mhz=1410.0,
        frequency_sustained_mhz=1310.0,

        vram_gb=80.0, memory_type="HBM2e",
        memory_channels=5,
        memory_bus_width=5120,
        memory_clock_mhz=1512.0,
        memory_bw_gbps=2000.0,             # HBM2e 80GB: 2000 GB/s

        memory_latency_ns=82.0,
        ecc_mode="full_in_band",
        ecc_bw_overhead=0.022,

        l2_cache_mb=40.0,
        l1_cache_kb=192.0,
        shared_mem_kb=164.0,
        shared_mem_max_kb=164.0,
        l2_bandwidth_tbps=19.5,

        nvlink_version=3, nvlink_links=12,
        nvlink_bw_gbps=600.0,
        nvlink_latency_us=0.6,
        nvlink_congestion_onset=0.72,
        pcie_version="4.0", pcie_lanes=16, pcie_bw_gbps=64.0,

        max_power_w=400, tdp_sustained_w=375,
        thermal_factor=0.94,

        kernel_launch_us=3.5,
        mem_alloc_overhead=0.022,
        typical_sm_occupancy=0.76,

        mfu_typical_min=0.36, mfu_typical_max=0.54,

        flash_attn_speedup=3.0,
        runtime_variance_pct=1.5,
        dataloader_stall_frac=0.022,
        mig_max_instances=7,

        hbm_efficiency_streaming=0.91,
        hbm_efficiency_random=0.70,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [8] NVIDIA RTX 4090  —  Ada Lovelace AD102, PCIe, Consumer Flagship
    # ─────────────────────────────────────────────────────────────────────────
    # Consumer flagship 2022-2024. 4th gen TC: FP8, BF16, INT8.
    # L2 cache 72 MB (!) — 12× RTX-3090 → membantu model kecil-menengah.
    # GDDR6X 1008 GB/s — tertinggi GDDR. Tidak ada NVLink.
    # Real: MFU ~28-45% untuk model <30B.
    "RTX-4090": GPUSpec(
        name="NVIDIA RTX 4090", arch="Ada Lovelace (AD102)",
        vendor="NVIDIA", process_node_nm=4, transistors_b=76.3, die_size_mm2=608.4,

        cuda_cores=16384, sm_count=128, tensor_core_gen=4,
        warp_size=32, max_warps_per_sm=48,     # Ada: 48 warps/SM

        fp16_tflops=165.0, bf16_tflops=165.0,
        fp32_tflops=82.6, int8_tops=330.0,
        sparsity_bf16_tflops=330.0,

        frequency_base_mhz=2235.0,
        frequency_boost_mhz=2520.0,
        frequency_sustained_mhz=2190.0,        # ~2100-2200 MHz saat training

        vram_gb=24.0, memory_type="GDDR6X",
        memory_channels=12,
        memory_bus_width=384,
        memory_clock_mhz=10501.0,
        memory_bw_gbps=1008.0,

        memory_latency_ns=400.0,
        ecc_mode="optional_secded",
        ecc_bw_overhead=0.00,

        l2_cache_mb=72.0,          # L2 MASSIVE: 72 MB! (key differentiator Ada)
        l1_cache_kb=128.0,
        shared_mem_kb=100.0,
        shared_mem_max_kb=100.0,
        l2_bandwidth_tbps=20.0,

        nvlink_version=0, nvlink_links=0, nvlink_bw_gbps=0.0,
        nvlink_latency_us=0.0, nvlink_congestion_onset=1.0,
        pcie_version="4.0", pcie_lanes=16, pcie_bw_gbps=32.0,

        max_power_w=450, tdp_sustained_w=380,
        thermal_factor=0.87,       # 2190/2520 = 0.869, power throttle juga

        kernel_launch_us=4.0,
        mem_alloc_overhead=0.03,
        typical_sm_occupancy=0.70,

        mfu_typical_min=0.26, mfu_typical_max=0.45,   # L2 72MB membantu MFU

        flash_attn_speedup=2.8,
        runtime_variance_pct=2.5,
        dataloader_stall_frac=0.04,
        mig_max_instances=0,

        hbm_efficiency_streaming=0.86,
        hbm_efficiency_random=0.62,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [9] NVIDIA H100 PCIe  —  Hopper GH100, PCIe, HBM2e
    # ─────────────────────────────────────────────────────────────────────────
    # Hopper dalam form factor PCIe (lebih murah, tanpa NVLink, tanpa HBM3).
    # 114 SM (vs 132 SXM). Memory HBM2e 2000 GB/s (bukan HBM3).
    # Transformer Engine: FP8 training native.
    # PCIe 5.0: 64 GB/s bidirectional (2× PCIe 4.0).
    "H100-PCIe": GPUSpec(
        name="NVIDIA H100 PCIe", arch="Hopper (GH100)",
        vendor="NVIDIA", process_node_nm=4, transistors_b=80.0, die_size_mm2=814.0,

        cuda_cores=14592, sm_count=114, tensor_core_gen=4,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=800.0, bf16_tflops=800.0,
        fp32_tflops=51.0, int8_tops=1600.0,
        sparsity_bf16_tflops=1600.0,

        frequency_base_mhz=1095.0,
        frequency_boost_mhz=1980.0,
        frequency_sustained_mhz=1880.0,

        vram_gb=80.0, memory_type="HBM2e",  # PCIe: HBM2e, bukan HBM3!
        memory_channels=5,
        memory_bus_width=5120,
        memory_clock_mhz=1512.0,
        memory_bw_gbps=2000.0,

        memory_latency_ns=82.0,
        ecc_mode="full_in_band",
        ecc_bw_overhead=0.020,

        l2_cache_mb=50.0,          # H100: 50 MB L2
        l1_cache_kb=256.0,         # Hopper: 256 KB per SM
        shared_mem_kb=228.0,
        shared_mem_max_kb=228.0,   # Hopper SMEM besar → FA tile optimal!
        l2_bandwidth_tbps=25.0,

        nvlink_version=0, nvlink_links=0, nvlink_bw_gbps=0.0,
        nvlink_latency_us=0.0, nvlink_congestion_onset=1.0,
        pcie_version="5.0", pcie_lanes=16, pcie_bw_gbps=64.0,  # PCIe 5.0!

        max_power_w=350, tdp_sustained_w=330,
        thermal_factor=0.95,       # 1880/1980 = 0.949 ≈ 0.95

        kernel_launch_us=2.5,      # Hopper: overhead kernel terendah
        mem_alloc_overhead=0.020,
        typical_sm_occupancy=0.78,

        mfu_typical_min=0.36, mfu_typical_max=0.55,

        flash_attn_speedup=3.5,    # 228 KB SMEM → tile sangat besar
        runtime_variance_pct=1.5,
        dataloader_stall_frac=0.020,
        mig_max_instances=7,

        hbm_efficiency_streaming=0.92,
        hbm_efficiency_random=0.72,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [10] NVIDIA H100 SXM5  —  Hopper GH100, SXM5, HBM3
    # ─────────────────────────────────────────────────────────────────────────
    # GPU flagship LLM pretraining 2023-2024. "Holy Grail" training GPU.
    # HBM3 3350 GB/s: 1.67× dari A100-80GB → memory wall turun drastis.
    # NVLink 4.0 900 GB/s: all-reduce sangat efisien (NVSwitch 3rd gen).
    # 132 SM (full die) + Transformer Engine (FP8 native).
    # Real: Meta LLaMA-3 70B → MFU ~47%; Mistral 7B → MFU ~52%.
    "H100-SXM": GPUSpec(
        name="NVIDIA H100 SXM5", arch="Hopper (GH100)",
        vendor="NVIDIA", process_node_nm=4, transistors_b=80.0, die_size_mm2=814.0,

        cuda_cores=16896, sm_count=132, tensor_core_gen=4,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=989.0, bf16_tflops=989.0,
        fp32_tflops=67.0, int8_tops=1979.0,
        sparsity_bf16_tflops=1979.0,

        frequency_base_mhz=1095.0,
        frequency_boost_mhz=1980.0,
        frequency_sustained_mhz=1880.0,        # ~1850-1910 MHz

        vram_gb=80.0, memory_type="HBM3",
        memory_channels=5,
        memory_bus_width=5120,
        memory_clock_mhz=2619.0,
        memory_bw_gbps=3350.0,

        memory_latency_ns=75.0,    # HBM3: ~70-80 ns
        ecc_mode="full_in_band",
        ecc_bw_overhead=0.018,

        l2_cache_mb=50.0,
        l1_cache_kb=256.0,
        shared_mem_kb=228.0,
        shared_mem_max_kb=228.0,
        l2_bandwidth_tbps=33.0,    # H100 SXM: 33 TB/s — terbaik!

        nvlink_version=4, nvlink_links=18,
        nvlink_bw_gbps=900.0,
        nvlink_latency_us=0.4,
        nvlink_congestion_onset=0.68,
        pcie_version="5.0", pcie_lanes=16, pcie_bw_gbps=64.0,

        max_power_w=700, tdp_sustained_w=670,
        thermal_factor=0.95,       # SXM5 cooling sangat baik

        kernel_launch_us=2.0,
        mem_alloc_overhead=0.018,
        typical_sm_occupancy=0.80,

        mfu_typical_min=0.40, mfu_typical_max=0.58,

        flash_attn_speedup=3.8,    # 228 KB SMEM + L2 33 TB/s: speedup maksimal
        runtime_variance_pct=1.2,
        dataloader_stall_frac=0.018,
        mig_max_instances=7,

        hbm_efficiency_streaming=0.93,
        hbm_efficiency_random=0.74,
    ),

    # ─────────────────────────────────────────────────────────────────────────
    # [11] NVIDIA H200 SXM  —  Hopper GH100 + HBM3e, SXM5
    # ─────────────────────────────────────────────────────────────────────────
    # H100-SXM + HBM3e: 141 GB + 4800 GB/s (+43% BW vs H100).
    # Compute identik dengan H100-SXM (GH100 die sama).
    # BW 4800 GB/s → model memory-bound pada H100 jadi compute-bound di H200.
    # 141 GB: LLaMA-3 70B full precision + optimizer states dalam 1 GPU.
    # Real: H200 vs H100 inference → ~1.9× throughput; training MFU ~50-70%.
    "H200-SXM": GPUSpec(
        name="NVIDIA H200 SXM", arch="Hopper (GH100 + HBM3e)",
        vendor="NVIDIA", process_node_nm=4, transistors_b=80.0, die_size_mm2=814.0,

        cuda_cores=16896, sm_count=132, tensor_core_gen=4,
        warp_size=32, max_warps_per_sm=64,

        fp16_tflops=989.0, bf16_tflops=989.0,   # compute identik H100-SXM
        fp32_tflops=67.0, int8_tops=1979.0,
        sparsity_bf16_tflops=1979.0,

        frequency_base_mhz=1095.0,
        frequency_boost_mhz=1980.0,
        frequency_sustained_mhz=1900.0,         # sedikit lebih stabil

        vram_gb=141.0, memory_type="HBM3e",     # 141 GB HBM3e!
        memory_channels=8,       # 8 stack HBM3e (6-high die stacking)
        memory_bus_width=6144,   # 6144-bit (8 × 768-bit)
        memory_clock_mhz=3125.0,
        memory_bw_gbps=4800.0,   # 4800 GB/s — +43% dari H100!

        memory_latency_ns=70.0,  # HBM3e: ~65-75 ns — terbaik saat ini
        ecc_mode="full_in_band",
        ecc_bw_overhead=0.015,

        l2_cache_mb=50.0,        # sama dengan H100 (die sama)
        l1_cache_kb=256.0,
        shared_mem_kb=228.0,
        shared_mem_max_kb=228.0,
        l2_bandwidth_tbps=33.0,

        nvlink_version=4, nvlink_links=18,
        nvlink_bw_gbps=900.0,
        nvlink_latency_us=0.4,
        nvlink_congestion_onset=0.65,
        pcie_version="5.0", pcie_lanes=16, pcie_bw_gbps=64.0,

        max_power_w=700, tdp_sustained_w=680,
        thermal_factor=0.96,     # 1900/1980 = 0.959 — paling stabil

        kernel_launch_us=2.0,
        mem_alloc_overhead=0.015,
        typical_sm_occupancy=0.82,  # tertinggi: BW besar → compute unit lebih aktif

        mfu_typical_min=0.50, mfu_typical_max=0.70,  # memory wall hilang!

        flash_attn_speedup=4.0,   # BW 4800 GB/s + SMEM 228 KB: maksimal
        runtime_variance_pct=1.0,
        dataloader_stall_frac=0.015,  # compute cepat → IO jadi bottleneck relatif
        mig_max_instances=7,

        hbm_efficiency_streaming=0.94,
        hbm_efficiency_random=0.76,
    ),
}


# ─── Utility Functions ────────────────────────────────────────────────────────

def get_ridge_point(gpu: GPUSpec) -> float:
    """
    Hitung hardware ridge point (FLOPs/byte).

    Ridge = peak_BF16_FLOPs / peak_eff_BW
    Di atas ridge: compute-bound    Di bawah ridge: memory-bound
    Ref: Williams et al. 2009 "Roofline"
    """
    eff_bw = gpu.effective_memory_bw_gbps * 1e9
    return (gpu.bf16_tflops * 1e12) / eff_bw if eff_bw > 0 else 0.0


def get_effective_bandwidth(gpu: GPUSpec, access_pattern: str = "streaming") -> float:
    """
    BW efektif (GB/s) setelah ECC dan access pattern.

    access_pattern: "streaming" | "random" | "mixed"
    """
    base_bw = gpu.effective_memory_bw_gbps
    if access_pattern == "streaming":
        return base_bw * gpu.hbm_efficiency_streaming
    elif access_pattern == "random":
        return base_bw * gpu.hbm_efficiency_random
    else:
        return base_bw * (0.7 * gpu.hbm_efficiency_streaming +
                          0.3 * gpu.hbm_efficiency_random)


def gpu_pretraining_summary(gpu: GPUSpec) -> dict:
    """Ringkasan performa GPU untuk pretraining — berguna untuk display."""
    return {
        "name":                  gpu.name,
        "ridge_flop_per_byte":   round(get_ridge_point(gpu), 1),
        "eff_bw_streaming_gbps": round(get_effective_bandwidth(gpu, "streaming"), 1),
        "peak_sustained_tflops": round(gpu.peak_sustained_tflops, 1),
        "mfu_range":             f"{gpu.mfu_typical_min*100:.0f}–{gpu.mfu_typical_max*100:.0f}%",
        "memory_type":           gpu.memory_type,
        "latency_ns":            gpu.memory_latency_ns,
        "l2_cache_mb":           gpu.l2_cache_mb,
        "smem_max_kb":           gpu.shared_mem_max_kb,
        "nvlink":                (f"NVLink v{gpu.nvlink_version} {gpu.nvlink_bw_gbps:.0f}GB/s"
                                  if gpu.has_nvlink else "PCIe only"),
        "flash_attn_optimal":    gpu.shared_mem_max_kb >= 100,
        "mig_capable":           gpu.mig_max_instances > 0,
        "pcie_gen":              gpu.pcie_version,
        "thermal_throttle_pct":  round((1 - gpu.thermal_factor) * 100, 1),
        "power_eff_tflops_w":    round(gpu.power_efficiency_tflops_per_w, 3),
        "ecc_mode":              gpu.ecc_mode,
        "ecc_bw_loss_pct":       round(gpu.ecc_bw_overhead * 100, 1),
    }
