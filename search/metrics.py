import math, random
from typing import Dict, Any, Optional
import numpy as np

from hardware import GPUSpec
from arch_types import ArchConfig, FFNType, AttentionType, OptimizerType
from generator import ArchitectureGenerator


class MetricsReport:
    """Assembles full profiling report for an architecture + GPU pair."""

    def __init__(self, cfg: ArchConfig, gpu: GPUSpec, profiler_results: Dict):
        self.cfg = cfg
        self.gpu = gpu
        self.pr  = profiler_results
        # Shared analytics helper (deterministic seed for reproducibility)
        self._gen = ArchitectureGenerator(gpu, rng_seed=42)
        self._gen.rng   = random.Random(42)
        self._gen.nprng = np.random.default_rng(42)

    # ── Memory Breakdown ──────────────────────────────────────────────────────

    def memory_breakdown(self) -> Dict[str, float]:
        cfg = self.cfg
        return {
            "Weights (BF16)":                              cfg.vram_weights_gb,
            "Activations":                                 cfg.vram_activations_gb,
            f"Optimizer ({cfg.optimizer_type.name})":      cfg.vram_optimizer_gb,
            "KV Cache (full seq)":                         cfg.vram_kv_cache_gb,
            "Gradient Buffers":                            round(cfg.vram_weights_gb * 0.50, 3),
            "CUDA Malloc Fragmentation":                   cfg.vram_fragmentation_gb,
        }

    def kv_cache_growth_curve(self) -> Dict[int, float]:
        """KV cache GB as token count increases (up to 2× max seq)."""
        cfg = self.cfg
        bytes_tok_layer = 2 * cfg.num_kv_heads * cfg.head_dim * 2  # BF16
        result = {}
        for t in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            if t <= cfg.seq_len * 2:
                gb = bytes_tok_layer * cfg.num_layers * t * cfg.batch_size / 1e9
                result[t] = round(gb, 5)
        return result

    # ── Compute Breakdown ─────────────────────────────────────────────────────

    def compute_breakdown(self) -> Dict[str, Any]:
        cfg = self.cfg; gen = self._gen
        attn_fwd = cfg.flops_attn_fwd
        ffn_fwd  = cfg.flops_ffn_fwd
        tot_fwd  = cfg.flops_per_token_fwd

        D   = cfg.hidden_dim; H = cfg.num_heads; Hkv = cfg.num_kv_heads; Hd = cfg.head_dim
        L   = cfg.num_layers; S = cfg.seq_len
        qkv_fwd   = 2 * S * D * (H * Hd + Hkv * Hd * 2) * L / S
        o_fwd     = 2 * S * (H * Hd) * D * L / S
        score_fwd = max(0.0, attn_fwd - qkv_fwd - o_fwd)

        eff_flops  = gen._effective_flops_efficiency(cfg)
        tc_util    = gen._tensor_core_utilization(cfg)
        gran       = gen._granularity_penalty(cfg)
        sched      = gen._scheduler_inefficiency(cfg)
        ovlp       = gen._async_compute_overlap_factor(cfg)

        effective_flops_pct = eff_flops * tc_util * gran * 100
        ideal_tflops = round((tot_fwd + cfg.flops_per_token_bwd) * S * cfg.batch_size
                             / 1e12 / max(1e-9, cfg.ms_per_step / 1000), 2)

        return {
            "QKV Projection GFLOP/tok":     round(qkv_fwd / 1e9, 3),
            "Attention Score+V GFLOP/tok":  round(score_fwd / 1e9, 3),
            "Output Projection GFLOP/tok":  round(o_fwd / 1e9, 3),
            "Total Attention GFLOP/tok":    round(attn_fwd / 1e9, 3),
            "FFN GFLOP/tok":                round(ffn_fwd / 1e9, 3),
            "Total Fwd GFLOP/tok":          round(tot_fwd / 1e9, 3),
            "Total Bwd GFLOP/tok":          round(cfg.flops_per_token_bwd / 1e9, 3),
            "Attn/FFN FLOPs Ratio":         round(attn_fwd / max(1.0, ffn_fwd), 3),
            "─── Hardware FLOPs Efficiency ───": "",
            "Effective FLOPs Efficiency":   f"{effective_flops_pct:.1f}%",
            "  └ Exec Efficiency Factor":   f"{eff_flops:.3f}",
            "  └ Tensor Core Utilization":  f"{tc_util:.3f}",
            "  └ Granularity/Size Factor":  f"{gran:.3f}",
            "  └ Scheduler Inefficiency":   f"{sched:.3f}× slowdown",
            "  └ Async Overlap Credit":     f"{ovlp:.3f}× (lower=better)",
            "Achieved TFLOPS (est.)":       f"{ideal_tflops:.2f}",
        }

    # ── Cache Efficiency ──────────────────────────────────────────────────────

    def cache_efficiency(self) -> Dict[str, str]:
        cfg = self.cfg; gpu = self.gpu
        hbm = self._gen._hbm_bandwidth_model(cfg)

        sat = hbm["saturation_pct"]
        if sat > 85:
            hbm_s = f"✗ HBM saturated ({sat:.0f}%) — bandwidth bottleneck"
        elif sat > 60:
            hbm_s = f"~ HBM moderate ({sat:.0f}%) — partial bottleneck"
        else:
            hbm_s = f"✓ HBM comfortable ({sat:.0f}%) — not bottleneck"

        l2_hit = hbm["l2_hit_est_pct"]
        l1_hit = hbm["l1_hit_est_pct"]
        l2_s   = (f"✓ L2 hot ({l2_hit:.0f}% est. hit)" if l2_hit > 60
                  else f"~ L2 moderate ({l2_hit:.0f}%)" if l2_hit > 40
                  else f"✗ L2 cold ({l2_hit:.0f}%) — streaming")
        l1_s   = (f"✓ L1 warm ({l1_hit:.0f}% est.)" if l1_hit > 55
                  else f"~ L1 partial ({l1_hit:.0f}%)")

        return {
            "HBM BW Saturation":     hbm_s,
            "HBM Required (GB/s)":   str(hbm["required_bw_gbps"]),
            "HBM Peak (GB/s)":       str(hbm["peak_bw_gbps"]),
            "L2 Cache Hit Est.":     l2_s,
            "L1 / Shared Mem":       l1_s,
            "Tensor Core Alignment": ("✓ Excellent (128-aligned)" if cfg.hidden_dim % 128 == 0
                                      else "~ Good (64-aligned)" if cfg.hidden_dim % 64 == 0
                                      else "✗ Suboptimal"),
            "Flash Attention":       ("✓ Enabled — O(1) HBM reads for attention"
                                       if cfg.use_flash_attn else "✗ Disabled"),
            "TC Utilization Factor": f"{self._gen._tensor_core_utilization(cfg):.3f}",
            "HBM Stream Efficiency": f"{gpu.hbm_efficiency_streaming*100:.0f}%",
            "HBM Random Efficiency": f"{gpu.hbm_efficiency_random*100:.0f}%",
        }

    # ── Kernel Occupancy ──────────────────────────────────────────────────────

    def kernel_occupancy(self) -> Dict[str, Any]:
        cfg = self.cfg; gpu = self.gpu
        occ    = self.pr.get("kernel_occupancy_est", cfg.sm_occupancy)
        sm_pct = self.pr.get("sm_active_pct", cfg.sm_occupancy * 100)
        warp_eff = self.pr.get("warp_efficiency_est", 1.0 - cfg.warp_divergence_pct / 100)
        wd = self._gen._estimate_warp_divergence(cfg)

        regs_per_thread = 64
        max_blocks_reg  = gpu.max_warps_per_sm * gpu.warp_size // max(1, regs_per_thread)
        smem_kb_used    = min(gpu.shared_mem_kb, cfg.head_dim * 4 / 1024)
        smem_occ_limit  = gpu.shared_mem_kb / max(0.1, smem_kb_used)

        gran_factor  = self._gen._granularity_penalty(cfg)
        sched_factor = self._gen._scheduler_inefficiency(cfg)
        model_occ    = gpu.typical_sm_occupancy * gran_factor

        return {
            "SM Active Pct (est.)":    f"{sm_pct:.1f}%",
            "Kernel Occupancy (est.)": f"{occ*100:.1f}%",
            "Model SM Occupancy":      f"{model_occ*100:.1f}% (size-adjusted)",
            "SM Count":                gpu.sm_count,
            "Warp Efficiency":         f"{warp_eff*100:.1f}%",
            "Warp Divergence":         f"{wd['warp_divergence_pct']:.1f}%",
            "Divergence Cause":        wd["primary_cause"],
            "Granularity Factor":      f"{gran_factor:.3f}  (1.0=optimal GPU fill)",
            "Scheduler Inefficiency":  f"{sched_factor:.3f}× slowdown",
            "L1 Cache / SM":           f"{gpu.l1_cache_kb:.0f} KB",
            "Shared Mem / SM":         f"{gpu.shared_mem_kb:.0f} KB",
            "Est. SMEM Used":          f"{smem_kb_used:.1f} KB",
            "Kernel Ops (profiler)":   str(self.pr.get("kernel_ops", "N/A")),
        }

    # ── MoE Routing Analysis ──────────────────────────────────────────────────

    def moe_routing(self) -> Optional[Dict]:
        cfg = self.cfg
        if cfg.ffn_type not in (FFNType.MOE, FFNType.MOE_TOPK):
            return None
        tokens_per_expert = (cfg.seq_len * cfg.batch_size * cfg.top_k_experts) / cfg.num_experts
        imbalance    = abs(cfg.top_k_experts / cfg.num_experts - 0.25) * 2.0
        overflow_rate = max(0, (tokens_per_expert * (1 + imbalance) /
                                (cfg.seq_len * cfg.batch_size / cfg.num_experts
                                 * cfg.expert_capacity_factor)) - 1.0)
        dispatch_us   = cfg.num_experts * 2.5 + tokens_per_expert * 0.12
        moe_comm_gb   = (tokens_per_expert * cfg.hidden_dim * 2 / 1e9 * cfg.num_experts * 2)
        load_score    = round(1.0 - imbalance, 3)
        routing_overhead_ms = dispatch_us / 1000 * cfg.num_layers

        return {
            "Experts":                      cfg.num_experts,
            "Top-K Active":                 cfg.top_k_experts,
            "Active Ratio":                 f"{cfg.top_k_experts/cfg.num_experts*100:.1f}%",
            "Tokens/Expert (avg)":          round(tokens_per_expert, 1),
            "Expert Capacity Factor":       cfg.expert_capacity_factor,
            "Load Imbalance Score":         round(imbalance, 3),
            "Load Balance Score (1=ideal)": load_score,
            "Overflow Rate (est.)":         f"{overflow_rate*100:.1f}%",
            "Router Dispatch Latency (µs)": round(dispatch_us, 1),
            "Routing Overhead / step (ms)": round(routing_overhead_ms, 2),
            "MoE All-to-All BW (GB/step)":  round(moe_comm_gb, 4),
            "Recommendation":               (
                "✓ Good balance" if imbalance < 0.15
                else "⚠ Increase capacity factor" if overflow_rate > 0.05
                else "~ Acceptable imbalance"
            ),
        }

    # ── Multi-GPU Scaling ─────────────────────────────────────────────────────

    def multi_gpu_estimate(self, n_gpus: int = 4) -> Dict[str, Any]:
        gpu = self.gpu
        if gpu.nvlink_bw_gbps > 0:
            comm_bw   = gpu.nvlink_bw_gbps; comm_type = "NVLink"
            lat_us    = gpu.nvlink_latency_us
        else:
            comm_bw   = gpu.pcie_bw_gbps * 0.78; comm_type = "PCIe"
            lat_us    = 8.5

        grad_gb      = self.cfg.vram_weights_gb
        allreduce_ms = (grad_gb * 1e9 * 2 * (n_gpus - 1) / n_gpus) / (comm_bw * 1e9) * 1000
        compute_ms   = self.pr.get("est_total_ms", self.cfg.ms_per_step)

        bw_demand  = grad_gb * 1e9 * 2 / max(1e-3, compute_ms / 1000)
        sat_ratio  = bw_demand / (comm_bw * 1e9)
        if sat_ratio > gpu.nvlink_congestion_onset:
            congestion = max(0.55, 1.0 - (sat_ratio - gpu.nvlink_congestion_onset) * 0.45)
        else:
            congestion = 1.0
        effective_bw     = comm_bw * congestion
        allreduce_eff_ms = (grad_gb * 1e9 * 2 * (n_gpus - 1) / n_gpus) / (effective_bw * 1e9) * 1000
        ring_lat_ms      = lat_us * math.log2(max(2, n_gpus)) / 1000

        overlap = comm_bw > 100 and gpu.nvlink_bw_gbps > 0
        if overlap:
            step_ms = max(compute_ms, allreduce_eff_ms + ring_lat_ms)
        else:
            step_ms = compute_ms + allreduce_eff_ms + ring_lat_ms

        io_ms = compute_ms * gpu.dataloader_stall_frac
        aol   = self._gen._async_overlap_model(self.cfg, compute_ms, allreduce_eff_ms, io_ms)

        ideal_sp  = n_gpus
        actual_sp = compute_ms / step_ms * n_gpus
        eff       = actual_sp / ideal_sp

        return {
            "GPUs":                     n_gpus,
            "Interconnect":             comm_type,
            "Raw BW (GB/s)":            comm_bw,
            "Congestion Factor":        round(congestion, 3),
            "Effective BW (GB/s)":      round(effective_bw, 1),
            "AllReduce (ideal) ms":     round(allreduce_ms, 2),
            "AllReduce (eff.) ms":      round(allreduce_eff_ms, 2),
            "Ring Latency ms":          round(ring_lat_ms, 3),
            "Compute ms":               round(compute_ms, 2),
            "Async Overlap":            "Yes" if overlap else "No",
            "Total Step ms":            round(step_ms, 2),
            "Effective Speedup":        round(actual_sp, 2),
            f"vs Ideal ({n_gpus}×)":   f"{eff*100:.1f}% efficiency",
            "Async Overlap Benefit ms": aol["overlap_benefit_ms"],
        }

    # ── Sequence Scaling ──────────────────────────────────────────────────────

    def sequence_scaling_estimate(self) -> Dict[int, float]:
        """
        FIX-METRICS-1: Sequence length scaling.

        FLOPs breakdown:
          FFN:       scales O(S)   — linear with token count
          Attention: scales O(S²)  — quadratic regardless of FlashAttention
            FlashAttn does NOT reduce FLOPs, only HBM traffic (no S×S matrix stored).
            Time ≈ max(FLOPs_time, Memory_time) from roofline.

        Memory scaling:
          FFN activations:  O(S)
          Attention matrix: O(S²) without FlashAttn (stored softmax weights)
                            O(S)  with FlashAttn    (recomputed in tiled blocks)
          KV cache:         O(S)

        Combined time:
          Without FA: dominated by O(S²) in both FLOPs and memory
          With FA:    FLOPs still O(S²), but memory O(S) — wall time bottleneck shifts
                      to FLOPs for large S (compute-bound at long context with FA)
        """
        import math
        cfg      = self.cfg
        base_ms  = self.pr.get("est_total_ms", cfg.ms_per_step)
        base_seq = cfg.seq_len

        # Fraction of FLOPs from attention (quadratic) vs FFN (linear)
        attn_flop_frac = cfg.flops_attn_fwd / max(1.0, cfg.flops_per_token_fwd)
        ffn_flop_frac  = 1.0 - attn_flop_frac

        # HBM traffic fraction from attention matrix (S×S) vs other
        if cfg.use_flash_attn:
            # FlashAttn: no S×S matrix in HBM → attention memory is O(S), not O(S²)
            # Memory is dominated by weights (constant) + activations (O(S)) + KV (O(S))
            attn_mem_frac = 0.0      # S×S matrix NOT in HBM; attention memory scales O(S)
            other_mem_frac = 1.0     # everything else: linear
        else:
            # Naive attention: S×S softmax weights in HBM → O(S²) memory
            # Estimate: attn S×S traffic vs total traffic
            s     = base_seq
            b     = cfg.batch_size
            h     = cfg.num_heads
            hd    = cfg.head_dim
            d     = cfg.hidden_dim
            attn_matrix_gb  = b * h * s * s * 4 / 1e9   # fp32 softmax
            total_act_gb    = cfg.vram_activations_gb
            attn_mem_frac   = min(0.80, attn_matrix_gb / max(0.001, total_act_gb))
            other_mem_frac  = 1.0 - attn_mem_frac

        results = {}
        for seq in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            r = seq / max(1, base_seq)

            # FLOPs time: FFN linear, attention quadratic (always O(S²))
            flop_scale = ffn_flop_frac * r + attn_flop_frac * r ** 2

            # Memory time: depends on FlashAttn
            if cfg.use_flash_attn:
                # All memory traffic O(S): KV cache, activations, weights (constant)
                mem_scale = r   # linear — FA eliminates O(S²) memory bottleneck
            else:
                # S×S attention matrix: O(S²); rest: O(S)
                mem_scale = attn_mem_frac * r ** 2 + other_mem_frac * r

            # Roofline: time ≈ max(compute_time, memory_time)
            # Approximate: weight by arithmetic intensity vs ridge
            ai    = cfg.arithmetic_intensity
            ridge = (self.gpu.bf16_tflops * 1e12) / (self.gpu.memory_bw_gbps * 1e9)
            if ai >= ridge:
                # Compute-bound: FLOPs dominate
                total_scale = flop_scale
            elif ai < ridge * 0.4:
                # Memory-bound: memory traffic dominates
                total_scale = mem_scale
            else:
                # Balanced: weighted average
                w_compute = (ai / ridge)
                total_scale = w_compute * flop_scale + (1 - w_compute) * mem_scale

            # KV cache pressure: if new KV exceeds 10% of VRAM, add stall overhead
            kv_extra_gb = (seq - base_seq) * cfg.num_kv_heads * cfg.head_dim * cfg.num_layers * 2 / 1e9
            if kv_extra_gb > self.gpu.vram_gb * 0.10:
                total_scale *= 1.12   # HBM pressure from KV cache swapping

            t_ms = base_ms * max(0.0, total_scale)
            if t_ms > 0:
                results[seq] = int(seq * cfg.batch_size / (t_ms / 1000.0))
            else:
                results[seq] = 0

        return results

    # ── Batch Scaling ─────────────────────────────────────────────────────────

    def batch_scaling_estimate(self) -> Dict[int, float]:
        cfg        = self.cfg
        base_vram  = cfg.vram_weights_gb + cfg.vram_optimizer_gb
        act_sample = cfg.vram_activations_gb / max(1, cfg.batch_size)
        kv_sample  = cfg.vram_kv_cache_gb   / max(1, cfg.batch_size)
        base_ms    = self.pr.get("est_total_ms", cfg.ms_per_step)
        serial_frac = 0.06 + 0.008 * math.log2(max(1, cfg.num_layers))

        results = {}
        for bs in [1, 2, 4, 8, 16, 32, 64]:
            vram_needed = (base_vram + act_sample * bs + kv_sample * bs
                           + cfg.vram_fragmentation_gb * (bs / max(1, cfg.batch_size)) ** 0.7)
            # Batch scaling berhenti di ≤80% VRAM (sinkron dengan VRAM_LIMIT_PCT)
            if vram_needed > self.gpu.vram_gb * 0.80:
                break
            amdahl_sp  = 1.0 / (serial_frac + (1 - serial_frac) / max(1, bs))
            mem_penalty = max(0.85, 1.0 - 0.02 * math.log2(max(1, bs)))
            eff_sp     = amdahl_sp * mem_penalty
            t_ms       = base_ms * (bs / max(1, cfg.batch_size)) / eff_sp
            results[bs] = int(cfg.seq_len * bs / max(1e-6, t_ms / 1000)) if t_ms > 0 else 0
        return results

    # ── Compiler Effects ──────────────────────────────────────────────────────

    def compiler_effects(self) -> Dict[str, Any]:
        cfg    = self.cfg
        fusion = self._gen._kernel_fusion_analysis(cfg)
        return {
            "torch.compile Enabled":   "Yes" if cfg.use_torch_compile else "No",
            "Compiler Speedup (est.)": f"{fusion['compiler_speedup']:.3f}×",
            "Fusion Ratio":            f"{fusion['fusion_ratio']*100:.0f}%",
            "CUDA Graph Compatible":   "Yes" if fusion["cuda_graph_compatible"] else "No (MoE routing breaks graph)",
            "Fused Kernels":           len(fusion["fused_kernels"]),
            "Unfused Kernels":         len(fusion["unfused_kernels"]),
            "Inductor / Triton":       "Triton + Inductor" if cfg.use_torch_compile else "PyTorch eager",
        }

    # ── Thermal & Noise ───────────────────────────────────────────────────────

    def thermal_and_noise(self) -> Dict[str, Any]:
        thermal = self._gen._thermal_model()
        noise   = self._gen._stochastic_noise_model(self.cfg)
        return {**thermal, **noise}

    # ── Activation Recomputation ──────────────────────────────────────────────

    def recomputation_cost(self) -> Optional[Dict]:
        return self._gen._activation_recomputation_cost(self.cfg)

    # ── Optimizer State Detail ────────────────────────────────────────────────

    def optimizer_detail(self) -> Dict[str, Any]:
        cfg = self.cfg; ot = cfg.optimizer_type; gb = cfg.vram_optimizer_gb
        bytes_per_param = gb * 1e9 / max(1, cfg.param_count)
        return {
            "Type":           ot.value,
            "Total GB":       round(gb, 3),
            "Bytes / Param":  round(bytes_per_param, 2),
            "vs Weights":     f"{gb / max(0.001, cfg.vram_weights_gb):.1f}×",
            "Supported Dtypes": ("FP32" if ot in (OptimizerType.ADAM_FP32,
                                                    OptimizerType.ZERO1,
                                                    OptimizerType.ZERO2,
                                                    OptimizerType.ZERO3)
                                  else "INT8+FP32" if ot == OptimizerType.ADAM_8BIT
                                  else "BF16" if ot == OptimizerType.ADAMW_BF16
                                  else "FP32"),
            "ZeRO Sharding":  ("Yes" if ot in (OptimizerType.ZERO1,
                                                 OptimizerType.ZERO2,
                                                 OptimizerType.ZERO3)
                                else "No"),
        }
