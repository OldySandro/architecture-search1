import time, math, random, hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# ─── VRAM Budget ──────────────────────────────────────────────────────────────
VRAM_LIMIT_PCT: float = 0.80   # arsitektur diterima hanya jika VRAM ≤ 80%

from hardware import GPUSpec
from arch_types import (
    ArchConfig, AttentionType, FFNType, NormType, PosEncType, OptimizerType
)


class ArchitectureGenerator:
    """
    Generates novel pretraining architectures dynamically.
    GPU-aware: tiap GPU menghasilkan range parameter yang berbeda.
    """

    # ── 7 Family Profiles ──────────────────────────────────────────────────────
    FAMILY_PROFILES = {
        "CoT-Optimizer": dict(
            desc="Deep narrow: chain-of-thought reasoning, reasoning-heavy tasks",
            # Expanded: layer lebih dalam (hingga 64), hidden lebih lebar (hingga 2048),
            # ffn_mult lebih tinggi untuk chain-of-thought dense
            layer_range=(20, 64), hidden_range=(512, 2048),
            head_range=(8, 32), ffn_mult_range=(2.5, 5.0),
            attn_candidates=[AttentionType.GQA, AttentionType.ROPE, AttentionType.MHA],
            ffn_candidates=[FFNType.GEGLU, FFNType.GATED, FFNType.DENSE],
            seq_range=(2048, 8192), gc_prob=0.7,
            opt_candidates=[OptimizerType.ADAM_FP32, OptimizerType.ADAMW_BF16,
                            OptimizerType.ZERO1],
        ),
        "Speed-Demon": dict(
            desc="Wide shallow: maximum tokens/sec throughput on limited VRAM",
            # Expanded: layer lebih bervariasi, hidden hingga 3072 untuk throughput
            layer_range=(4, 20), hidden_range=(512, 3072),
            head_range=(4, 16), ffn_mult_range=(1.5, 4.0),
            attn_candidates=[AttentionType.MQA, AttentionType.GQA, AttentionType.LINEAR],
            ffn_candidates=[FFNType.DENSE, FFNType.GATED, FFNType.GEGLU],
            seq_range=(256, 4096), gc_prob=0.0,
            opt_candidates=[OptimizerType.ADAM_8BIT, OptimizerType.LION,
                            OptimizerType.ADAMW_BF16],
        ),
        "Balanced-Pro": dict(
            desc="Balanced depth/width: general-purpose pretraining",
            # Expanded: range lebih lebar di kedua ujung untuk diversitas kandidat
            layer_range=(8, 40), hidden_range=(768, 3072),
            head_range=(8, 32), ffn_mult_range=(3.0, 5.5),
            attn_candidates=[AttentionType.GQA, AttentionType.MHA,
                             AttentionType.ROPE, AttentionType.HYBRID],
            ffn_candidates=[FFNType.DENSE, FFNType.GEGLU, FFNType.GATED],
            seq_range=(1024, 8192), gc_prob=0.3,
            opt_candidates=[OptimizerType.ADAM_FP32, OptimizerType.ZERO1,
                            OptimizerType.ADAMW_BF16],
        ),
        "MoE-Sparse": dict(
            desc="Mixture-of-Experts: massive capacity with sparse compute",
            # Expanded: layer lebih dalam, hidden lebih variasi, ffn_mult lebih lebar
            # Expert count 4–16 tergantung VRAM (diatur di generate_one)
            layer_range=(6, 28), hidden_range=(512, 1792),
            head_range=(8, 24), ffn_mult_range=(0.5, 2.5),
            attn_candidates=[AttentionType.GQA, AttentionType.MQA, AttentionType.ROPE],
            ffn_candidates=[FFNType.MOE, FFNType.MOE_TOPK],
            seq_range=(1024, 4096), gc_prob=0.6,
            opt_candidates=[OptimizerType.ADAM_8BIT, OptimizerType.LION,
                            OptimizerType.ZERO2, OptimizerType.ZERO3],
        ),
        "Long-Horizon": dict(
            desc="Extended context: long-range dependencies, hybrid attention",
            # Expanded: seq_len hingga 32k, hidden lebih bervariasi
            layer_range=(8, 36), hidden_range=(768, 3072),
            head_range=(8, 32), ffn_mult_range=(2.5, 4.5),
            attn_candidates=[AttentionType.SLIDE, AttentionType.HYBRID,
                             AttentionType.ALIBI, AttentionType.ROPE],
            ffn_candidates=[FFNType.GEGLU, FFNType.DENSE, FFNType.GATED],
            seq_range=(4096, 32768), gc_prob=0.85,
            opt_candidates=[OptimizerType.ADAM_8BIT, OptimizerType.ZERO2,
                            OptimizerType.ADAMW_BF16],
        ),
        "Nano-Efficient": dict(
            desc="Ultra-small: maximum quality per VRAM byte, embedded/edge pretrain",
            # Expanded: sedikit lebih dalam untuk diversitas, hidden lebih lebar
            layer_range=(2, 16), hidden_range=(128, 1024),
            head_range=(2, 12), ffn_mult_range=(1.5, 4.0),
            attn_candidates=[AttentionType.MQA, AttentionType.LINEAR,
                             AttentionType.GQA],
            ffn_candidates=[FFNType.GATED, FFNType.DENSE, FFNType.GEGLU],
            seq_range=(256, 4096), gc_prob=0.0,
            opt_candidates=[OptimizerType.LION, OptimizerType.ADAM_8BIT,
                            OptimizerType.ADAMW_BF16],
        ),
        "Compute-Dense": dict(
            desc="High FLOP/byte ratio: exploits tensor cores maximally on large GPU",
            # Expanded: hingga 96 layers untuk benar-benar dense, hidden hingga 6144
            layer_range=(24, 96), hidden_range=(1536, 6144),
            head_range=(16, 64), ffn_mult_range=(3.5, 7.0),
            attn_candidates=[AttentionType.GQA, AttentionType.MHA, AttentionType.ROPE],
            ffn_candidates=[FFNType.DENSE, FFNType.GEGLU, FFNType.GATED],
            seq_range=(2048, 16384), gc_prob=0.95,
            opt_candidates=[OptimizerType.ZERO1, OptimizerType.ZERO2,
                            OptimizerType.ADAM_FP32],
        ),
    }

    def __init__(self, gpu: GPUSpec, rng_seed: Optional[int] = None):
        self.gpu   = gpu
        self.rng   = random.Random(rng_seed or int(time.time() * 1000) % 99999)
        self.nprng = np.random.default_rng(rng_seed or 42)
        self._arch_counter = 0

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Parameter Count
    # ═════════════════════════════════════════════════════════════════════════

    def _make_divisible(self, n, divisor=64):
        return max(divisor, (n // divisor) * divisor)

    def _compute_params(self, cfg: ArchConfig) -> int:
        """Total trainable parameters (exact formula)."""
        D = cfg.hidden_dim; L = cfg.num_layers; V = cfg.vocab_size
        H = cfg.num_heads; Hkv = cfg.num_kv_heads; Hd = cfg.head_dim
        ffn = int(D * cfg.ffn_multiplier)
        E   = cfg.num_experts if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK) else 1

        emb    = V * D
        q_proj = D * (H   * Hd)
        k_proj = D * (Hkv * Hd)
        v_proj = D * (Hkv * Hd)
        o_proj = (H * Hd) * D
        attn   = q_proj + k_proj + v_proj + o_proj

        if cfg.ffn_type in (FFNType.GEGLU, FFNType.GATED):
            ffn_params = D * ffn * 2 + ffn * D
        elif cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            ffn_params = E * (D * ffn * 2 + ffn * D) + D * E
        else:
            ffn_params = D * ffn + ffn * D

        norm_mult = 1 if cfg.norm_type == NormType.RMSNORM else 2
        norms     = 2 * norm_mult * D
        layer_par = attn + ffn_params + norms
        total     = emb + L * layer_par
        if not cfg.tie_embeddings:
            total += D * V
        return int(total)

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Memory Model
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_fragmentation(self, cfg: ArchConfig) -> float:
        gpu       = self.gpu
        base_frag = gpu.mem_alloc_overhead
        # Fragmentation bertambah dengan jumlah layer (banyak tensor kecil)
        layer_frag = min(0.06, 0.001 * cfg.num_layers)
        # MoE menambah fragmentasi karena expert dispatch buffer
        moe_frag   = 0.025 if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK) else 0.0
        raw_gb     = cfg.vram_weights_gb + cfg.vram_activations_gb + cfg.vram_optimizer_gb
        pressure   = raw_gb / max(1.0, gpu.vram_gb)
        # Pressure fragmentation: makin penuh VRAM → allocator makin sering gagal
        pres_frag  = max(0.0, (pressure - 0.70) * 0.12) if pressure > 0.70 else 0.0
        return min(0.15, base_frag + layer_frag + moe_frag + pres_frag)

    def _compute_optimizer_memory(self, cfg: ArchConfig, n_gpus: int = 1) -> float:
        P = cfg.param_count; ot = cfg.optimizer_type
        if ot == OptimizerType.ADAM_FP32:
            gb = P * 4 * 3 / 1e9
        elif ot == OptimizerType.ADAM_8BIT:
            gb = P * (4 + 1 + 1) / 1e9
        elif ot == OptimizerType.LION:
            gb = P * 4 * 2 / 1e9
        elif ot == OptimizerType.ADAMW_BF16:
            gb = P * 2 * 3 / 1e9
        elif ot == OptimizerType.ZERO1:
            gb = P * 4 * 3 / 1e9 / max(1, n_gpus)
        elif ot == OptimizerType.ZERO2:
            gb = (P * 4 * 3 + P * 2) / 1e9 / max(1, n_gpus)
        elif ot == OptimizerType.ZERO3:
            gb = (P * 4 * 3 + P * 2 + P * 2) / 1e9 / max(1, n_gpus)
        else:
            gb = P * 4 * 3 / 1e9
        return gb

    def _compute_kv_cache_gb(self, cfg: ArchConfig) -> float:
        bytes_tok_layer = 2 * cfg.num_kv_heads * cfg.head_dim * 2  # BF16
        return bytes_tok_layer * cfg.num_layers * cfg.seq_len * cfg.batch_size / 1e9

    def _compute_memory(self, cfg: ArchConfig) -> Tuple[float, float, float, float]:
        """
        Returns (weights_gb, activations_gb, optimizer_gb, kv_cache_gb).
        FIX-3: Exact per-tensor sizes (Q=H×Hd, K/V=Hkv×Hd).
        FIX-4: S×S softmax matrix jika tidak ada FlashAttention.
        FIX-5: +50% overhead untuk gradient activation buffers.
        """
        weights_gb = cfg.param_count * 2 / 1e9  # BF16

        B, S, L = cfg.batch_size, cfg.seq_len, cfg.num_layers
        D, H, Hkv, Hd = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        DTYPE = 2  # BF16 bytes

        prenorm_b    = B * S * D * DTYPE
        q_b          = B * S * H   * Hd * DTYPE
        k_b          = B * S * Hkv * Hd * DTYPE
        v_b          = B * S * Hkv * Hd * DTYPE
        o_in_b       = B * S * D * DTYPE
        attn_layer_bytes = prenorm_b + q_b + k_b + v_b + o_in_b

        if not cfg.use_flash_attn:
            # S×S softmax untuk backward pass
            attn_layer_bytes += B * H * S * S * 4

        ffn_dim = int(D * cfg.ffn_multiplier)
        if cfg.ffn_type in (FFNType.GEGLU, FFNType.GATED):
            ffn_layer_bytes = B * S * ffn_dim * DTYPE * 3
        elif cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            active_bytes    = B * S * ffn_dim * DTYPE * cfg.top_k_experts
            router_bytes    = B * S * cfg.num_experts * 4
            ffn_layer_bytes = active_bytes + router_bytes
        else:
            ffn_layer_bytes = B * S * ffn_dim * DTYPE * 2

        total_act_bytes = L * (attn_layer_bytes + ffn_layer_bytes)
        total_act_bytes *= 1.50   # FIX-5: backward grad buffers

        total_act = total_act_bytes / 1e9

        if cfg.use_gradient_checkpointing:
            segs      = max(1, int(math.sqrt(L)))
            save_frac = segs / L
            total_act = total_act * save_frac + (total_act / 1.50) * (1 - save_frac) * 0.08

        optimizer_gb = self._compute_optimizer_memory(cfg)
        kv_cache_gb  = self._compute_kv_cache_gb(cfg)
        return weights_gb, total_act, optimizer_gb, kv_cache_gb

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — FLOPs
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_flops(self, cfg: ArchConfig) -> Tuple[float, float, float, float]:
        """
        Returns (fwd_total, bwd_total, attn_fwd, ffn_fwd) per token.
        FIX-1: bwd = attn_fwd×2.5 + ffn_fwd×2.0.
        FIX-2: Linear attention → kernel-trick O(S·H·Hd²).
        """
        D = cfg.hidden_dim; L = cfg.num_layers
        H = cfg.num_heads; Hkv = cfg.num_kv_heads; Hd = cfg.head_dim
        S = cfg.seq_len; ffn = int(D * cfg.ffn_multiplier)
        attn = cfg.attn_type

        q_flops = 2 * S * D * (H   * Hd)
        k_flops = 2 * S * D * (Hkv * Hd)
        v_flops = 2 * S * D * (Hkv * Hd)
        o_flops = 2 * S * (H * Hd) * D

        if attn in (AttentionType.MHA, AttentionType.ALIBI, AttentionType.ROPE,
                    AttentionType.GQA, AttentionType.MQA):
            score_flops  = 2 * S * S * H * Hd
            attn_v_flops = 2 * S * S * H * Hd
        elif attn == AttentionType.SLIDE:
            w = min(cfg.window_size, S)
            score_flops  = 2 * w * S * H * Hd
            attn_v_flops = 2 * w * S * H * Hd
        elif attn == AttentionType.HYBRID:
            n_global = cfg.global_attn_layers
            n_local  = L - n_global
            w = min(cfg.window_size, S)
            score_flops  = (n_global * 2 * S * S * H * Hd + n_local * 2 * w * S * H * Hd) / L
            attn_v_flops = score_flops
        elif attn == AttentionType.LINEAR:
            score_flops  = 2 * S * H * Hd * Hd
            attn_v_flops = 2 * S * H * Hd * Hd
        else:
            score_flops  = 2 * S * S * H * Hd
            attn_v_flops = 2 * S * S * H * Hd

        attn_fwd = L * (q_flops + k_flops + v_flops + score_flops + attn_v_flops + o_flops) / S

        if cfg.ffn_type in (FFNType.GEGLU, FFNType.GATED):
            ffn_total_layer = 2 * S * (D * ffn + D * ffn + ffn * D)
        elif cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            ffn_total_layer = 2 * S * (D * ffn + ffn * D) * cfg.top_k_experts
        else:
            ffn_total_layer = 2 * S * (D * ffn + ffn * D)

        ffn_fwd   = L * ffn_total_layer / S
        fwd_total = attn_fwd + ffn_fwd
        bwd_total = attn_fwd * 2.5 + ffn_fwd * 2.0

        return fwd_total, bwd_total, attn_fwd, ffn_fwd

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Arithmetic Intensity
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_arithmetic_intensity(self, cfg: ArchConfig) -> float:
        """
        FIX-6: Include gradient traffic + weight ×2 reads.
        F14-NEW: Gunakan effective_memory_bw (ECC-adjusted).
        """
        gpu = self.gpu
        flops_per_token   = cfg.flops_per_token_fwd + cfg.flops_per_token_bwd
        tokens_per_step   = cfg.seq_len * cfg.batch_size

        weight_traffic    = cfg.param_count * 2 * 2   # BF16, 2 reads
        grad_traffic      = cfg.param_count * 2 * 1   # BF16, 1 write
        act_traffic       = cfg.vram_activations_gb * 2.0 * 1e9

        total_bytes_per_step  = weight_traffic + grad_traffic + act_traffic
        total_bytes_per_token = total_bytes_per_step / max(1, tokens_per_step)
        theoretical_ai        = flops_per_token / max(total_bytes_per_token, 1.0)

        # F2-NEW: ECC overhead mengurangi effective BW → AI efektif naik
        # (lebih banyak FLOP per byte efektif)
        ecc_factor = 1.0 / max(0.5, 1.0 - gpu.ecc_bw_overhead)

        kv_frac = cfg.vram_kv_cache_gb / max(0.01,
                  cfg.vram_kv_cache_gb + cfg.vram_activations_gb + cfg.vram_weights_gb)
        hbm_eff = (gpu.hbm_efficiency_streaming * (1.0 - kv_frac) +
                   gpu.hbm_efficiency_random     * kv_frac)

        attn_bw = {
            AttentionType.MHA:    1.00,
            AttentionType.GQA:    1.04,
            AttentionType.MQA:    1.06,
            AttentionType.SLIDE:  0.90,
            AttentionType.HYBRID: 0.95,
            AttentionType.LINEAR: 1.02,
            AttentionType.ALIBI:  1.00,
            AttentionType.ROPE:   1.00,
        }
        attn_factor = attn_bw.get(cfg.attn_type, 1.00)
        moe_factor  = 0.82 if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK) else 1.0

        return theoretical_ai * hbm_eff * attn_factor * moe_factor * ecc_factor

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Warp Divergence
    # ═════════════════════════════════════════════════════════════════════════

    def _estimate_warp_divergence(self, cfg: ArchConfig) -> Dict:
        """
        Warp divergence dari causal masking, MoE routing, sliding window.
        F7-NEW: gunakan gpu.max_warps_per_sm untuk scaling.
        """
        base_div = 0.025

        if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            sparsity = 1.0 - cfg.top_k_experts / max(1, cfg.num_experts)
            moe_div  = 0.10 * sparsity
        else:
            moe_div = 0.0

        if cfg.attn_type == AttentionType.SLIDE:
            attn_div = 0.04
        elif cfg.attn_type == AttentionType.HYBRID:
            attn_div = 0.02
        elif cfg.attn_type == AttentionType.LINEAR:
            attn_div = 0.01
        else:
            attn_div = 0.01

        # GPU dengan max_warps_per_sm lebih sedikit (Turing=32) lebih sensitif divergence
        # karena lebih sedikit warp untuk hide latency
        gpu = self.gpu
        warp_sensitivity = max(0.8, 64.0 / max(1, gpu.max_warps_per_sm))
        total_div = (base_div + moe_div + attn_div) * warp_sensitivity
        warp_eff  = max(0.55, 1.0 - total_div)

        if moe_div > 0.05:
            cause = "MoE expert routing (token dispatch branches)"
        elif attn_div >= 0.04:
            cause = "Sliding-window attention boundary tokens"
        else:
            cause = "Causal mask / padding"

        return {
            "warp_divergence_pct": round(total_div * 100, 2),
            "warp_efficiency":     round(warp_eff, 3),
            "primary_cause":       cause,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Kernel Fusion
    # ═════════════════════════════════════════════════════════════════════════

    def _kernel_fusion_analysis(self, cfg: ArchConfig) -> Dict:
        """
        FIX-14: Triangular distribution [1.02, 1.28] dari PyTorch 2.0 benchmarks.
        F4-NEW: FlashAttention tile feasibility berdasarkan shared_mem_max_kb.
        """
        gpu = self.gpu
        fused, unfused = [], []

        fused.append("RMSNorm + QKV Linear (fused CUDA kernel)")

        # F4-NEW: cek apakah FA bisa pakai tile optimal di GPU ini
        fa_tile_feas = gpu.flash_attn_tile_feasibility(cfg.head_dim)
        if cfg.use_flash_attn:
            if fa_tile_feas >= 0.9:
                fused.append(f"FlashAttention: optimal tile (SMEM={gpu.shared_mem_max_kb:.0f}KB)")
            else:
                fused.append(f"FlashAttention: reduced tile ({fa_tile_feas*100:.0f}% of optimal)")
        else:
            unfused += ["Q@K^T matmul", "Softmax (separate)", "@V matmul"]

        if cfg.ffn_type in (FFNType.GEGLU, FFNType.GATED):
            fused.append("SwiGLU/GeGLU: gate×SiLU×up (Triton kernel)")
        elif cfg.ffn_type == FFNType.DENSE:
            fused.append("FFN Up + SiLU activation (Inductor fused)")
        elif cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            unfused += ["MoE Router softmax", "Expert token scatter", "Expert gather"]

        if cfg.pos_enc == PosEncType.ROPE:
            fused.append("RoPE: rotary embedding fused into Q/K projection")

        if cfg.use_torch_compile:
            fused.append("torch.compile: Inductor graph-level fusion")

        fused.append("Residual add + norm (fused via Inductor)")

        cuda_graph_ok = cfg.ffn_type not in (FFNType.MOE, FFNType.MOE_TOPK)
        total_ops     = len(fused) + len(unfused)
        fusion_ratio  = len(fused) / max(1, total_ops)

        if cfg.use_torch_compile:
            mode = min(1.22, 1.07 + 0.12 * fusion_ratio)
            compiler_spd = float(self.nprng.triangular(1.02, mode, 1.28))
            if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
                compiler_spd = max(1.0, compiler_spd * 0.85)
        else:
            compiler_spd = 1.0

        return {
            "fused_kernels":         fused,
            "unfused_kernels":       unfused,
            "fusion_ratio":          round(fusion_ratio, 2),
            "cuda_graph_compatible": cuda_graph_ok,
            "compiler_speedup":      round(compiler_spd, 3),
            "fa_tile_feasibility":   round(fa_tile_feas, 3),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Ridge Point
    # ═════════════════════════════════════════════════════════════════════════

    def _hardware_ridge_point(self) -> float:
        """
        FIX-7: Ridge = peak_compute / peak_bw — konstanta hardware.
        F2-NEW: BW = memory_bw_gbps × (1 - ecc_bw_overhead).
        """
        gpu = self.gpu
        eff_bw = gpu.memory_bw_gbps * (1.0 - gpu.ecc_bw_overhead)
        return (gpu.bf16_tflops * 1e12) / (eff_bw * 1e9)

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Dynamic MFU
    # ═════════════════════════════════════════════════════════════════════════

    def _compute_dynamic_mfu(self, cfg: ArchConfig, ai: float, ridge: float) -> float:
        """
        FIX-8: Clean roofline model.
        F3-NEW: Latency overhead untuk HBM vs GDDR berbeda.
        """
        gpu      = self.gpu
        ai_ratio = ai / max(ridge, 1e-9)

        if ai_ratio >= 1.0:
            sat = 1.0 - math.exp(-2.0 * (ai_ratio - 1.0))
            mfu = gpu.mfu_typical_min + sat * (gpu.mfu_typical_max - gpu.mfu_typical_min)
        else:
            mfu = ai_ratio * gpu.mfu_typical_max

        wd = self._estimate_warp_divergence(cfg)
        mfu *= wd["warp_efficiency"]

        # F3-NEW: GDDR memory latency penalty untuk random-access pattern
        # HBM: ~80-105 ns → latency mostly hidden
        # GDDR6/6X: ~400-450 ns → latency visible untuk KV cache scatter
        if not gpu.is_hbm:
            kv_frac = cfg.vram_kv_cache_gb / max(0.01,
                      cfg.vram_total_gb + 0.001)
            # Latency penalty proporsional ke KV fraction dan latency GPU
            latency_ratio = gpu.memory_latency_ns / 100.0  # normalized to HBM2
            latency_pen   = max(0.0, min(0.12, kv_frac * (latency_ratio - 1.0) * 0.04))
            mfu *= (1.0 - latency_pen)

        return float(np.clip(mfu, 0.05, 0.90))

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Hardware Efficiency Factors
    # ═════════════════════════════════════════════════════════════════════════

    def _tensor_core_utilization(self, cfg: ArchConfig) -> float:
        """
        F5-NEW: Gunakan gpu.optimal_tile_size per generasi.
        Gen4 (Hopper): tile 64 → penalti lebih keras untuk dim tidak aligned.
        """
        gpu          = self.gpu
        optimal_tile = gpu.optimal_tile_size   # dari GPU property, bukan hardcoded
        d_aligned    = (cfg.hidden_dim % optimal_tile == 0)
        hd_aligned   = (cfg.head_dim   % optimal_tile == 0)
        ffn_aligned  = (int(cfg.hidden_dim * cfg.ffn_multiplier) % optimal_tile == 0)
        batch_ok     = (cfg.batch_size >= 4)
        score        = 0.40 * d_aligned + 0.25 * hd_aligned + 0.20 * ffn_aligned + 0.15 * batch_ok
        return float(np.clip(0.60 + 0.40 * score, 0.55, 1.00))

    def _granularity_penalty(self, cfg: ArchConfig) -> float:
        """
        SM occupancy penalty hanya untuk model yang BENAR-BENAR kekurangan parallelism.
        F7-NEW: tokens_per_sm dihitung menggunakan max_warps_per_sm GPU aktual.
        """
        gpu = self.gpu
        tokens_per_sm = (cfg.batch_size * cfg.seq_len) / max(1, gpu.sm_count)

        # Threshold occupancy minimum berdasarkan max_warps_per_sm GPU
        # GPU dengan max_warps_per_sm rendah (Turing=32) butuh token lebih banyak
        min_tokens = 2.0 * (gpu.max_warps_per_sm / 64.0)  # scale by warp capacity

        if cfg.hidden_dim >= 256 and tokens_per_sm >= min_tokens:
            return 1.0

        work_factor   = min(1.0, tokens_per_sm / max(0.1, min_tokens * 2))
        hidden_factor = min(1.0, cfg.hidden_dim / 256.0)
        combined      = 0.50 * work_factor + 0.50 * hidden_factor
        return float(np.clip(combined, 0.65, 1.00))

    def _effective_flops_efficiency(self, cfg: ArchConfig) -> float:
        """Scheduled FLOPs yang mencapai execution unit dengan throughput penuh."""
        base = 0.90
        if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            sparsity = 1.0 - cfg.top_k_experts / max(1, cfg.num_experts)
            base -= 0.07 * sparsity
        if cfg.use_flash_attn:
            base = min(0.97, base + 0.03)
        if cfg.hidden_dim < 512:
            base -= 0.09
        elif cfg.hidden_dim < 1024:
            base -= 0.04
        return float(np.clip(base, 0.70, 0.97))

    def _scheduler_inefficiency(self, cfg: ArchConfig) -> float:
        """
        Warp stall + pipeline bubble slowdown (>= 1.0).
        F7-NEW: scaled by typical_sm_occupancy GPU aktual.
        """
        gpu        = self.gpu
        occ        = gpu.typical_sm_occupancy
        stall_base = max(0.0, 0.75 - occ) * 0.18
        if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            stall_base += 0.04
        if cfg.attn_type == AttentionType.SLIDE:
            stall_base += 0.03
        if cfg.use_gradient_checkpointing:
            stall_base += 0.015
        return float(np.clip(1.0 + stall_base, 1.00, 1.20))

    def _async_compute_overlap_factor(self, cfg: ArchConfig) -> float:
        """
        Compute-memory pipeline overlap credit (factor <= 1.0).
        F11-NEW: NVLink kredit HANYA jika gpu.nvlink_bw_gbps > 0.
        F4-NEW: FA kredit bergantung tile feasibility.
        """
        gpu = self.gpu
        fa_tile_feas  = gpu.flash_attn_tile_feasibility(cfg.head_dim)
        fa_overlap    = 0.04 * fa_tile_feas if cfg.use_flash_attn else 0.0
        compile_olap  = 0.02 if cfg.use_torch_compile else 0.0
        # NVLink kredit: hanya GPU yang punya NVLink (bukan T4, RTX-3090, RTX-4090)
        nvlink_olap   = 0.03 if gpu.nvlink_bw_gbps > 100 else 0.0
        moe_anti      = 0.04 if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK) else 0.0
        total         = fa_overlap + compile_olap + nvlink_olap - moe_anti
        return float(np.clip(1.0 - total, 0.88, 1.00))

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Bottleneck Classifier
    # ═════════════════════════════════════════════════════════════════════════

    def _classify_bottleneck(
        self, cfg: ArchConfig, ai: float, ridge: float,
        compute_sat: float, mem_sat: float, kl_frac: float,
    ) -> Tuple[str, Dict]:
        factors = {
            "compute_saturation": round(compute_sat, 3),
            "memory_saturation":  round(mem_sat, 3),
            "kernel_launch_frac": round(kl_frac, 3),
            "ai_vs_ridge":        round(ai / max(ridge, 1e-9), 3),
        }
        ai_ratio = ai / max(ridge, 1e-9)

        if ai_ratio >= 1.0:
            if mem_sat >= 0.75:
                label = "compute+memory-co-bound"
            else:
                label = "compute-bound"
        elif mem_sat >= 0.80:
            label = "memory-bandwidth-bound"
        elif kl_frac >= 0.18:
            label = "kernel-launch-overhead-bound"
        elif ai < ridge * 0.35:
            # F3-NEW: untuk GDDR GPU, latency-bound lebih sering terjadi
            gpu = self.gpu
            if not gpu.is_hbm and cfg.vram_kv_cache_gb > 0.1:
                label = "memory-latency-bound (GDDR KV scatter)"
            else:
                label = "memory-latency-bound"
        elif compute_sat >= 0.65:
            label = "compute-bound (overhead-limited)"
        else:
            label = "latency-bound (balanced)"

        return label, factors

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — HBM Bandwidth Model
    # ═════════════════════════════════════════════════════════════════════════

    def _hbm_bandwidth_model(self, cfg: ArchConfig) -> Dict:
        """
        FIX-12: Full training BW = 2×weights + 1×grads + 2×activations + kv.
        FIX-13: L1/L2 hit rate via exponential working-set model.
        F2-NEW: Gunakan effective_memory_bw_gbps (ECC-adjusted).
        F6-NEW: L2 bandwidth tbps untuk hit rate model yang lebih akurat.
        """
        gpu = self.gpu
        # F2-NEW: BW efektif setelah ECC overhead
        eff_bw  = gpu.effective_memory_bw_gbps   # GB/s, setelah ECC

        weight_rd  = cfg.vram_weights_gb * 2.0
        grad_wr    = cfg.vram_weights_gb * 1.0
        kv_rw      = cfg.vram_kv_cache_gb
        act_rw     = cfg.vram_activations_gb * 2.0
        total_traf = weight_rd + grad_wr + kv_rw + act_rw

        step_ms     = max(1e-3, cfg.ms_per_step)
        required_bw = total_traf / (step_ms / 1000.0)
        # Saturation dihitung terhadap BW efektif (setelah ECC)
        saturation  = min(1.0, required_bw / (eff_bw * gpu.hbm_efficiency_streaming))

        # FIX-13: Exponential working-set L2/L1 hit rate
        # F6-NEW: Pertimbangkan l2_bandwidth_tbps untuk L2 throughput feasibility
        l2_hot_mb  = (cfg.num_heads * cfg.head_dim * cfg.seq_len * 2) / 1e6
        l2_ratio   = gpu.l2_cache_mb / max(0.1, l2_hot_mb)
        l2_hit_pct = float(np.clip((1.0 - math.exp(-l2_ratio)) * 100, 15, 92))

        # L2 BW sanity check: jika L2 BW tidak cukup, hit rate efektif turun
        if gpu.l2_bandwidth_tbps > 0:
            l2_demand_tbps = total_traf / max(1e-3, step_ms / 1000) / 1e3
            if l2_demand_tbps > gpu.l2_bandwidth_tbps * 0.8:
                l2_hit_pct *= 0.85  # L2 BW saturasi → effective hit rate turun

        l1_hot_kb  = (cfg.head_dim * 32 * 2) / 1024
        l1_ratio   = gpu.l1_cache_kb / max(0.1, l1_hot_kb)
        l1_hit_pct = float(np.clip((1.0 - math.exp(-l1_ratio * 0.5)) * 100, 10, 88))

        return {
            "peak_bw_gbps":         eff_bw,   # F2-NEW: efektif, bukan theoretical
            "required_bw_gbps":     round(required_bw, 1),
            "saturation_pct":       round(saturation * 100, 1),
            "weight_rd_traffic_gb": round(weight_rd, 2),
            "grad_wr_traffic_gb":   round(grad_wr, 2),
            "kv_traffic_gb":        round(kv_rw, 3),
            "activation_rw_gb":     round(act_rw, 2),
            "total_traffic_gb":     round(total_traf, 2),
            "hbm_bottleneck":       saturation > 0.85,
            "l2_hit_est_pct":       round(l2_hit_pct, 1),
            "l1_hit_est_pct":       round(l1_hit_pct, 1),
            "ecc_bw_overhead_pct":  round(gpu.ecc_bw_overhead * 100, 1),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Thermal Model
    # ═════════════════════════════════════════════════════════════════════════

    def _thermal_model(self) -> Dict:
        """
        Exponential decay dari burst ke sustained.
        F1-NEW: gunakan frequency_sustained_mhz dan frequency_boost_mhz.
        """
        gpu   = self.gpu
        tau   = 5.0
        times = [0, 1, 2, 5, 10, 20, 30]
        curve = {}
        for t in times:
            p = gpu.thermal_factor + (1.0 - gpu.thermal_factor) * math.exp(-t / tau)
            curve[f"{t}min"] = f"{p*100:.1f}%"

        # F1-NEW: informasi clock yang lebih detail
        clock_drop = 0.0
        if gpu.frequency_boost_mhz > 0 and gpu.frequency_sustained_mhz > 0:
            clock_drop = (1.0 - gpu.thermal_factor) * 100

        return {
            "burst_performance":      "100%",
            "sustained_performance":  f"{gpu.thermal_factor*100:.0f}%",
            "boost_clock_mhz":        gpu.frequency_boost_mhz,
            "sustained_clock_mhz":    gpu.frequency_sustained_mhz,
            "clock_drop_pct":         round(clock_drop, 1),
            "tau_minutes":            tau,
            "performance_curve":      curve,
            "clock_jitter_pct":       round(gpu.runtime_variance_pct * 0.40, 1),
            "tdp_sustained_w":        gpu.tdp_sustained_w,
            "power_eff_tflops_w":     round(gpu.power_efficiency_tflops_per_w, 3),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Stochastic Noise
    # ═════════════════════════════════════════════════════════════════════════

    def _stochastic_noise_model(self, cfg: ArchConfig, n: int = 50) -> Dict:
        """
        FIX-11: Log-normal + Poisson spike process.
        GPU step times bersifat multiplicative dan right-skewed.
        """
        rng      = self.nprng
        base_ms  = max(1.0, cfg.ms_per_step)
        var_frac = self.gpu.runtime_variance_pct / 100.0

        log_sigma   = var_frac * 0.60
        log_samples = rng.normal(0.0, log_sigma, n)
        samples     = base_ms * np.exp(log_samples)

        spike_rate = 0.04
        spike_mask = rng.random(n) < spike_rate
        spike_mult = 1.0 + rng.uniform(0.15, 0.40, n) * spike_mask
        samples   *= spike_mult

        arr = np.array(samples)
        return {
            "mean_ms":        round(float(arr.mean()), 2),
            "median_ms":      round(float(np.median(arr)), 2),
            "std_ms":         round(float(arr.std()), 2),
            "p50_ms":         round(float(np.percentile(arr, 50)), 2),
            "p95_ms":         round(float(np.percentile(arr, 95)), 2),
            "p99_ms":         round(float(np.percentile(arr, 99)), 2),
            "cv_pct":         round(float(arr.std() / arr.mean() * 100), 2),
            "spike_rate_pct": f"{spike_rate*100:.0f}%",
            "distribution":   "log-normal + Poisson spikes",
        }

    def _activation_recomputation_cost(self, cfg: ArchConfig):
        if not cfg.use_gradient_checkpointing:
            return None
        L    = cfg.num_layers
        segs = max(1, int(math.sqrt(L)))
        save_frac = segs / L
        full_act   = cfg.vram_activations_gb / max(0.01, save_frac)
        saved_gb   = full_act - cfg.vram_activations_gb
        extra_ms   = cfg.ms_per_step * max(0.0, (segs - 1) / L) / 3.0
        return {
            "enabled":          True,
            "segments":         segs,
            "memory_saved_gb":  round(max(0, saved_gb), 3),
            "memory_used_gb":   round(cfg.vram_activations_gb, 3),
            "extra_forward_ms": round(extra_ms, 2),
            "overhead_pct":     round(extra_ms / max(1e-3, cfg.ms_per_step) * 100, 1),
            "tradeoff_summary": f"Save {max(0,saved_gb):.2f} GB VRAM, cost +{extra_ms:.1f}ms/step",
        }

    def _dataloader_stall(self, cfg: ArchConfig) -> Dict:
        gpu = self.gpu
        base_stall_pct = gpu.dataloader_stall_frac * 100
        bs_penalty     = min(2.5, math.log2(max(1, cfg.batch_size)) * 0.7)
        seq_penalty    = min(2.0, math.log2(max(1, cfg.seq_len / 1024)) * 0.5)
        # F12-NEW: PCIe version mempengaruhi IO stall
        # PCIe 3.0 (T4, V100): bottleneck lebih parah untuk multi-GPU setup
        pcie_penalty = 1.0
        if gpu.pcie_version == "3.0":
            pcie_penalty = 1.3
        elif gpu.pcie_version == "5.0":
            pcie_penalty = 0.7
        total_stall_pct = (base_stall_pct + bs_penalty + seq_penalty) * pcie_penalty
        stall_ms        = cfg.ms_per_step * total_stall_pct / 100
        net_stall_ms    = stall_ms * 0.30
        return {
            "raw_stall_pct":      round(total_stall_pct, 1),
            "prefetch_hidden_ms": round(stall_ms * 0.70, 2),
            "net_stall_ms":       round(net_stall_ms, 2),
            "cpu_bound_risk":     total_stall_pct > 8.0,
            "pcie_version":       gpu.pcie_version,
        }

    def _async_overlap_model(
        self, cfg: ArchConfig, compute_ms: float, comm_ms: float, io_ms: float,
    ) -> Dict:
        gpu        = self.gpu
        has_nvlink = gpu.nvlink_bw_gbps > 0
        if has_nvlink:
            # NVLink overlap: hanya 20% comm menambah latency
            compute_comm = compute_ms + comm_ms * 0.20
        else:
            # PCIe: full serialized (no duplex compute-comm overlap)
            compute_comm = compute_ms + comm_ms

        total_w  = compute_comm + io_ms * 0.30
        total_wo = compute_ms + comm_ms + io_ms
        return {
            "compute_ms":               round(compute_ms, 2),
            "comm_ms":                  round(comm_ms, 2),
            "io_ms":                    round(io_ms, 2),
            "total_with_overlap_ms":    round(total_w, 2),
            "total_without_overlap_ms": round(total_wo, 2),
            "overlap_benefit_ms":       round(total_wo - total_w, 2),
            "overlap_efficiency_pct":   round((total_wo - total_w) / max(1e-3, total_wo) * 100, 1),
            "nvlink_async":             has_nvlink,
            "io_prefetch_active":       True,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Throughput Estimation
    # ═════════════════════════════════════════════════════════════════════════

    def _estimate_throughput(self, cfg: ArchConfig) -> Dict[str, Any]:
        """
        Physics-based throughput model.

        F1-NEW: peak_flops = bf16_tflops × thermal_factor (sustained, bukan burst).
        F2-NEW: peak_bw = effective_memory_bw_gbps × thermal_factor.
        F4-NEW: FlashAttention speedup bergantung tile feasibility per GPU.
        F5-NEW: TC alignment dari gpu.optimal_tile_size.
        F13-NEW: Kernel launch overhead dari gpu.kernel_launch_us per GPU.
        """
        gpu          = self.gpu
        total_tokens = cfg.seq_len * cfg.batch_size
        total_flops  = (cfg.flops_per_token_fwd + cfg.flops_per_token_bwd) * total_tokens

        # ── 1. Peak compute — SUSTAINED (thermal-derated) ─────────────────────
        # F1-NEW: Gunakan thermal_factor yang mencerminkan sustained clock
        peak_flops_s = gpu.bf16_tflops * 1e12 * gpu.thermal_factor
        # F2-NEW: BW efektif = BW × (1 - ECC_overhead) × thermal_factor
        peak_bw_s    = gpu.effective_memory_bw_gbps * 1e9 * gpu.thermal_factor

        # ── 2. Ridge point (ECC-adjusted hardware constant) ───────────────────
        ridge = peak_flops_s / peak_bw_s if peak_bw_s > 0 else 1.0
        ai    = cfg.arithmetic_intensity

        # ── 3. Roofline MFU ───────────────────────────────────────────────────
        if ai >= ridge:
            excess       = (ai / ridge - 1.0)
            saturation   = 1.0 - math.exp(-1.2 * excess)
            mfu_roofline = gpu.mfu_typical_min + saturation * (gpu.mfu_typical_max - gpu.mfu_typical_min)
        else:
            mfu_roofline = (ai / ridge) * gpu.mfu_typical_min

        # ── 4. Tensor core alignment (F5-NEW: optimal_tile_size per GPU) ──────
        tile     = gpu.optimal_tile_size   # Gen4: 64, Gen3: 32, Gen1/2: 16
        n_align  = sum([
            cfg.hidden_dim % tile == 0,
            cfg.head_dim   % tile == 0,
            int(cfg.hidden_dim * cfg.ffn_multiplier) % tile == 0,
        ])
        # Gen4 lebih keras penaltinya karena tile 64 sulit di-align untuk model kecil
        if gpu.tensor_core_gen >= 4:
            tc_align = 0.82 + 0.06 * n_align   # range: [0.82, 1.00]
        else:
            tc_align = 0.85 + 0.05 * n_align   # range: [0.85, 1.00]

        # ── 5. MoE dispatch penalty ───────────────────────────────────────────
        if cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            sparsity    = 1.0 - cfg.top_k_experts / max(1, cfg.num_experts)
            # FIX-MOE-8: Penalty 0.40 terlalu besar. Mixtral 8x7B pada H100 mencapai
            # ~0.38-0.42 MFU — routing overhead nyata ≈ 15-25%, bukan 40%.
            # Reference: MegaBlocks (Gale et al., 2022), dMoE benchmarks.
            moe_penalty = 1.0 - 0.22 * sparsity
        else:
            moe_penalty = 1.0

        # ── 6. SM occupancy penalty ───────────────────────────────────────────
        # F7-NEW: tokens_per_sm scaling berdasarkan max_warps_per_sm GPU aktual
        tokens_per_sm = total_tokens / max(1, gpu.sm_count)
        min_tps = 2.0 * (gpu.max_warps_per_sm / 64.0)
        if cfg.hidden_dim < 256 or tokens_per_sm < min_tps:
            sm_penalty = max(0.65, (cfg.hidden_dim / 256.0) * 0.65
                                   + min(1.0, tokens_per_sm / max(0.1, min_tps * 2)) * 0.35)
        else:
            sm_penalty = 1.0

        # ── 7. Effective MFU ──────────────────────────────────────────────────
        mfu_eff = float(np.clip(
            mfu_roofline * tc_align * moe_penalty * sm_penalty,
            0.05, gpu.mfu_typical_max))

        # ── 8. Compute time ───────────────────────────────────────────────────
        fusion_info      = self._kernel_fusion_analysis(cfg)
        compiler_speedup = fusion_info["compiler_speedup"]
        compute_ms       = total_flops / max(1.0, peak_flops_s * mfu_eff) * 1000.0
        compute_ms      /= compiler_speedup

        # ── 9. Gradient checkpointing ─────────────────────────────────────────
        gc_ms = compute_ms * 0.30 if cfg.use_gradient_checkpointing else 0.0

        # ── 10. Additive overheads ────────────────────────────────────────────
        # F13-NEW: kernel_launch_us PER GPU (T4=9µs, H100=2µs)
        num_kernels = cfg.num_layers * 10
        kl_ms       = num_kernels * gpu.kernel_launch_us / 1000.0

        mem_ms = compute_ms * gpu.mem_alloc_overhead
        io_ms  = compute_ms * gpu.dataloader_stall_frac * 0.5

        # F3-NEW: GDDR memory latency overhead untuk scatter/gather patterns
        # (KV cache, MoE expert dispatch: random access → latency exposed)
        lat_penalty_ms = 0.0
        if not gpu.is_hbm and cfg.vram_kv_cache_gb > 0.05:
            # Latency overhead: normalized ke T4 baseline (450 ns)
            lat_norm  = gpu.memory_latency_ns / 100.0  # per 100 ns
            kv_ops    = cfg.num_layers * cfg.num_kv_heads * cfg.seq_len / max(1, cfg.batch_size)
            lat_us    = kv_ops * gpu.memory_latency_ns / 1e6 * 0.001  # µs → ms
            lat_penalty_ms = min(compute_ms * 0.05, lat_us)  # cap 5% compute

        # ── 11. Total step time + jitter ──────────────────────────────────────
        ms_base     = compute_ms + gc_ms + kl_ms + mem_ms + io_ms + lat_penalty_ms
        jitter      = float(np.clip(
            self.nprng.lognormal(0.0, gpu.runtime_variance_pct / 200.0),
            0.90, 1.15))
        ms_per_step = max(0.5, ms_base * jitter)

        # ── 12. Derived metrics ───────────────────────────────────────────────
        tokens_per_sec = total_tokens / (ms_per_step / 1000.0)
        mfu_actual     = total_flops / (ms_per_step / 1000.0) / (gpu.bf16_tflops * 1e12)

        # ── 13. Bottleneck classification ─────────────────────────────────────
        weight_bytes   = cfg.param_count * 2 * 2
        grad_bytes     = cfg.param_count * 2
        act_bytes      = cfg.vram_activations_gb * 1e9
        data_bytes     = weight_bytes + grad_bytes + act_bytes
        mem_time_s     = data_bytes / max(1.0, peak_bw_s)
        compute_time_s = total_flops / max(1.0, peak_flops_s * mfu_eff)

        compute_sat = min(1.0, compute_time_s / max(1e-9, ms_per_step / 1000.0))
        mem_sat     = min(1.0, mem_time_s     / max(1e-9, ms_per_step / 1000.0))
        kl_frac     = kl_ms / max(1e-3, ms_per_step)

        bottleneck, bt_factors = self._classify_bottleneck(
            cfg, ai, ridge, compute_sat, mem_sat, kl_frac)

        model_sm_occ = float(np.clip(gpu.typical_sm_occupancy * sm_penalty, 0.3, 1.0))
        bt_factors.update({
            "mfu_roofline":         round(mfu_roofline,     3),
            "tc_align":             round(tc_align,          3),
            "moe_penalty":          round(moe_penalty,       3),
            "sm_penalty":           round(sm_penalty,        3),
            "mfu_effective":        round(mfu_eff,           3),
            "sm_occ_model":         round(model_sm_occ,      3),
            "compiler_speedup":     round(compiler_speedup,  3),
            "fa_tile_feasibility":  round(fusion_info.get("fa_tile_feasibility", 1.0), 3),
            "lat_penalty_ms":       round(lat_penalty_ms,    3),
            "ecc_overhead_pct":     round(gpu.ecc_bw_overhead * 100, 1),
            "optimal_tile_size":    gpu.optimal_tile_size,
        })

        return {
            "ms_per_step":        round(ms_per_step, 2),
            "tokens_per_sec":     int(max(1, tokens_per_sec)),
            "mfu":                round(float(np.clip(mfu_actual, 0.0, 1.0)), 4),
            "bottleneck":         bottleneck,
            "bottleneck_factors": bt_factors,
            "ai":                 round(ai, 2),
            "ridge_point":        round(ridge, 2),
            "compiler_speedup":   round(compiler_speedup, 3),
            "warp_divergence":    self._estimate_warp_divergence(cfg),
            "sm_occupancy":       round(model_sm_occ, 3),
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MATH ENGINE — Fitness Score
    # ═════════════════════════════════════════════════════════════════════════

    def _fitness_score(self, cfg: ArchConfig) -> float:
        """
        Multi-objective fitness normalized ke GPU-relative ceiling.
        Speed cap dari GPU peak throughput (bukan fixed 100K tok/s).
        """
        if not cfg.fits_gpu:
            return 0.0

        flops_per_tok = max(1.0, cfg.flops_per_token_fwd + cfg.flops_per_token_bwd)
        gpu_tok_ceil  = max(10_000.0,
                            self.gpu.bf16_tflops * 1e12 * self.gpu.mfu_typical_max
                            / flops_per_tok)

        vram_budget_gb  = self.gpu.vram_gb * VRAM_LIMIT_PCT
        vram_efficiency = cfg.vram_total_gb / max(vram_budget_gb, 1e-6)
        vram_score      = float(np.clip(vram_efficiency, 0.0, 1.0))

        mfu_score   = cfg.mfu_estimate
        speed_score = min(1.0, cfg.tokens_per_sec_estimate / gpu_tok_ceil)
        param_dens  = min(1.0, math.log10(max(cfg.param_count, 1)) / math.log10(10e9))
        gc_pen      = 0.05 if cfg.use_gradient_checkpointing else 0.0
        warp_pen    = cfg.warp_divergence_pct / 100 * 0.06

        return round(
            0.30 * mfu_score   +
            0.25 * speed_score +
            0.18 * vram_score  +
            0.15 * param_dens  -
            0.07 * gc_pen      -
            0.05 * warp_pen, 4)

    # ═════════════════════════════════════════════════════════════════════════
    #  GENERATOR
    # ═════════════════════════════════════════════════════════════════════════

    def generate_one(self, family_name: str) -> ArchConfig:
        self._arch_counter += 1
        rng = self.rng; gpu = self.gpu
        fp  = self.FAMILY_PROFILES[family_name]

        cfg = ArchConfig(); cfg.arch_family = family_name

        # F8-NEW: VRAM scaling proporsional ke VRAM GPU aktual
        # hidden_dim ∝ sqrt(VRAM) untuk jumlah layer tetap
        # vram_scale = sqrt(gpu.vram_gb / 16), cap 4.0× (H200: √(141/16)≈3.0)
        _vram_scale = min(4.0, (gpu.vram_gb / 16.0) ** 0.5)
        _l_scale    = min(1.8, _vram_scale)  # layer scale lebih konservatif

        _h_lo = self._make_divisible(int(fp["hidden_range"][0] * _vram_scale), 128)
        _h_hi = self._make_divisible(int(fp["hidden_range"][1] * _vram_scale), 128)
        _l_lo = max(4,         round(fp["layer_range"][0] * _l_scale))
        _l_hi = max(_l_lo + 2, round(fp["layer_range"][1] * _l_scale))

        # Pastikan hidden tidak terlalu besar untuk GPU kecil
        # (T4/V100-16GB: jangan sampai model tidak muat bahkan dengan bs=1)
        _vram_gb_safe   = gpu.vram_gb * VRAM_LIMIT_PCT
        _max_hidden_dim = self._make_divisible(
            int(math.sqrt(_vram_gb_safe * 1e9 / (2.0 * max(1, _l_lo)))), 128)
        _h_hi = min(_h_hi, _max_hidden_dim)
        _h_lo = min(_h_lo, _h_hi)

        num_layers = rng.randint(_l_lo, _l_hi)
        hidden_raw = rng.randint(_h_lo, _h_hi)
        hidden_dim = self._make_divisible(hidden_raw, 128)

        # FIX-MOE-3: Terapkan hard cap hidden_dim untuk MoE sebelum heads dihitung.
        # MoE params ≈ E × 3 × D × ffn_per_expert × L → D² scaling yang brutal.
        # Dengan E=8, D=2048, ffn_mult=1.5, L=16: params ≈ 8×3×2048×3072×16 = 2.4B
        # Weights saja = 4.8GB BF16 + optimizer states bisa 10-20GB → OOM di GPU ≤40GB.
        # Formula cap: D_max = sqrt(vram_gb × 1e9 × 0.12 / (E_est × 3 × ffn_est × L))
        # Simplified conservative cap berdasarkan VRAM GPU.
        if family_name == "MoE-Sparse":
            _moe_e_est   = 8         # estimasi expert count sebelum dipilih
            _moe_f_est   = 1.25      # estimasi ffn_mult tengah
            _moe_bgt     = gpu.vram_gb * 1e9 * 0.18   # 18% VRAM untuk weights
            _moe_d2      = _moe_bgt / max(1, 2 * _moe_e_est * 3 * _moe_f_est * num_layers)
            _moe_dmax    = self._make_divisible(int(math.sqrt(max(1, _moe_d2))), 128)
            _moe_dmax    = max(384, min(_moe_dmax, 1280))   # absolute bounds
            if hidden_dim > _moe_dmax:
                hidden_dim = self._make_divisible(
                    rng.randint(max(384, _moe_dmax // 2), _moe_dmax), 128)

        # F5-NEW: pilih num_heads yang aligned dengan optimal_tile_size GPU
        tile       = gpu.optimal_tile_size
        head_range = list(range(fp["head_range"][0], fp["head_range"][1] + 1, 4)) or [8]
        num_heads  = rng.choice(head_range)
        head_dim   = self._make_divisible(hidden_dim // max(1, num_heads), 32)
        num_heads  = hidden_dim // max(1, head_dim)
        if num_heads < 4:
            num_heads = 4; head_dim = hidden_dim // 4

        # Snap num_heads ke power-of-2 terdekat yang membagi hidden_dim
        _std_heads = [4, 8, 16, 32, 64]
        if num_heads not in _std_heads:
            valid = [h for h in _std_heads if hidden_dim % h == 0]
            if valid:
                num_heads = min(valid, key=lambda h: abs(h - num_heads))
                head_dim  = hidden_dim // num_heads

        # Pastikan head_dim aligned ke optimal_tile_size GPU
        if head_dim % tile != 0:
            head_dim = max(tile, (head_dim // tile) * tile)
            if hidden_dim % head_dim != 0:
                head_dim = tile  # fallback
            num_heads = hidden_dim // head_dim

        attn_type   = rng.choice(fp["attn_candidates"])
        ffn_type    = rng.choice(fp["ffn_candidates"])
        ffn_mult    = round(rng.uniform(*fp["ffn_mult_range"]), 1)
        seq_len     = rng.choice([s for s in [512, 1024, 2048, 4096, 8192, 16384]
                                   if fp["seq_range"][0] <= s <= fp["seq_range"][1]] or [2048])
        kv_heads    = rng.choice([h for h in [1, 2, 4, 8]
                                   if num_heads % h == 0 and h <= num_heads])
        # FIX-MOE-4: num_experts dibatasi berdasarkan VRAM GPU.
        # Setiap expert menambah E×(3×D×ffn) params. Semakin besar VRAM → boleh lebih banyak experts.
        if ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            if gpu.vram_gb < 20:
                num_experts = 4                          # T4, V100-16GB, A10G: max 4 experts
            elif gpu.vram_gb < 48:
                num_experts = rng.choice([4, 8])         # A100-40GB, A30: 4 atau 8 experts
            else:
                num_experts = rng.choice([4, 8, 16])    # A100-80GB, H100, H200: boleh 16
        else:
            num_experts = 1
        top_k       = rng.choice([1, 2])     if ffn_type in (FFNType.MOE, FFNType.MOE_TOPK) else 1
        use_gc      = rng.random() < fp["gc_prob"]
        norm_t      = rng.choice([NormType.RMSNORM, NormType.LAYERNORM])
        pos_enc     = rng.choice([PosEncType.ROPE, PosEncType.ALIBI, PosEncType.LEARNED])
        opt_type    = rng.choice(fp.get("opt_candidates", [OptimizerType.ADAM_FP32]))

        # FIX-MOE-5: Override optimizer untuk MoE pada GPU kecil/menengah.
        # ZERO2/ZERO3 hanya hemat VRAM bila ada multi-GPU (sharding). Pada single GPU
        # mereka sama mahalnya dengan Adam FP32 tetapi dengan overhead komunikasi.
        # Adam 8-bit / Lion jauh lebih hemat: 6 bytes/param vs 12-16 bytes/param.
        if ffn_type in (FFNType.MOE, FFNType.MOE_TOPK):
            if gpu.vram_gb < 48:
                opt_type = rng.choice([OptimizerType.ADAM_8BIT, OptimizerType.LION])
            # GPU ≥ 48GB (A100-80, H100, H200): ZERO2/ZERO3 masuk akal untuk multi-GPU cluster
        use_compile = rng.random() < 0.65

        # F9-NEW: batch_size proporsional ke VRAM GPU aktual
        # Base batch=4 untuk T4 (16 GB), scale dengan VRAM ratio
        _vram_ratio = gpu.vram_gb / 16.0
        _batch_raw  = int(4 * _vram_ratio)
        _batch_raw  = max(1, min(_batch_raw, 64))
        batch_size  = 2 ** round(math.log2(max(1, _batch_raw)))

        cfg.hidden_dim     = hidden_dim; cfg.num_layers      = num_layers
        cfg.num_heads      = num_heads;  cfg.num_kv_heads    = (
            kv_heads if attn_type in (AttentionType.GQA, AttentionType.MQA) else num_heads)
        cfg.head_dim       = head_dim
        cfg.attn_type      = attn_type;  cfg.ffn_type        = ffn_type
        cfg.ffn_multiplier = ffn_mult;   cfg.num_experts     = num_experts
        cfg.top_k_experts  = top_k;      cfg.seq_len         = seq_len
        cfg.batch_size     = batch_size; cfg.norm_type       = norm_t
        cfg.pos_enc        = pos_enc
        cfg.use_gradient_checkpointing = use_gc
        cfg.optimizer_type    = opt_type
        cfg.use_torch_compile = use_compile
        cfg.window_size       = (rng.choice([128, 256, 512])
                                  if attn_type == AttentionType.SLIDE else seq_len)
        cfg.global_attn_layers = max(1, num_layers // 4)

        # ── Derived computations ──────────────────────────────────────────────
        cfg.param_count = self._compute_params(cfg)
        w_gb, a_gb, o_gb, kv_gb = self._compute_memory(cfg)
        cfg.vram_weights_gb      = round(w_gb, 3)
        cfg.vram_activations_gb  = round(a_gb, 3)
        cfg.vram_optimizer_gb    = round(o_gb, 3)
        cfg.vram_kv_cache_gb     = round(kv_gb, 3)
        cfg.vram_fragmentation_gb = round(
            self._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb), 3)
        total_gb = w_gb + a_gb + o_gb + kv_gb + cfg.vram_fragmentation_gb

        # Tahap 1: kurangi batch_size bertahap jika melebihi 80% VRAM
        tries = 0
        while total_gb > (gpu.vram_gb * VRAM_LIMIT_PCT) and batch_size > 1 and tries < 6:
            batch_size = max(1, batch_size // 2); cfg.batch_size = batch_size
            w_gb, a_gb, o_gb, kv_gb = self._compute_memory(cfg)
            cfg.vram_fragmentation_gb = round(
                self._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb), 3)
            total_gb = w_gb + a_gb + o_gb + kv_gb + cfg.vram_fragmentation_gb
            tries += 1

        # Tahap 2: aktifkan GC sebagai last resort
        if total_gb > (gpu.vram_gb * VRAM_LIMIT_PCT) and not use_gc:
            cfg.use_gradient_checkpointing = True
            w_gb, a_gb, o_gb, kv_gb = self._compute_memory(cfg)
            cfg.vram_fragmentation_gb = round(
                self._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb), 3)
            total_gb = w_gb + a_gb + o_gb + kv_gb + cfg.vram_fragmentation_gb

        # FIX-MOE-6: Tahap 3 — MoE saja: kurangi num_experts jika masih OOM.
        # Ini adalah senjata terakhir setelah batch_size dan GC. Mengurangi expert
        # mengurangi params secara masif (proporsi terhadap E).
        if (total_gb > (gpu.vram_gb * VRAM_LIMIT_PCT)
                and cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
                and cfg.num_experts > 4):
            cfg.num_experts = max(4, cfg.num_experts // 2)
            cfg.param_count = self._compute_params(cfg)
            w_gb, a_gb, o_gb, kv_gb = self._compute_memory(cfg)
            cfg.vram_fragmentation_gb = round(
                self._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb), 3)
            total_gb = w_gb + a_gb + o_gb + kv_gb + cfg.vram_fragmentation_gb

        # FIX-MOE-7: Tahap 4 — MoE saja: kurangi hidden_dim satu step jika masih OOM.
        # Sangat jarang terjadi setelah Tahap 1-3, tapi jaga-jaga.
        if (total_gb > (gpu.vram_gb * VRAM_LIMIT_PCT)
                and cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
                and cfg.hidden_dim > 384):
            new_hd = self._make_divisible(cfg.hidden_dim * 3 // 4, 128)
            new_hd = max(384, new_hd)
            # Recompute heads agar num_heads × head_dim = hidden_dim tetap valid
            _valid_hd = [h for h in [32, 64, 128] if new_hd % h == 0 and new_hd // h >= 4]
            if _valid_hd:
                _best_hd = min(_valid_hd, key=lambda h: abs(h - cfg.head_dim))
                cfg.hidden_dim   = new_hd
                cfg.head_dim     = _best_hd
                cfg.num_heads    = new_hd // _best_hd
                cfg.num_kv_heads = min(cfg.num_kv_heads, cfg.num_heads)
                cfg.param_count  = self._compute_params(cfg)
                w_gb, a_gb, o_gb, kv_gb = self._compute_memory(cfg)
                cfg.vram_fragmentation_gb = round(
                    self._compute_fragmentation(cfg) * (w_gb + a_gb + o_gb + kv_gb), 3)
                total_gb = w_gb + a_gb + o_gb + kv_gb + cfg.vram_fragmentation_gb

        cfg.vram_weights_gb      = round(w_gb, 3)
        cfg.vram_activations_gb  = round(a_gb, 3)
        cfg.vram_optimizer_gb    = round(o_gb, 3)
        cfg.vram_kv_cache_gb     = round(kv_gb, 3)
        cfg.vram_total_gb        = round(total_gb, 3)
        cfg.vram_usage_pct       = round(total_gb / gpu.vram_gb * 100, 2)
        cfg.fits_gpu             = total_gb <= (gpu.vram_gb * VRAM_LIMIT_PCT)

        fwd, bwd, attn_fwd, ffn_fwd = self._compute_flops(cfg)
        cfg.flops_per_token_fwd  = round(fwd, 0)
        cfg.flops_per_token_bwd  = round(bwd, 0)
        cfg.flops_attn_fwd       = round(attn_fwd, 0)
        cfg.flops_ffn_fwd        = round(ffn_fwd, 0)
        cfg.arithmetic_intensity = round(self._compute_arithmetic_intensity(cfg), 2)

        thr = self._estimate_throughput(cfg)
        cfg.tokens_per_sec_estimate = thr["tokens_per_sec"]
        cfg.mfu_estimate            = thr["mfu"]
        cfg.ms_per_step             = thr["ms_per_step"]
        cfg.bottleneck              = thr["bottleneck"]
        cfg.bottleneck_factors      = thr["bottleneck_factors"]
        cfg.compiler_speedup        = thr["compiler_speedup"]
        cfg.warp_divergence_pct     = thr["warp_divergence"]["warp_divergence_pct"]
        cfg.sm_occupancy            = thr["sm_occupancy"]
        cfg.fitness_score           = self._fitness_score(cfg)

        param_str = (f"{cfg.param_count/1e6:.0f}M" if cfg.param_count < 1e9
                     else f"{cfg.param_count/1e9:.2f}B")
        uid = hashlib.md5(
            f"{family_name}{hidden_dim}{num_layers}{attn_type}".encode()
        ).hexdigest()[:4].upper()
        cfg.arch_id   = f"ARC-{uid}"
        cfg.arch_name = (f"{family_name} | {param_str} | "
                         f"{attn_type.name}/{ffn_type.name} | L{num_layers}×D{hidden_dim}")
        return cfg

    def generate_all_families(self, n_per_family: int = 2) -> List[ArchConfig]:
        all_archs = []
        for family in self.FAMILY_PROFILES.keys():
            for _ in range(n_per_family):
                all_archs.append(self.generate_one(family))
        return sorted(all_archs, key=lambda x: x.fitness_score, reverse=True)
