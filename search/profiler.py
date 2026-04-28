import time, math, warnings
from typing import Dict, Any
import numpy as np

warnings.filterwarnings("ignore")

# ── PyTorch availability ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.profiler import profile, record_function, ProfilerActivity
    TORCH  = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH  = False
    DEVICE = "cpu"

from hardware import GPUSpec
from arch_types import ArchConfig, FFNType


class TorchProfiler:
    """
    Builds a minimal but representative model and runs real torch.profiler.
    Captures: forward/backward timing, memory, kernel stats, occupancy estimate.
    Falls back to analytical roofline model when GPU unavailable.
    """

    def __init__(self, cfg: ArchConfig, gpu_spec: GPUSpec):
        self.cfg    = cfg
        self.gpu    = gpu_spec
        self.device = DEVICE

    # ── Minimal Model Builder ─────────────────────────────────────────────────

    def _build_minimal_model(self):
        """Constructs a single representative transformer layer for profiling."""
        cfg = self.cfg

        class SingleLayer(nn.Module):
            def __init__(self, d, h, ffn_dim, seq, use_moe=False, n_exp=1, topk=1):
                super().__init__()
                self.norm1 = nn.RMSNorm(d) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d)
                self.norm2 = nn.LayerNorm(d)
                self.q    = nn.Linear(d, d, bias=False)
                self.k    = nn.Linear(d, d // max(1, h // 4), bias=False)
                self.v    = nn.Linear(d, d // max(1, h // 4), bias=False)
                self.o    = nn.Linear(d, d, bias=False)
                self.gate = nn.Linear(d, ffn_dim, bias=False)
                self.up   = nn.Linear(d, ffn_dim, bias=False)
                self.down = nn.Linear(ffn_dim, d, bias=False)
                self.use_moe = use_moe; self.n_exp = n_exp; self.topk = topk
                if use_moe and n_exp > 1:
                    self.router  = nn.Linear(d, n_exp, bias=False)
                    self.experts = nn.ModuleList([
                        nn.Sequential(nn.Linear(d, ffn_dim), nn.SiLU(), nn.Linear(ffn_dim, d))
                        for _ in range(n_exp)
                    ])

            def forward(self, x):
                B, S, D = x.shape
                r    = self.norm1(x)
                q    = self.q(r); k = self.k(r); v = self.v(r)
                scale = math.sqrt(q.shape[-1])
                attn  = torch.bmm(q, k.transpose(-2, -1)) / scale
                attn  = F.softmax(attn.float(), dim=-1).to(q.dtype)
                out   = torch.bmm(attn, v)
                x     = x + self.o(out)
                r2    = self.norm2(x)
                if self.use_moe and self.n_exp > 1:
                    logits = self.router(r2)
                    tv, ti = torch.topk(logits, self.topk, dim=-1)
                    gates  = F.softmax(tv, dim=-1)
                    out2   = torch.zeros_like(r2)
                    for i in range(self.topk):
                        idx = ti[..., i:i+1]
                        for ei, exp in enumerate(self.experts):
                            mask = (idx == ei).float().unsqueeze(-1)
                            out2 += mask * exp(r2) * gates[..., i:i+1]
                else:
                    g    = F.silu(self.gate(r2))
                    out2 = self.down(g * self.up(r2))
                return x + out2

        D     = min(cfg.hidden_dim, 512)
        H     = max(4, min(cfg.num_heads, 8))
        ffn_d = min(int(D * cfg.ffn_multiplier), 2048)
        S     = min(cfg.seq_len, 512)
        is_moe = cfg.ffn_type in (FFNType.MOE, FFNType.MOE_TOPK)
        n_exp  = min(cfg.num_experts, 4) if is_moe else 1
        model  = SingleLayer(D, H, ffn_d, S, is_moe, n_exp, cfg.top_k_experts)
        return model, D, S

    # ── Real GPU Profiling ────────────────────────────────────────────────────

    def run(self, warmup_iters: int = 3, profile_iters: int = 5) -> Dict[str, Any]:
        """
        Run real torch.profiler on GPU.
        Falls back to analytical model if no CUDA device is available.
        """
        if not TORCH or self.device == "cpu":
            return self._analytical_fallback()

        try:
            model, D, S = self._build_minimal_model()
            model = model.to(self.device).to(torch.bfloat16)
            opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
            B     = min(self.cfg.batch_size, 2)
            x     = torch.randn(B, S, D, device=self.device, dtype=torch.bfloat16)

            # Warmup
            for _ in range(warmup_iters):
                y = model(x); loss = y.mean(); loss.backward()
                opt.step(); opt.zero_grad()

            torch.cuda.synchronize()
            fwd_times, bwd_times, mem_used = [], [], []
            cuda_events = []

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True, with_stack=False,
                profile_memory=True, with_flops=True,
            ) as prof:
                for i in range(profile_iters):
                    e_start = torch.cuda.Event(enable_timing=True)
                    e_end   = torch.cuda.Event(enable_timing=True)
                    e_start.record()

                    with record_function("forward"):
                        t0 = time.perf_counter()
                        y  = model(x)
                        torch.cuda.synchronize()
                        fwd_times.append((time.perf_counter() - t0) * 1000)

                    with record_function("backward"):
                        t1 = time.perf_counter()
                        loss = y.mean(); loss.backward()
                        torch.cuda.synchronize()
                        bwd_times.append((time.perf_counter() - t1) * 1000)

                    e_end.record()
                    torch.cuda.synchronize()
                    cuda_events.append(e_start.elapsed_time(e_end))

                    opt.step(); opt.zero_grad()
                    mem_used.append(torch.cuda.memory_allocated(self.device) / 1e6)

            key_avg      = prof.key_averages()
            cuda_ms_tot  = sum(e.cuda_time_total for e in key_avg) / 1e3 / profile_iters
            num_kernels  = len([e for e in key_avg if e.cuda_time_total > 0])
            total_flops_prof = sum(e.flops for e in key_avg if hasattr(e, "flops"))

            # Scale to full model
            scale_layers = self.cfg.num_layers
            scale_dim    = (self.cfg.hidden_dim / D) ** 2
            scale_seq    = self.cfg.seq_len / S
            scale_batch  = self.cfg.batch_size / B
            scale_total  = scale_layers * scale_dim * scale_seq * scale_batch

            fwd_m = float(np.mean(fwd_times)); bwd_m = float(np.mean(bwd_times))
            fwd_s = float(np.std(fwd_times));  bwd_s = float(np.std(bwd_times))

            est_fwd   = fwd_m * scale_total
            est_bwd   = bwd_m * scale_total
            est_total = est_fwd + est_bwd
            tok_per_s = (self.cfg.seq_len * self.cfg.batch_size) / (est_total / 1000)
            total_flops_full = (self.cfg.flops_per_token_fwd + self.cfg.flops_per_token_bwd) * \
                               self.cfg.seq_len * self.cfg.batch_size
            achieved_mfu = (total_flops_full / (est_total / 1000)) / (self.gpu.bf16_tflops * 1e12)

            theoretical_ms = total_flops_full / (self.gpu.bf16_tflops * 1e12) * 1000
            occupancy_est  = min(1.0, theoretical_ms / max(1e-3, est_total))
            sm_active_pct  = min(100, occupancy_est * self.gpu.typical_sm_occupancy * 100)

            return {
                "source":               "torch.profiler (real measured)",
                "fwd_ms_mini":          round(fwd_m, 3),
                "bwd_ms_mini":          round(bwd_m, 3),
                "fwd_std_ms":           round(fwd_s, 3),
                "bwd_std_ms":           round(bwd_s, 3),
                "fwd_bwd_ratio":        round(bwd_m / max(fwd_m, 1e-9), 2),
                "cuda_total_ms":        round(cuda_ms_tot, 3),
                "cuda_event_ms":        round(float(np.mean(cuda_events)), 3),
                "mem_mb":               round(float(np.mean(mem_used)), 1),
                "mem_peak_mb":          round(float(np.max(mem_used)), 1),
                "est_fwd_full_ms":      round(est_fwd, 1),
                "est_bwd_full_ms":      round(est_bwd, 1),
                "est_total_ms":         round(est_total, 1),
                "est_tokens_per_s":     int(tok_per_s),
                "est_mfu":              round(achieved_mfu, 4),
                "kernel_ops":           num_kernels,
                "profiler_flops":       int(total_flops_prof),
                "scale_factor":         round(scale_total, 2),
                "variance_pct":         round(fwd_s / max(fwd_m, 1e-9) * 100, 2),
                "kernel_occupancy_est": round(occupancy_est, 3),
                "sm_active_pct":        round(sm_active_pct, 1),
                "warp_efficiency_est":  round(min(1.0, achieved_mfu / max(0.1, self.gpu.mfu_typical_max)), 3),
            }

        except Exception as exc:
            return {**self._analytical_fallback(), "profiler_error": str(exc)}

    # ── Analytical Fallback ───────────────────────────────────────────────────

    def _analytical_fallback(self) -> Dict[str, Any]:
        """Returns roofline-based estimates when GPU profiling is not available."""
        cfg = self.cfg; gpu = self.gpu
        var = 1 + gpu.runtime_variance_pct / 100 * float(np.random.randn())
        return {
            "source":               "Analytical roofline model (no GPU)",
            "est_fwd_full_ms":      round(cfg.ms_per_step / 3 * var, 1),
            "est_bwd_full_ms":      round(cfg.ms_per_step * 2 / 3 * var, 1),
            "est_total_ms":         round(cfg.ms_per_step * var, 1),
            "est_tokens_per_s":     int(cfg.tokens_per_sec_estimate / var),
            "est_mfu":              round(cfg.mfu_estimate, 4),
            "bottleneck":           cfg.bottleneck,
            "variance_pct":         round(gpu.runtime_variance_pct, 2),
            "kernel_occupancy_est": round(cfg.sm_occupancy, 3),
            "sm_active_pct":        round(cfg.sm_occupancy * 100, 1),
            "warp_efficiency_est":  round(1.0 - cfg.warp_divergence_pct / 100, 3),
        }
