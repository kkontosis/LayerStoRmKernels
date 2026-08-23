"""
Q4_K Tensor-Core GEMM Speed Benchmark — TC vs Phase 2 vs NVFP4 vs cuBLAS BF16.

Usage:
  python benchmarks/benchmark_q4k_cutlass_gemm.py
"""

import torch
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sm120_mla_kernels
from tests.test_q4k_gemm_reference import generate_q4k_weight, PROJECTION_SHAPES
from tests.test_nvfp4_gemm_reference import generate_nvfp4_weight

WARMUP = 50
TIMED = 200
BATCH_SIZES = [1, 8, 32, 128, 512, 1024]


def _time_kernel(fn, warmup=WARMUP, timed=TIMED):
    """Time a CUDA kernel with proper synchronization. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Q4_K Tensor-Core GEMM Speed Benchmark — {gpu_name}")
    print("=" * 140)

    header = (f"{'proj':>6} {'M':>5} | {'Q4K-TC (us)':>11} {'Q4K-P2 (us)':>11} {'NVFP4 (us)':>10} {'BF16 (us)':>10} |"
              f" {'TC TFLOPS':>10} {'P2 TFLOPS':>10} {'NVFP4 TFLOPS':>12} | {'TC/P2':>6}")
    print(f"\n{header}")
    print("-" * 140)

    results = []

    for name, N, K in PROJECTION_SHAPES:
        # Prepare Q4_K weight
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        # Prepare NVFP4 weight
        w_nvfp4_uint8, w_nvfp4_scale, w_nvfp4_scale2 = generate_nvfp4_weight(N, K)
        w_nvfp4_uint8_gpu = w_nvfp4_uint8.cuda()
        w_nvfp4_scale_gpu = w_nvfp4_scale.cuda()
        nvfp4_scale2_val = w_nvfp4_scale2.item()
        w_nvfp4_scale_pp = sm120_mla_kernels.nvfp4_gemm_preprocess(w_nvfp4_scale_gpu, N, K)

        # BF16 weight for baseline
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        for M in BATCH_SIZES:
            torch.manual_seed(42)
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

            # Q4_K tensor-core GEMM (Phase 3)
            t_tc = _time_kernel(
                lambda: sm120_mla_kernels.q4k_cutlass_gemm(x, w_q4k_gpu)
            )

            # Q4_K dequant-GEMM (Phase 2)
            t_p2 = _time_kernel(
                lambda: sm120_mla_kernels.q4k_dequant_gemm(x, w_q4k_gpu)
            )

            # NVFP4 GEMM (Phase 1)
            t_nvfp4 = _time_kernel(
                lambda: sm120_mla_kernels.nvfp4_gemm(x, w_nvfp4_uint8_gpu, w_nvfp4_scale_pp, nvfp4_scale2_val)
            )

            # BF16 GEMM baseline
            t_bf16 = _time_kernel(
                lambda: torch.mm(x, w_bf16.T)
            )

            flops = 2.0 * M * N * K
            tc_us = t_tc * 1000
            p2_us = t_p2 * 1000
            nvfp4_us = t_nvfp4 * 1000
            bf16_us = t_bf16 * 1000
            tc_tflops = flops / (t_tc * 1e-3) / 1e12
            p2_tflops = flops / (t_p2 * 1e-3) / 1e12
            nvfp4_tflops = flops / (t_nvfp4 * 1e-3) / 1e12
            speedup_tc_p2 = p2_us / tc_us

            print(f"{name:>6} {M:>5} | {tc_us:>11.1f} {p2_us:>11.1f} {nvfp4_us:>10.1f} {bf16_us:>10.1f} |"
                  f" {tc_tflops:>10.2f} {p2_tflops:>10.2f} {nvfp4_tflops:>12.2f} | {speedup_tc_p2:>5.2f}x")

            results.append({
                "projection": name,
                "M": M, "N": N, "K": K,
                "q4k_tc_us": round(tc_us, 1),
                "q4k_p2_us": round(p2_us, 1),
                "nvfp4_us": round(nvfp4_us, 1),
                "bf16_us": round(bf16_us, 1),
                "speedup_tc_vs_p2": round(speedup_tc_p2, 3),
                "q4k_tc_tflops": round(tc_tflops, 2),
                "q4k_p2_tflops": round(p2_tflops, 2),
                "nvfp4_tflops": round(nvfp4_tflops, 2),
            })

        print()

    # Weight memory summary
    print("Weight memory per layer:")
    for name, N, K in PROJECTION_SHAPES:
        q4k_mb = N * K * 144 / 256 / 1e6   # 4.5 bpw
        nvfp4_mb = (N * K / 2 + N * K / 16) / 1e6  # packed + scales
        bf16_mb = N * K * 2 / 1e6
        print(f"  {name:>6} [{N}, {K}]: Q4_K {q4k_mb:.1f} MB, NVFP4 {nvfp4_mb:.1f} MB, BF16 {bf16_mb:.1f} MB"
              f" (BF16/Q4_K = {bf16_mb/q4k_mb:.1f}x)")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_q4k_cutlass_gemm_results.json")
    with open(out_path, "w") as f:
        json.dump({"gpu": gpu_name, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
