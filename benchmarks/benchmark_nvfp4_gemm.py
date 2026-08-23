"""
NVFP4 GEMM Speed Benchmark — CUTLASS NVFP4 vs cuBLAS BF16 for attention weight projections.

Usage:
  python benchmarks/benchmark_nvfp4_gemm.py
"""

import torch
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sm120_mla_kernels
from tests.test_nvfp4_gemm_reference import generate_nvfp4_weight, PROJECTION_SHAPES

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
    print(f"NVFP4 GEMM Speed Benchmark — {gpu_name}")
    print("=" * 90)

    header = f"{'proj':>6} {'M':>5} | {'NVFP4 (us)':>10} {'BF16 (us)':>10} {'Speedup':>8} | {'NVFP4 TFLOPS':>12} {'BF16 TFLOPS':>12}"
    print(f"\n{header}")
    print("-" * 90)

    results = []

    for name, N, K in PROJECTION_SHAPES:
        # Prepare NVFP4 weight (once per shape)
        w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K)
        w_uint8_gpu = w_uint8.cuda()
        w_scale_gpu = w_scale.cuda()
        scale2_val = w_scale2.item()

        # Preprocess weight scales (done once)
        w_scale_pp = sm120_mla_kernels.nvfp4_gemm_preprocess(w_scale_gpu, N, K)

        # BF16 weight for baseline
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        for M in BATCH_SIZES:
            torch.manual_seed(42)
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

            # NVFP4 GEMM (with preprocessed scales)
            t_nvfp4 = _time_kernel(
                lambda: sm120_mla_kernels.nvfp4_gemm(x, w_uint8_gpu, w_scale_pp, scale2_val)
            )

            # BF16 GEMM baseline
            t_bf16 = _time_kernel(
                lambda: torch.mm(x, w_bf16.T)
            )

            flops = 2.0 * M * N * K
            nvfp4_us = t_nvfp4 * 1000
            bf16_us = t_bf16 * 1000
            nvfp4_tflops = flops / (t_nvfp4 * 1e-3) / 1e12
            bf16_tflops = flops / (t_bf16 * 1e-3) / 1e12
            speedup = bf16_us / nvfp4_us

            print(f"{name:>6} {M:>5} | {nvfp4_us:>10.1f} {bf16_us:>10.1f} {speedup:>7.2f}x | {nvfp4_tflops:>12.2f} {bf16_tflops:>12.2f}")

            results.append({
                "projection": name,
                "M": M, "N": N, "K": K,
                "nvfp4_us": round(nvfp4_us, 1),
                "bf16_us": round(bf16_us, 1),
                "speedup": round(speedup, 3),
                "nvfp4_tflops": round(nvfp4_tflops, 2),
                "bf16_tflops": round(bf16_tflops, 2),
            })

        print()

    # Weight memory summary
    print("Weight memory per layer:")
    for name, N, K in PROJECTION_SHAPES:
        nvfp4_mb = (N * K / 2 + N * K / 16) / 1e6  # packed + scales
        bf16_mb = N * K * 2 / 1e6
        print(f"  {name:>6} [{N}, {K}]: NVFP4 {nvfp4_mb:.1f} MB, BF16 {bf16_mb:.1f} MB ({bf16_mb/nvfp4_mb:.1f}x)")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_nvfp4_gemm_results.json")
    with open(out_path, "w") as f:
        json.dump({"gpu": gpu_name, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
