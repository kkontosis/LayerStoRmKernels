"""
GGUF GEMM speed benchmark — int (mmvq) vs dequant vs cuBLAS BF16, for the
attention weight projections, across all 6 GGUF weight types.

Usage:
  python benchmarks/benchmark_gguf_gemm.py
"""

import torch
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sm120_mla_kernels as K
from tests.test_gguf_gemm_reference import (
    generate_gguf_weight, PROJECTION_SHAPES, BLOCK_BYTES, BLOCK_VALUES,
)

WARMUP = 20
TIMED = 50
BATCH_SIZES = [1, 8, 32, 128, 512, 1024]
TYPES = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"]


def _time_kernel(fn, warmup=WARMUP, timed=TIMED):
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
    print(f"GGUF GEMM Speed Benchmark — {gpu_name}")
    print("=" * 110)

    results = []

    for qt in TYPES:
        bpw = 8.0 * BLOCK_BYTES[qt] / BLOCK_VALUES[qt]
        print(f"\n### {qt}  ({bpw:.2f} bpw)")
        header = (f"{'proj':>6} {'M':>5} | {'int us':>9} {'dequant us':>11} {'bf16 us':>9} | "
                  f"{'int TF':>8} {'deq TF':>8} {'bf16 TF':>8} | {'int/bf16':>8} {'deq/bf16':>8}")
        print(header)
        print("-" * 110)

        for name, N, Kdim in PROJECTION_SHAPES:
            if Kdim % BLOCK_VALUES[qt] != 0:
                continue
            packed = generate_gguf_weight(qt, N, Kdim, seed=N + Kdim)
            w_gpu = torch.from_numpy(packed).cuda()
            w_bf16 = torch.randn(N, Kdim, dtype=torch.bfloat16, device="cuda")

            for M in BATCH_SIZES:
                torch.manual_seed(42)
                x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

                t_int = _time_kernel(lambda: K.gguf_mul_mat(x, w_gpu, qt, "int"))
                t_deq = _time_kernel(lambda: K.gguf_mul_mat(x, w_gpu, qt, "dequant"))
                t_bf16 = _time_kernel(lambda: torch.mm(x, w_bf16.T))

                flops = 2.0 * M * N * Kdim
                int_us, deq_us, bf16_us = t_int * 1e3, t_deq * 1e3, t_bf16 * 1e3
                int_tf = flops / (t_int * 1e-3) / 1e12
                deq_tf = flops / (t_deq * 1e-3) / 1e12
                bf16_tf = flops / (t_bf16 * 1e-3) / 1e12

                print(f"{name:>6} {M:>5} | {int_us:>9.1f} {deq_us:>11.1f} {bf16_us:>9.1f} | "
                      f"{int_tf:>8.2f} {deq_tf:>8.2f} {bf16_tf:>8.2f} | "
                      f"{bf16_us/int_us:>7.2f}x {bf16_us/deq_us:>7.2f}x")

                results.append({
                    "type": qt, "projection": name, "M": M, "N": N, "K": Kdim,
                    "int_us": round(int_us, 1), "dequant_us": round(deq_us, 1),
                    "bf16_us": round(bf16_us, 1),
                    "int_tflops": round(int_tf, 2), "dequant_tflops": round(deq_tf, 2),
                    "bf16_tflops": round(bf16_tf, 2),
                    "int_speedup_vs_bf16": round(bf16_us / int_us, 3),
                    "dequant_speedup_vs_bf16": round(bf16_us / deq_us, 3),
                })

    # Weight-memory summary per type (vs BF16).
    print("\nWeight bytes/element and VRAM vs BF16 (q_a 1536x7168):")
    N, Kdim = 1536, 7168
    for qt in TYPES:
        be = BLOCK_BYTES[qt] / BLOCK_VALUES[qt]
        mb = N * Kdim * be / 1e6
        bf16_mb = N * Kdim * 2 / 1e6
        print(f"  {qt:>5}: {be:.4f} B/elt  {mb:6.1f} MB  (BF16/{qt} = {bf16_mb/mb:.1f}x)")

    out_path = os.path.join(os.path.dirname(__file__), "benchmark_gguf_gemm_results.json")
    with open(out_path, "w") as f:
        json.dump({"gpu": gpu_name, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
