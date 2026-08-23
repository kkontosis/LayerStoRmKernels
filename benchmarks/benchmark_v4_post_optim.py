"""
V4K-17a: Post-optimization re-benchmark

Compares Phase 11 baseline (nsp=1, fused rotation) against optimized
(optimal nsp, GEMM rotation) for all V4 kernels. Reports per-kernel speedup.

Usage:
  python benchmarks/benchmark_v4_post_optim.py -v
"""

import sys
import os
import math
import json
import argparse
import datetime
import statistics

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE,
    V4_FP8_BYTES_PER_ENTRY, V4_TQ_BYTES_PER_ENTRY,
    load_codebook, generate_rotation_matrix,
)

try:
    import sm120_mla_kernels as k
    HAS_KERNELS = torch.cuda.is_available()
except ImportError:
    HAS_KERNELS = False

WARMUP = 10
ITERS = 100
H_Q = 64

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_post_optim_results.json")


def _time(fn):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(times)


def _alloc_fp8_cache(n):
    np_ = (n + PAGE_SIZE - 1) // PAGE_SIZE
    c = torch.zeros(np_ * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    kn = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    kr = torch.randn(n, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    vn = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    sm = torch.arange(n, dtype=torch.int32, device='cuda')
    k.v4_fp8_k_append(kn, kr, vn, c, sm)
    return c


def _alloc_tq_cache(n, Pi, centroids, boundaries):
    np_ = (n + PAGE_SIZE - 1) // PAGE_SIZE
    c = torch.zeros(np_ * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    kn = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    kr = torch.randn(n, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    vn = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    sm = torch.arange(n, dtype=torch.int32, device='cuda')
    k.v4_tq_k_append(kn, kr, vn, c, sm, Pi, centroids, boundaries)
    return c


def bench_decode(verbose=False):
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
    swa_c = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_bt = torch.zeros(1, 1, dtype=torch.int32, device='cuda')
    swa_sl = torch.zeros(1, dtype=torch.int32, device='cuda')

    results = []

    configs = [
        ("CSA FP8", 256, 256),
        ("CSA FP8", 1024, 1024),
        ("CSA FP8", 4096, 1024),
        ("CSA FP8", 16384, 1024),
    ]

    for label, s_kv, topk in configs:
        cache = _alloc_fp8_cache(s_kv)
        qn = torch.randn(1, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        idx = torch.randint(0, s_kv, (1, 1, topk), dtype=torch.int32, device='cuda')

        t_base = _time(lambda: k.v4_csa_fp8_decode(
            qn, qr, cache, idx, swa_c, swa_bt, swa_sl,
            sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 1))

        t_opt = _time(lambda: k.v4_csa_fp8_decode(
            qn, qr, cache, idx, swa_c, swa_bt, swa_sl,
            sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 32))

        results.append((f"{label} s={s_kv} tk={topk}", t_base, t_opt, 1, 32))
        del cache; torch.cuda.empty_cache()

    centroids_cpu, boundaries_cpu = load_codebook()
    Pi = generate_rotation_matrix().cuda()
    centroids = centroids_cpu.cuda()
    boundaries = boundaries_cpu[1:-1].cuda()

    tq_configs = [
        ("CSA TQ", 256, 256),
        ("CSA TQ", 1024, 1024),
        ("CSA TQ", 4096, 1024),
        ("CSA TQ", 16384, 1024),
    ]

    for label, s_kv, topk in tq_configs:
        cache = _alloc_tq_cache(s_kv, Pi, centroids, boundaries)
        qr_rot = torch.randn(1, 1, H_Q, HEAD_DIM, device='cuda')
        qr_rope = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        idx = torch.randint(0, s_kv, (1, 1, topk), dtype=torch.int32, device='cuda')

        t_base = _time(lambda: k.v4_csa_tq_decode(
            qr_rot, qr_rope, cache, idx, centroids, sm_scale, 1))

        t_opt = _time(lambda: k.v4_csa_tq_decode(
            qr_rot, qr_rope, cache, idx, centroids, sm_scale, 8))

        results.append((f"{label} s={s_kv} tk={topk}", t_base, t_opt, 1, 8))
        del cache; torch.cuda.empty_cache()

    if verbose:
        print(f"\n{'Kernel':<30s}  {'Base(us)':>9s}  {'Opt(us)':>9s}  {'Speedup':>8s}  nsp")
        print("-" * 72)
        for name, tb, to, nb, no in results:
            print(f"  {name:<28s}  {tb:>9.1f}  {to:>9.1f}  {tb/to:>7.2f}x  {nb}→{no}")

    return results


def bench_tq_k_append(verbose=False):
    centroids_cpu, boundaries_cpu = load_codebook()
    Pi = generate_rotation_matrix().cuda()
    Pi_bf16 = Pi.to(torch.bfloat16)
    centroids = centroids_cpu.cuda()
    boundaries = boundaries_cpu[1:-1].cuda()

    results = []
    for n in [1, 16, 64, 256, 1024]:
        kn = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        kr = torch.randn(n, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        vn = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        np_ = (n + PAGE_SIZE - 1) // PAGE_SIZE
        c = torch.zeros(np_ * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        sm = torch.arange(n, dtype=torch.int32, device='cuda')

        t_base = _time(lambda: k.v4_tq_k_append(kn, kr, vn, c, sm, Pi, centroids, boundaries))
        t_opt = _time(lambda: k.v4_tq_k_append_gemm(kn, kr, vn, c, sm, Pi_bf16, centroids, boundaries))
        results.append((f"TQ k_append N={n}", t_base, t_opt))

    if verbose:
        print(f"\n{'Kernel':<30s}  {'Base(us)':>9s}  {'Opt(us)':>9s}  {'Speedup':>8s}")
        print("-" * 60)
        for name, tb, to in results:
            print(f"  {name:<28s}  {tb:>9.1f}  {to:>9.1f}  {tb/to:>7.2f}x")

    return results


def bench_e2e(verbose=False):
    centroids_cpu, boundaries_cpu = load_codebook()
    Pi = generate_rotation_matrix().cuda()
    Pi_bf16 = Pi.to(torch.bfloat16)
    centroids = centroids_cpu.cuda()
    boundaries = boundaries_cpu[1:-1].cuda()
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
    WINDOW, STRIDE = 8, 1

    swa_c = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_bt = torch.zeros(1, 1, dtype=torch.int32, device='cuda')
    swa_sl = torch.zeros(1, dtype=torch.int32, device='cuda')

    results = []
    for nc, topk in [(1024, 512), (4096, 1024), (16384, 1024)]:
        nt = WINDOW + nc * STRIDE
        inp_k = torch.randn(nt, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_kr = torch.randn(nt, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_v = torch.randn(nt, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(nt, WINDOW, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(nt, WINDOW, dtype=torch.bfloat16, device='cuda')
        cos_ = torch.randn(nt, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin_ = torch.randn(nt, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

        sm_map = torch.arange(nc, dtype=torch.int32, device='cuda')
        qn = torch.randn(1, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        qr = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        idx = torch.randint(0, nc, (1, 1, topk), dtype=torch.int32, device='cuda')

        # --- FP8 E2E (baseline: nsp=1, optimized: nsp=32) ---
        np_ = (nc + PAGE_SIZE - 1) // PAGE_SIZE
        fp8_c = torch.zeros(np_ * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')

        def fp8_base():
            kn, kr, vn = k.v4_csa_compress(inp_k, inp_kr, inp_v, gate, pos_bias, cos_, sin_,
                                             HEAD_DIM, QK_ROPE_HEAD_DIM, WINDOW, STRIDE)
            k.v4_fp8_k_append(kn, kr, vn, fp8_c, sm_map)
            k.v4_csa_fp8_decode(qn, qr, fp8_c, idx, swa_c, swa_bt, swa_sl,
                                sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 1)

        def fp8_opt():
            kn, kr, vn = k.v4_csa_compress(inp_k, inp_kr, inp_v, gate, pos_bias, cos_, sin_,
                                             HEAD_DIM, QK_ROPE_HEAD_DIM, WINDOW, STRIDE)
            k.v4_fp8_k_append(kn, kr, vn, fp8_c, sm_map)
            k.v4_csa_fp8_decode(qn, qr, fp8_c, idx, swa_c, swa_bt, swa_sl,
                                sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 32)

        t_fp8_base = _time(fp8_base)
        t_fp8_opt = _time(fp8_opt)

        # --- TQ E2E (baseline: nsp=1/fused, optimized: nsp=8/GEMM) ---
        tq_c = torch.zeros(np_ * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')

        def tq_base():
            kn, kr, vn = k.v4_csa_compress(inp_k, inp_kr, inp_v, gate, pos_bias, cos_, sin_,
                                             HEAD_DIM, QK_ROPE_HEAD_DIM, WINDOW, STRIDE)
            k.v4_tq_k_append(kn, kr, vn, tq_c, sm_map, Pi, centroids, boundaries)
            q_rot = (qn.float() @ Pi.T).contiguous()
            out, lse = k.v4_csa_tq_decode(q_rot, qr, tq_c, idx, centroids, sm_scale, 1)
            _ = (out.float() @ Pi).to(torch.bfloat16)

        def tq_opt():
            kn, kr, vn = k.v4_csa_compress(inp_k, inp_kr, inp_v, gate, pos_bias, cos_, sin_,
                                             HEAD_DIM, QK_ROPE_HEAD_DIM, WINDOW, STRIDE)
            k.v4_tq_k_append_gemm(kn, kr, vn, tq_c, sm_map, Pi_bf16, centroids, boundaries)
            q_rot = (qn.float() @ Pi.T).contiguous()
            out, lse = k.v4_csa_tq_decode(q_rot, qr, tq_c, idx, centroids, sm_scale, 8)
            _ = (out.float() @ Pi).to(torch.bfloat16)

        t_tq_base = _time(tq_base)
        t_tq_opt = _time(tq_opt)

        results.append((nc, topk, t_fp8_base, t_fp8_opt, t_tq_base, t_tq_opt))

        del fp8_c, tq_c
        torch.cuda.empty_cache()

    if verbose:
        print(f"\n{'Config':<20s}  {'FP8 base':>9s}  {'FP8 opt':>9s}  {'FP8 spd':>8s}  "
              f"{'TQ base':>9s}  {'TQ opt':>9s}  {'TQ spd':>8s}")
        print("-" * 90)
        for nc, tk, fb, fo, tb, to in results:
            print(f"  nc={nc:<5d} tk={tk:<4d}  {fb:>9.1f}  {fo:>9.1f}  {fb/fo:>7.2f}x  "
                  f"{tb:>9.1f}  {to:>9.1f}  {tb/to:>7.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="V4K-17a: Post-optimization re-benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not HAS_KERNELS:
        print("ERROR: sm120_mla_kernels not available")
        sys.exit(1)

    if args.verbose:
        print("=" * 90)
        print("V4K-17a: Post-Optimization Re-Benchmark")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 90)

    if args.verbose:
        print("\n--- Decode kernels: nsp=1 (baseline) vs optimal nsp ---")
    decode_results = bench_decode(args.verbose)

    if args.verbose:
        print("\n--- TQ k_append: fused scalar (baseline) vs GEMM ---")
    k_append_results = bench_tq_k_append(args.verbose)

    if args.verbose:
        print("\n--- End-to-End CSA pipelines: all optimizations combined ---")
    e2e_results = bench_e2e(args.verbose)

    all_results = {
        "env": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "decode": [{"name": n, "baseline_us": b, "optimized_us": o} for n, b, o, *_ in decode_results],
        "tq_k_append": [{"name": n, "baseline_us": b, "optimized_us": o} for n, b, o in k_append_results],
        "e2e": [{"nc": nc, "topk": tk, "fp8_base": fb, "fp8_opt": fo, "tq_base": tb, "tq_opt": to}
                for nc, tk, fb, fo, tb, to in e2e_results],
    }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if args.verbose:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
