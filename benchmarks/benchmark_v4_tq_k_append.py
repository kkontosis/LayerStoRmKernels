"""
V4K-16a: TQ k_append microbenchmark — baseline + decomposed rotation comparison

Measures:
  1. Current fused kernel: normalize → rotate (scalar FP32 matvec) → quantize → pack → write
  2. Decomposed: normalize → cuBLAS BF16 GEMM rotation → quantize+pack kernel
  3. Individual components (norm, rotation GEMM, quant+pack)

The 512×512 Pi rotation is the bottleneck in fused TQ k_append.

Usage:
  python benchmarks/benchmark_v4_tq_k_append.py -v
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
    V4_TQ_BYTES_PER_ENTRY,
    load_codebook, generate_rotation_matrix,
)

try:
    import sm120_mla_kernels as k
    HAS_KERNELS = torch.cuda.is_available()
except ImportError:
    HAS_KERNELS = False

WARMUP_ITERS = 10
TIMED_ITERS = 100

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_tq_k_append_results.json")


def _time_kernel(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return times


def _stats(times):
    ts = sorted(times)
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "p95_us": ts[int(len(ts) * 0.95)],
        "mean_us": statistics.mean(times),
    }


def _collect_env():
    env = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
    return env


def bench_fused(num_tokens, Pi, centroids, boundaries, verbose=False):
    """Current fused kernel: normalize+rotate+quantize+pack per CTA."""
    k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    tq_cache = torch.zeros(num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

    def run():
        k.v4_tq_k_append(k_nope, k_rope, v_nope, tq_cache, slot_mapping,
                          Pi, centroids, boundaries)

    times = _time_kernel(run)
    st = _stats(times)
    st["num_tokens"] = num_tokens
    if verbose:
        print(f"  Fused   (N={num_tokens:>5d}): median={st['median_us']:>8.1f} us  "
              f"min={st['min_us']:>8.1f} us  per_token={st['median_us']/num_tokens:>6.1f} us")
    return st


def bench_decomposed_gemm(num_tokens, Pi, centroids, boundaries, verbose=False):
    """Decomposed: norm → cuBLAS BF16 GEMM → quant+pack (Python-side, no custom kernel for quant yet)."""
    k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    Pi_bf16 = Pi.to(torch.bfloat16)
    boundaries_full = boundaries

    def run():
        # K: normalize → rotate via GEMM
        k_norm = k_nope.float().norm(dim=1, keepdim=True).clamp(min=1e-10)
        k_unit = (k_nope.float() / k_norm).to(torch.bfloat16)
        k_rot = k_unit @ Pi_bf16.T  # cuBLAS GEMM: [N, 512] @ [512, 512]

        # V: normalize → rotate via GEMM
        v_norm = v_nope.float().norm(dim=1, keepdim=True).clamp(min=1e-10)
        v_unit = (v_nope.float() / v_norm).to(torch.bfloat16)
        v_rot = v_unit @ Pi_bf16.T

    times = _time_kernel(run)
    st = _stats(times)
    st["num_tokens"] = num_tokens
    if verbose:
        print(f"  GEMM-only (N={num_tokens:>5d}): median={st['median_us']:>8.1f} us  "
              f"min={st['min_us']:>8.1f} us  per_token={st['median_us']/num_tokens:>6.1f} us")
    return st


def bench_gemm_binding(num_tokens, Pi, centroids, boundaries, verbose=False):
    """New GEMM-based binding: normalize kernel → cuBLAS BF16 GEMM → quant+pack kernel."""
    k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    Pi_bf16 = Pi.to(torch.bfloat16)

    num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    tq_cache = torch.zeros(num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

    def run():
        k.v4_tq_k_append_gemm(k_nope, k_rope, v_nope, tq_cache, slot_mapping,
                               Pi_bf16, centroids, boundaries)

    times = _time_kernel(run)
    st = _stats(times)
    st["num_tokens"] = num_tokens
    if verbose:
        print(f"  GEMM    (N={num_tokens:>5d}): median={st['median_us']:>8.1f} us  "
              f"min={st['min_us']:>8.1f} us  per_token={st['median_us']/num_tokens:>6.1f} us")
    return st


def bench_gemm_only(num_tokens, Pi, verbose=False):
    """Just the GEMM rotation, no norm/quant — to isolate rotation cost."""
    x = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    Pi_bf16 = Pi.to(torch.bfloat16)

    def run():
        # 2 rotations (K + V)
        torch.mm(x, Pi_bf16.T)
        torch.mm(x, Pi_bf16.T)

    times = _time_kernel(run)
    st = _stats(times)
    st["num_tokens"] = num_tokens
    if verbose:
        print(f"  GEMM×2  (N={num_tokens:>5d}): median={st['median_us']:>8.1f} us  "
              f"min={st['min_us']:>8.1f} us  per_token={st['median_us']/num_tokens:>6.1f} us")
    return st


def main():
    parser = argparse.ArgumentParser(description="V4K-16a: TQ k_append microbenchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not HAS_KERNELS:
        print("ERROR: sm120_mla_kernels not available")
        sys.exit(1)

    centroids_cpu, boundaries_cpu = load_codebook()
    Pi_cpu = generate_rotation_matrix()
    Pi = Pi_cpu.cuda()
    centroids = centroids_cpu.cuda()
    boundaries = boundaries_cpu[1:-1].cuda()

    token_counts = [1, 4, 16, 64, 256, 1024]

    all_results = {"env": _collect_env()}

    if args.verbose:
        print("=" * 72)
        print("V4K-16a: TQ k_append Microbenchmark")
        print("=" * 72)

    # Benchmark current fused kernel
    if args.verbose:
        print("\n--- Current fused kernel (scalar rotation) ---")
    fused = {}
    for n in token_counts:
        fused[str(n)] = bench_fused(n, Pi, centroids, boundaries, args.verbose)
    all_results["fused"] = fused

    # Benchmark new GEMM-based binding
    if args.verbose:
        print("\n--- GEMM binding (normalize → cuBLAS → quant+pack) ---")
    gemm_bind = {}
    for n in token_counts:
        gemm_bind[str(n)] = bench_gemm_binding(n, Pi, centroids, boundaries, args.verbose)
    all_results["gemm_binding"] = gemm_bind

    # Benchmark GEMM only (2× for K+V) — theoretical floor
    if args.verbose:
        print("\n--- GEMM-only: 2× [N,512]@[512,512] (theoretical floor) ---")
    gemm = {}
    for n in token_counts:
        gemm[str(n)] = bench_gemm_only(n, Pi, args.verbose)
    all_results["gemm_only"] = gemm

    # Summary
    if args.verbose:
        print(f"\n{'='*72}")
        print("Summary: Fused (baseline) vs GEMM binding (optimized)")
        print(f"{'N':>6s}  {'Fused(us)':>10s}  {'GEMM(us)':>10s}  {'speedup':>8s}  {'floor(us)':>10s}")
        for n in token_counts:
            f = fused[str(n)]["median_us"]
            g = gemm_bind[str(n)]["median_us"]
            fl = gemm[str(n)]["median_us"]
            print(f"{n:>6d}  {f:>10.1f}  {g:>10.1f}  {f/g:>7.2f}x  {fl:>10.1f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if args.verbose:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
