"""
V4K-11c/11d: V4 FP8 and TQ Decode Speed Benchmarks

Measures CSA FP8, HCA FP8, SWA, CSA TQ, and HCA TQ decode kernel latency
at various context lengths.

Usage:
  python benchmarks/benchmark_v4_decode.py -v
  python benchmarks/benchmark_v4_decode.py -o results.json
"""

import sys
import os
import math
import json
import argparse
import platform
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

WARMUP_ITERS = 10
TIMED_ITERS = 100
H_Q = 64
NUM_SM_PARTS = 1

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_decode_results.json")


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
        "std_us": statistics.stdev(times) if len(times) > 1 else 0.0,
        "n_iters": len(times),
    }


def _collect_env():
    env = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda or "N/A"
    return env


def _alloc_fp8_cache(num_entries):
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    cache_bytes = num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY
    cache = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_entries, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot_mapping = torch.arange(num_entries, dtype=torch.int32, device='cuda')
    k.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot_mapping)
    return cache


def _alloc_tq_cache(num_entries, Pi, centroids, boundaries):
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    cache_bytes = num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY
    cache = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_entries, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot_mapping = torch.arange(num_entries, dtype=torch.int32, device='cuda')
    k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot_mapping,
                      Pi, centroids, boundaries)
    return cache


def bench_csa_fp8_decode(context_lengths, verbose=False):
    results = {}
    b, s_q = 1, 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
    compressed_pbs = PAGE_SIZE
    swa_pbs = PAGE_SIZE

    # Minimal SWA cache (no SWA tokens)
    swa_cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_block_table = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
    swa_seqlens = torch.zeros(b, dtype=torch.int32, device='cuda')

    for s_kv in context_lengths:
        topk = min(s_kv, 1024)
        cache = _alloc_fp8_cache(s_kv)
        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.randint(0, s_kv, (b, s_q, topk), dtype=torch.int32, device='cuda')

        fn = lambda: k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, indices,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, topk, compressed_pbs, swa_pbs, NUM_SM_PARTS)

        times = _time_kernel(fn)
        st = _stats(times)
        st["s_kv"] = s_kv
        st["topk"] = topk
        results[f"csa_fp8_{s_kv}"] = st

        if verbose:
            print(f"  CSA FP8 (s_kv={s_kv:>7d}, topk={topk:>4d}): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
        del cache
        torch.cuda.empty_cache()

    return results


def bench_hca_fp8_decode(context_lengths, verbose=False):
    results = {}
    b, s_q = 1, 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
    swa_cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_block_table = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
    swa_seqlens = torch.zeros(b, dtype=torch.int32, device='cuda')

    for s_kv in context_lengths:
        cache = _alloc_fp8_cache(s_kv)
        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')

        fn = lambda: k.v4_hca_fp8_decode(
            q_nope, q_rope, cache, s_kv,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE, NUM_SM_PARTS)

        times = _time_kernel(fn)
        st = _stats(times)
        st["s_kv"] = s_kv
        results[f"hca_fp8_{s_kv}"] = st

        if verbose:
            print(f"  HCA FP8 (s_kv={s_kv:>7d}): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
        del cache
        torch.cuda.empty_cache()

    return results


def bench_csa_tq_decode(context_lengths, Pi, centroids, boundaries, verbose=False):
    results = {}
    b, s_q = 1, 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    for s_kv in context_lengths:
        topk = min(s_kv, 1024)
        cache = _alloc_tq_cache(s_kv, Pi, centroids, boundaries)
        q_rot = torch.randn(b, s_q, H_Q, HEAD_DIM, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.randint(0, s_kv, (b, s_q, topk), dtype=torch.int32, device='cuda')

        fn = lambda: k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, centroids, sm_scale)

        times = _time_kernel(fn)
        st = _stats(times)
        st["s_kv"] = s_kv
        st["topk"] = topk
        results[f"csa_tq_{s_kv}"] = st

        if verbose:
            print(f"  CSA TQ  (s_kv={s_kv:>7d}, topk={topk:>4d}): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
        del cache
        torch.cuda.empty_cache()

    return results


def bench_hca_tq_decode(context_lengths, Pi, centroids, boundaries, verbose=False):
    results = {}
    b, s_q = 1, 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    for s_kv in context_lengths:
        cache = _alloc_tq_cache(s_kv, Pi, centroids, boundaries)
        q_rot = torch.randn(b, s_q, H_Q, HEAD_DIM, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(s_kv, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).contiguous()

        fn = lambda: k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, centroids, sm_scale)

        times = _time_kernel(fn)
        st = _stats(times)
        st["s_kv"] = s_kv
        results[f"hca_tq_{s_kv}"] = st

        if verbose:
            print(f"  HCA TQ  (s_kv={s_kv:>7d}, dense): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
        del cache
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="V4 Decode Speed Benchmark")
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

    fp8_sizes = [256, 1024, 4096, 16384]
    tq_sizes = [256, 1024, 4096, 16384]
    hca_sizes = [64, 256, 1024]

    all_results = {"env": _collect_env()}

    if args.verbose:
        print("=" * 60)
        print("V4 Decode Speed Benchmarks")
        print("=" * 60)
        print(f"\nCSA FP8 Decode (h_q={H_Q}, split-KV parts={NUM_SM_PARTS}):")
    all_results["csa_fp8_decode"] = bench_csa_fp8_decode(fp8_sizes, args.verbose)

    if args.verbose:
        print(f"\nHCA FP8 Decode (h_q={H_Q}, dense):")
    all_results["hca_fp8_decode"] = bench_hca_fp8_decode(hca_sizes, args.verbose)

    if args.verbose:
        print(f"\nCSA TQ Decode (h_q={H_Q}, single CTA/head):")
    all_results["csa_tq_decode"] = bench_csa_tq_decode(tq_sizes, Pi, centroids, boundaries, args.verbose)

    if args.verbose:
        print(f"\nHCA TQ Decode (h_q={H_Q}, dense indices):")
    all_results["hca_tq_decode"] = bench_hca_tq_decode(hca_sizes, Pi, centroids, boundaries, args.verbose)

    # Print comparison table
    if args.verbose:
        print(f"\n{'='*60}")
        print("Comparison: CSA FP8 vs CSA TQ at same context lengths")
        print(f"{'s_kv':>8s}  {'FP8 (us)':>10s}  {'TQ (us)':>10s}  {'speedup':>8s}")
        for s_kv in tq_sizes:
            fp8_key = f"csa_fp8_{s_kv}"
            tq_key = f"csa_tq_{s_kv}"
            if fp8_key in all_results["csa_fp8_decode"] and tq_key in all_results["csa_tq_decode"]:
                fp8_t = all_results["csa_fp8_decode"][fp8_key]["median_us"]
                tq_t = all_results["csa_tq_decode"][tq_key]["median_us"]
                speedup = fp8_t / tq_t if tq_t > 0 else float('inf')
                print(f"{s_kv:>8d}  {fp8_t:>10.1f}  {tq_t:>10.1f}  {speedup:>7.2f}x")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if args.verbose:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
