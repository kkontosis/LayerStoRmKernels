"""
V4K-11b: V4 Lightning Indexer Speed Benchmarks

Measures scoring latency vs number of compressed blocks and top-k selection latency.
Lightning Indexer: FP8 per-block score GEMM → weighted sum → top-k selection.

Usage:
  python benchmarks/benchmark_v4_indexer.py -v
"""

import sys
import os
import json
import argparse
import platform
import datetime
import statistics

import torch

try:
    import sm120_mla_kernels as k
    HAS_KERNELS = torch.cuda.is_available()
except ImportError:
    HAS_KERNELS = False

INDEX_N_HEADS = 4
INDEX_HEAD_DIM = 128

WARMUP_ITERS = 10
TIMED_ITERS = 100

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_indexer_results.json")


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


def bench_lightning_score(block_counts, verbose=False):
    results = {}
    q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    score_proj = torch.randn(INDEX_N_HEADS, dtype=torch.float32, device="cuda")

    for num_blocks in block_counts:
        indexer_k = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM,
                                dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        k_scales = torch.ones(num_blocks, dtype=torch.float32, device="cuda")

        fn = lambda: k.v4_lightning_score(q_proj, indexer_k, k_scales, score_proj)

        times = _time_kernel(fn)
        st = _stats(times)
        st["num_blocks"] = num_blocks
        st["us_per_block"] = st["median_us"] / num_blocks
        results[f"score_{num_blocks}"] = st

        if verbose:
            print(f"  Score ({num_blocks:>6d} blocks): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us  "
                  f"per_block={st['us_per_block']:.4f} us")
    return results


def bench_lightning_topk(block_count, topk_values, verbose=False):
    results = {}
    scores = torch.randn(block_count, dtype=torch.float32, device="cuda")
    block_endpoints = torch.arange(1, block_count + 1, dtype=torch.int32, device="cuda") * 8
    query_pos = block_count * 8 - 1

    for topk in topk_values:
        fn = lambda: k.v4_lightning_topk(scores, block_endpoints, query_pos, topk)

        times = _time_kernel(fn)
        st = _stats(times)
        st["num_blocks"] = block_count
        st["topk"] = topk
        results[f"topk_{topk}_at_{block_count}"] = st

        if verbose:
            print(f"  TopK (k={topk:>4d}, {block_count:>6d} blocks): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
    return results


def main():
    parser = argparse.ArgumentParser(description="V4 Lightning Indexer Speed Benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not HAS_KERNELS:
        print("ERROR: sm120_mla_kernels not available")
        sys.exit(1)

    all_results = {"env": _collect_env()}

    block_counts = [1000, 4000, 16000, 64000, 250000]
    topk_values = [512, 1024]

    if args.verbose:
        print("=" * 60)
        print("V4 Lightning Indexer Speed Benchmarks")
        print("=" * 60)
        print(f"\nScoring latency (INDEX_N_HEADS={INDEX_N_HEADS}, INDEX_HEAD_DIM={INDEX_HEAD_DIM}):")
    all_results["lightning_score"] = bench_lightning_score(block_counts, args.verbose)

    if args.verbose:
        print(f"\nTop-K selection latency ({block_counts[-1]} blocks):")
    all_results["lightning_topk"] = bench_lightning_topk(block_counts[-1], topk_values, args.verbose)

    if args.verbose:
        print(f"\nTop-K at various block counts (k=1024):")
    topk_by_size = {}
    for nb in block_counts:
        r = bench_lightning_topk(nb, [1024], args.verbose)
        topk_by_size.update(r)
    all_results["lightning_topk_scaling"] = topk_by_size

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if args.verbose:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
