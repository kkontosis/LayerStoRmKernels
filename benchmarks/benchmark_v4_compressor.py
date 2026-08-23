"""
V4K-11a: V4 Compressor Speed Benchmarks

Measures CSA and HCA compressor kernel latency at various batch sizes.
CSA: 8-token window, stride 1 (gated softmax pooling).
HCA: 128-token window, stride 128 (averaging).

Usage:
  python benchmarks/benchmark_v4_compressor.py -v
  python benchmarks/benchmark_v4_compressor.py -o results.json
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

try:
    import sm120_mla_kernels as k
    HAS_KERNELS = torch.cuda.is_available()
except ImportError:
    HAS_KERNELS = False

HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64

WARMUP_ITERS = 10
TIMED_ITERS = 100

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_compressor_results.json")


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


def _make_compressor_inputs(num_tokens, device="cuda"):
    return {
        "input_k_nope": torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "input_k_rope_raw": torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device),
        "input_v": torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device),
        "compress_cos": torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device=device),
        "compress_sin": torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device=device),
    }


def bench_csa_compressor(batch_sizes, verbose=False):
    results = {}
    window, stride = 8, 1
    for num_tokens in batch_sizes:
        inp = _make_compressor_inputs(num_tokens)
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device="cuda")
        pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device="cuda")

        fn = lambda: k.v4_csa_compress(
            inp["input_k_nope"], inp["input_k_rope_raw"], inp["input_v"],
            gate, pos_bias, inp["compress_cos"], inp["compress_sin"],
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)

        times = _time_kernel(fn)
        st = _stats(times)
        num_compressed = max(0, num_tokens - window) // stride
        st["num_tokens"] = num_tokens
        st["num_compressed"] = num_compressed
        st["us_per_token"] = st["median_us"] / max(num_compressed, 1)
        results[f"csa_{num_tokens}"] = st

        if verbose:
            print(f"  CSA compress ({num_tokens:>4d} tokens → {num_compressed:>4d} entries): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us  "
                  f"per_token={st['us_per_token']:.2f} us")
    return results


def bench_hca_compressor(batch_sizes, verbose=False):
    results = {}
    window, stride = 128, 128
    for num_tokens in batch_sizes:
        inp = _make_compressor_inputs(num_tokens)
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device="cuda")

        fn = lambda: k.v4_hca_compress(
            inp["input_k_nope"], inp["input_k_rope_raw"], inp["input_v"],
            gate, inp["compress_cos"], inp["compress_sin"],
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)

        times = _time_kernel(fn)
        st = _stats(times)
        num_compressed = num_tokens // stride
        st["num_tokens"] = num_tokens
        st["num_compressed"] = num_compressed
        st["us_per_token"] = st["median_us"] / max(num_compressed, 1)
        results[f"hca_{num_tokens}"] = st

        if verbose:
            print(f"  HCA compress ({num_tokens:>4d} tokens → {num_compressed:>4d} entries): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us  "
                  f"per_token={st['us_per_token']:.2f} us")
    return results


def bench_fused_csa_compress_insert(batch_sizes, verbose=False):
    results = {}
    window, stride = 8, 1
    entry_bytes = 1160
    page_size = 64
    for num_tokens in batch_sizes:
        num_compressed = max(0, num_tokens - window) // stride
        num_pages = (num_compressed + page_size - 1) // page_size
        cache = torch.zeros(num_pages * page_size * entry_bytes, dtype=torch.uint8, device="cuda")
        slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device="cuda")

        inp = _make_compressor_inputs(num_tokens)
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device="cuda")
        pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device="cuda")

        fn = lambda: k.v4_fused_csa_compress_insert(
            inp["input_k_nope"], inp["input_k_rope_raw"], inp["input_v"],
            gate, pos_bias, inp["compress_cos"], inp["compress_sin"],
            cache, slot_mapping, HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)

        times = _time_kernel(fn)
        st = _stats(times)
        st["num_tokens"] = num_tokens
        st["num_compressed"] = num_compressed
        st["us_per_token"] = st["median_us"] / max(num_compressed, 1)
        results[f"fused_csa_{num_tokens}"] = st

        if verbose:
            print(f"  Fused CSA ({num_tokens:>4d} tokens → {num_compressed:>4d} entries): "
                  f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us  "
                  f"per_token={st['us_per_token']:.2f} us")
    return results


def main():
    parser = argparse.ArgumentParser(description="V4 Compressor Speed Benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not HAS_KERNELS:
        print("ERROR: sm120_mla_kernels not available")
        sys.exit(1)

    csa_sizes = [8+1, 16, 32, 64, 128, 256]
    hca_sizes = [128, 256, 512, 1024]

    all_results = {"env": _collect_env()}

    if args.verbose:
        print("=" * 60)
        print("V4 Compressor Speed Benchmarks")
        print("=" * 60)
        print(f"\nCSA Compressor (window=8, stride=1):")
    all_results["csa_compress"] = bench_csa_compressor(csa_sizes, args.verbose)

    if args.verbose:
        print(f"\nHCA Compressor (window=128, stride=128):")
    all_results["hca_compress"] = bench_hca_compressor(hca_sizes, args.verbose)

    if args.verbose:
        print(f"\nFused CSA Compress+Insert:")
    all_results["fused_csa_compress_insert"] = bench_fused_csa_compress_insert(csa_sizes, args.verbose)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if args.verbose:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
