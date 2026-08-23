"""
SnapMLA Kernel Speed Benchmark

Measures kernel execution time using CUDA events for:
  - fused_q_quant (prep)
  - fused_k_append (prep)
  - dense_decode_v32 (attention)
  - sparse_decode_v32 (attention)

Usage:
    python benchmarks/benchmark_speed.py                    # default output
    python benchmarks/benchmark_speed.py -v                 # verbose
    python benchmarks/benchmark_speed.py -o results.json    # custom output

Prerequisites:
    pip install -e .   (build sm120_mla_kernels)
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

# Import kernel module
try:
    import sm120_mla_kernels
    HAS_KERNELS = torch.cuda.is_available()
except ImportError:
    HAS_KERNELS = False

# Constants (V3.2 model config)
D_C = 512
D_ROPE = 64
D_QK = D_C + D_ROPE  # 576
D_V = 512
H_Q = 64
PAGE_SIZE = 64
FP8_MAX = 448.0

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_speed_results.json")

WARMUP_ITERS = 10
TIMED_ITERS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alloc_paged_cache(num_pages, page_size, d_c, d_rope, device="cuda"):
    row_bytes = d_c + 4 + d_rope * 2
    return torch.zeros(num_pages * page_size * row_bytes, dtype=torch.uint8, device=device)


def _time_kernel(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    """Time a kernel function using CUDA events. Returns list of times in microseconds."""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us

    return times


def _stats(times):
    """Compute timing statistics from a list of microsecond measurements."""
    times_sorted = sorted(times)
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "p95_us": times_sorted[int(len(times_sorted) * 0.95)],
        "mean_us": statistics.mean(times),
        "std_us": statistics.stdev(times) if len(times) > 1 else 0.0,
        "n_iters": len(times),
    }


def _collect_env():
    env = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "has_kernels": HAS_KERNELS,
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "N/A"
        env["gpu_name"] = torch.cuda.get_device_name(0)
    return env


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_fused_q_quant(s_q=1, verbose=False):
    """Benchmark fused_q_quant kernel at various head counts."""
    results = []
    for h_q in [64]:
        q = torch.randn(s_q, h_q, D_QK, device="cuda", dtype=torch.bfloat16)

        def fn():
            sm120_mla_kernels.fused_q_quant(q, D_C)

        times = _time_kernel(fn)
        stats = _stats(times)
        stats["s_q"] = s_q
        stats["h_q"] = h_q
        results.append(stats)

        if verbose:
            print(f"  fused_q_quant  s_q={s_q:>5d} h_q={h_q:>3d}  "
                  f"median={stats['median_us']:>8.1f} us  min={stats['min_us']:>8.1f} us")

    return results


def bench_fused_k_append(verbose=False):
    """Benchmark fused_k_append kernel at various token counts."""
    results = []
    for n_tokens in [1, 64, 256]:
        c_kv = torch.randn(n_tokens, D_C, device="cuda", dtype=torch.bfloat16)
        k_rope = torch.randn(n_tokens, D_ROPE, device="cuda", dtype=torch.bfloat16)
        n_pages = (n_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
        slot_mapping = torch.arange(n_tokens, dtype=torch.int32, device="cuda")

        def fn():
            sm120_mla_kernels.fused_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                                              D_C, D_ROPE, PAGE_SIZE)

        times = _time_kernel(fn)
        stats = _stats(times)
        stats["n_tokens"] = n_tokens
        results.append(stats)

        if verbose:
            print(f"  fused_k_append n_tokens={n_tokens:>5d}  "
                  f"median={stats['median_us']:>8.1f} us  min={stats['min_us']:>8.1f} us")

    return results


def _build_decode_data(s_kv):
    """Build Q, KV cache, and metadata for decode benchmarks."""
    b, s_q = 1, 1
    q_bf16 = torch.randn(b, s_q, H_Q, D_QK, device="cuda", dtype=torch.bfloat16)
    c_kv = torch.randn(s_kv, D_C, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(s_kv, D_ROPE, device="cuda", dtype=torch.bfloat16)

    n_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device="cuda")
    sm120_mla_kernels.fused_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                                      D_C, D_ROPE, PAGE_SIZE)

    q_flat = q_bf16.view(-1, H_Q, D_QK)
    q_nope_fp8, q_rope_bf16, q_scales = sm120_mla_kernels.fused_q_quant(q_flat, D_C)
    q_nope_fp8 = q_nope_fp8.view(b, s_q, H_Q, D_C)
    q_rope_bf16 = q_rope_bf16.view(b, s_q, H_Q, D_ROPE)
    q_scales = q_scales.view(b, s_q, H_Q)

    sm_scale = 1.0 / math.sqrt(D_QK)

    block_table = torch.arange(n_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device="cuda")

    return {
        "q_nope_fp8": q_nope_fp8,
        "q_rope_bf16": q_rope_bf16,
        "q_scales": q_scales,
        "kv_cache": kv_cache,
        "block_table": block_table,
        "seqlens_k": seqlens_k,
        "sm_scale": sm_scale,
        "s_kv": s_kv,
        "n_pages": n_pages,
    }


def bench_dense_decode(verbose=False):
    """Benchmark dense_decode_v32 at various context lengths."""
    results = []
    for s_kv in [256, 1024, 4096, 16384, 32768]:
        data = _build_decode_data(s_kv)
        num_sm_parts = 1

        def fn():
            sm120_mla_kernels.dense_decode_v32(
                data["q_nope_fp8"], data["q_rope_bf16"], data["q_scales"],
                data["kv_cache"], data["block_table"], data["seqlens_k"],
                data["sm_scale"], PAGE_SIZE, num_sm_parts)

        times = _time_kernel(fn)
        stats = _stats(times)
        stats["s_kv"] = s_kv
        stats["num_sm_parts"] = num_sm_parts
        results.append(stats)

        if verbose:
            print(f"  dense_decode   s_kv={s_kv:>6d}  "
                  f"median={stats['median_us']:>8.1f} us  min={stats['min_us']:>8.1f} us")

    return results


def bench_sparse_decode(verbose=False):
    """Benchmark sparse_decode_v32 at various context lengths and topk values."""
    results = []

    configs = [
        # (s_kv, topk)
        (256, 256),
        (1024, 256),
        (1024, 1024),
        (4096, 1024),
        (4096, 2048),
        (16384, 2048),
        (32768, 2048),
    ]

    for s_kv, topk in configs:
        data = _build_decode_data(s_kv)
        b, s_q = 1, 1
        num_sm_parts = 1

        # Pad topk to multiple of 64
        topk_padded = ((topk + 63) // 64) * 64
        actual_topk = min(topk, s_kv)

        indices = torch.full((b, s_q, topk_padded), -1, dtype=torch.int32, device="cuda")
        # Select first `actual_topk` tokens (deterministic for benchmarking)
        indices[0, 0, :actual_topk] = torch.arange(actual_topk, dtype=torch.int32, device="cuda")

        def fn():
            sm120_mla_kernels.sparse_decode_v32(
                data["q_nope_fp8"], data["q_rope_bf16"], data["q_scales"],
                data["kv_cache"], indices, data["sm_scale"],
                PAGE_SIZE, topk_padded, num_sm_parts)

        times = _time_kernel(fn)
        stats = _stats(times)
        stats["s_kv"] = s_kv
        stats["topk"] = topk
        stats["topk_padded"] = topk_padded
        stats["num_sm_parts"] = num_sm_parts
        results.append(stats)

        if verbose:
            print(f"  sparse_decode  s_kv={s_kv:>6d} topk={topk:>5d}  "
                  f"median={stats['median_us']:>8.1f} us  min={stats['min_us']:>8.1f} us")

    return results


def bench_sparse_prefill(verbose=False):
    """Benchmark sparse_prefill_v32 at various s_q and s_kv."""
    results = []

    configs = [
        # (s_q, s_kv, topk)  — topk=s_kv means dense prefill via sparse kernel
        (1, 256, 256),
        (1, 1024, 1024),
        (1, 4096, 4096),
        (32, 1024, 1024),
        (128, 1024, 1024),
        (128, 4096, 4096),
    ]

    for s_q, s_kv, topk in configs:
        q = torch.randn(s_q, H_Q, D_QK, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(s_kv, 1, D_QK, device="cuda", dtype=torch.bfloat16)
        sm_scale = 1.0 / math.sqrt(D_QK)

        indices = torch.arange(topk, dtype=torch.int32, device="cuda")
        indices = indices.unsqueeze(0).unsqueeze(0).expand(s_q, 1, -1).contiguous()

        def fn():
            sm120_mla_kernels.sparse_prefill_v32(q, kv, indices, sm_scale, topk)

        times = _time_kernel(fn)
        stats = _stats(times)
        stats["s_q"] = s_q
        stats["s_kv"] = s_kv
        stats["topk"] = topk
        results.append(stats)

        if verbose:
            print(f"  sparse_prefill s_q={s_q:>4d} s_kv={s_kv:>6d} topk={topk:>5d}  "
                  f"median={stats['median_us']:>8.1f} us  min={stats['min_us']:>8.1f} us")

    return results


def bench_graph_dense_decode(verbose=False):
    """Benchmark CUDA graph dense decode replay (fused_q_quant + decode + combine).

    Captures the graph once per s_kv, then times update+replay only.
    This measures the production hot-path cost (no init/capture overhead).
    """
    if not hasattr(sm120_mla_kernels, 'DecodeGraphRunner'):
        return []

    results = []
    for s_kv in [256, 1024, 4096, 16384, 32768]:
        b, s_q = 1, 1
        q_bf16 = torch.randn(b, s_q, H_Q, D_QK, device="cuda", dtype=torch.bfloat16)
        c_kv = torch.randn(s_kv, D_C, device="cuda", dtype=torch.bfloat16)
        k_rope = torch.randn(s_kv, D_ROPE, device="cuda", dtype=torch.bfloat16)

        n_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
        kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
        slot_mapping = torch.arange(s_kv, dtype=torch.int32, device="cuda")
        sm120_mla_kernels.fused_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                                          D_C, D_ROPE, PAGE_SIZE)

        sm_scale = 1.0 / math.sqrt(D_QK)
        block_table = torch.arange(n_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
        seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device="cuda")
        num_sm_parts = 1

        # Capture once (persistent runner)
        runner = sm120_mla_kernels.DecodeGraphRunner()
        runner.init(kv_cache, b, s_q, H_Q, 1, D_QK, D_C, D_C,
                    PAGE_SIZE, block_table.size(1), sm_scale, num_sm_parts)
        runner.update_metadata(seqlens_k, num_sm_parts)

        # Time update + replay only (graph already captured)
        def fn():
            runner.update(q_bf16, seqlens_k, block_table)
            runner.replay()

        times = _time_kernel(fn)
        stats = _stats(times)
        stats["s_kv"] = s_kv
        stats["num_sm_parts"] = num_sm_parts
        results.append(stats)

        runner.destroy()

        if verbose:
            print(f"  graph_replay   s_kv={s_kv:>6d}  "
                  f"median={stats['median_us']:>8.1f} us  min={stats['min_us']:>8.1f} us")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SnapMLA Kernel Speed Benchmark")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if not HAS_KERNELS:
        print("ERROR: sm120_mla_kernels not available. Build with: pip install -e .")
        sys.exit(1)

    results = {"env": _collect_env(), "benchmarks": {}}

    if args.verbose:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Warmup: {WARMUP_ITERS}, Timed: {TIMED_ITERS}\n")

    # Prep kernels
    if args.verbose:
        print("=== Prep Kernels ===")
    results["benchmarks"]["fused_q_quant"] = bench_fused_q_quant(verbose=args.verbose)
    results["benchmarks"]["fused_k_append"] = bench_fused_k_append(verbose=args.verbose)

    # Decode kernels
    if args.verbose:
        print("\n=== Dense Decode ===")
    results["benchmarks"]["dense_decode"] = bench_dense_decode(verbose=args.verbose)

    if args.verbose:
        print("\n=== Sparse Decode ===")
    results["benchmarks"]["sparse_decode"] = bench_sparse_decode(verbose=args.verbose)

    # Prefill kernels
    if args.verbose:
        print("\n=== Sparse Prefill ===")
    results["benchmarks"]["sparse_prefill"] = bench_sparse_prefill(verbose=args.verbose)

    # CUDA graph decode
    if hasattr(sm120_mla_kernels, 'DecodeGraphRunner'):
        if args.verbose:
            print("\n=== Graph Dense Decode ===")
        results["benchmarks"]["graph_dense_decode"] = bench_graph_dense_decode(verbose=args.verbose)

    # Summary
    if args.verbose:
        print("\nDone.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
