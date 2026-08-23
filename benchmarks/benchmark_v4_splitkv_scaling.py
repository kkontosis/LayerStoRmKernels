"""
V4K-14f: Split-KV Scaling Optimization Benchmark

Profiles CSA FP8 decode at various num_sm_parts to find optimal split count.
Also tests at multiple batch sizes to understand scaling behavior.

Usage:
  python benchmarks/benchmark_v4_splitkv_scaling.py
  python benchmarks/benchmark_v4_splitkv_scaling.py -o results.json
"""

import sys
import os
import math
import json
import argparse
import datetime
import statistics
import platform

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE, V4_FP8_BYTES_PER_ENTRY,
)

import sm120_mla_kernels as K

WARMUP = 10
ITERS = 100
H_Q = 64
TOPK = 1024
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
S_KV = 4096

SM_PARTS_LIST = [1, 2, 4, 8, 16, 32, 64]
BATCH_LIST = [1, 4, 8, 32]

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_splitkv_results.json")


def _time_kernel(fn, warmup=WARMUP, iters=ITERS):
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
        "median_us": round(statistics.median(times), 1),
        "min_us": round(min(times), 1),
        "p95_us": round(ts[int(len(ts) * 0.95)], 1),
        "mean_us": round(statistics.mean(times), 1),
    }


def _alloc_fp8_cache(num_entries):
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                        dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_entries, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot = torch.arange(num_entries, dtype=torch.int32, device='cuda')
    K.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)
    return cache


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def bench_splitkv(verbose=True):
    torch.manual_seed(42)
    results = {}

    cache = _alloc_fp8_cache(S_KV)
    swa_cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')

    # Reference output at num_sm_parts=1 for accuracy check
    ref_outputs = {}

    for batch in BATCH_LIST:
        batch_results = {}
        q_nope = torch.randn(batch, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(batch, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.randint(0, S_KV, (batch, 1, TOPK), dtype=torch.int32, device='cuda')
        swa_bt = torch.zeros(batch, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(batch, dtype=torch.int32, device='cuda')

        for nsp in SM_PARTS_LIST:
            try:
                fn = lambda: K.v4_csa_fp8_decode(
                    q_nope, q_rope, cache, indices,
                    swa_cache, swa_bt, swa_sl,
                    SM_SCALE, TOPK, PAGE_SIZE, PAGE_SIZE, nsp)

                # Accuracy check
                out, lse = K.v4_csa_fp8_decode(
                    q_nope, q_rope, cache, indices,
                    swa_cache, swa_bt, swa_sl,
                    SM_SCALE, TOPK, PAGE_SIZE, PAGE_SIZE, nsp)
                torch.cuda.synchronize()

                if batch == 1 and nsp == 1:
                    ref_outputs[batch] = out.clone()

                cos = 1.0
                if batch in ref_outputs:
                    cos = cosine_sim(out, ref_outputs[batch])

                times = _time_kernel(fn)
                s = _stats(times)
                s["cosine_vs_nsp1"] = round(cos, 6)
                batch_results[f"nsp={nsp}"] = s

                if verbose:
                    speedup = ""
                    if f"nsp=1" in batch_results:
                        base = batch_results["nsp=1"]["median_us"]
                        speedup = f"  ({base/s['median_us']:.2f}x vs nsp=1)"
                    print(f"  batch={batch} nsp={nsp:>3d}: median={s['median_us']:>8.1f}us  "
                          f"cos={cos:.6f}{speedup}")

            except Exception as e:
                if verbose:
                    print(f"  batch={batch} nsp={nsp}: ERROR: {e}")
                batch_results[f"nsp={nsp}"] = {"error": str(e)}

        results[f"batch={batch}"] = batch_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: topk={TOPK}, h_q={H_Q}, s_kv={S_KV}, page_size={PAGE_SIZE}")
    print(f"Testing num_sm_parts: {SM_PARTS_LIST}")
    print(f"Testing batch sizes: {BATCH_LIST}")
    print()

    results = bench_splitkv(verbose=args.verbose)

    report = {
        "benchmark": "v4_splitkv_scaling",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "config": {
            "topk": TOPK, "h_q": H_Q, "s_kv": S_KV,
            "page_size": PAGE_SIZE, "warmup": WARMUP, "iters": ITERS,
        },
        "results": results,
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
