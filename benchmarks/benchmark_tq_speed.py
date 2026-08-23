"""
TurboQuant Speed Benchmark — TQ decode pipeline vs SnapMLA FP8.

Usage:
  python benchmarks/benchmark_tq_speed.py
"""

import torch
import math
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sm120_mla_kernels
from tests.test_tq_reference import (
    load_codebook, generate_rotation_matrix, compute_tq_row_bytes,
    D_C, D_ROPE, D_QK, D_V, H_Q, H_KV, TQ_BITS,
)

WARMUP = 10
TIMED = 100


def _time_kernel(fn, warmup=WARMUP, timed=TIMED):
    """Time a CUDA kernel with proper synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(timed):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / timed
    return ms


def _stats(times_ms):
    t = sorted(times_ms)
    return {
        "median_us": t[len(t)//2] * 1000,
        "min_us": t[0] * 1000,
        "p95_us": t[int(len(t)*0.95)] * 1000,
        "mean_us": sum(t) / len(t) * 1000,
    }


def bench_tq_pipeline(s_kv, page_size=64):
    """Benchmark full TQ pipeline: q_rotate + dense_decode + v_rotate_back."""
    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    # Populate cache
    c_kv = torch.randn(s_kv, D_C, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.bfloat16, device='cuda')
    kv_cache = torch.zeros(num_pages * page_size * row_bytes, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(c_kv, k_rope, kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')

    # Q inputs
    q_nope = torch.randn(1, 1, H_Q, D_C, dtype=torch.bfloat16, device='cuda')
    q_rope_bf16 = torch.randn(1, 1, H_Q, D_ROPE, dtype=torch.bfloat16, device='cuda')

    # Benchmark individual kernels
    def fn_q_rotate():
        return sm120_mla_kernels.tq_q_rotate(q_nope, Pi_gpu)

    # Get num_sm_parts for split-KV
    dev_props = torch.cuda.get_device_properties(0)
    num_sms = dev_props.multi_processor_count

    def fn_decode():
        q_rot = sm120_mla_kernels.tq_q_rotate(q_nope, Pi_gpu)
        return sm120_mla_kernels.tq_dense_decode_v32(
            q_rot, q_rope_bf16, kv_cache, block_table, seqlens_k,
            centroids_gpu, sm_scale, page_size, num_sms)

    def fn_full_pipeline():
        q_rot = sm120_mla_kernels.tq_q_rotate(q_nope, Pi_gpu)
        out_rot, lse = sm120_mla_kernels.tq_dense_decode_v32(
            q_rot, q_rope_bf16, kv_cache, block_table, seqlens_k,
            centroids_gpu, sm_scale, page_size, num_sms)
        return sm120_mla_kernels.tq_v_rotate_back(out_rot, Pi_gpu)

    t_rotate = _time_kernel(fn_q_rotate)
    t_decode = _time_kernel(fn_decode)
    t_full = _time_kernel(fn_full_pipeline)

    return {
        "s_kv": s_kv,
        "q_rotate_us": t_rotate * 1000,
        "decode_us": t_decode * 1000,
        "full_pipeline_us": t_full * 1000,
        "bytes_per_token": row_bytes,
        "total_cache_MB": (s_kv * row_bytes) / (1024 * 1024),
    }


def bench_snapmla_decode(s_kv, page_size=64):
    """Benchmark SnapMLA FP8 dense decode for comparison."""
    sm_scale = 1.0 / math.sqrt(D_QK)
    row_bytes = D_C + 4 + D_ROPE * 2  # 644 for V3.2

    torch.manual_seed(42)
    num_pages = (s_kv + page_size - 1) // page_size

    q = torch.randn(1, 1, H_Q, D_QK, dtype=torch.bfloat16, device='cuda')
    c_kv = torch.randn(s_kv, D_C, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.bfloat16, device='cuda')

    # FP8 cache
    kv_cache = torch.zeros(num_pages * page_size * row_bytes, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.fused_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                                      D_C, D_ROPE, page_size)

    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')

    q_nope_fp8, q_rope_bf16, q_scales = sm120_mla_kernels.fused_q_quant(
        q.squeeze(0), D_C)

    def fn():
        return sm120_mla_kernels.dense_decode_v32(
            q_nope_fp8.unsqueeze(0), q_rope_bf16.unsqueeze(0), q_scales.unsqueeze(0),
            kv_cache, block_table, seqlens_k, sm_scale, page_size, 1)

    t = _time_kernel(fn)
    return {
        "s_kv": s_kv,
        "decode_us": t * 1000,
        "bytes_per_token": row_bytes,
        "total_cache_MB": (s_kv * row_bytes) / (1024 * 1024),
    }


def main():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"TurboQuant Speed Benchmark — {gpu_name}")
    print("=" * 65)

    context_lengths = [256, 1024, 4096, 16384]

    print(f"\n{'s_kv':>7} | {'TQ full (us)':>12} | {'TQ decode (us)':>14} | {'SnapMLA (us)':>12} | {'Ratio':>6}")
    print("-" * 65)

    results = []
    for s_kv in context_lengths:
        tq = bench_tq_pipeline(s_kv)
        snap = bench_snapmla_decode(s_kv)
        ratio = tq["full_pipeline_us"] / snap["decode_us"] if snap["decode_us"] > 0 else float('inf')

        print(f"{s_kv:>7} | {tq['full_pipeline_us']:>12.1f} | {tq['decode_us']:>14.1f} | {snap['decode_us']:>12.1f} | {ratio:>6.2f}x")

        results.append({
            "s_kv": s_kv,
            "tq_full_us": tq["full_pipeline_us"],
            "tq_decode_us": tq["decode_us"],
            "tq_q_rotate_us": tq["q_rotate_us"],
            "snap_decode_us": snap["decode_us"],
            "ratio": ratio,
            "tq_bytes_per_token": tq["bytes_per_token"],
            "snap_bytes_per_token": snap["bytes_per_token"],
        })

    print(f"\nTQ cache: {results[0]['tq_bytes_per_token']}B/token, SnapMLA: {results[0]['snap_bytes_per_token']}B/token")
    print(f"Compression: {results[0]['snap_bytes_per_token'] / results[0]['tq_bytes_per_token']:.2f}x")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "tq_speed_results.json")
    with open(out_path, "w") as f:
        json.dump({"gpu": gpu_name, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
