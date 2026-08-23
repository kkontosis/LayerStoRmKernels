"""
V4K-11e: V4 End-to-End Per-Token Latency

Full decode pipeline per layer type:
  CSA: compress → FP8 k_append → lightning_score → lightning_topk → CSA decode → combine
  HCA: compress → FP8 k_append → HCA decode
  TQ:  compress → TQ k_append → CSA TQ decode → v_rotate_back

Reports critical-path latency (sequential, no multi-stream overlap).

Usage:
  python benchmarks/benchmark_v4_e2e.py -v
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
INDEX_N_HEADS = 4
INDEX_HEAD_DIM = 128
NUM_SM_PARTS = 1

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "benchmark_v4_e2e_results.json")


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


def bench_csa_fp8_e2e(num_compressed, topk, verbose=False):
    """Full CSA FP8 pipeline: compress → k_append → score → topk → decode."""
    b, s_q = 1, 1
    window, stride = 8, 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    # Pre-allocate inputs
    num_tokens = window + num_compressed * stride
    inp_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    inp_k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    inp_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
    pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
    sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

    # Pre-allocate FP8 cache
    num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
    fp8_cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device='cuda')

    # Indexer cache
    num_blocks = (num_compressed + 7) // 8
    indexer_k = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM,
                            dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    k_scales = torch.ones(num_blocks, dtype=torch.float32, device='cuda')
    q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    score_proj = torch.randn(INDEX_N_HEADS, dtype=torch.float32, device='cuda')
    block_endpoints = torch.arange(1, num_blocks + 1, dtype=torch.int32, device='cuda') * 8

    # Q tensors
    q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    # SWA (empty)
    swa_cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
    swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')

    def pipeline():
        # 1. Compress
        k_nope, k_rope, v_nope = k.v4_csa_compress(
            inp_k_nope, inp_k_rope, inp_v, gate, pos_bias, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        # 2. K append
        k.v4_fp8_k_append(k_nope, k_rope, v_nope, fp8_cache, slot_mapping)
        # 3. Lightning score
        scores = k.v4_lightning_score(q_proj, indexer_k, k_scales, score_proj)
        # 4. Lightning topk
        topk_result = k.v4_lightning_topk(scores, block_endpoints, num_compressed - 1, topk)
        indices = topk_result[0].unsqueeze(0).unsqueeze(0)
        # 5. CSA decode
        k.v4_csa_fp8_decode(
            q_nope, q_rope, fp8_cache, indices,
            swa_cache, swa_bt, swa_sl,
            sm_scale, topk, PAGE_SIZE, PAGE_SIZE, NUM_SM_PARTS)

    times = _time_kernel(pipeline)
    st = _stats(times)
    st["num_compressed"] = num_compressed
    st["topk"] = topk
    if verbose:
        print(f"  CSA FP8 E2E ({num_compressed:>6d} entries, topk={topk:>4d}): "
              f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
    return st


def bench_csa_tq_e2e(num_compressed, topk, Pi, centroids, boundaries, verbose=False):
    """Full CSA TQ pipeline: compress → TQ k_append → q_rotate → TQ decode → v_rotate_back."""
    b, s_q = 1, 1
    window, stride = 8, 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    num_tokens = window + num_compressed * stride
    inp_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    inp_k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    inp_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
    pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
    sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

    num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
    tq_cache = torch.zeros(num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device='cuda')

    q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    indices = torch.randint(0, num_compressed, (b, s_q, topk), dtype=torch.int32, device='cuda')

    def pipeline():
        # 1. Compress
        k_nope, k_rope, v_nope = k.v4_csa_compress(
            inp_k_nope, inp_k_rope, inp_v, gate, pos_bias, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        # 2. TQ k_append
        k.v4_tq_k_append(k_nope, k_rope, v_nope, tq_cache, slot_mapping,
                          Pi, centroids, boundaries)
        # 3. Q rotate
        q_rot = (q_nope.float() @ Pi.T).contiguous()
        # 4. CSA TQ decode
        out_rot, lse = k.v4_csa_tq_decode(q_rot, q_rope, tq_cache, indices, centroids, sm_scale)
        # 5. V rotate back
        out_final = (out_rot.float() @ Pi).to(torch.bfloat16)

    times = _time_kernel(pipeline)
    st = _stats(times)
    st["num_compressed"] = num_compressed
    st["topk"] = topk
    if verbose:
        print(f"  CSA TQ  E2E ({num_compressed:>6d} entries, topk={topk:>4d}): "
              f"median={st['median_us']:.1f} us  p95={st['p95_us']:.1f} us")
    return st


def main():
    parser = argparse.ArgumentParser(description="V4 End-to-End Per-Token Latency")
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

    all_results = {"env": _collect_env()}

    configs = [(1024, 512), (4096, 1024), (16384, 1024)]

    if args.verbose:
        print("=" * 60)
        print("V4 End-to-End Per-Token Latency")
        print("=" * 60)
        print(f"\nCSA FP8 E2E (compress → k_append → score → topk → decode):")
    fp8_e2e = {}
    for nc, tk in configs:
        fp8_e2e[f"fp8_{nc}_{tk}"] = bench_csa_fp8_e2e(nc, tk, args.verbose)
    all_results["csa_fp8_e2e"] = fp8_e2e

    if args.verbose:
        print(f"\nCSA TQ E2E (compress → TQ k_append → q_rotate → TQ decode → v_rotate_back):")
    tq_e2e = {}
    for nc, tk in configs:
        tq_e2e[f"tq_{nc}_{tk}"] = bench_csa_tq_e2e(nc, tk, Pi, centroids, boundaries, args.verbose)
    all_results["csa_tq_e2e"] = tq_e2e

    if args.verbose:
        print(f"\n{'='*60}")
        print("E2E Comparison: CSA FP8 vs CSA TQ")
        print(f"{'entries':>8s}  {'topk':>5s}  {'FP8 (us)':>10s}  {'TQ (us)':>10s}  {'speedup':>8s}")
        for nc, tk in configs:
            fp8 = fp8_e2e[f"fp8_{nc}_{tk}"]["median_us"]
            tq = tq_e2e[f"tq_{nc}_{tk}"]["median_us"]
            print(f"{nc:>8d}  {tk:>5d}  {fp8:>10.1f}  {tq:>10.1f}  {fp8/tq:>7.2f}x")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    if args.verbose:
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
