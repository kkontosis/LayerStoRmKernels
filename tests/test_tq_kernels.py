"""
TurboQuant CUDA Kernel Tests — validate GPU kernels against Python references.

Usage:
  python tests/test_tq_kernels.py -v
"""

import torch
import torch.nn.functional as F
import math
import json
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Must import torch before the extension
import sm120_mla_kernels

# Import reference functions
from tests.test_tq_reference import (
    load_codebook, generate_rotation_matrix,
    ref_tq_mse_k_append, ref_tq_mse_dequant_indexed,
    ref_tq_mse_q_rotate, ref_tq_mse_v_rotate_back,
    ref_tq_mse_sparse_decode,
    ref_tq_mse_full_pipeline, ref_mla_attention_bf16,
    allocate_tq_cache, compute_tq_row_bytes,
    D_C, D_ROPE, D_QK, D_V, H_Q, H_KV, TQ_BITS, TQ_SEED,
)


def compute_metrics(ref, test):
    ref_f = ref.float().flatten()
    test_f = test.float().flatten()
    cosine = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()
    mse = ((ref_f - test_f) ** 2).mean().item()
    nrmse = math.sqrt(mse) / (ref_f.norm().item() / math.sqrt(ref_f.numel()) + 1e-12)
    return {"cosine": cosine, "mse": mse, "nrmse": nrmse}


def fmt(m):
    return f"cos={m['cosine']:.6f} nrmse={m['nrmse']:.4f}"


# ===========================================================================
# Tests
# ===========================================================================

def test_tq_q_rotate(verbose=False):
    """GPU tq_q_rotate vs Python reference."""
    print("\n=== test_tq_q_rotate ===")

    Pi = generate_rotation_matrix(D_C)
    Pi_gpu = Pi.cuda()

    torch.manual_seed(123)
    q_nope = torch.randn(1, H_Q, D_C, dtype=torch.bfloat16, device='cuda')

    # GPU kernel
    q_rot_gpu = sm120_mla_kernels.tq_q_rotate(q_nope, Pi_gpu)

    # Reference
    q_rot_ref = ref_tq_mse_q_rotate(q_nope.float().cpu(), Pi)

    m = compute_metrics(q_rot_ref, q_rot_gpu.cpu())
    print(f"  GPU vs ref: {fmt(m)}")
    assert m["cosine"] > 0.999, f"q_rotate cosine {m['cosine']:.6f} < 0.999"
    print("  PASS")


def test_tq_v_rotate_back(verbose=False):
    """GPU tq_v_rotate_back vs Python reference."""
    print("\n=== test_tq_v_rotate_back ===")

    Pi = generate_rotation_matrix(D_C)
    Pi_gpu = Pi.cuda()

    torch.manual_seed(99)
    x = torch.randn(1, H_Q, D_C, dtype=torch.float32, device='cuda')

    # GPU kernel
    out_gpu = sm120_mla_kernels.tq_v_rotate_back(x, Pi_gpu)

    # Reference
    out_ref = ref_tq_mse_v_rotate_back(x.cpu(), Pi)

    m = compute_metrics(out_ref, out_gpu.float().cpu())
    print(f"  GPU vs ref: {fmt(m)}")
    assert m["cosine"] > 0.999, f"v_rotate_back cosine {m['cosine']:.6f} < 0.999"

    # Also test round-trip: rotate forward then back = identity
    q_nope_bf16 = torch.randn(1, H_Q, D_C, dtype=torch.bfloat16, device='cuda')
    q_rot = sm120_mla_kernels.tq_q_rotate(q_nope_bf16, Pi_gpu)  # forward
    q_back = sm120_mla_kernels.tq_v_rotate_back(q_rot, Pi_gpu)  # back

    m2 = compute_metrics(q_nope_bf16.float().cpu(), q_back.float().cpu())
    print(f"  Round-trip (forward+back): {fmt(m2)}")
    assert m2["cosine"] > 0.999, f"Round-trip cosine {m2['cosine']:.6f} < 0.999"
    print("  PASS")


def test_tq_k_append_dequant_roundtrip(verbose=False):
    """GPU tq_fused_k_append → tq_dequant_ckv_indexed round-trip."""
    print("\n=== test_tq_k_append_dequant_roundtrip ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    n_tokens = 128
    c_kv = torch.randn(n_tokens, D_C, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(n_tokens, D_ROPE, dtype=torch.bfloat16, device='cuda')

    page_size = 64
    num_pages = (n_tokens + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)
    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(n_tokens, dtype=torch.int32, device='cuda')

    # K append
    sm120_mla_kernels.tq_fused_k_append(
        c_kv, k_rope, kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu,
        D_C, D_ROPE, page_size)

    # Dequant indexed
    indices = torch.arange(n_tokens, dtype=torch.int32, device='cuda')
    k_out = sm120_mla_kernels.tq_dequant_ckv_indexed(
        kv_cache, indices, Pi_gpu, centroids_gpu,
        D_C, D_ROPE, page_size)

    # NOPE round-trip
    nope_out = k_out[:, :D_C].float().cpu()
    nope_ref = c_kv.float().cpu()

    per_vec_cos = F.cosine_similarity(nope_ref, nope_out, dim=-1)
    mean_cos = per_vec_cos.mean().item()
    min_cos = per_vec_cos.min().item()
    print(f"  NOPE per-vec cosine: mean={mean_cos:.4f} min={min_cos:.4f}")
    assert mean_cos > 0.75, f"NOPE mean cosine {mean_cos:.4f} < 0.75"

    # ROPE round-trip
    rope_out = k_out[:, D_C:].float().cpu()
    rope_ref = k_rope.float().cpu()
    rope_cos = F.cosine_similarity(rope_ref.flatten().unsqueeze(0),
                                    rope_out.flatten().unsqueeze(0)).item()
    print(f"  ROPE cosine: {rope_cos:.6f}")
    assert rope_cos > 0.999, f"ROPE cosine {rope_cos:.6f} < 0.999"

    print("  PASS")


def test_tq_k_append_vs_python_reference(verbose=False):
    """GPU tq_fused_k_append cache bytes vs Python reference cache bytes."""
    print("\n=== test_tq_k_append_vs_python_reference ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(77)
    n_tokens = 16
    c_kv = torch.randn(n_tokens, D_C, dtype=torch.bfloat16)
    k_rope = torch.randn(n_tokens, D_ROPE, dtype=torch.bfloat16)

    page_size = 64
    num_pages = (n_tokens + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    # GPU path
    kv_cache_gpu = torch.zeros(num_pages * page_size * row_bytes,
                                dtype=torch.uint8, device='cuda')
    slot_mapping_gpu = torch.arange(n_tokens, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv.cuda(), k_rope.cuda(), kv_cache_gpu, slot_mapping_gpu,
        Pi_gpu, centroids_gpu, boundaries_gpu,
        D_C, D_ROPE, page_size)

    # Python path
    kv_cache_cpu = allocate_tq_cache(num_pages, page_size, D_C, D_ROPE)
    slot_mapping_cpu = torch.arange(n_tokens, dtype=torch.int64)
    ref_tq_mse_k_append(c_kv.float(), k_rope, kv_cache_cpu, slot_mapping_cpu,
                          Pi, centroids, boundaries, D_C, D_ROPE, page_size)

    # Compare raw cache bytes
    gpu_bytes = kv_cache_gpu.cpu()
    cpu_bytes = kv_cache_cpu

    # Check packed nope bytes match for each token
    packed_nope_bytes = D_C // 2  # 256
    matches = 0
    total = 0
    for t in range(n_tokens):
        offset = t * row_bytes
        gpu_packed = gpu_bytes[offset:offset + packed_nope_bytes]
        cpu_packed = cpu_bytes[offset:offset + packed_nope_bytes]
        match = (gpu_packed == cpu_packed).sum().item()
        matches += match
        total += packed_nope_bytes

    match_rate = matches / total
    print(f"  Packed byte match rate: {match_rate:.4f} ({matches}/{total})")
    assert match_rate > 0.95, f"Packed byte match rate {match_rate:.4f} < 0.95"

    print("  PASS")


# ===========================================================================
# Main
# ===========================================================================

def test_tq_dense_decode_short(verbose=False):
    """GPU tq_dense_decode at s_kv=256 vs Python reference full pipeline."""
    print("\n=== test_tq_dense_decode_short (s_kv=256) ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    s_kv = 256
    q = torch.randn(1, H_Q, D_QK)
    c_kv = torch.randn(s_kv, D_C)
    k_rope = torch.randn(s_kv, D_ROPE)

    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    # Populate paged cache
    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv.bfloat16().cuda(), k_rope.bfloat16().cuda(), kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

    # Build block_table (identity: page i = block i)
    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')

    # q_rotate
    q_nope = q[..., :D_C].bfloat16().cuda()
    q_rope_q = q[..., D_C:].bfloat16().cuda()
    q_rot = sm120_mla_kernels.tq_q_rotate(q_nope.unsqueeze(0), Pi_gpu)  # [1, 1, h_q, d_c]

    # Dense decode
    out_rot, lse = sm120_mla_kernels.tq_dense_decode_v32(
        q_rot, q_rope_q.unsqueeze(0), kv_cache, block_table, seqlens_k,
        centroids_gpu, sm_scale, page_size, 1)

    # v_rotate_back
    out_gpu = sm120_mla_kernels.tq_v_rotate_back(out_rot, Pi_gpu)

    # Python reference full pipeline
    out_ref, lse_ref = ref_tq_mse_full_pipeline(
        q, c_kv, k_rope, sm_scale, Pi, centroids, boundaries)

    m = compute_metrics(out_ref, out_gpu.float().cpu())
    print(f"  GPU pipeline vs Python ref: {fmt(m)}")
    assert m["cosine"] > 0.95, f"cosine {m['cosine']:.4f} < 0.95"

    # Also compare vs BF16 ground truth
    out_bf16, lse_bf16 = ref_mla_attention_bf16(q, c_kv, k_rope, sm_scale)
    m_bf16 = compute_metrics(out_bf16, out_gpu.float().cpu())
    print(f"  GPU pipeline vs BF16 ground truth: {fmt(m_bf16)}")

    print("  PASS")


def test_tq_dense_decode_scaling(verbose=False):
    """GPU tq_dense_decode at s_kv=256/1024/4096."""
    print("\n=== test_tq_dense_decode_scaling ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()
    page_size = 64

    for s_kv in [256, 1024, 4096]:
        torch.manual_seed(42)
        q = torch.randn(1, H_Q, D_QK)
        c_kv = torch.randn(s_kv, D_C)
        k_rope = torch.randn(s_kv, D_ROPE)

        num_pages = (s_kv + page_size - 1) // page_size
        row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

        kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                                dtype=torch.uint8, device='cuda')
        slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
        sm120_mla_kernels.tq_fused_k_append(
            c_kv.bfloat16().cuda(), k_rope.bfloat16().cuda(), kv_cache, slot_mapping,
            Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

        block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
        seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')

        q_nope = q[..., :D_C].bfloat16().cuda()
        q_rope_q = q[..., D_C:].bfloat16().cuda()
        q_rot = sm120_mla_kernels.tq_q_rotate(q_nope.unsqueeze(0), Pi_gpu)

        out_rot, lse = sm120_mla_kernels.tq_dense_decode_v32(
            q_rot, q_rope_q.unsqueeze(0), kv_cache, block_table, seqlens_k,
            centroids_gpu, sm_scale, page_size, 1)

        out_gpu = sm120_mla_kernels.tq_v_rotate_back(out_rot, Pi_gpu)

        # Compare vs BF16 ground truth
        out_bf16, _ = ref_mla_attention_bf16(q, c_kv, k_rope, sm_scale)
        m = compute_metrics(out_bf16, out_gpu.float().cpu())
        print(f"  s_kv={s_kv:5d}: {fmt(m)}")

        if s_kv >= 1024:
            assert m["cosine"] > 0.90, f"s_kv={s_kv} cosine {m['cosine']:.4f} < 0.90"

    print("  PASS")


def test_tq_sparse_decode_equals_dense(verbose=False):
    """GPU sparse(topk=all) == dense — same result when attending to all tokens."""
    print("\n=== test_tq_sparse_decode_equals_dense ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    s_kv = 256
    q = torch.randn(1, H_Q, D_QK)
    c_kv = torch.randn(s_kv, D_C)
    k_rope = torch.randn(s_kv, D_ROPE)

    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    # Populate paged cache
    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv.bfloat16().cuda(), k_rope.bfloat16().cuda(), kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

    # Q inputs
    q_nope = q[..., :D_C].bfloat16().cuda()
    q_rope_q = q[..., D_C:].bfloat16().cuda()
    q_rot = sm120_mla_kernels.tq_q_rotate(q_nope.unsqueeze(0), Pi_gpu)

    # Dense decode
    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')
    out_dense, lse_dense = sm120_mla_kernels.tq_dense_decode_v32(
        q_rot, q_rope_q.unsqueeze(0), kv_cache, block_table, seqlens_k,
        centroids_gpu, sm_scale, page_size, 1)

    # Sparse decode with topk=all (indices = 0..s_kv-1)
    indices = torch.arange(s_kv, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
    # Shape: [1, 1, s_kv]
    out_sparse, lse_sparse = sm120_mla_kernels.tq_sparse_decode_v32(
        q_rot, q_rope_q.unsqueeze(0), kv_cache, indices,
        centroids_gpu, sm_scale, page_size)

    # Compare
    m = compute_metrics(out_dense.cpu(), out_sparse.cpu())
    print(f"  Sparse(topk=all) vs Dense: {fmt(m)}")
    assert m["cosine"] > 0.999, f"sparse==dense cosine {m['cosine']:.6f} < 0.999"

    # LSE comparison
    lse_diff = (lse_dense - lse_sparse).abs().max().item()
    print(f"  LSE max diff: {lse_diff:.4e}")
    assert lse_diff < 0.01, f"LSE diff {lse_diff} > 0.01"

    print("  PASS")


def test_tq_sparse_decode_topk(verbose=False):
    """GPU sparse(topk=2048) at s_kv=4096 — quality check."""
    print("\n=== test_tq_sparse_decode_topk (s_kv=4096, topk=2048) ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    s_kv = 4096
    topk = 2048
    q = torch.randn(1, H_Q, D_QK)
    c_kv = torch.randn(s_kv, D_C)
    k_rope = torch.randn(s_kv, D_ROPE)

    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    # Populate paged cache
    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv.bfloat16().cuda(), k_rope.bfloat16().cuda(), kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

    # Q inputs
    q_nope = q[..., :D_C].bfloat16().cuda()
    q_rope_q = q[..., D_C:].bfloat16().cuda()
    q_rot = sm120_mla_kernels.tq_q_rotate(q_nope.unsqueeze(0), Pi_gpu)

    # Dense decode (all tokens) — use as ground truth for this test
    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')
    out_dense, lse_dense = sm120_mla_kernels.tq_dense_decode_v32(
        q_rot, q_rope_q.unsqueeze(0), kv_cache, block_table, seqlens_k,
        centroids_gpu, sm_scale, page_size, 1)
    out_dense_final = sm120_mla_kernels.tq_v_rotate_back(out_dense, Pi_gpu)

    # Select topk random indices (sorted for cache locality)
    perm = torch.randperm(s_kv)[:topk].sort().values
    indices = perm.int().cuda().unsqueeze(0).unsqueeze(0)  # [1, 1, topk]

    # Sparse decode
    out_sparse, lse_sparse = sm120_mla_kernels.tq_sparse_decode_v32(
        q_rot, q_rope_q.unsqueeze(0), kv_cache, indices,
        centroids_gpu, sm_scale, page_size)
    out_sparse_final = sm120_mla_kernels.tq_v_rotate_back(out_sparse, Pi_gpu)

    m = compute_metrics(out_dense_final.cpu(), out_sparse_final.cpu())
    print(f"  Sparse(topk={topk}) vs Dense(all={s_kv}): {fmt(m)}")
    # With random indices selecting 50% of tokens, cosine should be reasonable
    # (random selection doesn't use actual attention scores for topk, so this
    #  tests correctness more than quality)
    assert m["cosine"] > 0.70, f"sparse topk cosine {m['cosine']:.4f} < 0.70"

    # Also test sparse vs Python reference
    kv_cache_cpu = kv_cache.cpu()
    q_rot_cpu = q_rot.squeeze(0).cpu()  # [s_q, h_q, d_c]
    q_rope_cpu = q_rope_q.float().cpu()  # [s_q, h_q, d_rope]
    indices_cpu = perm.long()

    out_ref, lse_ref = ref_tq_mse_sparse_decode(
        q_rot_cpu, q_rope_cpu, kv_cache_cpu, indices_cpu,
        Pi, centroids, sm_scale, D_C, D_ROPE, page_size)
    out_ref_back = out_ref @ Pi  # v_rotate_back

    m_ref = compute_metrics(out_ref_back, out_sparse_final.float().cpu())
    print(f"  Sparse GPU vs Python ref: {fmt(m_ref)}")
    assert m_ref["cosine"] > 0.95, f"sparse vs ref cosine {m_ref['cosine']:.4f} < 0.95"

    print("  PASS")


def test_tq_dense_decode_splitkv(verbose=False):
    """GPU split-KV decode matches non-split decode."""
    print("\n=== test_tq_dense_decode_splitkv ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    import torch.cuda
    dev_props = torch.cuda.get_device_properties(0)
    num_sms = dev_props.multi_processor_count

    page_size = 64
    for s_kv in [256, 1024, 4096]:
        torch.manual_seed(42)
        q = torch.randn(1, H_Q, D_QK)
        c_kv = torch.randn(s_kv, D_C)
        k_rope = torch.randn(s_kv, D_ROPE)

        num_pages = (s_kv + page_size - 1) // page_size
        row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

        kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                                dtype=torch.uint8, device='cuda')
        slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
        sm120_mla_kernels.tq_fused_k_append(
            c_kv.bfloat16().cuda(), k_rope.bfloat16().cuda(), kv_cache, slot_mapping,
            Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

        block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
        seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')

        q_nope = q[..., :D_C].bfloat16().cuda()
        q_rope_q = q[..., D_C:].bfloat16().cuda()
        q_rot = sm120_mla_kernels.tq_q_rotate(q_nope.unsqueeze(0), Pi_gpu)

        # Non-split
        out_nosplit, lse_nosplit = sm120_mla_kernels.tq_dense_decode_v32(
            q_rot, q_rope_q.unsqueeze(0), kv_cache, block_table, seqlens_k,
            centroids_gpu, sm_scale, page_size, 1)
        out_nosplit_final = sm120_mla_kernels.tq_v_rotate_back(out_nosplit, Pi_gpu)

        # Split-KV with num_sms partitions
        out_split, lse_split = sm120_mla_kernels.tq_dense_decode_v32(
            q_rot, q_rope_q.unsqueeze(0), kv_cache, block_table, seqlens_k,
            centroids_gpu, sm_scale, page_size, num_sms)
        out_split_final = sm120_mla_kernels.tq_v_rotate_back(out_split, Pi_gpu)

        m = compute_metrics(out_nosplit_final.cpu(), out_split_final.cpu())
        print(f"  s_kv={s_kv:5d}: split-KV({num_sms}) vs no-split: {fmt(m)}")
        assert m["cosine"] > 0.99, f"s_kv={s_kv} split-KV cosine {m['cosine']:.4f} < 0.99"

    print("  PASS")


def test_tq_dense_prefill(verbose=False):
    """GPU tq_dense_prefill vs BF16 ground truth."""
    print("\n=== test_tq_dense_prefill (s_q=4, s_kv=256) ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    s_q = 4
    s_kv = 256
    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.bfloat16, device='cuda')
    c_kv = torch.randn(s_kv, D_C, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.bfloat16, device='cuda')

    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    # Populate TQ cache
    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv, k_rope, kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

    # TQ dense prefill
    out_tq, lse_tq = sm120_mla_kernels.tq_dense_prefill_v32(
        q, kv_cache, Pi_gpu, centroids_gpu, s_kv, sm_scale,
        D_C, D_ROPE, page_size)

    # BF16 ground truth prefill (dequant first, then standard prefill)
    kv_bf16 = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)  # [s_kv, 1, d_qk]
    out_bf16, lse_bf16 = sm120_mla_kernels.dense_prefill_v32(q, kv_bf16, sm_scale)

    # TQ prefill should be close to BF16 (limited by 4-bit quantization)
    m = compute_metrics(out_bf16.cpu(), out_tq.cpu())
    print(f"  TQ prefill vs BF16 prefill: {fmt(m)}")
    # At s_kv=256, TQ 4-bit limits quality — but prefill quality should match
    # decode quality at the same context length
    assert m["cosine"] > 0.70, f"TQ prefill cosine {m['cosine']:.4f} < 0.70"

    print("  PASS")


def test_tq_sparse_prefill(verbose=False):
    """GPU tq_sparse_prefill vs tq_dense_prefill with all indices."""
    print("\n=== test_tq_sparse_prefill (s_q=4, s_kv=256, topk=256) ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    s_q = 4
    s_kv = 256
    topk = s_kv
    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.bfloat16, device='cuda')
    c_kv = torch.randn(s_kv, D_C, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.bfloat16, device='cuda')

    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(D_C, D_ROPE)

    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv, k_rope, kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, D_C, D_ROPE, page_size)

    # Dense prefill (ground truth for this test)
    out_dense, lse_dense = sm120_mla_kernels.tq_dense_prefill_v32(
        q, kv_cache, Pi_gpu, centroids_gpu, s_kv, sm_scale,
        D_C, D_ROPE, page_size)

    # Sparse prefill with all indices
    # indices shape: [s_q, h_kv=1, topk]
    indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
    indices = indices.expand(s_q, 1, topk).contiguous()

    out_sparse, lse_sparse = sm120_mla_kernels.tq_sparse_prefill_v32(
        q, kv_cache, indices, Pi_gpu, centroids_gpu, sm_scale, topk,
        D_C, D_ROPE, page_size)

    m = compute_metrics(out_dense.cpu(), out_sparse.cpu())
    print(f"  Sparse(topk=all) vs Dense prefill: {fmt(m)}")
    assert m["cosine"] > 0.99, f"sparse==dense prefill cosine {m['cosine']:.6f} < 0.99"

    print("  PASS")


def test_tq_model1_decode(verbose=False):
    """TQ decode with MODEL1 dimensions (d_c=448, d_rope=64)."""
    print("\n=== test_tq_model1_decode (d_c=448) ===")

    d_c_m1 = 448
    d_rope_m1 = 64
    d_qk_m1 = d_c_m1 + d_rope_m1
    h_q_m1 = 64

    Pi = generate_rotation_matrix(d_c_m1)
    centroids, boundaries = load_codebook(d_c_m1, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(d_qk_m1)

    Pi_gpu = Pi.cuda()
    centroids_gpu = centroids.cuda()
    boundaries_gpu = boundaries.cuda()

    torch.manual_seed(42)
    s_kv = 256
    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    row_bytes = compute_tq_row_bytes(d_c_m1, d_rope_m1)

    c_kv = torch.randn(s_kv, d_c_m1, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(s_kv, d_rope_m1, dtype=torch.bfloat16, device='cuda')

    kv_cache = torch.zeros(num_pages * page_size * row_bytes,
                            dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    sm120_mla_kernels.tq_fused_k_append(
        c_kv, k_rope, kv_cache, slot_mapping,
        Pi_gpu, centroids_gpu, boundaries_gpu, d_c_m1, d_rope_m1, page_size)

    block_table = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device='cuda')

    q_nope = torch.randn(1, 1, h_q_m1, d_c_m1, dtype=torch.bfloat16, device='cuda')
    q_rope_q = torch.randn(1, 1, h_q_m1, d_rope_m1, dtype=torch.bfloat16, device='cuda')

    q_rot = sm120_mla_kernels.tq_q_rotate(q_nope.squeeze(0), Pi_gpu)
    out_rot, lse = sm120_mla_kernels.tq_dense_decode_v32(
        q_rot.unsqueeze(0), q_rope_q, kv_cache, block_table, seqlens_k,
        centroids_gpu, sm_scale, page_size, 1)
    out = sm120_mla_kernels.tq_v_rotate_back(out_rot.squeeze(0), Pi_gpu)

    print(f"  d_c={d_c_m1}: out shape {out.shape}, non-zero={out.abs().sum().item():.2f}")
    assert out.shape == (1, h_q_m1, d_c_m1), f"Wrong output shape: {out.shape}"
    assert out.abs().sum().item() > 0, "Output is all zeros"

    # Round-trip dequant check
    indices = torch.arange(s_kv, dtype=torch.int32, device='cuda')
    k_out = sm120_mla_kernels.tq_dequant_ckv_indexed(
        kv_cache, indices, Pi_gpu, centroids_gpu,
        d_c_m1, d_rope_m1, page_size)
    nope_cos = F.cosine_similarity(c_kv.float().cpu().flatten().unsqueeze(0),
                                    k_out[:, :d_c_m1].float().cpu().flatten().unsqueeze(0)).item()
    print(f"  k_append+dequant round-trip NOPE cosine: {nope_cos:.4f}")
    assert nope_cos > 0.75, f"MODEL1 NOPE cosine {nope_cos:.4f} < 0.75"

    print("  PASS")


ALL_TESTS = [
    test_tq_q_rotate,
    test_tq_v_rotate_back,
    test_tq_k_append_dequant_roundtrip,
    test_tq_k_append_vs_python_reference,
    test_tq_dense_decode_short,
    test_tq_dense_decode_scaling,
    test_tq_sparse_decode_equals_dense,
    test_tq_sparse_decode_topk,
    test_tq_dense_decode_splitkv,
    test_tq_dense_prefill,
    test_tq_sparse_prefill,
    test_tq_model1_decode,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("TurboQuant CUDA Kernel Tests")
    print("=" * 45)

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn(verbose=args.verbose)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 45}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
