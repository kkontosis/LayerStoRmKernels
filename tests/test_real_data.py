"""
SnapMLA Kernel Validation — Real Model Data

Mirrors the 13-test structure of test_snapmla_reference.py, but uses
activations generated from real Kimi K2.5 and DeepSeek V3.2 attention
weights (via sample-data/generate_samples.py) instead of torch.randn.

Key difference: real model weights produce structured attention distributions,
making sparse topk tests meaningful. Random data gives cosine ~0.04 at 25%
topk; projected real weights give 0.55-0.80.

Usage:
    python tests/test_real_data.py           # runs all tests
    python tests/test_real_data.py -v        # verbose comparison tables

Prerequisites:
    python sample-data/generate_samples.py   # generate .pt files first
"""

import sys
import os
import math
import argparse

import torch
import torch.nn.functional as F

# Import helpers from sibling module
sys.path.insert(0, os.path.dirname(__file__))
from test_snapmla_reference import (
    D_C, D_ROPE, D_QK, D_V, H_Q, H_KV, PAGE_SIZE, FP8_MAX, HAS_KERNELS,
    ref_mla_attention_bf16, ref_fused_q_quant, ref_fused_k_append,
    ref_snapmla_decode_fp8,
    naive_fp8_attention, snapmla_fp8_attention,
    _fp8_roundtrip_rough, _fp8_roundtrip_real,
    _compute_metrics, _get_snapmla_row, _oracle_topk_indices, _topk_size,
    _quant_roundtrip_metrics,
    _prep_gpu_decode,
)

if HAS_KERNELS:
    import sm120_mla_kernels

# Import sample data loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.helpers.load_sample_data import list_samples, load_sample, has_samples

_NO_SAMPLES_MSG = "  [no sample data — run: python sample-data/generate_samples.py]"


# ---------------------------------------------------------------------------
# Sample selection helpers
# ---------------------------------------------------------------------------

# Kimi K2.5 is a dense-attention model; DeepSeek V3.2 uses sparse attention.
# Dense tests run Kimi only, sparse tests run DeepSeek only.
DENSE_MODELS = {'kimi_k2.5'}
SPARSE_MODELS = {'deepseek_v32'}


def _kernel_mode(model):
    """Return the natural kernel mode for a model."""
    return "dense" if model in DENSE_MODELS else "sparse"


def _one_per_model(mode='decode', s_kv=None, model_filter=None):
    """Return one sample per model, optionally filtered by s_kv and model set."""
    seen = set()
    for info in list_samples(mode=mode):
        if s_kv is not None and info['s_kv'] != s_kv:
            continue
        if model_filter and info['model'] not in model_filter:
            continue
        if info['model'] not in seen:
            seen.add(info['model'])
            yield info


def _scaling_samples(mode='decode', model_filter=None):
    """Return samples at s_kv=1024, 4096, 32768, optionally filtered by model set."""
    for info in list_samples(mode=mode):
        if info['s_kv'] in (1024, 4096, 16384, 32768):
            if model_filter and info['model'] not in model_filter:
                continue
            yield info


# ---------------------------------------------------------------------------
# NSA indexer (production-style topk selection)
# ---------------------------------------------------------------------------

def _nsa_topk_indices(q_index, k_index, importance, topk):
    """NSA-style topk: learned per-head importance gating + ReLU scoring.

    Mirrors the production NSA indexer from SGLang-FluentLLM:
    score(q,k) = sum_h(importance_h * relu(q_index_h @ k_index^T))
    """
    # q_index: [s_q, n_heads, head_dim], k_index: [s_kv, head_dim], importance: [s_q, n_heads]
    scores = torch.einsum('qhd,kd->qhk', q_index.float(), k_index.float())
    scores = F.relu(scores)
    scores = (scores * importance.float().unsqueeze(-1)).sum(dim=1)  # [s_q, s_kv]
    _, idx = scores.topk(topk, dim=-1)
    return idx.sort(dim=-1).values


# ---------------------------------------------------------------------------
# Attention runners (real data)
# ---------------------------------------------------------------------------

def _run_real_attention(sample, kernel_mode="dense"):
    """Run all FP8 attention approaches on real sample data.

    Returns: (rows, lse_diff, has_fp8)
    """
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    q = sample['q'].float()
    c_kv = sample['c_kv'].float()
    k_rope = sample['k_rope'].float()
    s_kv = sample['s_kv']
    s_q = sample['s_q']
    sm_scale = 1.0 / math.sqrt(D_QK)

    # BF16 reference (ground truth)
    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)
    out_bf16, lse_bf16 = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    rows = []

    # 1. Rough fallback
    out_rough = naive_fp8_attention(q, c_kv, k_rope, sm_scale, _fp8_roundtrip_rough)
    rows.append(("Rough fallback", _compute_metrics(out_bf16, out_rough)))

    if has_fp8:
        # 2. Naive real FP8
        out_naive = naive_fp8_attention(q, c_kv, k_rope, sm_scale, _fp8_roundtrip_real)
        rows.append(("Naive real FP8", _compute_metrics(out_bf16, out_naive)))

        # 3. SnapMLA FP8
        out_snap = snapmla_fp8_attention(q, c_kv, k_rope, sm_scale)
        rows.append(("SnapMLA FP8", _compute_metrics(out_bf16, out_snap)))

    # 4. GPU Kernel (if available)
    if HAS_KERNELS and kernel_mode:
        if s_q == 1:
            # Decode path
            q_gpu = q.to(torch.bfloat16).unsqueeze(0).cuda()
            c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
            k_rope_gpu = k_rope.to(torch.bfloat16).cuda()

            if kernel_mode == "sparse":
                out_gpu, _ = _run_gpu_sparse_decode(q_gpu, c_kv_gpu, k_rope_gpu, sm_scale)
                label = "GPU Sparse"
            else:
                out_gpu, _ = _run_gpu_dense_decode(q_gpu, c_kv_gpu, k_rope_gpu, sm_scale)
                label = "GPU Dense"
            out_gpu = out_gpu.view(s_q, H_Q, D_V)
            rows.append((label, _compute_metrics(out_bf16, out_gpu)))
        elif hasattr(sm120_mla_kernels, 'sparse_prefill_v32'):
            # Prefill path — always use topk=s_kv (all tokens) here.
            # Topk subset tests are in _run_real_topk_attention.
            out_gpu = _run_gpu_prefill(q, c_kv, k_rope, s_kv, sm_scale, dense=True)
            label = "GPU Sparse Prefill"
            rows.append((label, _compute_metrics(out_bf16, out_gpu)))

            if hasattr(sm120_mla_kernels, 'dense_prefill_v32'):
                out_gpu_dense = _run_gpu_dense_prefill(q, c_kv, k_rope, sm_scale)
                rows.append(("GPU Dense Prefill", _compute_metrics(out_bf16, out_gpu_dense)))
                # Verify dense == sparse(identity)
                cos_ds = F.cosine_similarity(
                    out_gpu_dense.flatten(), out_gpu.flatten(), dim=0).item()
                assert cos_ds > 0.999, (
                    f"Dense vs Sparse(identity) cosine={cos_ds:.6f}")

    # LSE diff
    _, lse_snap = ref_snapmla_decode_fp8(q, c_kv, k_rope, sm_scale)
    lse_diff = (lse_snap.float() - lse_bf16.float()).abs().max().item()

    return rows, lse_diff, has_fp8


def _run_real_topk_attention(sample, kernel_mode=None):
    """Run FP8 attention with oracle topk subset on real data.

    Returns: (rows, topk, has_fp8)
    """
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    q = sample['q'].float()
    c_kv = sample['c_kv'].float()
    k_rope = sample['k_rope'].float()
    s_kv = sample['s_kv']
    s_q = sample['s_q']
    sm_scale = 1.0 / math.sqrt(D_QK)
    topk = _topk_size(s_kv)

    # Dense BF16 reference (full attention — ground truth)
    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)
    out_bf16, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    # Oracle topk selection
    idx = _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk)  # [s_q, topk]

    if s_q == 1:
        idx_0 = idx[0]
        c_kv_sub = c_kv[idx_0]
        k_rope_sub = k_rope[idx_0]
    else:
        all_idx = idx.reshape(-1).unique().sort().values
        if len(all_idx) > topk:
            all_idx = all_idx[:topk]
        c_kv_sub = c_kv[all_idx]
        k_rope_sub = k_rope[all_idx]
        idx_0 = all_idx

    # Sparse BF16 baseline — same topk subset, no quantization.
    k_sub = torch.cat([c_kv_sub, k_rope_sub], dim=-1).unsqueeze(1)
    v_sub = c_kv_sub.unsqueeze(1)
    out_sparse_bf16, _ = ref_mla_attention_bf16(q, k_sub, v_sub, sm_scale)

    # Rows are 3-tuples: (name, metrics_vs_sparse, metrics_vs_dense)
    # Sparse BF16 row: vs_sparse=None (it IS the sparse baseline), vs_dense=gap
    # FP8/GPU rows: vs_sparse=kernel accuracy, vs_dense=combined gap
    rows = []
    rows.append(("Sparse BF16", None, _compute_metrics(out_bf16, out_sparse_bf16)))

    out_rough = naive_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale, _fp8_roundtrip_rough)
    rows.append(("Rough fallback",
                 _compute_metrics(out_sparse_bf16, out_rough),
                 _compute_metrics(out_bf16, out_rough)))

    if has_fp8:
        out_naive = naive_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale, _fp8_roundtrip_real)
        rows.append(("Naive real FP8",
                     _compute_metrics(out_sparse_bf16, out_naive),
                     _compute_metrics(out_bf16, out_naive)))

        out_snap = snapmla_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale)
        rows.append(("SnapMLA FP8",
                     _compute_metrics(out_sparse_bf16, out_snap),
                     _compute_metrics(out_bf16, out_snap)))

    # GPU Sparse Kernel
    if HAS_KERNELS and kernel_mode == "sparse" and s_q > 1 and hasattr(sm120_mla_kernels, 'sparse_prefill_v32'):
        q_gpu = q.to(torch.bfloat16).cuda()
        kv_gpu = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1).to(torch.bfloat16).cuda()
        # Use the same shared index set as the BF16 baseline (idx_0), not per-query
        # indices (idx). Per-query indices produce different attention than the shared
        # subset, making cos(sparse) comparison meaningless.
        shared_topk = len(idx_0)
        gpu_idx = idx_0.unsqueeze(0).unsqueeze(0).expand(s_q, 1, -1).contiguous()
        gpu_idx = gpu_idx.to(torch.int32).cuda()  # [s_q, 1, shared_topk]

        out_gpu, _ = sm120_mla_kernels.sparse_prefill_v32(
            q_gpu, kv_gpu, gpu_idx, sm_scale, shared_topk
        )
        out_gpu = out_gpu.float().cpu()
        rows.append(("GPU Sparse Prefill",
                     _compute_metrics(out_sparse_bf16, out_gpu),
                     _compute_metrics(out_bf16, out_gpu)))

    if HAS_KERNELS and kernel_mode == "sparse" and s_q == 1:
        q_gpu = q.to(torch.bfloat16).unsqueeze(0).cuda()
        c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
        k_rope_gpu = k_rope.to(torch.bfloat16).cuda()

        kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages = \
            _prep_gpu_decode(q_gpu, c_kv_gpu, k_rope_gpu)

        topk_padded = ((topk + 63) // 64) * 64
        gpu_indices = torch.full((1, s_q, topk_padded), -1, dtype=torch.int32, device="cuda")
        gpu_indices[0, 0, :topk] = idx[0].to(torch.int32).cuda()

        out_gpu, _ = sm120_mla_kernels.sparse_decode_v32(
            q_nope_fp8, q_rope_bf16, q_scales,
            kv_cache, gpu_indices, sm_scale, PAGE_SIZE, topk_padded, 1
        )
        out_gpu = out_gpu.float().cpu().view(s_q, H_Q, D_V)
        rows.append(("GPU Sparse",
                     _compute_metrics(out_sparse_bf16, out_gpu),
                     _compute_metrics(out_bf16, out_gpu)))

    # NSA indexer rows (DeepSeek V3.2 only — has pre-computed NSA fields)
    if 'k_index' in sample:
        nsa_idx = _nsa_topk_indices(
            sample['q_index'].float(), sample['k_index'].float(),
            sample['importance'].float(), topk)
        if s_q == 1:
            nsa_idx_0 = nsa_idx[0]
        else:
            nsa_idx_0 = nsa_idx.reshape(-1).unique().sort().values
            if len(nsa_idx_0) > topk:
                nsa_idx_0 = nsa_idx_0[:topk]
        c_kv_nsa = c_kv[nsa_idx_0]
        k_rope_nsa = k_rope[nsa_idx_0]
        k_nsa = torch.cat([c_kv_nsa, k_rope_nsa], dim=-1).unsqueeze(1)
        v_nsa = c_kv_nsa.unsqueeze(1)
        out_nsa_bf16, _ = ref_mla_attention_bf16(q, k_nsa, v_nsa, sm_scale)
        rows.append(("NSA BF16", None, _compute_metrics(out_bf16, out_nsa_bf16)))

    return rows, topk, has_fp8


def _run_gpu_prefill(q, c_kv, k_rope, s_kv, sm_scale, dense=True):
    """GPU sparse prefill (BF16 absorbed path). dense=True uses topk=s_kv."""
    s_q = q.shape[0]
    q_gpu = q.to(torch.bfloat16).cuda()
    kv_gpu = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1).to(torch.bfloat16).cuda()

    if dense:
        topk = s_kv
        idx = torch.arange(s_kv, dtype=torch.int32)
        idx = idx.unsqueeze(0).unsqueeze(0).expand(s_q, 1, -1).contiguous().cuda()
    else:
        topk = _topk_size(s_kv)
        idx_oracle = _oracle_topk_indices(q.float(), c_kv.float(), k_rope.float(), sm_scale, topk)
        idx = idx_oracle.unsqueeze(1).to(torch.int32).cuda()  # [s_q, 1, topk]

    out_gpu, _ = sm120_mla_kernels.sparse_prefill_v32(
        q_gpu, kv_gpu, idx, sm_scale, topk
    )
    return out_gpu.float().cpu()


def _run_gpu_dense_prefill(q, c_kv, k_rope, sm_scale):
    """GPU dense prefill (BF16 absorbed path). No indices needed."""
    q_gpu = q.to(torch.bfloat16).cuda()
    kv_gpu = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1).to(torch.bfloat16).cuda()
    out_gpu, _ = sm120_mla_kernels.dense_prefill_v32(q_gpu, kv_gpu, sm_scale)
    return out_gpu.float().cpu()


def _run_gpu_dense_decode(q_bf16_4d, c_kv, k_rope, sm_scale, num_sm_parts=1):
    """GPU dense decode."""
    b, s_q = q_bf16_4d.shape[:2]
    s_kv = c_kv.shape[0]
    kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages = \
        _prep_gpu_decode(q_bf16_4d, c_kv, k_rope)

    block_table = torch.arange(n_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device="cuda")

    out, lse = sm120_mla_kernels.dense_decode_v32(
        q_nope_fp8, q_rope_bf16, q_scales,
        kv_cache, block_table, seqlens_k,
        sm_scale, PAGE_SIZE, num_sm_parts
    )
    return out.float().cpu(), lse.float().cpu()


def _run_gpu_sparse_decode(q_bf16_4d, c_kv, k_rope, sm_scale, num_sm_parts=1):
    """GPU sparse decode (topk=all)."""
    b, s_q = q_bf16_4d.shape[:2]
    s_kv = c_kv.shape[0]
    kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages = \
        _prep_gpu_decode(q_bf16_4d, c_kv, k_rope)

    topk = ((s_kv + 63) // 64) * 64
    indices = torch.full((b, s_q, topk), -1, dtype=torch.int32, device="cuda")
    indices[0, 0, :s_kv] = torch.arange(s_kv, dtype=torch.int32, device="cuda")

    out, lse = sm120_mla_kernels.sparse_decode_v32(
        q_nope_fp8, q_rope_bf16, q_scales,
        kv_cache, indices, sm_scale, PAGE_SIZE, topk, num_sm_parts
    )
    return out.float().cpu(), lse.float().cpu()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_attention_table(label, rows, lse_diff, verbose):
    if verbose:
        print(f"\n  {label}:")
        print(f"  {'Method':<20s} {'cosine':>10s} {'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
        print(f"  {'-'*60}")
        for name, m in rows:
            print(f"  {name:<20s} {m['cosine']:>10.6f} {m['nrmse']:>10.4f} "
                  f"{m['mean_rel']:>10.4f} {m['max_rel']:>10.4f}")
        print(f"  LSE diff (SnapMLA pipeline): {lse_diff:.4f}")


def _print_topk_table(label, topk, s_kv, rows, verbose):
    """Print topk table with dual baselines: cos(sparse) + cos(dense).

    Rows are 3-tuples: (name, m_vs_sparse_or_None, m_vs_dense).
    """
    if verbose:
        print(f"\n  {label}, topk={topk} ({100*topk/s_kv:.0f}%):")
        print(f"  {'Method':<20s} {'cos(sparse)':>12s} {'cos(dense)':>12s} "
              f"{'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
        print(f"  {'-'*74}")
        for name, m_sparse, m_dense in rows:
            cos_s = f"{m_sparse['cosine']:>12.6f}" if m_sparse else f"{'—':>12s}"
            m = m_sparse if m_sparse else m_dense
            print(f"  {name:<20s} {cos_s} {m_dense['cosine']:>12.6f} "
                  f"{m['nrmse']:>10.4f} {m['mean_rel']:>10.4f} {m['max_rel']:>10.4f}")


def _print_prefill_table(label, rows, verbose):
    if verbose:
        print(f"\n  {label}:")
        print(f"  {'Method':<20s} {'cosine':>10s} {'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
        print(f"  {'-'*60}")
        for name, m in rows:
            print(f"  {name:<20s} {m['cosine']:>10.6f} {m['nrmse']:>10.4f} "
                  f"{m['mean_rel']:>10.4f} {m['max_rel']:>10.4f}")


# ---------------------------------------------------------------------------
# Test 1: fused_q_quant round-trip (mirrors reference test 1)
# ---------------------------------------------------------------------------

def test_q_quant_roundtrip(verbose=False):
    """fused_q_quant: Rough vs Naive FP8 vs SnapMLA round-trip (real data)."""
    samples = list(_one_per_model(mode='decode', s_kv=1024))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    all_passed = True

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        q_nope = q[..., :D_C]

        if verbose:
            print(f"\n  {info['model']}:")
            print(f"  {'Method':<20s} {'NOPE max_rel':>12s} {'NOPE mean_rel':>14s} {'ROPE err':>10s}")
            print(f"  {'-'*58}")

        # 1. Rough fallback
        max_r, mean_r = _quant_roundtrip_metrics(q_nope, _fp8_roundtrip_rough)
        if verbose:
            print(f"  {'Rough fallback':<20s} {max_r:>12.4f} {mean_r:>14.4f} {'N/A':>10s}")

        results = [('Rough', max_r)]

        if has_fp8:
            # 2. Naive real FP8
            max_n, mean_n = _quant_roundtrip_metrics(q_nope, _fp8_roundtrip_real)
            if verbose:
                print(f"  {'Naive real FP8':<20s} {max_n:>12.4f} {mean_n:>14.4f} {'N/A':>10s}")
            results.append(('Naive', max_n))

            # 3. SnapMLA (only NOPE quantized, ROPE stays BF16)
            q_nope_deq, q_rope_ps, scale = ref_fused_q_quant(q)
            q_rope_expected = q[..., D_C:] / scale.unsqueeze(-1)
            rope_err = (q_rope_ps - q_rope_expected).abs().max().item()
            significant = q_nope.abs() > 0.01
            if significant.any():
                rel = (q_nope_deq - q[..., :D_C]).abs()[significant] / q[..., :D_C].abs()[significant]
                max_s, mean_s = rel.max().item(), rel.mean().item()
            else:
                max_s, mean_s = 0.0, 0.0
            if verbose:
                print(f"  {'SnapMLA FP8':<20s} {max_s:>12.4f} {mean_s:>14.4f} {rope_err:>10.2e}")
            results.append(('SnapMLA', max_s))

        # 4. GPU Kernel (if available)
        if HAS_KERNELS:
            q_gpu = q.to(torch.bfloat16).cuda()
            q_nope_fp8, q_rope_bf16, q_scales_gpu = sm120_mla_kernels.fused_q_quant(q_gpu, D_C)
            q_bf16_cpu = q_gpu.float().cpu()
            nope_deq_gpu = q_nope_fp8.float().cpu() * q_scales_gpu.cpu().unsqueeze(-1)
            significant = q_bf16_cpu[..., :D_C].abs() > 0.01
            if significant.any():
                rel = (nope_deq_gpu - q_bf16_cpu[..., :D_C]).abs()[significant] / q_bf16_cpu[..., :D_C].abs()[significant]
                max_g, mean_g = rel.max().item(), rel.mean().item()
            else:
                max_g, mean_g = 0.0, 0.0
            q_rope_expected = q_bf16_cpu[..., D_C:] / q_scales_gpu.cpu().unsqueeze(-1)
            rope_err_gpu = (q_rope_bf16.float().cpu() - q_rope_expected).abs().max().item()
            if verbose:
                print(f"  {'GPU Kernel':<20s} {max_g:>12.4f} {mean_g:>14.4f} {rope_err_gpu:>10.2e}")
            results.append(('GPU Kernel', max_g))

        best = min(results, key=lambda x: x[1])
        passed = best[1] < 0.07
        all_passed &= passed
        if verbose:
            print(f"  Best NOPE precision: {best[0]} ({best[1]:.4f}) — {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 2: fused_k_append round-trip (mirrors reference test 2)
# ---------------------------------------------------------------------------

def test_k_append_dequant_roundtrip(verbose=False):
    """fused_k_append + dequant_ckv: Rough vs Naive FP8 vs SnapMLA round-trip (real data)."""
    samples = list(_one_per_model(mode='decode', s_kv=1024))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    all_passed = True

    for info in samples:
        sample = load_sample(info['path'])
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()

        if verbose:
            print(f"\n  {info['model']}:")
            print(f"  {'Method':<20s} {'NOPE max_rel':>12s} {'NOPE mean_rel':>14s} {'ROPE err':>10s}")
            print(f"  {'-'*58}")

        # 1. Rough fallback
        max_r, mean_r = _quant_roundtrip_metrics(c_kv, _fp8_roundtrip_rough)
        if verbose:
            print(f"  {'Rough fallback':<20s} {max_r:>12.4f} {mean_r:>14.4f} {'N/A':>10s}")

        results = [('Rough', max_r)]

        if has_fp8:
            # 2. Naive real FP8
            max_n, mean_n = _quant_roundtrip_metrics(c_kv, _fp8_roundtrip_real)
            if verbose:
                print(f"  {'Naive real FP8':<20s} {max_n:>12.4f} {mean_n:>14.4f} {'N/A':>10s}")
            results.append(('Naive', max_n))

            # 3. SnapMLA
            c_kv_deq, k_rope_ps, scale = ref_fused_k_append(c_kv, k_rope)
            k_rope_recovered = k_rope_ps * scale.unsqueeze(-1)
            rope_err = (k_rope_recovered - k_rope).abs().max().item()
            significant = c_kv.abs() > 0.01
            if significant.any():
                rel = (c_kv_deq - c_kv).abs()[significant] / c_kv.abs()[significant]
                max_s, mean_s = rel.max().item(), rel.mean().item()
            else:
                max_s, mean_s = 0.0, 0.0
            if verbose:
                print(f"  {'SnapMLA FP8':<20s} {max_s:>12.4f} {mean_s:>14.4f} {rope_err:>10.2e}")
            results.append(('SnapMLA', max_s))

        # 4. GPU Kernel roundtrip
        if HAS_KERNELS:
            from test_snapmla_reference import _alloc_paged_cache
            c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
            k_rope_gpu = k_rope.to(torch.bfloat16).cuda()
            n_tokens = c_kv.shape[0]

            n_pages = (n_tokens + PAGE_SIZE - 1) // PAGE_SIZE
            kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
            slot_mapping = torch.arange(n_tokens, dtype=torch.int32, device="cuda")
            sm120_mla_kernels.fused_k_append(c_kv_gpu, k_rope_gpu, kv_cache, slot_mapping,
                                              D_C, D_ROPE, PAGE_SIZE)
            indices = torch.arange(n_tokens, dtype=torch.int32, device="cuda")
            k_out = sm120_mla_kernels.dequant_ckv_indexed(kv_cache, indices, D_C, D_ROPE, PAGE_SIZE)
            nope_out = k_out.float().cpu()[:, :D_C]

            significant = c_kv.abs() > 0.01
            if significant.any():
                rel = (nope_out - c_kv).abs()[significant] / c_kv.abs()[significant]
                max_g, mean_g = rel.max().item(), rel.mean().item()
            else:
                max_g, mean_g = 0.0, 0.0
            rope_out = k_out.float().cpu()[:, D_C:]
            # dequant_ckv_indexed now un-scales ROPE (multiplies by scale),
            # so output should approximate original k_rope
            rope_cos = F.cosine_similarity(rope_out.flatten(), k_rope.flatten(), dim=0).item()
            rope_err_gpu = 1.0 - rope_cos
            if verbose:
                print(f"  {'GPU Kernel':<20s} {max_g:>12.4f} {mean_g:>14.4f} {rope_err_gpu:>10.2e}")
            results.append(('GPU Kernel', max_g))

        best = min(results, key=lambda x: x[1])
        passed = best[1] < 0.07
        all_passed &= passed
        if verbose:
            print(f"  Best NOPE precision: {best[0]} ({best[1]:.4f}) — {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 3: Decode attention short (mirrors reference test 3)
# ---------------------------------------------------------------------------

def test_decode_attention_short(verbose=False):
    """Decode attention (s_kv=256): Rough vs Naive FP8 vs SnapMLA vs BF16 (real data)."""
    samples = list(_one_per_model(mode='decode', s_kv=256))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    for info in samples:
        sample = load_sample(info['path'])
        label = f"{info['model']} s_kv=256"
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode=_kernel_mode(info['model']))
        _print_attention_table(label, rows, lse_diff, verbose)

        m = _get_snapmla_row(rows, has_fp8)
        passed = m['cosine'] > 0.995 and m['nrmse'] < 0.05
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 4: Dense decode scaling (mirrors reference test 4)
# ---------------------------------------------------------------------------

def test_dense_decode_scaling(verbose=False):
    """Dense decode scaling (1K→32K): Kimi K2.5 (dense model) only."""
    all_passed = True
    found_any = False

    for info in _scaling_samples(mode='decode', model_filter=DENSE_MODELS):
        found_any = True
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        label = f"{info['model']} s_kv={s_kv}"
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode="dense")
        _print_attention_table(label, rows, lse_diff, verbose)

        m = _get_snapmla_row(rows, has_fp8)
        if s_kv >= 4096:
            thr_cos, thr_nrmse = 0.990, 0.15
        else:
            thr_cos, thr_nrmse = 0.995, 0.05
        passed = m['cosine'] > thr_cos and m['nrmse'] < thr_nrmse
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    if not found_any:
        print(_NO_SAMPLES_MSG)
    return all_passed


# ---------------------------------------------------------------------------
# Test 5: Sparse decode scaling (mirrors reference test 5)
# ---------------------------------------------------------------------------

def test_sparse_decode_scaling(verbose=False):
    """Sparse decode scaling (1K→32K): DeepSeek V3.2 (sparse model) only."""
    all_passed = True
    found_any = False

    for info in _scaling_samples(mode='decode', model_filter=SPARSE_MODELS):
        found_any = True
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        label = f"{info['model']} s_kv={s_kv}"
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode="sparse")
        _print_attention_table(label, rows, lse_diff, verbose)

        m = _get_snapmla_row(rows, has_fp8)
        if s_kv >= 4096:
            thr_cos, thr_nrmse = 0.990, 0.15
        else:
            thr_cos, thr_nrmse = 0.995, 0.05
        passed = m['cosine'] > thr_cos and m['nrmse'] < thr_nrmse
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    if not found_any:
        print(_NO_SAMPLES_MSG)
    return all_passed


# ---------------------------------------------------------------------------
# Test 6: Sparse topk decode scaling (mirrors reference test 6) — CRITICAL
# ---------------------------------------------------------------------------

def _get_topk_snapmla(rows, has_fp8):
    """Get SnapMLA row metrics from 3-tuple topk rows. Returns (m_sparse, m_dense)."""
    for name, m_s, m_d in rows:
        if name == "SnapMLA FP8":
            return m_s, m_d
    last = rows[-1] if has_fp8 else rows[0]
    return last[1], last[2]


def test_sparse_topk_decode_scaling(verbose=False):
    """Sparse topk decode scaling (256→32K): DeepSeek V3.2 (sparse model) only.

    With real projected weights, oracle topk captures meaningful attention mass
    (cosine 0.55-0.80 vs random's ~0.04). Threshold: cosine >0.50.
    """
    samples = list(list_samples(mode='decode', model='deepseek_v32'))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    for info in samples:
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        rows, topk, has_fp8 = _run_real_topk_attention(sample, kernel_mode="sparse")
        label = f"{info['model']} s_kv={s_kv}"
        _print_topk_table(label, topk, s_kv, rows, verbose)

        m_sparse, m_dense = _get_topk_snapmla(rows, has_fp8)
        # Threshold scales with sparsity: 6% topk on random data gives cos ~0.22
        ratio = topk / s_kv
        thr = 0.50 if ratio >= 0.25 else 0.15
        passed = m_dense['cosine'] > thr and not math.isnan(m_dense['nrmse'])
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}, topk={topk} ({100*topk/s_kv:.0f}%): "
                  f"cos(dense)={m_dense['cosine']:.4f} — {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 7: SnapMLA beats naive (mirrors reference test 7)
# ---------------------------------------------------------------------------

def test_snapmla_decode_beats_naive(verbose=False):
    """SnapMLA decode: does NOPE/ROPE split beat naive all-FP8 at longest context? (real data)"""
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    if not has_fp8:
        if verbose:
            print("  [torch.float8_e4m3fn not available — skipped]")
        return True

    # Use longest available context per model
    samples = list(_one_per_model(mode='decode', s_kv=4096))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    for info in samples:
        sample = load_sample(info['path'])
        label = f"{info['model']} s_kv={info['s_kv']}"
        rows, lse_diff, _ = _run_real_attention(sample, kernel_mode=None)
        _print_attention_table(label, rows, lse_diff, verbose)

        # rows[1] = Naive real FP8, rows[2] = SnapMLA FP8
        m_snap = rows[2][1]
        m_naive = rows[1][1]
        snap_better = m_snap['cosine'] >= m_naive['cosine'] and m_snap['nrmse'] <= m_naive['nrmse']
        all_passed &= snap_better
        if verbose or not snap_better:
            print(f"  {label}: SnapMLA beats naive? {'YES' if snap_better else 'NO'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 8: Dense prefill scaling (mirrors reference test 8)
# ---------------------------------------------------------------------------

def test_dense_prefill_scaling(verbose=False):
    """Dense prefill (s_q=128): Kimi K2.5 (dense model) only + GPU."""
    samples = list(list_samples(mode='prefill', model='kimi_k2.5'))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    for info in samples:
        sample = load_sample(info['path'])
        label = f"{info['model']} s_q={info['s_q']} s_kv={info['s_kv']}"
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode="dense")
        _print_attention_table(label, rows, lse_diff, verbose)

        m = _get_snapmla_row(rows, has_fp8)
        passed = m['cosine'] > 0.995 and m['nrmse'] < 0.08
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 9: Sparse prefill scaling (mirrors reference test 9)
# ---------------------------------------------------------------------------

def test_sparse_prefill_scaling(verbose=False):
    """Sparse prefill (s_q=128): DeepSeek V3.2 (sparse model) only + GPU."""
    samples = list(list_samples(mode='prefill', model='deepseek_v32'))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    for info in samples:
        sample = load_sample(info['path'])
        label = f"{info['model']} s_q={info['s_q']} s_kv={info['s_kv']}"
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode="sparse")
        _print_prefill_table(label, rows, verbose)

        m = _get_snapmla_row(rows, has_fp8)
        passed = m['cosine'] > 0.995 and m['nrmse'] < 0.08
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 10: Sparse topk prefill scaling (mirrors reference test 10)
# ---------------------------------------------------------------------------

def test_sparse_topk_prefill_scaling(verbose=False):
    """Sparse topk prefill (s_q=128): DeepSeek V3.2 (sparse model) only."""
    samples = list(list_samples(mode='prefill', model='deepseek_v32'))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    for info in samples:
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        rows, topk, has_fp8 = _run_real_topk_attention(sample, kernel_mode="sparse")
        label = f"{info['model']} s_q={info['s_q']} s_kv={s_kv}"
        _print_topk_table(label, topk, s_kv, rows, verbose)

        m_sparse, m_dense = _get_topk_snapmla(rows, has_fp8)
        ratio = topk / s_kv
        thr = 0.50 if ratio >= 0.25 else 0.15
        passed = m_dense['cosine'] > thr and not math.isnan(m_dense['nrmse'])
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}, topk={topk} ({100*topk/s_kv:.0f}%): "
                  f"cos(dense)={m_dense['cosine']:.4f} — {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 11: Chunked prefill consistency (mirrors reference test 11)
# ---------------------------------------------------------------------------

def test_chunked_prefill_consistency(verbose=False):
    """Chunked prefill: LSE merge consistency (real data).

    Splits s_kv into two chunks and verifies LSE-based merging produces
    the same output as single-pass attention.
    """
    samples = list(list_samples(mode='prefill'))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    chunk_size = 512
    sm_scale = 1.0 / math.sqrt(D_QK)
    all_passed = True

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']
        s_q = sample['s_q']

        k = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)   # [s_kv, 1, D_QK]
        v = c_kv.unsqueeze(1)                                  # [s_kv, 1, D_V]

        # Single-pass reference
        out_full, lse_full = ref_mla_attention_bf16(q, k, v, sm_scale)

        # Chunked with LSE merge
        out_accum = torch.zeros(s_q, H_Q, D_V, dtype=torch.float32)
        lse_accum = torch.full((s_q, H_Q), float('-inf'), dtype=torch.float32)

        for start in range(0, s_kv, chunk_size):
            end = min(start + chunk_size, s_kv)
            out_chunk, lse_chunk = ref_mla_attention_bf16(q, k[start:end], v[start:end], sm_scale)

            new_lse = torch.logaddexp(lse_accum, lse_chunk.float())
            old_w = torch.exp(lse_accum - new_lse).unsqueeze(-1)
            new_w = torch.exp(lse_chunk.float() - new_lse).unsqueeze(-1)
            out_accum = old_w * out_accum + new_w * out_chunk.float()
            lse_accum = new_lse

        max_diff = (out_accum.float() - out_full.float()).abs().max().item()
        lse_diff = (lse_accum - lse_full.float()).abs().max().item()

        passed = max_diff < 1e-4 and lse_diff < 1e-4
        all_passed &= passed
        if verbose or not passed:
            print(f"  {info['model']}: out_max_diff={max_diff:.2e}, "
                  f"lse_max_diff={lse_diff:.2e} — {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 12: Sparse vs Dense identity (mirrors reference test 12)
# ---------------------------------------------------------------------------

def test_sparse_vs_dense(verbose=False):
    """Sparse vs Dense attention: topk=all identity (real data).

    When topk = s_kv (all tokens selected), sparse and dense produce
    identical results. Also tests that random permutation doesn't change output.
    """
    samples = list(_one_per_model(mode='decode', s_kv=256))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    sm_scale = 1.0 / math.sqrt(D_QK)
    all_passed = True

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']

        k = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v = c_kv.unsqueeze(1)

        # Dense: all tokens
        out_dense, _ = ref_mla_attention_bf16(q, k, v, sm_scale)

        # Sparse: identity permutation
        indices = torch.arange(s_kv)
        out_sparse, _ = ref_mla_attention_bf16(q, k[indices], v[indices], sm_scale)
        max_diff = (out_dense.float() - out_sparse.float()).abs().max().item()
        id_ok = max_diff < 1e-6

        if verbose or not id_ok:
            print(f"  {info['model']} Sparse(topk=all) vs Dense: "
                  f"max_diff={max_diff:.2e} — {'PASS' if id_ok else 'FAIL'}")

        # Sparse: random permutation
        torch.manual_seed(42)
        perm = torch.randperm(s_kv)
        out_perm, _ = ref_mla_attention_bf16(q, k[perm], v[perm], sm_scale)
        perm_diff = (out_dense.float() - out_perm.float()).abs().max().item()
        perm_ok = perm_diff < 1e-5

        if verbose:
            print(f"  {info['model']} Sparse(permuted) vs Dense: "
                  f"max_diff={perm_diff:.2e} — {'PASS' if perm_ok else 'FAIL'}")

        all_passed &= id_ok and perm_ok

    return all_passed


# ---------------------------------------------------------------------------
# Test 13: Sparse topk vs Dense quality (mirrors reference test 13)
# ---------------------------------------------------------------------------

def test_sparse_topk_vs_dense(verbose=False):
    """Sparse topk vs Dense: DeepSeek V3.2 (sparse model) only.

    Shows oracle topk quality and compares with random subset baseline.
    """
    samples = list(_one_per_model(mode='decode', s_kv=1024, model_filter=SPARSE_MODELS))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    sm_scale = 1.0 / math.sqrt(D_QK)
    all_passed = True

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']
        topk = _topk_size(s_kv)

        k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v_mla = c_kv.unsqueeze(1)

        # Dense: all tokens
        out_dense, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

        # Oracle topk
        idx = _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk)
        idx_0 = idx[0]
        out_topk, _ = ref_mla_attention_bf16(q, k_mla[idx_0], v_mla[idx_0], sm_scale)
        m = _compute_metrics(out_dense, out_topk)

        passed = m['cosine'] > 0.50
        all_passed &= passed
        if verbose or not passed:
            print(f"  {info['model']} Sparse(topk={topk}/{s_kv}={100*topk/s_kv:.0f}%) vs Dense: "
                  f"cosine={m['cosine']:.6f}, nrmse={m['nrmse']:.4f} — {'PASS' if passed else 'FAIL'}")

        # Random subset (for comparison — expected to be similar with uniform-ish attention)
        torch.manual_seed(42)
        rand_idx = torch.randperm(s_kv)[:topk].sort().values
        out_rand, _ = ref_mla_attention_bf16(q, k_mla[rand_idx], v_mla[rand_idx], sm_scale)
        m_rand = _compute_metrics(out_dense, out_rand)
        if verbose:
            print(f"  {info['model']} Random(topk={topk}/{s_kv}={100*topk/s_kv:.0f}%) vs Dense: "
                  f"cosine={m_rand['cosine']:.6f}, nrmse={m_rand['nrmse']:.4f}")

        # NSA indexer (production-style, uses trained weights)
        if 'k_index' in sample:
            nsa_idx = _nsa_topk_indices(
                sample['q_index'].float(), sample['k_index'].float(),
                sample['importance'].float(), topk)
            nsa_idx_0 = nsa_idx[0]
            out_nsa, _ = ref_mla_attention_bf16(q, k_mla[nsa_idx_0], v_mla[nsa_idx_0], sm_scale)
            m_nsa = _compute_metrics(out_dense, out_nsa)
            if verbose:
                print(f"  {info['model']} NSA(topk={topk}/{s_kv}={100*topk/s_kv:.0f}%) vs Dense: "
                      f"cosine={m_nsa['cosine']:.6f}, nrmse={m_nsa['nrmse']:.4f}")

    return all_passed


# ---------------------------------------------------------------------------
# Test 14: FP8 cache roundtrip prefill (production path)
# ---------------------------------------------------------------------------

def test_fp8_cache_roundtrip_prefill(verbose=False):
    """FP8 cache roundtrip prefill: fused_k_append → dequant → BF16 prefill.

    Tests the production prefill path: KV is quantized to FP8 paged cache
    (SnapMLA format: NOPE FP8 + scale + ROPE BF16), then dequanted back to
    BF16, then used for BF16 prefill attention. This measures the combined
    quantization + dequantization + attention error.
    """
    if not HAS_KERNELS:
        print("  [no GPU kernels — skipped]")
        return True
    if not hasattr(sm120_mla_kernels, 'sparse_prefill_v32'):
        print("  [no sparse_prefill_v32 — skipped]")
        return True

    from test_snapmla_reference import _alloc_paged_cache

    samples = list(list_samples(mode='prefill', model='deepseek_v32'))
    if not samples:
        print(_NO_SAMPLES_MSG)
        return True

    all_passed = True
    sm_scale = 1.0 / math.sqrt(D_QK)

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']
        s_q = sample['s_q']
        topk = _topk_size(s_kv)

        # Dense BF16 reference (ground truth)
        k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v_mla = c_kv.unsqueeze(1)
        out_bf16, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

        # --- Path A: Direct BF16 prefill (no FP8 roundtrip) ---
        q_gpu = q.to(torch.bfloat16).cuda()
        kv_direct = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1).to(torch.bfloat16).cuda()
        idx_all = torch.arange(s_kv, dtype=torch.int32).unsqueeze(0).unsqueeze(0)
        idx_all = idx_all.expand(s_q, 1, -1).contiguous().cuda()
        out_direct, _ = sm120_mla_kernels.sparse_prefill_v32(
            q_gpu, kv_direct, idx_all, sm_scale, s_kv)
        out_direct = out_direct.float().cpu()

        # --- Path A2: Dense BF16 prefill (verify matches sparse path A) ---
        has_dense_prefill = hasattr(sm120_mla_kernels, 'dense_prefill_v32')
        if has_dense_prefill:
            out_dense_direct, _ = sm120_mla_kernels.dense_prefill_v32(
                q_gpu, kv_direct, sm_scale)
            out_dense_direct = out_dense_direct.float().cpu()

        # --- Path B: FP8 cache roundtrip → BF16 prefill ---
        c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
        k_rope_gpu = k_rope.to(torch.bfloat16).cuda()

        n_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
        kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
        slot_mapping = torch.arange(s_kv, dtype=torch.int32, device="cuda")
        sm120_mla_kernels.fused_k_append(
            c_kv_gpu, k_rope_gpu, kv_cache, slot_mapping, D_C, D_ROPE, PAGE_SIZE)

        # Dequant all tokens back to BF16
        fetch_indices = torch.arange(s_kv, dtype=torch.int32, device="cuda")
        kv_dequant = sm120_mla_kernels.dequant_ckv_indexed(
            kv_cache, fetch_indices, D_C, D_ROPE, PAGE_SIZE)  # [s_kv, 576] BF16

        # dequant_ckv_indexed now un-scales ROPE (multiplies by scale),
        # so output is directly usable for BF16 prefill — no manual
        # scale extraction needed.
        kv_recovered = kv_dequant.unsqueeze(1).cuda()  # [s_kv, 1, 576]

        out_roundtrip, _ = sm120_mla_kernels.sparse_prefill_v32(
            q_gpu, kv_recovered, idx_all, sm_scale, s_kv)
        out_roundtrip = out_roundtrip.float().cpu()

        m_direct = _compute_metrics(out_bf16, out_direct)
        m_roundtrip = _compute_metrics(out_bf16, out_roundtrip)

        label = f"{info['model']} s_q={s_q} s_kv={s_kv}"
        if verbose:
            print(f"\n  {label}:")
            print(f"  {'Path':<30s} {'cosine':>10s} {'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
            print(f"  {'-'*70}")
            print(f"  {'Sparse BF16 prefill':<30s} {m_direct['cosine']:>10.6f} {m_direct['nrmse']:>10.4f} "
                  f"{m_direct['mean_rel']:>10.4f} {m_direct['max_rel']:>10.4f}")
            if has_dense_prefill:
                m_dense_direct = _compute_metrics(out_bf16, out_dense_direct)
                cos_ds = F.cosine_similarity(
                    out_dense_direct.flatten(), out_direct.flatten(), dim=0).item()
                print(f"  {'Dense BF16 prefill':<30s} {m_dense_direct['cosine']:>10.6f} {m_dense_direct['nrmse']:>10.4f} "
                      f"{m_dense_direct['mean_rel']:>10.4f} {m_dense_direct['max_rel']:>10.4f}")
                print(f"  {'Dense vs Sparse(identity)':<30s} {'cos=' + f'{cos_ds:.6f}':>10s}")
            print(f"  {'FP8 cache roundtrip prefill':<30s} {m_roundtrip['cosine']:>10.6f} {m_roundtrip['nrmse']:>10.4f} "
                  f"{m_roundtrip['mean_rel']:>10.4f} {m_roundtrip['max_rel']:>10.4f}")

        passed = m_roundtrip['cosine'] > 0.99 and m_roundtrip['nrmse'] < 0.10
        if has_dense_prefill:
            cos_ds = F.cosine_similarity(
                out_dense_direct.flatten(), out_direct.flatten(), dim=0).item()
            passed &= cos_ds > 0.999
        all_passed &= passed
        if verbose or not passed:
            print(f"  {label}: {'PASS' if passed else 'FAIL'}")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SnapMLA real-data tests')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print("SnapMLA Real-Data Tests")
    print("=" * 60)

    if not has_samples():
        print("\nNo sample data found. Generate it first:")
        print("  python sample-data/generate_samples.py")
        return 1

    tests = [
        ("fused_q_quant: Rough vs Naive FP8 vs SnapMLA round-trip (real data)",
         test_q_quant_roundtrip),
        ("fused_k_append + dequant_ckv: round-trip (real data)",
         test_k_append_dequant_roundtrip),
        ("Decode attention (s_kv=256): real data",
         test_decode_attention_short),
        ("Dense decode scaling (1K→32K): Kimi (dense)",
         test_dense_decode_scaling),
        ("Sparse decode scaling (1K→32K): DeepSeek (sparse)",
         test_sparse_decode_scaling),
        ("Sparse topk decode scaling (256→32K): DeepSeek (sparse)",
         test_sparse_topk_decode_scaling),
        ("SnapMLA decode: beats Naive FP8 at longest context? (real data)",
         test_snapmla_decode_beats_naive),
        ("Dense prefill (s_q=128): Kimi (dense) + GPU",
         test_dense_prefill_scaling),
        ("Sparse prefill (s_q=128): DeepSeek (sparse) + GPU",
         test_sparse_prefill_scaling),
        ("Sparse topk prefill (s_q=128): DeepSeek (sparse) + GPU",
         test_sparse_topk_prefill_scaling),
        ("Chunked prefill: LSE merge consistency (real data)",
         test_chunked_prefill_consistency),
        ("Sparse vs Dense attention: topk=all identity (real data)",
         test_sparse_vs_dense),
        ("Sparse topk vs Dense: DeepSeek (sparse)",
         test_sparse_topk_vs_dense),
        ("FP8 cache roundtrip prefill (production path)",
         test_fp8_cache_roundtrip_prefill),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        passed = test_fn(verbose=args.verbose)
        results.append((name, passed))
        if not args.verbose:
            print(f"  {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")

    if n_pass < n_total:
        print("\nFailed tests:")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
