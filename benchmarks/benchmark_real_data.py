"""
SnapMLA Kernel Benchmark — Real Model Data (JSON output)

Runs the same 14 tests as test_real_data.py but captures every numeric
metric into a structured JSON file for tracking regressions.

Usage:
    python tests/benchmark_real_data.py                         # default output: benchmarks/benchmark_results.json
    python tests/benchmark_real_data.py -o results.json         # custom output path
    python tests/benchmark_real_data.py -v                      # verbose (print tables to stdout too)
    python tests/benchmark_real_data.py -v -o my_run.json       # both

Prerequisites:
    python sample-data/generate_samples.py   # generate .pt files first
"""

import sys
import os
import math
import json
import time
import argparse
import platform
import datetime

import torch
import torch.nn.functional as F

# Import helpers from tests/
_tests_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
sys.path.insert(0, _tests_dir)
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
from tests.helpers.load_sample_data import list_samples, load_sample, has_samples  # noqa: E402

# Re-use real-data test helpers
from test_real_data import (
    DENSE_MODELS, SPARSE_MODELS,
    _kernel_mode, _one_per_model, _scaling_samples,
    _nsa_topk_indices,
    _run_real_attention, _run_real_topk_attention,
    _run_gpu_prefill, _run_gpu_dense_decode, _run_gpu_sparse_decode,
    _get_topk_snapmla,
)

DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(__file__), "benchmark_real_data_results.json"
)


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------

def _collect_env():
    """Gather reproducibility metadata."""
    env = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "has_fp8": hasattr(torch, "float8_e4m3fn"),
        "has_kernels": HAS_KERNELS,
    }
    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda or "N/A"
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    return env


# ---------------------------------------------------------------------------
# Per-test benchmark functions
# Each returns a dict with test-specific metrics.
# ---------------------------------------------------------------------------

def bench_q_quant_roundtrip(verbose):
    """Test 1: fused_q_quant round-trip."""
    samples = list(_one_per_model(mode='decode', s_kv=1024))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    records = []

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        q_nope = q[..., :D_C]
        rec = {"model": info['model'], "methods": {}}

        # Rough fallback
        max_r, mean_r = _quant_roundtrip_metrics(q_nope, _fp8_roundtrip_rough)
        rec["methods"]["rough_fallback"] = {"nope_max_rel": max_r, "nope_mean_rel": mean_r}

        if has_fp8:
            max_n, mean_n = _quant_roundtrip_metrics(q_nope, _fp8_roundtrip_real)
            rec["methods"]["naive_real_fp8"] = {"nope_max_rel": max_n, "nope_mean_rel": mean_n}

            q_nope_deq, q_rope_ps, scale = ref_fused_q_quant(q)
            q_rope_expected = q[..., D_C:] / scale.unsqueeze(-1)
            rope_err = (q_rope_ps - q_rope_expected).abs().max().item()
            significant = q_nope.abs() > 0.01
            if significant.any():
                rel = (q_nope_deq - q[..., :D_C]).abs()[significant] / q[..., :D_C].abs()[significant]
                max_s, mean_s = rel.max().item(), rel.mean().item()
            else:
                max_s, mean_s = 0.0, 0.0
            rec["methods"]["snapmla_fp8"] = {
                "nope_max_rel": max_s, "nope_mean_rel": mean_s, "rope_err": rope_err,
            }

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
            rec["methods"]["gpu_kernel"] = {
                "nope_max_rel": max_g, "nope_mean_rel": mean_g, "rope_err": rope_err_gpu,
            }

        best = min(
            ((k, v["nope_max_rel"]) for k, v in rec["methods"].items()),
            key=lambda x: x[1],
        )
        rec["best_method"] = best[0]
        rec["best_nope_max_rel"] = best[1]
        rec["passed"] = best[1] < 0.07
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_k_append_roundtrip(verbose):
    """Test 2: fused_k_append + dequant round-trip."""
    samples = list(_one_per_model(mode='decode', s_kv=1024))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    records = []

    for info in samples:
        sample = load_sample(info['path'])
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        rec = {"model": info['model'], "methods": {}}

        max_r, mean_r = _quant_roundtrip_metrics(c_kv, _fp8_roundtrip_rough)
        rec["methods"]["rough_fallback"] = {"nope_max_rel": max_r, "nope_mean_rel": mean_r}

        if has_fp8:
            max_n, mean_n = _quant_roundtrip_metrics(c_kv, _fp8_roundtrip_real)
            rec["methods"]["naive_real_fp8"] = {"nope_max_rel": max_n, "nope_mean_rel": mean_n}

            c_kv_deq, k_rope_ps, scale = ref_fused_k_append(c_kv, k_rope)
            k_rope_recovered = k_rope_ps * scale.unsqueeze(-1)
            rope_err = (k_rope_recovered - k_rope).abs().max().item()
            significant = c_kv.abs() > 0.01
            if significant.any():
                rel = (c_kv_deq - c_kv).abs()[significant] / c_kv.abs()[significant]
                max_s, mean_s = rel.max().item(), rel.mean().item()
            else:
                max_s, mean_s = 0.0, 0.0
            rec["methods"]["snapmla_fp8"] = {
                "nope_max_rel": max_s, "nope_mean_rel": mean_s, "rope_err": rope_err,
            }

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
            # dequant_ckv_indexed now un-scales ROPE, so compare against original k_rope
            rope_cos = F.cosine_similarity(rope_out.flatten(), k_rope.flatten(), dim=0).item()
            rec["methods"]["gpu_kernel"] = {
                "nope_max_rel": max_g, "nope_mean_rel": mean_g, "rope_cos": rope_cos,
            }

        best = min(
            ((k, v["nope_max_rel"]) for k, v in rec["methods"].items()),
            key=lambda x: x[1],
        )
        rec["best_method"] = best[0]
        rec["best_nope_max_rel"] = best[1]
        rec["passed"] = best[1] < 0.07
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def _bench_attention_scaling(test_name, mode, model_filter, kernel_mode_fn, verbose):
    """Shared logic for tests 3-5: run attention at various s_kv and collect metrics."""
    if mode == 'scaling':
        samples_iter = _scaling_samples(mode='decode', model_filter=model_filter)
    elif isinstance(mode, int):
        # mode is s_kv value for _one_per_model
        samples_iter = _one_per_model(mode='decode', s_kv=mode, model_filter=model_filter)
    else:
        samples_iter = _scaling_samples(mode=mode, model_filter=model_filter)

    samples = list(samples_iter)
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    records = []
    for info in samples:
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        km = kernel_mode_fn(info['model']) if callable(kernel_mode_fn) else kernel_mode_fn
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode=km)

        rec = {
            "model": info['model'],
            "s_kv": s_kv,
            "lse_diff": lse_diff,
            "methods": {},
        }
        for name, m in rows:
            key = name.lower().replace(" ", "_")
            rec["methods"][key] = m

        m = _get_snapmla_row(rows, has_fp8)
        rec["snapmla_cosine"] = m["cosine"]
        rec["snapmla_nrmse"] = m["nrmse"]

        if s_kv >= 4096:
            thr_cos, thr_nrmse = 0.990, 0.15
        else:
            thr_cos, thr_nrmse = 0.995, 0.05
        rec["passed"] = m["cosine"] > thr_cos and m["nrmse"] < thr_nrmse
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_decode_short(verbose):
    """Test 3: Decode attention short (s_kv=256)."""
    return _bench_attention_scaling("decode_short", 256, None, _kernel_mode, verbose)


def bench_dense_decode_scaling(verbose):
    """Test 4: Dense decode scaling (1K->32K), Kimi only."""
    return _bench_attention_scaling("dense_decode_scaling", 'scaling', DENSE_MODELS, "dense", verbose)


def bench_sparse_decode_scaling(verbose):
    """Test 5: Sparse decode scaling (1K->32K), DeepSeek only."""
    return _bench_attention_scaling("sparse_decode_scaling", 'scaling', SPARSE_MODELS, "sparse", verbose)


def bench_sparse_topk_decode_scaling(verbose):
    """Test 6: Sparse topk decode scaling."""
    samples = list(list_samples(mode='decode', model='deepseek_v32'))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    records = []
    for info in samples:
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        rows, topk, has_fp8 = _run_real_topk_attention(sample, kernel_mode="sparse")

        rec = {
            "model": info['model'],
            "s_kv": s_kv,
            "topk": topk,
            "topk_ratio": topk / s_kv,
            "methods": {},
        }
        for name, m_sparse, m_dense in rows:
            key = name.lower().replace(" ", "_")
            entry = {"vs_dense": m_dense}
            if m_sparse is not None:
                entry["vs_sparse"] = m_sparse
            rec["methods"][key] = entry

        m_sparse, m_dense = _get_topk_snapmla(rows, has_fp8)
        rec["snapmla_cos_dense"] = m_dense["cosine"]
        rec["snapmla_cos_sparse"] = m_sparse["cosine"] if m_sparse else None

        ratio = topk / s_kv
        thr = 0.50 if ratio >= 0.25 else 0.15
        rec["passed"] = m_dense["cosine"] > thr and not math.isnan(m_dense["nrmse"])
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_snapmla_beats_naive(verbose):
    """Test 7: SnapMLA beats naive FP8."""
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    if not has_fp8:
        return {"skipped": True, "reason": "no torch.float8_e4m3fn"}

    samples = list(_one_per_model(mode='decode', s_kv=4096))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    records = []
    for info in samples:
        sample = load_sample(info['path'])
        rows, lse_diff, _ = _run_real_attention(sample, kernel_mode=None)

        rec = {
            "model": info['model'],
            "s_kv": info['s_kv'],
            "methods": {},
        }
        for name, m in rows:
            key = name.lower().replace(" ", "_")
            rec["methods"][key] = m

        m_snap = rows[2][1]
        m_naive = rows[1][1]
        snap_better = m_snap["cosine"] >= m_naive["cosine"] and m_snap["nrmse"] <= m_naive["nrmse"]
        rec["snapmla_cosine"] = m_snap["cosine"]
        rec["naive_cosine"] = m_naive["cosine"]
        rec["snapmla_nrmse"] = m_snap["nrmse"]
        rec["naive_nrmse"] = m_naive["nrmse"]
        rec["snapmla_wins"] = snap_better
        rec["passed"] = snap_better
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def _bench_prefill_scaling(model_filter, kernel_mode, verbose):
    """Shared logic for tests 8-9: prefill scaling."""
    model = list(model_filter)[0] if len(model_filter) == 1 else None
    samples = list(list_samples(mode='prefill', model=model))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    records = []
    for info in samples:
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        rows, lse_diff, has_fp8 = _run_real_attention(sample, kernel_mode=kernel_mode)

        rec = {
            "model": info['model'],
            "s_q": info['s_q'],
            "s_kv": s_kv,
            "lse_diff": lse_diff,
            "methods": {},
        }
        for name, m in rows:
            key = name.lower().replace(" ", "_")
            rec["methods"][key] = m

        m = _get_snapmla_row(rows, has_fp8)
        rec["snapmla_cosine"] = m["cosine"]
        rec["snapmla_nrmse"] = m["nrmse"]
        rec["passed"] = m["cosine"] > 0.995 and m["nrmse"] < 0.08
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_dense_prefill(verbose):
    """Test 8: Dense prefill (Kimi)."""
    return _bench_prefill_scaling(DENSE_MODELS, "dense", verbose)


def bench_sparse_prefill(verbose):
    """Test 9: Sparse prefill (DeepSeek)."""
    return _bench_prefill_scaling(SPARSE_MODELS, "sparse", verbose)


def bench_sparse_topk_prefill(verbose):
    """Test 10: Sparse topk prefill."""
    samples = list(list_samples(mode='prefill', model='deepseek_v32'))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    records = []
    for info in samples:
        sample = load_sample(info['path'])
        s_kv = info['s_kv']
        rows, topk, has_fp8 = _run_real_topk_attention(sample, kernel_mode="sparse")

        rec = {
            "model": info['model'],
            "s_q": info['s_q'],
            "s_kv": s_kv,
            "topk": topk,
            "topk_ratio": topk / s_kv,
            "methods": {},
        }
        for name, m_sparse, m_dense in rows:
            key = name.lower().replace(" ", "_")
            entry = {"vs_dense": m_dense}
            if m_sparse is not None:
                entry["vs_sparse"] = m_sparse
            rec["methods"][key] = entry

        m_sparse, m_dense = _get_topk_snapmla(rows, has_fp8)
        rec["snapmla_cos_dense"] = m_dense["cosine"]
        rec["snapmla_cos_sparse"] = m_sparse["cosine"] if m_sparse else None

        ratio = topk / s_kv
        thr = 0.50 if ratio >= 0.25 else 0.15
        rec["passed"] = m_dense["cosine"] > thr and not math.isnan(m_dense["nrmse"])
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_chunked_prefill(verbose):
    """Test 11: Chunked prefill LSE merge consistency."""
    samples = list(list_samples(mode='prefill'))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    chunk_size = 512
    sm_scale = 1.0 / math.sqrt(D_QK)
    records = []

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']
        s_q = sample['s_q']

        k = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v = c_kv.unsqueeze(1)

        out_full, lse_full = ref_mla_attention_bf16(q, k, v, sm_scale)

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

        rec = {
            "model": info['model'],
            "s_q": s_q,
            "s_kv": s_kv,
            "chunk_size": chunk_size,
            "out_max_diff": max_diff,
            "lse_max_diff": lse_diff,
            "passed": max_diff < 1e-4 and lse_diff < 1e-4,
        }
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_sparse_vs_dense(verbose):
    """Test 12: Sparse vs Dense identity (topk=all)."""
    samples = list(_one_per_model(mode='decode', s_kv=256))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    sm_scale = 1.0 / math.sqrt(D_QK)
    records = []

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']

        k = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v = c_kv.unsqueeze(1)

        out_dense, _ = ref_mla_attention_bf16(q, k, v, sm_scale)

        indices = torch.arange(s_kv)
        out_sparse, _ = ref_mla_attention_bf16(q, k[indices], v[indices], sm_scale)
        id_diff = (out_dense.float() - out_sparse.float()).abs().max().item()

        torch.manual_seed(42)
        perm = torch.randperm(s_kv)
        out_perm, _ = ref_mla_attention_bf16(q, k[perm], v[perm], sm_scale)
        perm_diff = (out_dense.float() - out_perm.float()).abs().max().item()

        rec = {
            "model": info['model'],
            "s_kv": s_kv,
            "identity_max_diff": id_diff,
            "permuted_max_diff": perm_diff,
            "passed": id_diff < 1e-6 and perm_diff < 1e-5,
        }
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_sparse_topk_quality(verbose):
    """Test 13: Sparse topk vs Dense quality."""
    samples = list(_one_per_model(mode='decode', s_kv=1024, model_filter=SPARSE_MODELS))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    sm_scale = 1.0 / math.sqrt(D_QK)
    records = []

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']
        topk = _topk_size(s_kv)

        k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v_mla = c_kv.unsqueeze(1)

        out_dense, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

        idx = _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk)
        idx_0 = idx[0]
        out_topk, _ = ref_mla_attention_bf16(q, k_mla[idx_0], v_mla[idx_0], sm_scale)
        m_oracle = _compute_metrics(out_dense, out_topk)

        torch.manual_seed(42)
        rand_idx = torch.randperm(s_kv)[:topk].sort().values
        out_rand, _ = ref_mla_attention_bf16(q, k_mla[rand_idx], v_mla[rand_idx], sm_scale)
        m_random = _compute_metrics(out_dense, out_rand)

        rec = {
            "model": info['model'],
            "s_kv": s_kv,
            "topk": topk,
            "topk_ratio": topk / s_kv,
            "oracle": m_oracle,
            "random": m_random,
            "passed": m_oracle["cosine"] > 0.50,
        }

        if 'k_index' in sample:
            nsa_idx = _nsa_topk_indices(
                sample['q_index'].float(), sample['k_index'].float(),
                sample['importance'].float(), topk)
            nsa_idx_0 = nsa_idx[0]
            out_nsa, _ = ref_mla_attention_bf16(q, k_mla[nsa_idx_0], v_mla[nsa_idx_0], sm_scale)
            rec["nsa"] = _compute_metrics(out_dense, out_nsa)

        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


def bench_fp8_cache_roundtrip_prefill(verbose):
    """Test 14: FP8 cache roundtrip prefill."""
    if not HAS_KERNELS:
        return {"skipped": True, "reason": "no GPU kernels"}
    if not hasattr(sm120_mla_kernels, 'sparse_prefill_v32'):
        return {"skipped": True, "reason": "no sparse_prefill_v32"}

    from test_snapmla_reference import _alloc_paged_cache

    samples = list(list_samples(mode='prefill', model='deepseek_v32'))
    if not samples:
        return {"skipped": True, "reason": "no sample data"}

    sm_scale = 1.0 / math.sqrt(D_QK)
    records = []

    for info in samples:
        sample = load_sample(info['path'])
        q = sample['q'].float()
        c_kv = sample['c_kv'].float()
        k_rope = sample['k_rope'].float()
        s_kv = sample['s_kv']
        s_q = sample['s_q']

        k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
        v_mla = c_kv.unsqueeze(1)
        out_bf16, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

        # Path A: Direct BF16 prefill
        q_gpu = q.to(torch.bfloat16).cuda()
        kv_direct = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1).to(torch.bfloat16).cuda()
        idx_all = torch.arange(s_kv, dtype=torch.int32).unsqueeze(0).unsqueeze(0)
        idx_all = idx_all.expand(s_q, 1, -1).contiguous().cuda()
        out_direct, _ = sm120_mla_kernels.sparse_prefill_v32(
            q_gpu, kv_direct, idx_all, sm_scale, s_kv)
        out_direct = out_direct.float().cpu()

        # Path B: FP8 cache roundtrip
        c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
        k_rope_gpu = k_rope.to(torch.bfloat16).cuda()
        n_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
        kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
        slot_mapping = torch.arange(s_kv, dtype=torch.int32, device="cuda")
        sm120_mla_kernels.fused_k_append(
            c_kv_gpu, k_rope_gpu, kv_cache, slot_mapping, D_C, D_ROPE, PAGE_SIZE)

        fetch_indices = torch.arange(s_kv, dtype=torch.int32, device="cuda")
        kv_dequant = sm120_mla_kernels.dequant_ckv_indexed(
            kv_cache, fetch_indices, D_C, D_ROPE, PAGE_SIZE)

        # dequant_ckv_indexed now un-scales ROPE, so output is directly usable
        kv_recovered = kv_dequant.unsqueeze(1).cuda()

        out_roundtrip, _ = sm120_mla_kernels.sparse_prefill_v32(
            q_gpu, kv_recovered, idx_all, sm_scale, s_kv)
        out_roundtrip = out_roundtrip.float().cpu()

        m_direct = _compute_metrics(out_bf16, out_direct)
        m_roundtrip = _compute_metrics(out_bf16, out_roundtrip)

        rec = {
            "model": info['model'],
            "s_q": s_q,
            "s_kv": s_kv,
            "direct_bf16": m_direct,
            "fp8_roundtrip": m_roundtrip,
            "passed": m_roundtrip["cosine"] > 0.99 and m_roundtrip["nrmse"] < 0.10,
        }
        records.append(rec)

    return {"records": records, "passed": all(r["passed"] for r in records)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("01_q_quant_roundtrip", "fused_q_quant round-trip", bench_q_quant_roundtrip),
    ("02_k_append_roundtrip", "fused_k_append + dequant round-trip", bench_k_append_roundtrip),
    ("03_decode_short", "Decode attention (s_kv=256)", bench_decode_short),
    ("04_dense_decode_scaling", "Dense decode scaling (1K-32K)", bench_dense_decode_scaling),
    ("05_sparse_decode_scaling", "Sparse decode scaling (1K-32K)", bench_sparse_decode_scaling),
    ("06_sparse_topk_decode_scaling", "Sparse topk decode scaling", bench_sparse_topk_decode_scaling),
    ("07_snapmla_beats_naive", "SnapMLA beats naive FP8", bench_snapmla_beats_naive),
    ("08_dense_prefill", "Dense prefill (Kimi)", bench_dense_prefill),
    ("09_sparse_prefill", "Sparse prefill (DeepSeek)", bench_sparse_prefill),
    ("10_sparse_topk_prefill", "Sparse topk prefill", bench_sparse_topk_prefill),
    ("11_chunked_prefill", "Chunked prefill LSE merge", bench_chunked_prefill),
    ("12_sparse_vs_dense", "Sparse vs Dense identity", bench_sparse_vs_dense),
    ("13_sparse_topk_quality", "Sparse topk quality", bench_sparse_topk_quality),
    ("14_fp8_cache_roundtrip_prefill", "FP8 cache roundtrip prefill", bench_fp8_cache_roundtrip_prefill),
]


def main():
    parser = argparse.ArgumentParser(description='SnapMLA real-data benchmark (JSON output)')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT,
                        help=f'Output JSON path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print progress and summary to stdout')
    args = parser.parse_args()

    if not has_samples():
        print("No sample data found. Generate it first:")
        print("  python sample-data/generate_samples.py")
        return 1

    results = {
        "env": _collect_env(),
        "tests": {},
        "summary": {},
    }

    n_pass = 0
    n_total = 0
    t_total_start = time.time()

    for key, desc, fn in BENCHMARKS:
        if args.verbose:
            print(f"  [{key}] {desc} ... ", end="", flush=True)

        t0 = time.time()
        data = fn(verbose=False)  # never print tables; JSON captures everything
        elapsed = time.time() - t0

        data["elapsed_s"] = round(elapsed, 3)
        results["tests"][key] = data

        skipped = data.get("skipped", False)
        passed = data.get("passed", False)
        if not skipped:
            n_total += 1
            if passed:
                n_pass += 1

        if args.verbose:
            if skipped:
                print(f"SKIP ({data.get('reason', '?')})")
            else:
                status = "PASS" if passed else "FAIL"
                print(f"{status}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_total_start
    results["summary"] = {
        "passed": n_pass,
        "total": n_total,
        "skipped": sum(1 for k in results["tests"] if results["tests"][k].get("skipped")),
        "all_passed": n_pass == n_total,
        "elapsed_s": round(total_elapsed, 3),
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"Results: {n_pass}/{n_total} passed, "
              f"{results['summary']['skipped']} skipped  ({total_elapsed:.1f}s)")
        print(f"Output:  {args.output}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    exit(main())
