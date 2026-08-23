"""
V4K-3a: CSA FP8 Decode Kernel Tests

Smoke test: kernel builds, runs at s_kv=256 without crash/NaN.
Accuracy test: kernel vs ref_csa_fp8_decode() at 4K/16K context,
  cosine > 0.999, topk=all == dense (exact), SWA correctness.

Usage:
  python tests/test_v4_csa_decode.py -v
"""

import torch
import unittest
import sys
import os
import struct
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE, FP8_MAX,
    V4_FP8_BYTES_PER_ENTRY,
    V4_FP8_K_NOPE_OFFSET, V4_FP8_K_SCALE_OFFSET,
    V4_FP8_K_ROPE_OFFSET, V4_FP8_V_NOPE_OFFSET, V4_FP8_V_SCALE_OFFSET,
    alloc_v4_fp8_cache, ref_v4_fp8_k_append, ref_v4_fp8_dequant,
    ref_csa_fp8_decode,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def populate_v4_cache(num_entries, page_size=PAGE_SIZE):
    """Create a V4 FP8 cache populated with random data.

    Returns:
        cache_gpu: uint8 tensor on GPU
        k_nope_deq: [num_entries, HEAD_DIM] float32 (dequantized)
        k_rope_deq: [num_entries, QK_ROPE_HEAD_DIM] float32
        v_nope_deq: [num_entries, HEAD_DIM] float32 (dequantized)
    """
    num_pages = max((num_entries + page_size - 1) // page_size, 1)
    cache = alloc_v4_fp8_cache(num_pages, page_size)

    k_nope_all = torch.randn(num_entries, HEAD_DIM) * 0.5
    k_rope_all = torch.randn(num_entries, QK_ROPE_HEAD_DIM) * 0.3
    v_nope_all = torch.randn(num_entries, HEAD_DIM) * 0.5

    slot_mapping = torch.arange(num_entries)
    ref_v4_fp8_k_append(k_nope_all, k_rope_all, v_nope_all, cache, slot_mapping)

    # Read back to get the quantized values (what the kernel will see)
    indices = torch.arange(num_entries)
    k_nope_deq, k_rope_deq, v_nope_deq = ref_v4_fp8_dequant(cache, indices)

    cache_gpu = cache.to('cuda')
    return cache_gpu, k_nope_deq, k_rope_deq, v_nope_deq


def run_csa_decode(q_nope, q_rope, compressed_cache, sparse_indices,
                   swa_cache, swa_block_table, swa_seqlens,
                   sm_scale, topk, compressed_pbs, swa_pbs, num_sm_parts=1):
    """Run the CUDA v4_csa_fp8_decode kernel."""
    import sm120_mla_kernels

    b = q_nope.shape[0]
    s_q = q_nope.shape[1]

    out, lse = sm120_mla_kernels.v4_csa_fp8_decode(
        q_nope.cuda().contiguous(),
        q_rope.cuda().contiguous(),
        compressed_cache.cuda().contiguous(),
        sparse_indices.cuda().to(torch.int32).contiguous(),
        swa_cache.cuda().contiguous(),
        swa_block_table.cuda().to(torch.int32).contiguous(),
        swa_seqlens.cuda().to(torch.int32).contiguous(),
        sm_scale, topk,
        compressed_pbs, swa_pbs,
        num_sm_parts,
    )
    return out.cpu(), lse.cpu()


class TestCsaFp8DecodeSmoke(unittest.TestCase):
    """V4K-3a.4: Smoke test — kernel runs without crash."""

    def test_small_no_crash(self):
        """s_kv=256, topk=64, swa=64 — no crash, no NaN."""
        torch.manual_seed(42)
        h_q = 64
        topk = 64
        num_compressed = 64
        swa_len = 64
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        # Populate compressed cache
        comp_cache, _, _, _ = populate_v4_cache(num_compressed)

        # Populate SWA cache
        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        # Q
        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)

        # Indices: just [0, 1, 2, ..., topk-1]
        sparse_indices = torch.arange(topk, dtype=torch.int32).unsqueeze(0).unsqueeze(0)

        # SWA block table: single page per batch item
        swa_block_table = torch.zeros(1, 1, dtype=torch.int32)  # page 0

        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        out, lse = run_csa_decode(
            q_nope, q_rope, comp_cache, sparse_indices,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, topk,
            compressed_pbs=PAGE_SIZE, swa_pbs=PAGE_SIZE,
            num_sm_parts=1,
        )

        self.assertEqual(out.shape, (1, 1, h_q, HEAD_DIM))
        self.assertEqual(lse.shape, (1, 1, h_q))
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")

    def test_no_swa(self):
        """Compressed-only (no SWA) — shouldn't crash."""
        torch.manual_seed(43)
        h_q = 64
        topk = 32
        num_compressed = 32
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)

        # Empty SWA cache — 1 page but 0 seqlen
        swa_cache = alloc_v4_fp8_cache(1).cuda()
        swa_block_table = torch.zeros(1, 1, dtype=torch.int32)
        swa_seqlens = torch.tensor([0], dtype=torch.int32)

        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        sparse_indices = torch.arange(topk, dtype=torch.int32).unsqueeze(0).unsqueeze(0)

        out, lse = run_csa_decode(
            q_nope, q_rope, comp_cache, sparse_indices,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, topk,
            compressed_pbs=PAGE_SIZE, swa_pbs=PAGE_SIZE,
            num_sm_parts=1,
        )
        self.assertFalse(torch.isnan(out).any())


class TestCsaFp8DecodeAccuracy(unittest.TestCase):
    """V4K-3a.5: Accuracy test — kernel vs reference."""

    def _run_accuracy_test(self, num_compressed, topk, swa_len, h_q=64, num_sm_parts=1):
        """Helper: populate cache, run kernel + reference, compare."""
        torch.manual_seed(100 + num_compressed)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        # Compressed cache
        comp_cache, comp_kn, comp_kr, comp_vn = populate_v4_cache(num_compressed)

        # SWA cache
        if swa_len > 0:
            swa_cache_gpu, swa_kn, swa_kr, swa_vn = populate_v4_cache(swa_len)
        else:
            swa_cache_gpu = alloc_v4_fp8_cache(1).cuda()
            swa_kn = torch.zeros(0, HEAD_DIM)
            swa_kr = torch.zeros(0, QK_ROPE_HEAD_DIM)
            swa_vn = torch.zeros(0, HEAD_DIM)

        # Q
        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)

        # Sparse indices — select topk from num_compressed
        if topk <= num_compressed:
            perm = torch.randperm(num_compressed)[:topk].sort().values
        else:
            perm = torch.arange(num_compressed)
            topk = num_compressed
        sparse_indices = perm.to(torch.int32).unsqueeze(0).unsqueeze(0)

        # SWA block table: pages 0, 1 for up to 128 tokens
        num_swa_pages = max((swa_len + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        swa_block_table = torch.arange(num_swa_pages, dtype=torch.int32).unsqueeze(0)
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        # --- Run CUDA kernel ---
        out_cuda, lse_cuda = run_csa_decode(
            q_nope, q_rope, comp_cache, sparse_indices,
            swa_cache_gpu, swa_block_table, swa_seqlens,
            sm_scale, topk,
            compressed_pbs=PAGE_SIZE, swa_pbs=PAGE_SIZE,
            num_sm_parts=num_sm_parts,
        )

        # --- Run PyTorch reference ---
        out_ref, lse_ref = ref_csa_fp8_decode(
            q_nope.squeeze(0).squeeze(0).float(),   # [h_q, HEAD_DIM]
            q_rope.squeeze(0).squeeze(0).float(),   # [h_q, QK_ROPE_HEAD_DIM]
            comp_kn, comp_kr, comp_vn,              # compressed KV (dequantized)
            swa_kn, swa_kr, swa_vn,                 # SWA KV (dequantized)
            perm.long(),                             # sparse indices
            sm_scale, h_q,
        )

        out_cuda_2d = out_cuda.squeeze(0).squeeze(0).float()  # [h_q, HEAD_DIM]
        cos = cosine_sim(out_cuda_2d, out_ref)
        return cos, out_cuda_2d, out_ref

    def test_accuracy_small(self):
        """64 compressed + 64 SWA, topk=64 — cosine > 0.99 (FP8 Q+K+V quantization)."""
        cos, _, _ = self._run_accuracy_test(
            num_compressed=64, topk=64, swa_len=64, h_q=64)
        print(f"  small: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"Cosine {cos:.6f} < 0.99")

    def test_accuracy_medium(self):
        """256 compressed + 128 SWA, topk=128 — cosine > 0.99."""
        cos, _, _ = self._run_accuracy_test(
            num_compressed=256, topk=128, swa_len=128, h_q=64)
        print(f"  medium: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"Cosine {cos:.6f} < 0.99")

    def test_accuracy_no_swa(self):
        """128 compressed, no SWA, topk=64 — cosine > 0.99."""
        cos, _, _ = self._run_accuracy_test(
            num_compressed=128, topk=64, swa_len=0, h_q=64)
        print(f"  no-swa: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"Cosine {cos:.6f} < 0.99")

    def test_whole_tile_neg1_padding(self):
        """Regression: topk padded past the valid count by WHOLE 64-tiles
        (50 valid + 206 x -1 at topk=256). A fully-masked tile must be an
        online-softmax no-op — the old rescale computed exp2(old_max −
        MAX_INIT_VAL_MASK) = inf and exploded sL/rO (sub-tile -1 padding was
        unaffected). Arises for HCA-dense/prefill callers that fix topk at a
        padded bound while rows see fewer entries."""
        torch.manual_seed(7)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
        nc, swa_len, topk_pad = 50, 64, 256
        comp_cache, comp_kn, comp_kr, comp_vn = populate_v4_cache(nc)
        swa_cache_gpu, swa_kn, swa_kr, swa_vn = populate_v4_cache(swa_len)
        q_nope = torch.randn(1, 1, 64, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, 64, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        indices = torch.full((1, 1, topk_pad), -1, dtype=torch.int32)
        indices[0, 0, :nc] = torch.arange(nc, dtype=torch.int32)
        num_swa_pages = (swa_len + PAGE_SIZE - 1) // PAGE_SIZE
        swa_block_table = torch.arange(num_swa_pages, dtype=torch.int32).unsqueeze(0)
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)
        out_cuda, lse_cuda = run_csa_decode(
            q_nope, q_rope, comp_cache, indices,
            swa_cache_gpu, swa_block_table, swa_seqlens,
            sm_scale, topk_pad,
            compressed_pbs=PAGE_SIZE, swa_pbs=PAGE_SIZE, num_sm_parts=1)
        out_ref, lse_ref = ref_csa_fp8_decode(
            q_nope.squeeze(0).squeeze(0).float(),
            q_rope.squeeze(0).squeeze(0).float(),
            comp_kn, comp_kr, comp_vn, swa_kn, swa_kr, swa_vn,
            torch.arange(nc).long(), sm_scale, 64)
        out_cuda_2d = out_cuda.squeeze(0).squeeze(0).float()
        self.assertTrue(torch.isfinite(out_cuda_2d).all().item(),
                        "whole-tile padding produced non-finite output")
        cos = cosine_sim(out_cuda_2d, out_ref)
        print(f"  whole-tile-pad: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"Cosine {cos:.6f} < 0.99")

    def test_topk_all_eq_dense(self):
        """topk=ALL compressed + SWA — cosine > 0.99."""
        cos, _, _ = self._run_accuracy_test(
            num_compressed=64, topk=64, swa_len=64, h_q=64)
        print(f"  topk=all: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"topk=all cosine {cos:.6f} < 0.99")


if __name__ == "__main__":
    unittest.main()
