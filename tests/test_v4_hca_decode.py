"""
V4K-4a: HCA FP8 Decode Kernel Tests

Dense attention over ALL compressed blocks + SWA + LSE combine.
Reuses CSA decode kernel with topk=num_compressed (no sparse selection).

Usage:
  python tests/test_v4_hca_decode.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE, FP8_MAX,
    V4_FP8_BYTES_PER_ENTRY,
    alloc_v4_fp8_cache, ref_v4_fp8_k_append, ref_v4_fp8_dequant,
    ref_hca_fp8_decode,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def populate_v4_cache(num_entries, page_size=PAGE_SIZE):
    """Create V4 FP8 cache with random data, return GPU cache + dequanted CPU tensors."""
    num_pages = max((num_entries + page_size - 1) // page_size, 1)
    cache = alloc_v4_fp8_cache(num_pages, page_size)

    k_nope = torch.randn(num_entries, HEAD_DIM) * 0.5
    k_rope = torch.randn(num_entries, QK_ROPE_HEAD_DIM) * 0.3
    v_nope = torch.randn(num_entries, HEAD_DIM) * 0.5

    slot_mapping = torch.arange(num_entries)
    ref_v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot_mapping)

    indices = torch.arange(num_entries)
    k_nope_deq, k_rope_deq, v_nope_deq = ref_v4_fp8_dequant(cache, indices)

    return cache.to('cuda'), k_nope_deq, k_rope_deq, v_nope_deq


def run_hca_decode(q_nope, q_rope, compressed_cache, num_compressed,
                   swa_cache, swa_block_table, swa_seqlens,
                   sm_scale, compressed_pbs, swa_pbs, num_sm_parts=1):
    """Run the CUDA v4_hca_fp8_decode kernel."""
    import sm120_mla_kernels

    out, lse = sm120_mla_kernels.v4_hca_fp8_decode(
        q_nope.cuda().contiguous(),
        q_rope.cuda().contiguous(),
        compressed_cache.cuda().contiguous(),
        num_compressed,
        swa_cache.cuda().contiguous(),
        swa_block_table.cuda().to(torch.int32).contiguous(),
        swa_seqlens.cuda().to(torch.int32).contiguous(),
        sm_scale,
        compressed_pbs, swa_pbs,
        num_sm_parts,
    )
    return out.cpu(), lse.cpu()


class TestHcaFp8DecodeSmoke(unittest.TestCase):
    """V4K-4a: Smoke test — kernel runs without crash."""

    def test_small_no_crash(self):
        """64 compressed + 64 SWA — no crash, no NaN."""
        torch.manual_seed(42)
        h_q = 64
        num_compressed = 64
        swa_len = 64
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)
        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        swa_block_table = torch.zeros(1, 1, dtype=torch.int32)
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        out, lse = run_hca_decode(
            q_nope, q_rope, comp_cache, num_compressed,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE,
        )
        self.assertEqual(out.shape, (1, 1, h_q, HEAD_DIM))
        self.assertFalse(torch.isnan(out).any(), "NaN in output")

    def test_no_swa(self):
        """Compressed-only (no SWA) — shouldn't crash."""
        torch.manual_seed(43)
        h_q = 64
        num_compressed = 128
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)
        swa_cache, _, _, _ = populate_v4_cache(1)

        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        swa_block_table = torch.zeros(1, 1, dtype=torch.int32)
        swa_seqlens = torch.tensor([0], dtype=torch.int32)

        out, lse = run_hca_decode(
            q_nope, q_rope, comp_cache, num_compressed,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE,
        )
        self.assertFalse(torch.isnan(out).any())


class TestHcaFp8DecodeAccuracy(unittest.TestCase):
    """V4K-4a: Accuracy test — kernel vs ref_hca_fp8_decode()."""

    def _run_accuracy(self, num_compressed, swa_len, h_q=64, seed=100):
        torch.manual_seed(seed)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, k_nope, k_rope, v_nope = populate_v4_cache(num_compressed)
        swa_cache, swa_k_nope, swa_k_rope, swa_v = populate_v4_cache(max(swa_len, 1))

        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)

        swa_block_table = torch.zeros(1, max((swa_len + PAGE_SIZE - 1) // PAGE_SIZE, 1), dtype=torch.int32)
        for i in range(swa_block_table.shape[1]):
            swa_block_table[0, i] = i
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        # Kernel
        out_k, lse_k = run_hca_decode(
            q_nope, q_rope, comp_cache, num_compressed,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE,
        )

        # Reference (FP8-quantized values)
        out_r, lse_r = ref_hca_fp8_decode(
            q_nope[0, 0].float(), q_rope[0, 0].float(),
            k_nope.float(), k_rope.float(), v_nope.float(),
            swa_k_nope[:swa_len].float(), swa_k_rope[:swa_len].float(), swa_v[:swa_len].float(),
            sm_scale, h_q,
        )

        return out_k[0, 0], out_r, lse_k[0, 0], lse_r

    def test_small(self):
        """64 compressed + 64 SWA — cosine > 0.99."""
        out_k, out_r, _, _ = self._run_accuracy(64, 64)
        cos = cosine_sim(out_k, out_r)
        print(f"  small: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_medium(self):
        """256 compressed + 128 SWA — cosine > 0.99."""
        out_k, out_r, _, _ = self._run_accuracy(256, 128, seed=101)
        cos = cosine_sim(out_k, out_r)
        print(f"  medium: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_no_swa(self):
        """128 compressed, no SWA — cosine > 0.99."""
        out_k, out_r, _, _ = self._run_accuracy(128, 0, seed=102)
        cos = cosine_sim(out_k, out_r)
        print(f"  no-swa: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_h128(self):
        """h_q=128 (V4 Pro) — cosine > 0.99."""
        out_k, out_r, _, _ = self._run_accuracy(128, 64, h_q=128, seed=103)
        cos = cosine_sim(out_k, out_r)
        print(f"  h128: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_split_kv_consistency(self):
        """Split-KV (num_sm_parts=8) vs single — output should be close."""
        torch.manual_seed(200)
        h_q = 64
        num_compressed = 256
        swa_len = 128
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)
        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        swa_block_table = torch.zeros(1, (swa_len + PAGE_SIZE - 1) // PAGE_SIZE, dtype=torch.int32)
        for i in range(swa_block_table.shape[1]):
            swa_block_table[0, i] = i
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        out_1, _ = run_hca_decode(
            q_nope, q_rope, comp_cache, num_compressed,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE, num_sm_parts=1,
        )
        out_8, _ = run_hca_decode(
            q_nope, q_rope, comp_cache, num_compressed,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE, num_sm_parts=8,
        )

        cos = cosine_sim(out_1, out_8)
        print(f"  split-kv cosine: {cos:.6f}")
        self.assertGreater(cos, 0.99, f"split-KV cosine={cos:.6f}")


if __name__ == '__main__':
    unittest.main()
