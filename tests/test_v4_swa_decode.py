"""
V4K-5a: SWA-Only Decode Kernel Tests

Pure sliding-window attention (no compression). For compress_ratios=0
layers and MTP. Reuses CSA decode kernel with topk=0.

Usage:
  python tests/test_v4_swa_decode.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE,
    alloc_v4_fp8_cache, ref_v4_fp8_k_append, ref_v4_fp8_dequant,
    ref_swa_decode,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def populate_v4_cache(num_entries, page_size=PAGE_SIZE):
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


def run_swa_decode(q_nope, q_rope, swa_cache, swa_block_table, swa_seqlens,
                   sm_scale, swa_pbs, num_sm_parts=1):
    import sm120_mla_kernels
    out, lse = sm120_mla_kernels.v4_swa_decode(
        q_nope.cuda().contiguous(),
        q_rope.cuda().contiguous(),
        swa_cache.cuda().contiguous(),
        swa_block_table.cuda().to(torch.int32).contiguous(),
        swa_seqlens.cuda().to(torch.int32).contiguous(),
        sm_scale, swa_pbs, num_sm_parts,
    )
    return out.cpu(), lse.cpu()


class TestSwaDecodeSmoke(unittest.TestCase):
    """V4K-5a: Smoke test."""

    def test_small_no_crash(self):
        """64 SWA tokens — no crash, no NaN."""
        torch.manual_seed(42)
        h_q = 64
        swa_len = 64
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        swa_cache, _, _, _ = populate_v4_cache(swa_len)
        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        swa_block_table = torch.zeros(1, 1, dtype=torch.int32)
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        out, lse = run_swa_decode(q_nope, q_rope, swa_cache, swa_block_table,
                                  swa_seqlens, sm_scale, PAGE_SIZE)
        self.assertEqual(out.shape, (1, 1, h_q, HEAD_DIM))
        self.assertFalse(torch.isnan(out).any())

    def test_full_window(self):
        """128 tokens (full sliding window) — no crash."""
        torch.manual_seed(43)
        h_q = 64
        swa_len = 128
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        swa_cache, _, _, _ = populate_v4_cache(swa_len)
        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        swa_block_table = torch.zeros(1, 2, dtype=torch.int32)
        swa_block_table[0, 0] = 0
        swa_block_table[0, 1] = 1
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        out, lse = run_swa_decode(q_nope, q_rope, swa_cache, swa_block_table,
                                  swa_seqlens, sm_scale, PAGE_SIZE)
        self.assertFalse(torch.isnan(out).any())


class TestSwaDecodeAccuracy(unittest.TestCase):
    """V4K-5a: Accuracy test — vs ref_swa_decode()."""

    def _run_accuracy(self, swa_len, h_q=64, seed=100):
        torch.manual_seed(seed)
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        swa_cache, swa_k_nope, swa_k_rope, swa_v = populate_v4_cache(swa_len)
        q_nope = torch.randn(1, 1, h_q, HEAD_DIM, dtype=torch.bfloat16)
        q_rope = torch.randn(1, 1, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)

        num_pages = (swa_len + PAGE_SIZE - 1) // PAGE_SIZE
        swa_block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32)

        out_k, lse_k = run_swa_decode(q_nope, q_rope, swa_cache, swa_block_table,
                                      swa_seqlens, sm_scale, PAGE_SIZE)

        out_r, lse_r = ref_swa_decode(
            q_nope[0, 0].float(), q_rope[0, 0].float(),
            swa_k_nope[:swa_len].float(), swa_k_rope[:swa_len].float(),
            swa_v[:swa_len].float(),
            sm_scale, h_q,
        )

        return out_k[0, 0], out_r, lse_k[0, 0], lse_r

    def test_64_tokens(self):
        """64 SWA tokens — cosine > 0.99."""
        out_k, out_r, _, _ = self._run_accuracy(64)
        cos = cosine_sim(out_k, out_r)
        print(f"  64-token: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_128_tokens(self):
        """128 tokens (full window) — cosine > 0.99."""
        out_k, out_r, _, _ = self._run_accuracy(128, seed=101)
        cos = cosine_sim(out_k, out_r)
        print(f"  128-token: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_1_token(self):
        """1 SWA token — edge case."""
        out_k, out_r, _, _ = self._run_accuracy(1, seed=102)
        cos = cosine_sim(out_k, out_r)
        print(f"  1-token: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_h128(self):
        """h_q=128 (V4 Pro)."""
        out_k, out_r, _, _ = self._run_accuracy(64, h_q=128, seed=103)
        cos = cosine_sim(out_k, out_r)
        print(f"  h128: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_window_boundary(self):
        """Exactly at page boundary (64 tokens = 1 full page)."""
        out_k, out_r, _, _ = self._run_accuracy(PAGE_SIZE, seed=104)
        cos = cosine_sim(out_k, out_r)
        print(f"  page-boundary: cosine = {cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")


if __name__ == '__main__':
    unittest.main()
