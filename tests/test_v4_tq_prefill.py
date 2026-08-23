"""
V4K-8e/8f: V4 TQ Prefill Tests

TQ prefill = dequant TQ entries → BF16 K/V → standard CSA prefill.
Composes existing v4_tq_dequant_indexed + v4_csa_fp8_prefill.

Usage:
  python tests/test_v4_tq_prefill.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE,
    V4_TQ_BYTES_PER_ENTRY,
    load_codebook, generate_rotation_matrix,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def tq_prefill(k, q_bf16, tq_cache, indices, Pi, centroids, sm_scale):
    """Compose TQ dequant + CSA FP8 prefill."""
    results = k.v4_tq_dequant_indexed(
        tq_cache, indices, Pi, centroids, HEAD_DIM, QK_ROPE_HEAD_DIM)
    k_nope, k_rope, v_nope = results[0], results[1], results[2]

    k_full = torch.cat([k_nope, k_rope], dim=-1)  # [s_kv, 576]
    out, lse = k.v4_csa_fp8_prefill(q_bf16, k_full, v_nope, sm_scale)
    return out, lse


class TestV4TqPrefill(unittest.TestCase):
    """V4K-8e/8f: TQ prefill tests."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()

    def _populate_tq_cache(self, k_nope, k_rope, v_nope):
        num_tokens = k_nope.shape[0]
        num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        cache_bytes = num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY
        cache = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(
            k_nope.to(torch.bfloat16).cuda(),
            k_rope.to(torch.bfloat16).cuda(),
            v_nope.to(torch.bfloat16).cuda(),
            cache, slot_mapping,
            self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu,
        )
        return cache

    def test_smoke(self):
        """TQ prefill composition runs without crash."""
        torch.manual_seed(42)
        s_q, h_q, s_kv = 4, 8, 16
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(s_kv, HEAD_DIM)
        k_rope = torch.randn(s_kv, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(s_kv, HEAD_DIM)
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        d_qk = HEAD_DIM + QK_ROPE_HEAD_DIM
        q = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(s_kv, dtype=torch.int32, device='cuda')

        out, lse = tq_prefill(self.k, q, cache, indices,
                               self.Pi_gpu, self.centroids_gpu, sm_scale)

        self.assertEqual(out.shape, (s_q, h_q, HEAD_DIM))
        self.assertEqual(lse.shape, (s_q, h_q))
        self.assertTrue(torch.isfinite(out).all())

    def test_vs_fp8_prefill(self):
        """TQ prefill vs FP8 prefill (with same dequanted data)."""
        torch.manual_seed(42)
        s_q, h_q, s_kv = 4, 8, 32
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(s_kv, HEAD_DIM) * 0.3
        k_rope = torch.randn(s_kv, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(s_kv, HEAD_DIM) * 0.3
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        d_qk = HEAD_DIM + QK_ROPE_HEAD_DIM
        q = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(s_kv, dtype=torch.int32, device='cuda')

        # TQ prefill (dequant → prefill)
        out_tq, lse_tq = tq_prefill(self.k, q, cache, indices,
                                      self.Pi_gpu, self.centroids_gpu, sm_scale)

        # Direct prefill with same dequanted data
        results = self.k.v4_tq_dequant_indexed(
            cache, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)
        k_deq, kr_deq, v_deq = results[0], results[1], results[2]
        k_full = torch.cat([k_deq, kr_deq], dim=-1)
        out_direct, lse_direct = self.k.v4_csa_fp8_prefill(q, k_full, v_deq, sm_scale)

        cos = cosine_sim(out_tq, out_direct)
        print(f"  TQ prefill vs direct: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99999, f"cosine={cos}")

    def test_tq_vs_original(self):
        """TQ prefill output vs original (non-quantized) BF16 prefill."""
        torch.manual_seed(77)
        s_q, h_q, s_kv = 4, 8, 16
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(s_kv, HEAD_DIM) * 0.3
        k_rope = torch.randn(s_kv, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(s_kv, HEAD_DIM) * 0.3
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        d_qk = HEAD_DIM + QK_ROPE_HEAD_DIM
        q = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(s_kv, dtype=torch.int32, device='cuda')

        # TQ prefill
        out_tq, _ = tq_prefill(self.k, q, cache, indices,
                                self.Pi_gpu, self.centroids_gpu, sm_scale)

        # Original BF16 prefill (no TQ quantization)
        k_full_orig = torch.cat([
            k_nope.to(torch.bfloat16),
            k_rope.to(torch.bfloat16)
        ], dim=-1).cuda()
        v_orig = v_nope.to(torch.bfloat16).cuda()
        out_orig, _ = self.k.v4_csa_fp8_prefill(q, k_full_orig, v_orig, sm_scale)

        cos = cosine_sim(out_tq, out_orig)
        print(f"  TQ vs original BF16: cosine={cos:.6f}")
        # TQ quantization + dequant round-trip introduces ~1% error
        self.assertGreater(cos, 0.95, f"cosine={cos}")


if __name__ == '__main__':
    unittest.main()
