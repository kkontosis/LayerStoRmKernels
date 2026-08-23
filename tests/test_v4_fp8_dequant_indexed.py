"""
V4K-5d: V4 FP8 Dequant Indexed Kernel Tests

Accuracy test: write cache via k_append kernel, then gather+dequant via this
kernel, compare against reference round-trip. Cosine > 0.999.

Usage:
  python tests/test_v4_fp8_dequant_indexed.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM,
    alloc_v4_fp8_cache, ref_v4_fp8_k_append, ref_v4_fp8_dequant,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def populate_cache_gpu(k_nope, k_rope, v_nope, slot_mapping, num_pages=1):
    """Write entries to GPU cache via the k_append kernel, return GPU cache."""
    import sm120_mla_kernels

    cache = alloc_v4_fp8_cache(num_pages, device='cpu')
    k_gpu = k_nope.to(device='cuda', dtype=torch.bfloat16).contiguous()
    r_gpu = k_rope.to(device='cuda', dtype=torch.bfloat16).contiguous()
    v_gpu = v_nope.to(device='cuda', dtype=torch.bfloat16).contiguous()
    c_gpu = cache.cuda().contiguous()
    s_gpu = slot_mapping.to(device='cuda', dtype=torch.int32).contiguous()

    sm120_mla_kernels.v4_fp8_k_append(k_gpu, r_gpu, v_gpu, c_gpu, s_gpu)
    return c_gpu


def run_dequant_kernel(cache_gpu, indices):
    """Run the CUDA v4_fp8_dequant_indexed kernel."""
    import sm120_mla_kernels

    idx_gpu = indices.to(device='cuda', dtype=torch.int32).contiguous()
    k_n, k_r, v_n = sm120_mla_kernels.v4_fp8_dequant_indexed(
        cache_gpu, idx_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)
    return k_n.cpu(), k_r.cpu(), v_n.cpu()


class TestV4Fp8DequantSmoke(unittest.TestCase):
    """V4K-5d.1: Smoke test — dequant runs without crash."""

    def test_smoke_basic(self):
        """Write 8 tokens, read back all 8."""
        torch.manual_seed(42)
        n = 8
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)
        slots = torch.arange(n, dtype=torch.int32)

        cache = populate_cache_gpu(k_nope, k_rope, v_nope, slots)
        k_n, k_r, v_n = run_dequant_kernel(cache, slots)

        self.assertEqual(k_n.shape, (n, HEAD_DIM))
        self.assertEqual(k_r.shape, (n, QK_ROPE_HEAD_DIM))
        self.assertEqual(v_n.shape, (n, HEAD_DIM))
        self.assertTrue(torch.isfinite(k_n).all())

    def test_smoke_subset(self):
        """Write 16 tokens, read back 4 (non-sequential)."""
        torch.manual_seed(55)
        n = 16
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)
        slots = torch.arange(n, dtype=torch.int32)

        cache = populate_cache_gpu(k_nope, k_rope, v_nope, slots)
        fetch_indices = torch.tensor([3, 7, 0, 15], dtype=torch.int32)
        k_n, k_r, v_n = run_dequant_kernel(cache, fetch_indices)

        self.assertEqual(k_n.shape, (4, HEAD_DIM))


class TestV4Fp8DequantAccuracy(unittest.TestCase):
    """V4K-5d.2: Accuracy test — round-trip cosine > 0.999."""

    def test_roundtrip(self):
        """k_append → dequant round-trip for K NOPE, K ROPE, V NOPE."""
        torch.manual_seed(42)
        n = 16
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)
        slots = torch.arange(n, dtype=torch.int32)

        cache = populate_cache_gpu(k_nope, k_rope, v_nope, slots)
        k_n, k_r, v_n = run_dequant_kernel(cache, slots)

        for i in range(n):
            cos_k = cosine_sim(k_nope[i], k_n[i])
            self.assertGreater(cos_k, 0.999,
                f"Token {i} K NOPE cosine {cos_k:.6f}")

            k_rope_bf16 = k_rope[i].to(torch.bfloat16).float()
            diff_r = (k_r[i].float() - k_rope_bf16).abs().max().item()
            self.assertLess(diff_r, 1e-3,
                f"Token {i} K ROPE diff {diff_r:.2e}")

            cos_v = cosine_sim(v_nope[i], v_n[i])
            self.assertGreater(cos_v, 0.999,
                f"Token {i} V NOPE cosine {cos_v:.6f}")

    def test_matches_reference(self):
        """Kernel dequant matches ref_v4_fp8_dequant."""
        torch.manual_seed(77)
        n = 8
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)
        slots = torch.arange(n, dtype=torch.int32)

        # Reference path: ref write + ref read
        ref_cache = alloc_v4_fp8_cache(1)
        ref_v4_fp8_k_append(k_nope, k_rope, v_nope, ref_cache, slots)
        ref_kn, ref_kr, ref_vn = ref_v4_fp8_dequant(ref_cache, list(range(n)))

        # Kernel path: kernel write + kernel read
        gpu_cache = populate_cache_gpu(k_nope, k_rope, v_nope, slots)
        k_n, k_r, v_n = run_dequant_kernel(gpu_cache, slots)

        # Compare kernel dequant vs reference dequant
        cos_kn = cosine_sim(ref_kn, k_n)
        self.assertGreater(cos_kn, 0.999,
            f"K NOPE kernel vs ref cosine {cos_kn:.6f}")

        diff_kr = (k_r.float() - ref_kr.float()).abs().max().item()
        self.assertLess(diff_kr, 1e-3,
            f"K ROPE kernel vs ref diff {diff_kr:.2e}")

        cos_vn = cosine_sim(ref_vn, v_n)
        self.assertGreater(cos_vn, 0.999,
            f"V NOPE kernel vs ref cosine {cos_vn:.6f}")

    def test_gather_subset(self):
        """Gather non-sequential indices, verify correct tokens fetched."""
        torch.manual_seed(33)
        n = 32
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)
        slots = torch.arange(n, dtype=torch.int32)

        cache = populate_cache_gpu(k_nope, k_rope, v_nope, slots)

        fetch = [5, 10, 20, 31]
        k_n, k_r, v_n = run_dequant_kernel(
            cache, torch.tensor(fetch, dtype=torch.int32))

        for out_i, slot in enumerate(fetch):
            cos_k = cosine_sim(k_nope[slot], k_n[out_i])
            self.assertGreater(cos_k, 0.999,
                f"Gather slot {slot} K NOPE cosine {cos_k:.6f}")
            cos_v = cosine_sim(v_nope[slot], v_n[out_i])
            self.assertGreater(cos_v, 0.999,
                f"Gather slot {slot} V NOPE cosine {cos_v:.6f}")


if __name__ == "__main__":
    unittest.main()
