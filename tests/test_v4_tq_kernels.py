"""
V4K-8a: V4 TQ k_append Kernel Tests

CUDA kernel round-trip accuracy: TQ quantize → dequant via reference,
comparing against original BF16 compressed vectors.

Usage:
  python tests/test_v4_tq_kernels.py -v
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
    alloc_v4_tq_cache, ref_v4_tq_dequant,
    _make_csa_compressed_vectors, _make_hca_compressed_vectors,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


class TestV4TqKAppend(unittest.TestCase):
    """V4K-8a: V4 TQ k_append kernel accuracy tests."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        # Interior boundaries only (15 values)
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()

    def _run_kernel_roundtrip(self, k_nope, k_rope, v_nope):
        """Run kernel k_append then reference dequant, return dequanted tensors."""
        num_tokens = k_nope.shape[0]
        num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE

        # Allocate GPU cache
        cache_bytes = num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY
        cache = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

        # Run CUDA kernel
        self.k.v4_tq_k_append(
            k_nope.to(torch.bfloat16).cuda(),
            k_rope.to(torch.bfloat16).cuda(),
            v_nope.to(torch.bfloat16).cuda(),
            cache, slot_mapping,
            self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu,
        )
        torch.cuda.synchronize()

        # Dequant via reference (CPU)
        cache_cpu = cache.cpu()
        k_deq, k_rope_deq, v_deq = ref_v4_tq_dequant(
            cache_cpu, list(range(num_tokens)), self.Pi, self.centroids)

        return k_deq, k_rope_deq, v_deq

    def test_smoke(self):
        """Kernel builds, runs, does not crash."""
        torch.manual_seed(42)
        n = 4
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)

        k_deq, k_rope_deq, v_deq = self._run_kernel_roundtrip(k_nope, k_rope, v_nope)

        self.assertEqual(k_deq.shape, (n, HEAD_DIM))
        self.assertEqual(k_rope_deq.shape, (n, QK_ROPE_HEAD_DIM))
        self.assertEqual(v_deq.shape, (n, HEAD_DIM))
        self.assertTrue(torch.isfinite(k_deq).all())
        self.assertTrue(torch.isfinite(v_deq).all())

    def test_csa_roundtrip(self):
        """CSA compressed vectors: CUDA kernel vs original, cosine > 0.99."""
        num_tokens = 32
        k_nope, v_nope = _make_csa_compressed_vectors(num_tokens, seed=42)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)

        k_deq, k_rope_deq, v_deq = self._run_kernel_roundtrip(k_nope, k_rope, v_nope)

        k_cos = cosine_sim(k_nope, k_deq)
        v_cos = cosine_sim(v_nope, v_deq)
        print(f"  CSA K cosine: {k_cos:.6f}, V cosine: {v_cos:.6f}")
        self.assertGreater(k_cos, 0.99, f"K cosine={k_cos}")
        self.assertGreater(v_cos, 0.99, f"V cosine={v_cos}")

        # K ROPE should be exact (BF16 round-trip)
        rope_bf16 = k_rope.to(torch.bfloat16).float()
        rope_diff = (rope_bf16 - k_rope_deq).abs().max().item()
        print(f"  K ROPE max diff: {rope_diff:.2e}")
        self.assertLess(rope_diff, 1e-3)

    def test_hca_roundtrip(self):
        """HCA compressed vectors: CUDA kernel vs original, cosine > 0.99."""
        num_tokens = 16
        k_nope, v_nope = _make_hca_compressed_vectors(num_tokens, seed=77)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)

        k_deq, k_rope_deq, v_deq = self._run_kernel_roundtrip(k_nope, k_rope, v_nope)

        k_cos = cosine_sim(k_nope, k_deq)
        v_cos = cosine_sim(v_nope, v_deq)
        print(f"  HCA K cosine: {k_cos:.6f}, V cosine: {v_cos:.6f}")
        self.assertGreater(k_cos, 0.99, f"K cosine={k_cos}")
        self.assertGreater(v_cos, 0.99, f"V cosine={v_cos}")

    def test_kernel_vs_reference(self):
        """CUDA kernel output matches Python reference exactly (same codebook, same cache)."""
        torch.manual_seed(99)
        num_tokens = 8
        k_nope = torch.randn(num_tokens, HEAD_DIM) * 0.5
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM) * 0.5

        # CUDA kernel path
        k_deq_cuda, k_rope_cuda, v_deq_cuda = self._run_kernel_roundtrip(
            k_nope, k_rope, v_nope)

        # Pure reference path
        from test_v4_reference import ref_v4_tq_k_append, alloc_v4_tq_cache
        num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        cache_ref = alloc_v4_tq_cache(num_pages)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
        ref_v4_tq_k_append(k_nope, k_rope, v_nope, cache_ref, slot_mapping,
                            self.Pi, self.centroids, self.boundaries)
        k_deq_ref, k_rope_ref, v_deq_ref = ref_v4_tq_dequant(
            cache_ref, list(range(num_tokens)), self.Pi, self.centroids)

        # Both should dequant to same values (same quantization algorithm)
        k_cos = cosine_sim(k_deq_cuda, k_deq_ref)
        v_cos = cosine_sim(v_deq_cuda, v_deq_ref)
        print(f"  CUDA vs ref: K cosine={k_cos:.6f}, V cosine={v_cos:.6f}")
        self.assertGreater(k_cos, 0.999, f"K cosine={k_cos}")
        self.assertGreater(v_cos, 0.999, f"V cosine={v_cos}")


class TestV4TqKAppendGemm(unittest.TestCase):
    """V4K-16a: GEMM-based TQ k_append — accuracy vs original fused kernel."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        cls.Pi_bf16_gpu = cls.Pi.to(torch.bfloat16).cuda()
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()

    def _run_both(self, k_nope, k_rope, v_nope):
        num_tokens = k_nope.shape[0]
        num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        cache_bytes = num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

        kn = k_nope.to(torch.bfloat16).cuda()
        kr = k_rope.to(torch.bfloat16).cuda()
        vn = v_nope.to(torch.bfloat16).cuda()

        cache_old = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
        self.k.v4_tq_k_append(kn, kr, vn, cache_old, slot_mapping,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

        cache_new = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
        self.k.v4_tq_k_append_gemm(kn, kr, vn, cache_new, slot_mapping,
                                    self.Pi_bf16_gpu, self.centroids_gpu, self.boundaries_gpu)
        return cache_old, cache_new

    def test_roundtrip_accuracy(self):
        """GEMM kernel round-trip cosine matches fused kernel (> 0.99)."""
        num_tokens = 32
        k_nope, v_nope = _make_csa_compressed_vectors(num_tokens, seed=42)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)

        cache_old, cache_new = self._run_both(k_nope, k_rope, v_nope)
        indices = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

        k_old, kr_old, v_old = self.k.v4_tq_dequant_indexed(
            cache_old, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)
        k_new, kr_new, v_new = self.k.v4_tq_dequant_indexed(
            cache_new, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        k_cos_old = cosine_sim(k_nope, k_old.cpu())
        k_cos_new = cosine_sim(k_nope, k_new.cpu())
        v_cos_old = cosine_sim(v_nope, v_old.cpu())
        v_cos_new = cosine_sim(v_nope, v_new.cpu())
        print(f"  Fused: K={k_cos_old:.6f} V={v_cos_old:.6f}")
        print(f"  GEMM:  K={k_cos_new:.6f} V={v_cos_new:.6f}")
        self.assertGreater(k_cos_new, 0.99)
        self.assertGreater(v_cos_new, 0.99)

    def test_old_vs_new_agreement(self):
        """GEMM vs fused kernel: dequanted output cosine > 0.999."""
        torch.manual_seed(77)
        num_tokens = 64
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)

        cache_old, cache_new = self._run_both(k_nope, k_rope, v_nope)
        indices = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

        k_old, kr_old, v_old = self.k.v4_tq_dequant_indexed(
            cache_old, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)
        k_new, kr_new, v_new = self.k.v4_tq_dequant_indexed(
            cache_new, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        k_cos = cosine_sim(k_old, k_new)
        v_cos = cosine_sim(v_old, v_new)
        kr_cos = cosine_sim(kr_old, kr_new)
        print(f"  Old vs new: K={k_cos:.6f} V={v_cos:.6f} K_rope={kr_cos:.6f}")
        self.assertGreater(k_cos, 0.999)
        self.assertGreater(v_cos, 0.999)
        self.assertAlmostEqual(kr_cos, 1.0, places=5)

    def test_single_token(self):
        """GEMM path works for N=1 (edge case for cuBLAS)."""
        torch.manual_seed(42)
        k_nope = torch.randn(1, HEAD_DIM)
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(1, HEAD_DIM)

        cache_old, cache_new = self._run_both(k_nope, k_rope, v_nope)
        indices = torch.arange(1, dtype=torch.int32, device='cuda')

        k_old, _, v_old = self.k.v4_tq_dequant_indexed(
            cache_old, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)
        k_new, _, v_new = self.k.v4_tq_dequant_indexed(
            cache_new, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        self.assertGreater(cosine_sim(k_old, k_new), 0.999)
        self.assertGreater(cosine_sim(v_old, v_new), 0.999)


class TestV4TqDequantIndexed(unittest.TestCase):
    """V4K-8b: V4 TQ dequant indexed kernel accuracy tests."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()

    def _populate_cache(self, k_nope, k_rope, v_nope):
        """Populate TQ cache via CUDA k_append kernel, return cache tensor."""
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
        """Dequant kernel builds, runs, does not crash."""
        torch.manual_seed(42)
        n = 4
        k_nope = torch.randn(n, HEAD_DIM)
        k_rope = torch.randn(n, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(n, HEAD_DIM)

        cache = self._populate_cache(k_nope, k_rope, v_nope)
        indices = torch.arange(n, dtype=torch.int32, device='cuda')

        results = self.k.v4_tq_dequant_indexed(
            cache, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, (n, HEAD_DIM))
        self.assertEqual(results[1].shape, (n, QK_ROPE_HEAD_DIM))
        self.assertEqual(results[2].shape, (n, HEAD_DIM))
        self.assertTrue(torch.isfinite(results[0]).all())

    def test_roundtrip(self):
        """TQ k_append → TQ dequant round-trip, cosine > 0.98 (BF16 in + TQ + BF16 out)."""
        torch.manual_seed(42)
        num_tokens = 16
        k_nope, v_nope = _make_csa_compressed_vectors(num_tokens, seed=42)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)

        cache = self._populate_cache(k_nope, k_rope, v_nope)
        indices = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

        k_deq, k_rope_deq, v_deq = self.k.v4_tq_dequant_indexed(
            cache, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        k_cos = cosine_sim(k_nope, k_deq.cpu())
        v_cos = cosine_sim(v_nope, v_deq.cpu())
        print(f"  Roundtrip K cosine: {k_cos:.6f}, V cosine: {v_cos:.6f}")
        # Full pipeline BF16→TQ→BF16 adds rounding on both ends
        self.assertGreater(k_cos, 0.98, f"K cosine={k_cos}")
        self.assertGreater(v_cos, 0.98, f"V cosine={v_cos}")

        # K ROPE exact
        rope_bf16 = k_rope.to(torch.bfloat16).float()
        rope_diff = (rope_bf16 - k_rope_deq.cpu().float()).abs().max().item()
        print(f"  K ROPE max diff: {rope_diff:.2e}")
        self.assertLess(rope_diff, 1e-3)

    def test_dequant_vs_reference(self):
        """CUDA dequant matches Python reference dequant."""
        torch.manual_seed(55)
        num_tokens = 8
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)

        cache = self._populate_cache(k_nope, k_rope, v_nope)
        indices = torch.arange(num_tokens, dtype=torch.int32, device='cuda')

        # CUDA dequant
        k_cuda, kr_cuda, v_cuda = self.k.v4_tq_dequant_indexed(
            cache, indices, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        # Reference dequant (from same cache)
        k_ref, kr_ref, v_ref = ref_v4_tq_dequant(
            cache.cpu(), list(range(num_tokens)), self.Pi, self.centroids)

        k_cos = cosine_sim(k_cuda.cpu(), k_ref)
        v_cos = cosine_sim(v_cuda.cpu(), v_ref)
        print(f"  CUDA vs ref dequant: K cos={k_cos:.6f}, V cos={v_cos:.6f}")
        self.assertGreater(k_cos, 0.999, f"K cosine={k_cos}")
        self.assertGreater(v_cos, 0.999, f"V cosine={v_cos}")

    def test_gather_subset(self):
        """Dequant a subset of indices from populated cache."""
        torch.manual_seed(42)
        num_tokens = 32
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)

        cache = self._populate_cache(k_nope, k_rope, v_nope)

        # Fetch only indices [5, 10, 20, 31]
        subset = torch.tensor([5, 10, 20, 31], dtype=torch.int32, device='cuda')
        k_deq, kr_deq, v_deq = self.k.v4_tq_dequant_indexed(
            cache, subset, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        self.assertEqual(k_deq.shape, (4, HEAD_DIM))
        self.assertTrue(torch.isfinite(k_deq).all())

        # Fetch all then compare subset
        all_idx = torch.arange(num_tokens, dtype=torch.int32, device='cuda')
        k_all, _, v_all = self.k.v4_tq_dequant_indexed(
            cache, all_idx, self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)

        for i, s in enumerate([5, 10, 20, 31]):
            k_diff = (k_deq[i] - k_all[s]).abs().max().item()
            self.assertLess(k_diff, 1e-5, f"K mismatch at idx {s}: diff={k_diff}")


if __name__ == '__main__':
    unittest.main()
