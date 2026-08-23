"""
V4K-12a/12b: V4 Stress Tests — Edge Cases and Numerical Extremes

Tests that all V4 kernels handle boundary conditions without crashes or NaN.

Usage:
  python tests/test_v4_stress.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE,
    V4_FP8_BYTES_PER_ENTRY, V4_TQ_BYTES_PER_ENTRY,
    load_codebook, generate_rotation_matrix,
)


def _alloc_fp8_cache(num_entries, k_mod):
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    cache_bytes = num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY
    cache = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_entries, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot = torch.arange(num_entries, dtype=torch.int32, device='cuda')
    k_mod.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)
    torch.cuda.synchronize()
    return cache


def _alloc_tq_cache(num_entries, k_mod, Pi, centroids, boundaries):
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    cache_bytes = num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY
    cache = torch.zeros(cache_bytes, dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(num_entries, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(num_entries, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot = torch.arange(num_entries, dtype=torch.int32, device='cuda')
    k_mod.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot, Pi, centroids, boundaries)
    torch.cuda.synchronize()
    return cache


class TestV4StressEdgeCases(unittest.TestCase):
    """V4K-12a: Edge case stress tests."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels
        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()
        cls.sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    def setUp(self):
        torch.cuda.synchronize()

    def tearDown(self):
        torch.cuda.synchronize()

    def _check_finite(self, out, name="output"):
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(out).all(), f"{name} has NaN/Inf")

    # --- CSA FP8 decode edge cases ---

    def test_csa_fp8_topk_1(self):
        """CSA FP8 decode with topk=1."""
        b, s_q, h_q, topk = 1, 1, 8, 1
        cache = _alloc_fp8_cache(64, self.k)
        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.zeros(b, s_q, topk, dtype=torch.int32, device='cuda')
        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')
        out, lse = self.k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, indices, swa, swa_bt, swa_sl,
            self.sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 1)
        self._check_finite(out, "topk=1 out")
        self._check_finite(lse, "topk=1 lse")

    def test_csa_fp8_topk_equals_all(self):
        """CSA FP8 decode with topk=all entries (HCA-style dense)."""
        b, s_q, h_q = 1, 1, 8
        num = 32
        topk = num
        cache = _alloc_fp8_cache(num, self.k)
        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(num, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')
        out, lse = self.k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, indices, swa, swa_bt, swa_sl,
            self.sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 1)
        self._check_finite(out)

    def test_csa_fp8_exactly_one_page(self):
        """s_kv = PAGE_SIZE (exactly one page boundary), sparse topk subset."""
        b, s_q, h_q = 1, 1, 8
        num = PAGE_SIZE
        topk = 16  # sparse subset (split-KV metadata needs topk < page_size for single split)
        cache = _alloc_fp8_cache(num, self.k)
        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')
        out, lse = self.k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, indices, swa, swa_bt, swa_sl,
            self.sm_scale, topk, PAGE_SIZE, PAGE_SIZE, 1)
        self._check_finite(out)

    # --- CSA TQ decode edge cases ---

    def test_csa_tq_topk_1(self):
        """CSA TQ decode with topk=1."""
        b, s_q, h_q, topk = 1, 1, 8, 1
        cache = _alloc_tq_cache(64, self.k, self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)
        q_rot = torch.randn(b, s_q, h_q, HEAD_DIM, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.zeros(b, s_q, topk, dtype=torch.int32, device='cuda')
        out, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, self.sm_scale)
        self._check_finite(out, "TQ topk=1 out")
        self._check_finite(lse, "TQ topk=1 lse")

    def test_csa_tq_topk_all(self):
        """CSA TQ decode with topk=all."""
        b, s_q, h_q = 1, 1, 8
        num = PAGE_SIZE
        cache = _alloc_tq_cache(num, self.k, self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)
        q_rot = torch.randn(b, s_q, h_q, HEAD_DIM, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(num, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        out, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, self.sm_scale)
        self._check_finite(out)

    def test_csa_tq_large_topk(self):
        """CSA TQ decode with topk=1024."""
        b, s_q, h_q = 1, 1, 8
        num = 1024
        cache = _alloc_tq_cache(num, self.k, self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)
        q_rot = torch.randn(b, s_q, h_q, HEAD_DIM, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(num, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        out, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, self.sm_scale)
        self._check_finite(out)
        self.assertEqual(out.shape, (b, s_q, h_q, HEAD_DIM))

    # --- K append edge cases ---

    def test_fp8_k_append_single_token(self):
        """FP8 k_append with 1 token."""
        cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        k_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        slot = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.k.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)

    def test_tq_k_append_single_token(self):
        """TQ k_append with 1 token."""
        cache = torch.zeros(PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        k_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        slot = torch.zeros(1, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

    # --- Dequant edge cases ---

    def test_fp8_dequant_single_index(self):
        """FP8 dequant with 1 index."""
        cache = _alloc_fp8_cache(16, self.k)
        indices = torch.tensor([0], dtype=torch.int32, device='cuda')
        results = self.k.v4_fp8_dequant_indexed(cache, indices, HEAD_DIM, QK_ROPE_HEAD_DIM)
        self.assertEqual(results[0].shape, (1, HEAD_DIM))

    def test_tq_dequant_single_index(self):
        """TQ dequant with 1 index."""
        cache = _alloc_tq_cache(16, self.k, self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)
        indices = torch.tensor([0], dtype=torch.int32, device='cuda')
        results = self.k.v4_tq_dequant_indexed(cache, indices, self.Pi_gpu, self.centroids_gpu,
                                                 HEAD_DIM, QK_ROPE_HEAD_DIM)
        self.assertEqual(results[0].shape, (1, HEAD_DIM))

    # --- Lightning Indexer edge cases ---

    def test_lightning_score_single_block(self):
        """Lightning score with 1 block."""
        q_proj = torch.randn(4, 128, dtype=torch.bfloat16, device='cuda')
        cache = torch.randn(1, 4, 128, dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        scales = torch.ones(1, dtype=torch.float32, device='cuda')
        proj = torch.ones(4, dtype=torch.float32, device='cuda')
        scores = self.k.v4_lightning_score(q_proj, cache, scales, proj)
        self.assertEqual(scores.shape, (1,))
        self._check_finite(scores)

    def test_lightning_topk_k_equals_1(self):
        """Lightning topk with k=1."""
        scores = torch.randn(100, dtype=torch.float32, device='cuda')
        endpoints = torch.arange(1, 101, dtype=torch.int32, device='cuda') * 8
        result = self.k.v4_lightning_topk(scores, endpoints, 799, 1)
        self.assertEqual(result[0].shape, (1,))

    # --- Graph runner edge cases ---

    def test_tq_graph_topk_1(self):
        """TQ graph runner with topk=1."""
        b, s_q, h_q, topk = 1, 1, 8, 1
        cache = _alloc_tq_cache(64, self.k, self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)
        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu, b, s_q, h_q, topk, self.sm_scale)
        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.zeros(b, s_q, topk, dtype=torch.int32, device='cuda')
        runner.update(q_nope, q_rope, indices)
        runner.replay()
        out, lse = runner.get_output(cache)
        self._check_finite(out)
        runner.destroy()


class TestV4StressNumerical(unittest.TestCase):
    """V4K-12b: Numerical edge case stress tests."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels
        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()
        cls.sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    def setUp(self):
        torch.cuda.synchronize()

    def tearDown(self):
        torch.cuda.synchronize()

    def _check_finite(self, out, name="output"):
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(out).all(), f"{name} has NaN/Inf")

    def test_all_zero_q(self):
        """CSA TQ decode with all-zero Q — should produce finite output."""
        b, s_q, h_q, topk = 1, 1, 8, 16
        cache = _alloc_tq_cache(topk, self.k, self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)
        q_rot = torch.zeros(b, s_q, h_q, HEAD_DIM, device='cuda')
        q_rope = torch.zeros(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        out, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, self.sm_scale)
        self._check_finite(out, "zero-Q out")

    def test_near_zero_kv(self):
        """FP8 k_append + decode with near-zero K/V values."""
        num = 32
        cache = torch.zeros((num + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')
        k_nope = torch.ones(num, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 1e-6
        k_rope = torch.ones(num, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 1e-6
        v_nope = torch.ones(num, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 1e-6
        slot = torch.arange(num, dtype=torch.int32, device='cuda')
        self.k.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)

        b, s_q, h_q = 1, 1, 8
        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(num, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')
        out, lse = self.k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, indices, swa, swa_bt, swa_sl,
            self.sm_scale, num, PAGE_SIZE, PAGE_SIZE, 1)
        self._check_finite(out, "near-zero KV out")

    def test_large_magnitude_kv(self):
        """FP8 k_append with large values (near BF16 max)."""
        num = 16
        cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        k_nope = torch.ones(num, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 100.0
        k_rope = torch.ones(num, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 100.0
        v_nope = torch.ones(num, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 100.0
        slot = torch.arange(num, dtype=torch.int32, device='cuda')
        self.k.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)

        results = self.k.v4_fp8_dequant_indexed(
            cache, torch.arange(num, dtype=torch.int32, device='cuda'),
            HEAD_DIM, QK_ROPE_HEAD_DIM)
        self._check_finite(results[0], "large K dequant")
        self._check_finite(results[2], "large V dequant")

    def test_tq_all_zero_vectors(self):
        """TQ k_append with all-zero vectors — norm=0 edge case."""
        num = 8
        cache = torch.zeros(PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        k_nope = torch.zeros(num, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.zeros(num, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.zeros(num, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        slot = torch.arange(num, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

        # Dequant should give zeros back
        results = self.k.v4_tq_dequant_indexed(
            cache, torch.arange(num, dtype=torch.int32, device='cuda'),
            self.Pi_gpu, self.centroids_gpu, HEAD_DIM, QK_ROPE_HEAD_DIM)
        self._check_finite(results[0], "zero TQ K dequant")
        self._check_finite(results[2], "zero TQ V dequant")

    def test_uniform_attention(self):
        """CSA TQ decode where all K are identical → uniform attention."""
        b, s_q, h_q, topk = 1, 1, 8, 16
        cache = torch.zeros(PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        # All identical K/V
        k_nope = torch.ones(topk, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 0.1
        k_rope = torch.ones(topk, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 0.1
        v_nope = torch.ones(topk, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 0.1
        slot = torch.arange(topk, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

        q_rot = torch.randn(b, s_q, h_q, HEAD_DIM, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        out, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, self.sm_scale)
        self._check_finite(out, "uniform attention out")
        self._check_finite(lse, "uniform attention lse")

    def test_one_hot_attention(self):
        """CSA TQ: one key has huge score, rest are near-zero."""
        b, s_q, h_q, topk = 1, 1, 8, 16
        cache = torch.zeros(PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        k_nope = torch.randn(topk, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 0.01
        k_rope = torch.randn(topk, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 0.01
        # Make first key dominant
        k_nope[0] = torch.ones(HEAD_DIM, dtype=torch.bfloat16) * 10.0
        v_nope = torch.randn(topk, HEAD_DIM, dtype=torch.bfloat16, device='cuda') * 0.1
        slot = torch.arange(topk, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

        # Q aligned with first key
        q_rot = torch.ones(b, s_q, h_q, HEAD_DIM, device='cuda') * 5.0
        q_rope = torch.zeros(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        out, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, self.sm_scale)
        self._check_finite(out, "one-hot attention out")
        self._check_finite(lse, "one-hot attention lse")

    def test_compressor_minimal_input(self):
        """CSA compressor with minimum viable input (window=8, 9 tokens)."""
        num_tokens = 9
        window, stride = 8, 1
        k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        out_k, out_kr, out_v = self.k.v4_csa_compress(
            k_nope, k_rope, v, gate, pos_bias, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        self.assertEqual(out_k.shape[0], 1)
        self._check_finite(out_k)

    def test_compressor_all_zero_gates(self):
        """CSA compressor with zero gate weights."""
        num_tokens = 16
        window, stride = 8, 1
        k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.zeros(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.zeros(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        out_k, out_kr, out_v = self.k.v4_csa_compress(
            k_nope, k_rope, v, gate, pos_bias, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        self._check_finite(out_k, "zero-gate compress K")
        self._check_finite(out_v, "zero-gate compress V")


if __name__ == '__main__':
    unittest.main()
