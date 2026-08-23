"""
V4K-13c: End-to-End V4 Integration Smoke Tests

Full V4 decode pipeline for each layer type:
  CSA: compress → FP8 k_append → lightning_score → topk → CSA FP8 decode
  HCA: compress → FP8 k_append → HCA FP8 decode (dense, no indexer)
  SWA: FP8 k_append → SWA decode (sliding window only)
  TQ:  compress → TQ k_append → TQ decode → v_rotate_back

Each test exercises the complete pipeline end-to-end from Python,
verifying outputs are finite and shapes are correct.

Usage:
  python tests/test_v4_integration.py -v
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

H_Q = 64
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
INDEX_N_HEADS = 4
INDEX_HEAD_DIM = 128


class TestCsaFp8Pipeline(unittest.TestCase):
    """Full CSA FP8 pipeline: compress → cache → index → decode."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

    def test_csa_fp8_full_pipeline(self):
        """End-to-end CSA FP8 decode with all stages."""
        torch.manual_seed(42)
        num_tokens = 128
        window, stride = 8, 1
        topk = 64  # multiple of 64

        # Stage 1: Compress raw tokens → compressed K/V
        inp_k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_kr = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

        k_nope, k_rope, v_nope = self.k.v4_csa_compress(
            inp_k, inp_kr, inp_v, gate, pos_bias, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        num_compressed = k_nope.shape[0]
        self.assertGreater(num_compressed, 0)

        # Stage 2: FP8 k_append → paged cache
        num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
        cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')
        slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
        self.k.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot_mapping)

        # Stage 3: Lightning Indexer (score + topk)
        num_blocks = (num_compressed + 7) // 8
        indexer_k = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM,
                                dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
        k_scales = torch.ones(num_blocks, dtype=torch.float32, device='cuda')
        q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        score_proj = torch.ones(INDEX_N_HEADS, dtype=torch.float32, device='cuda')
        block_endpoints = torch.arange(1, num_blocks + 1, dtype=torch.int32, device='cuda') * 8

        scores = self.k.v4_lightning_score(q_proj, indexer_k, k_scales, score_proj)
        actual_topk = min(topk, num_compressed)
        topk_result = self.k.v4_lightning_topk(scores, block_endpoints, num_compressed - 1, actual_topk)
        sparse_indices = topk_result[0]  # [topk] int32

        # Stage 4: CSA FP8 decode
        b, s_q = 1, 1
        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        idx_4d = sparse_indices.unsqueeze(0).unsqueeze(0)

        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')

        out, lse = self.k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, idx_4d, swa, swa_bt, swa_sl,
            SM_SCALE, actual_topk, PAGE_SIZE, PAGE_SIZE, 1)

        self.assertEqual(out.shape, (b, s_q, H_Q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all(), "CSA FP8 output has NaN/Inf")
        self.assertTrue(torch.isfinite(lse).all(), "CSA FP8 LSE has NaN/Inf")


class TestHcaFp8Pipeline(unittest.TestCase):
    """Full HCA FP8 pipeline: compress → cache → dense decode."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

    def test_hca_fp8_full_pipeline(self):
        """End-to-end HCA FP8 decode (dense, no indexer)."""
        torch.manual_seed(42)
        num_tokens = 256
        window, stride = 128, 128

        # Stage 1: HCA compress
        inp_k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_kr = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

        k_nope, k_rope, v_nope = self.k.v4_hca_compress(
            inp_k, inp_kr, inp_v, gate, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        num_compressed = k_nope.shape[0]

        # Stage 2: FP8 k_append
        num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
        cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')
        slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
        self.k.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)

        # Stage 3: HCA FP8 decode (dense — all compressed entries)
        b, s_q = 1, 1
        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')

        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')

        out, lse = self.k.v4_hca_fp8_decode(
            q_nope, q_rope, cache, num_compressed,
            swa, swa_bt, swa_sl, SM_SCALE, PAGE_SIZE, PAGE_SIZE, 1)

        self.assertEqual(out.shape, (b, s_q, H_Q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all(), "HCA FP8 output has NaN/Inf")


class TestSwaPipeline(unittest.TestCase):
    """SWA pipeline: k_append → SWA decode (sliding window only)."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

    def test_swa_full_pipeline(self):
        """End-to-end SWA-only decode."""
        torch.manual_seed(42)
        swa_len = 64
        b, s_q = 1, 1

        # Stage 1: k_append for SWA tokens
        k_nope = torch.randn(swa_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(swa_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        num_pages = (swa_len + PAGE_SIZE - 1) // PAGE_SIZE
        swa_cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                                dtype=torch.uint8, device='cuda')
        slot = torch.arange(swa_len, dtype=torch.int32, device='cuda')
        self.k.v4_fp8_k_append(k_nope, k_rope, v_nope, swa_cache, slot)

        # Stage 2: SWA decode (no compressed entries)
        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        swa_bt = torch.zeros(b, num_pages, dtype=torch.int32, device='cuda')
        for i in range(num_pages):
            swa_bt[0, i] = i
        swa_sl = torch.tensor([swa_len], dtype=torch.int32, device='cuda')

        out, lse = self.k.v4_swa_decode(
            q_nope, q_rope, swa_cache, swa_bt, swa_sl,
            SM_SCALE, PAGE_SIZE, 1)

        self.assertEqual(out.shape, (b, s_q, H_Q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all(), "SWA output has NaN/Inf")


class TestCsaTqPipeline(unittest.TestCase):
    """Full CSA TQ pipeline: compress → TQ k_append → TQ decode → rotate back."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels
        cls.centroids, cls.boundaries = load_codebook()
        cls.Pi = generate_rotation_matrix()
        cls.centroids_gpu = cls.centroids.cuda()
        cls.Pi_gpu = cls.Pi.cuda()
        cls.boundaries_gpu = cls.boundaries[1:-1].cuda()

    def test_csa_tq_full_pipeline(self):
        """End-to-end CSA TQ decode with compression + quantization."""
        torch.manual_seed(42)
        num_tokens = 128
        window, stride = 8, 1
        b, s_q = 1, 1

        # Stage 1: CSA compress
        inp_k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_kr = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

        k_nope, k_rope, v_nope = self.k.v4_csa_compress(
            inp_k, inp_kr, inp_v, gate, pos_bias, cos, sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
        num_compressed = k_nope.shape[0]

        # Stage 2: TQ k_append (quantize + cache write)
        num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
        cache = torch.zeros(num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')
        slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

        # Stage 3: Q rotate + TQ decode
        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')

        q_rot = (q_nope.float() @ self.Pi_gpu.T).contiguous()
        topk = min(num_compressed, 64)
        indices = torch.arange(topk, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).contiguous()

        out_rot, lse = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, SM_SCALE)

        # Stage 4: V rotate back
        out = (out_rot.float() @ self.Pi_gpu).to(torch.bfloat16)

        self.assertEqual(out.shape, (b, s_q, H_Q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all(), "TQ output has NaN/Inf")
        self.assertTrue(torch.isfinite(lse).all(), "TQ LSE has NaN/Inf")

    def test_csa_tq_graph_pipeline(self):
        """End-to-end CSA TQ via CUDA graph runner."""
        torch.manual_seed(42)
        num_compressed = 64
        topk = 64
        b, s_q = 1, 1

        # Populate TQ cache
        k_nope = torch.randn(num_compressed, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(num_compressed, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(num_compressed, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        cache = torch.zeros(PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
        self.k.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                               self.Pi_gpu, self.centroids_gpu, self.boundaries_gpu)

        # Graph runner
        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu, b, s_q, H_Q, topk, SM_SCALE)

        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).contiguous()

        runner.update(q_nope, q_rope, indices)
        runner.replay()
        out, lse = runner.get_output(cache)

        self.assertEqual(out.shape, (b, s_q, H_Q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all(), "Graph output has NaN/Inf")
        runner.destroy()


class TestFusedPipeline(unittest.TestCase):
    """Fused compress+insert pipeline (single kernel for write path)."""

    @classmethod
    def setUpClass(cls):
        import sm120_mla_kernels
        cls.k = sm120_mla_kernels

    def test_fused_csa_pipeline(self):
        """Fused CSA compress+insert → FP8 decode."""
        torch.manual_seed(42)
        num_tokens = 64
        window, stride = 8, 1
        b, s_q = 1, 1

        inp_k = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_kr = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        inp_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(num_tokens, window, dtype=torch.bfloat16, device='cuda')
        cos = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
        sin = torch.randn(num_tokens, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

        num_compressed = max(0, num_tokens - window) // stride
        num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
        cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')
        slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')

        # Single fused kernel: compress + FP8 quant + cache write
        self.k.v4_fused_csa_compress_insert(
            inp_k, inp_kr, inp_v, gate, pos_bias, cos, sin,
            cache, slot, HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)

        # Decode from fused cache
        topk = 64
        actual_topk = min(num_compressed, topk)
        # Pad to multiple of 64
        padded_topk = ((actual_topk + 63) // 64) * 64
        idx = torch.full((b, s_q, padded_topk), -1, dtype=torch.int32, device='cuda')
        idx[0, 0, :actual_topk] = torch.arange(actual_topk, dtype=torch.int32, device='cuda')

        q_nope = torch.randn(b, s_q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
        swa_bt = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_sl = torch.zeros(b, dtype=torch.int32, device='cuda')

        out, lse = self.k.v4_csa_fp8_decode(
            q_nope, q_rope, cache, idx, swa, swa_bt, swa_sl,
            SM_SCALE, padded_topk, PAGE_SIZE, PAGE_SIZE, 1)

        self.assertEqual(out.shape, (b, s_q, H_Q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all(), "Fused pipeline output has NaN/Inf")


if __name__ == '__main__':
    unittest.main()
