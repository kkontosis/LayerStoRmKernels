"""
V4K-9a.2 / V4K-9b.2: V4 TQ CUDA Graph Runner Tests

Tests that graph-captured TQ decode matches uncaptured (direct call) results.

Usage:
  python tests/test_v4_tq_graph.py -v
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


class TestCsaTqDecodeGraph(unittest.TestCase):
    """V4K-9a: CSA TQ decode graph runner tests."""

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
        """Graph runner builds, replays, returns correct shapes."""
        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 8
        topk = 16
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(topk, HEAD_DIM)
        k_rope = torch.randn(topk, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(topk, HEAD_DIM)
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu,
                    b, s_q, h_q, topk, sm_scale)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).expand(b, s_q, -1).contiguous()

        runner.update(q_nope, q_rope, indices)
        runner.replay()
        out, lse = runner.get_output(cache)

        self.assertEqual(out.shape, (b, s_q, h_q, HEAD_DIM))
        self.assertEqual(lse.shape, (b, s_q, h_q))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(lse).all())
        runner.destroy()

    def test_graph_vs_uncaptured(self):
        """Graph output matches direct (uncaptured) TQ decode pipeline."""
        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 8
        topk = 32
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(topk, HEAD_DIM) * 0.3
        k_rope = torch.randn(topk, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(topk, HEAD_DIM) * 0.3
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(topk, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).expand(b, s_q, -1).contiguous()

        # Direct (uncaptured) pipeline: q_rotate → csa_tq_decode → v_rotate_back
        q_rot = (q_nope.float() @ self.Pi_gpu.T).contiguous()
        out_rot, lse_direct = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, sm_scale)
        out_direct = (out_rot.float() @ self.Pi_gpu).to(torch.bfloat16)

        # Graph pipeline
        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu,
                    b, s_q, h_q, topk, sm_scale)
        runner.update(q_nope, q_rope, indices)
        runner.replay()
        out_graph, lse_graph = runner.get_output(cache)

        cos = cosine_sim(out_graph, out_direct)
        print(f"  Graph vs uncaptured: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99999, f"cosine={cos}")

        cos_lse = cosine_sim(lse_graph, lse_direct)
        print(f"  LSE graph vs direct: cosine={cos_lse:.6f}")
        self.assertGreater(cos_lse, 0.99999, f"LSE cosine={cos_lse}")
        runner.destroy()

    def test_multi_replay(self):
        """Multiple replays with different Q produce different results."""
        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 8
        topk = 16
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(topk, HEAD_DIM)
        k_rope = torch.randn(topk, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(topk, HEAD_DIM)
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu,
                    b, s_q, h_q, topk, sm_scale)

        indices = torch.arange(topk, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).expand(b, s_q, -1).contiguous()

        # Replay 1
        q1 = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        qr1 = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        runner.update(q1, qr1, indices)
        runner.replay()
        out1, _ = runner.get_output(cache)

        # Replay 2 with different Q
        q2 = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        qr2 = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        runner.update(q2, qr2, indices)
        runner.replay()
        out2, _ = runner.get_output(cache)

        cos = cosine_sim(out1, out2)
        self.assertLess(cos, 0.99, f"Different Q should give different outputs: cosine={cos}")
        runner.destroy()


class TestHcaTqDecodeGraph(unittest.TestCase):
    """V4K-9b: HCA TQ decode graph runner tests (reuses CSA TQ with topk=all)."""

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

    def test_hca_graph_smoke(self):
        """HCA graph = CSA TQ graph with topk=all (dense indices)."""
        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 8
        num_hca = 16
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(num_hca, HEAD_DIM)
        k_rope = torch.randn(num_hca, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_hca, HEAD_DIM)
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu,
                    b, s_q, h_q, num_hca, sm_scale)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(num_hca, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).expand(b, s_q, -1).contiguous()

        runner.update(q_nope, q_rope, indices)
        runner.replay()
        out, lse = runner.get_output(cache)

        self.assertEqual(out.shape, (b, s_q, h_q, HEAD_DIM))
        self.assertTrue(torch.isfinite(out).all())
        runner.destroy()

    def test_hca_graph_vs_uncaptured(self):
        """HCA graph output matches direct TQ decode."""
        torch.manual_seed(77)
        b, s_q, h_q = 1, 1, 8
        num_hca = 24
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(num_hca, HEAD_DIM) * 0.3
        k_rope = torch.randn(num_hca, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_hca, HEAD_DIM) * 0.3
        cache = self._populate_tq_cache(k_nope, k_rope, v_nope)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        indices = torch.arange(num_hca, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).expand(b, s_q, -1).contiguous()

        # Direct pipeline
        q_rot = (q_nope.float() @ self.Pi_gpu.T).contiguous()
        out_rot, lse_direct = self.k.v4_csa_tq_decode(
            q_rot, q_rope, cache, indices, self.centroids_gpu, sm_scale)
        out_direct = (out_rot.float() @ self.Pi_gpu).to(torch.bfloat16)

        # Graph pipeline
        runner = self.k.CsaTqDecodeGraphRunner()
        runner.init(cache, self.Pi_gpu, self.centroids_gpu,
                    b, s_q, h_q, num_hca, sm_scale)
        runner.update(q_nope, q_rope, indices)
        runner.replay()
        out_graph, lse_graph = runner.get_output(cache)

        cos = cosine_sim(out_graph, out_direct)
        print(f"  HCA graph vs uncaptured: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99999, f"cosine={cos}")
        runner.destroy()


if __name__ == '__main__':
    unittest.main()
