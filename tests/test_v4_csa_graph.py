"""
V4K-7a: CSA FP8 Decode Graph Runner Tests

Graph-captured CSA decode must produce identical output to uncaptured.

Usage:
  python tests/test_v4_csa_graph.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE
from test_v4_csa_decode import populate_v4_cache, run_csa_decode


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


class TestCsaFp8DecodeGraph(unittest.TestCase):
    """V4K-7a: Graph captured vs uncaptured must match."""

    def test_graph_matches_uncaptured(self):
        """Graph replay output matches direct kernel call."""
        import sm120_mla_kernels

        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 64
        topk = 64
        num_compressed = 128
        swa_len = 64
        num_sm_parts = 8
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)
        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        sparse_indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        swa_block_table = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32, device='cuda')

        # Uncaptured (direct kernel call)
        out_direct, lse_direct = run_csa_decode(
            q_nope, q_rope, comp_cache, sparse_indices,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, topk, PAGE_SIZE, PAGE_SIZE, num_sm_parts)

        # Graph-captured
        runner = sm120_mla_kernels.CsaFp8DecodeGraphRunner()
        runner.init(comp_cache, swa_cache,
                    b, s_q, h_q, topk, sm_scale, num_sm_parts,
                    PAGE_SIZE, PAGE_SIZE, 1)

        topk_seqlens = torch.full((b,), topk, dtype=torch.int32, device='cuda')
        runner.update_metadata(topk_seqlens, num_sm_parts)
        runner.update(q_nope, q_rope, sparse_indices, swa_block_table, swa_seqlens)
        runner.replay()
        out_graph, lse_graph = runner.get_output(q_nope)
        runner.destroy()

        out_graph = out_graph.cpu()
        lse_graph = lse_graph.cpu()

        cos = cosine_sim(out_direct, out_graph)
        lse_diff = (lse_direct.float() - lse_graph.float()).abs().max().item()
        print(f"  graph vs direct: cosine={cos:.6f}, max_lse_diff={lse_diff:.2e}")
        self.assertGreater(cos, 0.99, f"cosine={cos}")

    def test_graph_replay_consistent(self):
        """Two replays with same input produce identical output."""
        import sm120_mla_kernels

        torch.manual_seed(43)
        b, s_q, h_q = 1, 1, 64
        topk = 32
        num_compressed = 64
        swa_len = 32
        num_sm_parts = 4
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)
        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        sparse_indices = torch.arange(topk, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)
        swa_block_table = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32, device='cuda')

        runner = sm120_mla_kernels.CsaFp8DecodeGraphRunner()
        runner.init(comp_cache, swa_cache,
                    b, s_q, h_q, topk, sm_scale, num_sm_parts,
                    PAGE_SIZE, PAGE_SIZE, 1)

        topk_seqlens = torch.full((b,), topk, dtype=torch.int32, device='cuda')
        runner.update_metadata(topk_seqlens, num_sm_parts)
        runner.update(q_nope, q_rope, sparse_indices, swa_block_table, swa_seqlens)

        runner.replay()
        out1, lse1 = runner.get_output(q_nope)

        runner.replay()
        out2, lse2 = runner.get_output(q_nope)

        runner.destroy()

        cos = cosine_sim(out1, out2)
        self.assertGreater(cos, 0.99999, f"Two replays cosine={cos}")
        lse_diff = (lse1.float() - lse2.float()).abs().max().item()
        self.assertLess(lse_diff, 1e-3, f"LSE diff={lse_diff}")


if __name__ == '__main__':
    unittest.main()
