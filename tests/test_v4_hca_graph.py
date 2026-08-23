"""
V4K-7b: HCA FP8 Decode Graph Runner Tests

HCA graph reuses CSA graph runner with topk=num_compressed (dense).
Graph output must match uncaptured HCA decode.

Usage:
  python tests/test_v4_hca_graph.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE
from test_v4_hca_decode import populate_v4_cache, run_hca_decode


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


class TestHcaFp8DecodeGraph(unittest.TestCase):
    """V4K-7b: HCA graph vs uncaptured."""

    def test_graph_matches_uncaptured(self):
        import sm120_mla_kernels

        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 64
        num_compressed = 64
        swa_len = 64
        num_sm_parts = 4
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        comp_cache, _, _, _ = populate_v4_cache(num_compressed)
        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        swa_block_table = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32, device='cuda')

        # Uncaptured
        out_direct, lse_direct = run_hca_decode(
            q_nope, q_rope, comp_cache, num_compressed,
            swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, PAGE_SIZE, num_sm_parts)

        # Graph: use CSA graph runner with topk=num_compressed
        sparse_indices = torch.arange(num_compressed, dtype=torch.int32, device='cuda') \
            .unsqueeze(0).unsqueeze(0).expand(b, s_q, num_compressed).contiguous()

        runner = sm120_mla_kernels.CsaFp8DecodeGraphRunner()
        runner.init(comp_cache, swa_cache,
                    b, s_q, h_q, num_compressed, sm_scale, num_sm_parts,
                    PAGE_SIZE, PAGE_SIZE, 1)
        topk_seqlens = torch.full((b,), num_compressed, dtype=torch.int32, device='cuda')
        runner.update_metadata(topk_seqlens, num_sm_parts)
        runner.update(q_nope, q_rope, sparse_indices, swa_block_table, swa_seqlens)
        runner.replay()
        out_graph, lse_graph = runner.get_output(q_nope)
        runner.destroy()

        cos = cosine_sim(out_direct, out_graph.cpu())
        print(f"  HCA graph vs direct: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos}")


if __name__ == '__main__':
    unittest.main()
