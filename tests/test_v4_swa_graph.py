"""
V4K-7c: SWA Decode Graph Runner Tests

SWA graph reuses CSA graph runner with topk=0 (SWA-only).

Usage:
  python tests/test_v4_swa_graph.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE
from test_v4_swa_decode import populate_v4_cache, run_swa_decode


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


class TestSwaDecodeGraph(unittest.TestCase):
    """V4K-7c: SWA graph vs uncaptured."""

    def test_graph_matches_uncaptured(self):
        import sm120_mla_kernels

        torch.manual_seed(42)
        b, s_q, h_q = 1, 1, 64
        swa_len = 64
        num_sm_parts = 4
        sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

        swa_cache, _, _, _ = populate_v4_cache(swa_len)

        q_nope = torch.randn(b, s_q, h_q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q_rope = torch.randn(b, s_q, h_q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        swa_block_table = torch.zeros(b, 1, dtype=torch.int32, device='cuda')
        swa_seqlens = torch.tensor([swa_len], dtype=torch.int32, device='cuda')

        # Uncaptured
        out_direct, lse_direct = run_swa_decode(
            q_nope, q_rope, swa_cache, swa_block_table, swa_seqlens,
            sm_scale, PAGE_SIZE, num_sm_parts)

        # Graph: CSA graph runner with topk=0, empty compressed cache
        import sm120_mla_kernels as k
        entry_bytes = 1160
        empty_comp = torch.zeros(1, PAGE_SIZE * entry_bytes, dtype=torch.uint8, device='cuda')
        empty_indices = torch.zeros(b, s_q, 0, dtype=torch.int32, device='cuda')

        # For topk=0, we still need a valid graph. Use topk=1 with a dummy index.
        dummy_indices = torch.zeros(b, s_q, 1, dtype=torch.int32, device='cuda')

        runner = k.CsaFp8DecodeGraphRunner()
        runner.init(empty_comp, swa_cache,
                    b, s_q, h_q, 1, sm_scale, num_sm_parts,
                    PAGE_SIZE, PAGE_SIZE, 1)
        topk_seqlens = torch.full((b,), 1, dtype=torch.int32, device='cuda')
        runner.update_metadata(topk_seqlens, num_sm_parts)
        runner.update(q_nope, q_rope, dummy_indices, swa_block_table, swa_seqlens)
        runner.replay()
        out_graph, lse_graph = runner.get_output(q_nope)
        runner.destroy()

        cos = cosine_sim(out_direct, out_graph.cpu())
        print(f"  SWA graph vs direct: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos}")


if __name__ == '__main__':
    unittest.main()
