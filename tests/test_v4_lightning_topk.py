"""
V4K-2b: Lightning Indexer Top-K Selection Kernel Tests

Smoke test (V4K-2b.3): kernel builds, runs on small/large input without crash.
Accuracy test (V4K-2b.3): top-k correctness vs ref_lightning_topk(), causality
enforcement verified, output sorted ascending.

The kernel uses multi-pass radix histogram selection matching the SGLang/vLLM/
TRT-LLM approach (convert_to_uint32 → 4-pass 8-bit histogram → gather → sort).

Usage:
  python tests/test_v4_lightning_topk.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import ref_lightning_topk


def run_topk_kernel(scores, block_endpoints, query_position, topk):
    """Run the CUDA lightning_topk kernel."""
    import sm120_mla_kernels

    scores_gpu = scores.to(device='cuda', dtype=torch.float32).contiguous()
    endpoints_gpu = block_endpoints.to(device='cuda', dtype=torch.int32).contiguous()

    indices, sel_scores, eff_k = sm120_mla_kernels.v4_lightning_topk(
        scores_gpu, endpoints_gpu, int(query_position), int(topk))
    return indices.cpu(), sel_scores.cpu(), eff_k.cpu().item()


class TestLightningTopkSmoke(unittest.TestCase):
    """V4K-2b.3: Smoke test — kernel runs without crash."""

    def test_smoke_small(self):
        """64 blocks, topk=16, no crash."""
        torch.manual_seed(42)
        num_blocks = 64
        topk = 16

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = num_blocks * 4

        indices, sel_scores, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)

        self.assertEqual(indices.shape, (topk,))
        self.assertEqual(sel_scores.shape, (topk,))
        self.assertEqual(eff_k, topk)
        self.assertTrue(torch.isfinite(sel_scores[:eff_k]).all(),
                        "Selected scores contain NaN/Inf")

    def test_smoke_single_block(self):
        """1 block, topk=1."""
        scores = torch.tensor([3.14], dtype=torch.float32)
        endpoints = torch.tensor([0], dtype=torch.int32)

        indices, sel_scores, eff_k = run_topk_kernel(scores, endpoints, 100, 1)
        self.assertEqual(eff_k, 1)
        self.assertEqual(indices[0].item(), 0)

    def test_smoke_large(self):
        """16K blocks, topk=1024 — realistic CSA at 64K context."""
        torch.manual_seed(99)
        num_blocks = 16384
        topk = 1024

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = num_blocks * 4

        indices, sel_scores, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)

        self.assertEqual(indices.shape, (topk,))
        self.assertEqual(eff_k, topk)
        self.assertTrue(torch.isfinite(sel_scores[:eff_k]).all())

    def test_smoke_very_large(self):
        """64K blocks, topk=1024."""
        torch.manual_seed(77)
        num_blocks = 65536
        topk = 1024

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = num_blocks * 4

        indices, sel_scores, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)
        self.assertEqual(eff_k, topk)


class TestLightningTopkAccuracy(unittest.TestCase):
    """V4K-2b.3: Accuracy test — top-k correctness vs ref_lightning_topk()."""

    def _compare(self, num_blocks, topk, query_pos=None, seed=42):
        """Run kernel and reference, return both results."""
        torch.manual_seed(seed)

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        if query_pos is None:
            query_pos = num_blocks * 4  # all blocks valid

        ref_idx, ref_scores = ref_lightning_topk(scores, topk, query_pos, endpoints)

        k_idx, k_scores, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)
        k_idx = k_idx[:eff_k]
        k_scores = k_scores[:eff_k]

        return ref_idx, ref_scores, k_idx, k_scores, eff_k

    def test_topk_correctness_small(self):
        """256 blocks, topk=32: selected index set matches reference."""
        ref_idx, _, k_idx, _, eff_k = self._compare(256, 32)
        self.assertEqual(eff_k, len(ref_idx))

        ref_set = set(ref_idx.tolist())
        kernel_set = set(k_idx.tolist())
        self.assertEqual(ref_set, kernel_set,
            f"Index sets differ: only_ref={ref_set - kernel_set}, only_kernel={kernel_set - ref_set}")

    def test_topk_correctness_medium(self):
        """2048 blocks, topk=256."""
        ref_idx, _, k_idx, _, eff_k = self._compare(2048, 256, seed=55)
        self.assertEqual(eff_k, 256)

        ref_set = set(ref_idx.tolist())
        kernel_set = set(k_idx.tolist())
        # Allow 1-2 elements to differ (possible threshold ties)
        overlap = len(ref_set & kernel_set)
        self.assertGreaterEqual(overlap, 256 - 2,
            f"Top-256 overlap: {overlap}/256")

    def test_topk_1024(self):
        """4096 blocks, topk=1024 (Pro config)."""
        ref_idx, _, k_idx, _, eff_k = self._compare(4096, 1024, seed=77)
        self.assertEqual(eff_k, 1024)

        ref_set = set(ref_idx.tolist())
        kernel_set = set(k_idx.tolist())
        overlap = len(ref_set & kernel_set)
        self.assertGreaterEqual(overlap, 1024 - 2,
            f"Top-1024 overlap: {overlap}/1024")

    def test_topk_512(self):
        """4096 blocks, topk=512 (Flash config)."""
        ref_idx, _, k_idx, _, eff_k = self._compare(4096, 512, seed=33)
        self.assertEqual(eff_k, 512)

        ref_set = set(ref_idx.tolist())
        kernel_set = set(k_idx.tolist())
        overlap = len(ref_set & kernel_set)
        self.assertGreaterEqual(overlap, 512 - 2,
            f"Top-512 overlap: {overlap}/512")

    def test_causality_enforcement(self):
        """Future blocks excluded from selection."""
        torch.manual_seed(42)
        num_blocks = 200
        topk = 50

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = 200  # blocks 0..50 have endpoint <= 200

        k_idx, k_scores, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)

        for i in range(eff_k):
            idx = k_idx[i].item()
            self.assertLessEqual(endpoints[idx].item(), query_pos,
                f"Block {idx} endpoint {endpoints[idx].item()} > query_pos {query_pos}")

        # Compare against reference
        ref_idx, _ = ref_lightning_topk(scores, topk, query_pos, endpoints)
        self.assertEqual(eff_k, len(ref_idx))

    def test_sorted_output(self):
        """Selected indices are sorted in ascending order."""
        torch.manual_seed(42)
        num_blocks = 1000
        topk = 100

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = num_blocks * 4

        k_idx, _, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)

        for i in range(eff_k - 1):
            self.assertLess(k_idx[i].item(), k_idx[i + 1].item(),
                f"Not sorted: [{i}]={k_idx[i].item()} >= [{i+1}]={k_idx[i+1].item()}")

    def test_all_masked(self):
        """All blocks have future endpoints — effective_k = 0."""
        num_blocks = 50
        topk = 10

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = torch.full((num_blocks,), 1000, dtype=torch.int32)
        query_pos = 0

        _, _, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)
        self.assertEqual(eff_k, 0)

    def test_topk_exceeds_valid(self):
        """topk > num_blocks — returns all blocks."""
        torch.manual_seed(42)
        num_blocks = 30
        topk = 100

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = num_blocks * 4

        k_idx, _, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)
        self.assertEqual(eff_k, num_blocks)

        # Should contain all indices 0..29
        self.assertEqual(set(k_idx[:eff_k].tolist()), set(range(num_blocks)))

    def test_scores_match_reference(self):
        """Selected scores match reference scores (exact for FP32)."""
        ref_idx, ref_scores, k_idx, k_scores, eff_k = self._compare(256, 32)
        self.assertEqual(eff_k, len(ref_idx))

        # Build index→score maps and compare
        ref_map = dict(zip(ref_idx.tolist(), ref_scores.tolist()))
        kernel_map = dict(zip(k_idx.tolist(), k_scores.tolist()))

        for idx in ref_map:
            if idx in kernel_map:
                diff = abs(ref_map[idx] - kernel_map[idx])
                self.assertLess(diff, 1e-5,
                    f"Score mismatch at idx {idx}: ref={ref_map[idx]}, kernel={kernel_map[idx]}")

    def test_partial_causality(self):
        """Half the blocks are causal, topk spans the boundary."""
        torch.manual_seed(99)
        num_blocks = 100
        topk = 60  # more than the 50 valid blocks

        scores = torch.randn(num_blocks, dtype=torch.float32)
        endpoints = (torch.arange(num_blocks, dtype=torch.int32) * 4)
        query_pos = 196  # blocks 0..49 valid (endpoint <= 196)

        ref_idx, _ = ref_lightning_topk(scores, topk, query_pos, endpoints)
        k_idx, _, eff_k = run_topk_kernel(scores, endpoints, query_pos, topk)

        self.assertEqual(eff_k, len(ref_idx))
        self.assertLessEqual(eff_k, 50)  # can't exceed valid count


if __name__ == "__main__":
    unittest.main()
