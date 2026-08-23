"""
V4K-1b: HCA Compressor Kernel Tests

Smoke test (V4K-1b.3): kernel runs on 256-token input without crash/NaN.
Accuracy test (V4K-1b.4): kernel output vs ref_hca_compress() golden reference.

Usage:
  python tests/test_v4_hca_compressor.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, HCA_WINDOW, HCA_STRIDE,
    COMPRESS_ROPE_THETA, precompute_rope_freqs, ref_hca_compress,
)


def run_kernel(num_tokens, gate_weights, compress_cos, compress_sin,
               input_k_nope=None, input_k_rope_raw=None, input_v=None, seed=42):
    torch.manual_seed(seed)
    device = "cuda"

    if input_k_nope is None:
        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if input_k_rope_raw is None:
        input_k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
    if input_v is None:
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)

    gw = gate_weights.to(device=device, dtype=torch.bfloat16)
    cc = compress_cos.to(device=device, dtype=torch.float32)
    cs = compress_sin.to(device=device, dtype=torch.float32)

    import sm120_mla_kernels
    out_k_nope, out_k_rope, out_v = sm120_mla_kernels.v4_hca_compress(
        input_k_nope, input_k_rope_raw, input_v,
        gw, cc, cs,
        HEAD_DIM, QK_ROPE_HEAD_DIM, HCA_WINDOW, HCA_STRIDE,
    )
    return out_k_nope, out_k_rope, out_v, input_k_nope, input_k_rope_raw, input_v


class TestHcaCompressorSmoke(unittest.TestCase):
    """V4K-1b.3: Smoke test."""

    def test_smoke_256_tokens(self):
        """256 tokens → 2 compressed entries, no crash."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 512)
        gate_weights = torch.randn(HCA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            256, gate_weights, compress_cos, compress_sin)

        self.assertEqual(out_k_nope.shape, (2, HEAD_DIM))
        self.assertEqual(out_k_rope.shape, (2, QK_ROPE_HEAD_DIM))
        self.assertEqual(out_v.shape, (2, HEAD_DIM))
        self.assertFalse(torch.isnan(out_k_nope).any())
        self.assertFalse(torch.isnan(out_k_rope).any())
        self.assertFalse(torch.isnan(out_v).any())

    def test_smoke_128_tokens(self):
        """128 tokens → 1 compressed entry."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 256)
        gate_weights = torch.randn(HCA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            128, gate_weights, compress_cos, compress_sin)

        self.assertEqual(out_k_nope.shape[0], 1)

    def test_smoke_below_window(self):
        """64 tokens < window=128 → 0 compressed."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 256)
        gate_weights = torch.randn(HCA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            64, gate_weights, compress_cos, compress_sin)

        self.assertEqual(out_k_nope.shape[0], 0)

    def test_smoke_large(self):
        """4096 tokens → 32 compressed, no crash."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 8192)
        gate_weights = torch.randn(HCA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            4096, gate_weights, compress_cos, compress_sin)

        self.assertEqual(out_k_nope.shape[0], 32)
        self.assertFalse(torch.isnan(out_k_nope).any())


class TestHcaCompressorAccuracy(unittest.TestCase):
    """V4K-1b.4: Accuracy test — kernel vs ref_hca_compress()."""

    @classmethod
    def setUpClass(cls):
        cls.compress_cos, cls.compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 65536)
        cls.gate_weights = torch.randn(HCA_WINDOW)

    def _run_and_compare(self, num_tokens, label=""):
        torch.manual_seed(123)
        device = "cuda"
        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
        input_k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            num_tokens, self.gate_weights, self.compress_cos, self.compress_sin,
            input_k_nope=input_k_nope, input_k_rope_raw=input_k_rope_raw, input_v=input_v)

        ref_k_nope, ref_k_rope, ref_v = ref_hca_compress(
            input_k_nope.cpu().float(), input_k_rope_raw.cpu().float(), input_v.cpu().float(),
            self.gate_weights,
            self.compress_cos.cpu(), self.compress_sin.cpu())

        self.assertEqual(out_k_nope.shape[0], ref_k_nope.shape[0])
        if ref_k_nope.shape[0] == 0:
            return

        cos_k = torch.nn.functional.cosine_similarity(
            out_k_nope.cpu().float().flatten().unsqueeze(0),
            ref_k_nope.float().flatten().unsqueeze(0)).item()
        cos_v = torch.nn.functional.cosine_similarity(
            out_v.cpu().float().flatten().unsqueeze(0),
            ref_v.float().flatten().unsqueeze(0)).item()
        cos_r = torch.nn.functional.cosine_similarity(
            out_k_rope.cpu().float().flatten().unsqueeze(0),
            ref_k_rope.float().flatten().unsqueeze(0)).item()

        print(f"  {label} n={num_tokens} compressed={ref_k_nope.shape[0]}: "
              f"K_nope cos={cos_k:.6f}, K_rope cos={cos_r:.6f}, V cos={cos_v:.6f}")

        self.assertGreater(cos_k, 0.999, f"K_nope cosine {cos_k:.6f} < 0.999")
        self.assertGreater(cos_v, 0.999, f"V cosine {cos_v:.6f} < 0.999")
        self.assertGreater(cos_r, 0.999, f"K_rope cosine {cos_r:.6f} < 0.999")

    def test_accuracy_256_tokens(self):
        """256 tokens → 2 compressed."""
        self._run_and_compare(256, "256tok")

    def test_accuracy_1024_tokens(self):
        """1024 tokens → 8 compressed."""
        self._run_and_compare(1024, "1024tok")

    def test_accuracy_4096_tokens(self):
        """4096 tokens → 32 compressed."""
        self._run_and_compare(4096, "4096tok")

    def test_compression_ratio(self):
        """Verify 128:1 compression ratio."""
        self.assertEqual(1024 // HCA_STRIDE, 8)
        self.assertEqual(256 // HCA_STRIDE, 2)


if __name__ == "__main__":
    unittest.main()
