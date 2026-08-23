"""
V4K-5b: Inverse RoPE Kernel Tests

Accuracy test: apply RoPE then inverse RoPE = identity within BF16 tolerance.
Tests at multiple positions and with both standard and compressed RoPE theta.

Usage:
  python tests/test_v4_inverse_rope.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    QK_ROPE_HEAD_DIM, ROPE_THETA, COMPRESS_ROPE_THETA,
    precompute_rope_freqs, apply_rope, ref_inverse_rope,
)


def run_inverse_rope_kernel(x_bf16, cos_table, sin_table, positions):
    """Run the CUDA inverse_rope kernel."""
    import sm120_mla_kernels

    x_gpu = x_bf16.to(device='cuda', dtype=torch.bfloat16).contiguous()
    cos_gpu = cos_table.to(device='cuda', dtype=torch.float32).contiguous()
    sin_gpu = sin_table.to(device='cuda', dtype=torch.float32).contiguous()
    pos_gpu = positions.to(device='cuda', dtype=torch.int32).contiguous()

    out = sm120_mla_kernels.v4_inverse_rope(x_gpu, cos_gpu, sin_gpu, pos_gpu)
    return out.cpu()


class TestInverseRopeSmoke(unittest.TestCase):
    """V4K-5b.1: Smoke test — kernel runs without crash."""

    def test_smoke_basic(self):
        """4 rows, standard theta, no crash."""
        torch.manual_seed(42)
        dim = QK_ROPE_HEAD_DIM
        cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 1024)

        x = torch.randn(4, dim, dtype=torch.bfloat16)
        positions = torch.tensor([0, 1, 42, 500], dtype=torch.int32)

        out = run_inverse_rope_kernel(x, cos, sin, positions)
        self.assertEqual(out.shape, (4, dim))
        self.assertTrue(torch.isfinite(out).all())

    def test_smoke_single(self):
        """Single row."""
        dim = QK_ROPE_HEAD_DIM
        cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 64)
        x = torch.randn(1, dim, dtype=torch.bfloat16)
        positions = torch.tensor([0], dtype=torch.int32)

        out = run_inverse_rope_kernel(x, cos, sin, positions)
        self.assertEqual(out.shape, (1, dim))

    def test_smoke_large_batch(self):
        """16K rows — realistic batch*heads."""
        torch.manual_seed(99)
        dim = QK_ROPE_HEAD_DIM
        N = 16384
        cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 2048)

        x = torch.randn(N, dim, dtype=torch.bfloat16)
        positions = torch.randint(0, 2048, (N,), dtype=torch.int32)

        out = run_inverse_rope_kernel(x, cos, sin, positions)
        self.assertEqual(out.shape, (N, dim))
        self.assertTrue(torch.isfinite(out).all())


class TestInverseRopeAccuracy(unittest.TestCase):
    """V4K-5b.2: Accuracy test — round-trip and reference match."""

    def test_roundtrip_standard_theta(self):
        """RoPE then inverse = identity (standard theta=10000)."""
        torch.manual_seed(42)
        dim = QK_ROPE_HEAD_DIM
        cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 1024)

        for pos_val in [0, 1, 42, 500, 1023]:
            x = torch.randn(4, dim)
            positions = torch.full((4,), pos_val, dtype=torch.long)

            x_roped = apply_rope(x, cos, sin, positions)
            x_roped_bf16 = x_roped.to(torch.bfloat16)
            pos_i32 = positions.to(torch.int32)

            x_recovered = run_inverse_rope_kernel(x_roped_bf16, cos, sin, pos_i32)

            # Compare against original x (BF16 precision: ~1e-2 relative)
            max_diff = (x_recovered.float() - x.float()).abs().max().item()
            self.assertLess(max_diff, 0.05,
                f"Round-trip failed at pos={pos_val}: max_diff={max_diff:.4f}")

    def test_roundtrip_compressed_theta(self):
        """RoPE then inverse = identity (compressed theta=160000)."""
        torch.manual_seed(42)
        dim = QK_ROPE_HEAD_DIM
        cos, sin = precompute_rope_freqs(COMPRESS_ROPE_THETA, dim, 2048)

        x = torch.randn(8, dim)
        positions = torch.tensor([127, 255, 383, 511, 639, 767, 895, 1023], dtype=torch.long)

        x_roped = apply_rope(x, cos, sin, positions)
        x_roped_bf16 = x_roped.to(torch.bfloat16)
        pos_i32 = positions.to(torch.int32)

        x_recovered = run_inverse_rope_kernel(x_roped_bf16, cos, sin, pos_i32)

        max_diff = (x_recovered.float() - x.float()).abs().max().item()
        self.assertLess(max_diff, 0.05,
            f"Compressed theta round-trip max_diff={max_diff:.4f}")

    def test_matches_reference(self):
        """Kernel output matches ref_inverse_rope() within BF16 tolerance."""
        torch.manual_seed(55)
        dim = QK_ROPE_HEAD_DIM
        cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 1024)

        x = torch.randn(16, dim)
        positions = torch.randint(0, 1024, (16,))

        ref_out = ref_inverse_rope(x, cos, sin, positions)

        x_bf16 = x.to(torch.bfloat16)
        pos_i32 = positions.to(torch.int32)
        kernel_out = run_inverse_rope_kernel(x_bf16, cos, sin, pos_i32)

        # Kernel operates on BF16 input; ref on FP32. Tolerance for BF16 quantization.
        max_diff = (kernel_out.float() - ref_out.float()).abs().max().item()
        self.assertLess(max_diff, 0.05,
            f"Kernel vs reference max_diff={max_diff:.4f}")

        cos_sim = torch.nn.functional.cosine_similarity(
            kernel_out.float().flatten().unsqueeze(0),
            ref_out.float().flatten().unsqueeze(0)).item()
        self.assertGreater(cos_sim, 0.999,
            f"Cosine similarity {cos_sim:.6f} below 0.999")

    def test_position_zero_identity_ish(self):
        """At position 0, cos=1, sin=0 → inverse RoPE is identity."""
        dim = QK_ROPE_HEAD_DIM
        cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 8)

        x = torch.randn(2, dim, dtype=torch.bfloat16)
        positions = torch.zeros(2, dtype=torch.int32)

        out = run_inverse_rope_kernel(x, cos, sin, positions)

        max_diff = (out.float() - x.float()).abs().max().item()
        self.assertLess(max_diff, 1e-6,
            f"Position 0 should be near-identity, got max_diff={max_diff}")


if __name__ == "__main__":
    unittest.main()
