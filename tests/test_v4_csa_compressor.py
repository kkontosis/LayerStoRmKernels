"""
V4K-1a: CSA Compressor Kernel Tests

Smoke test (V4K-1a.3): kernel builds, runs on 16-token input without crash/NaN.
Accuracy test (V4K-1a.4): kernel output vs ref_csa_compress() golden reference.

Usage:
  python tests/test_v4_csa_compressor.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, CSA_WINDOW, CSA_STRIDE,
    COMPRESS_ROPE_THETA, precompute_rope_freqs, ref_csa_compress,
)


def run_kernel(num_tokens, gate_weights, positional_bias, compress_cos, compress_sin,
               input_k_nope=None, input_k_rope_raw=None, input_v=None, seed=42):
    """Helper: generate random inputs (if not provided) and run the CUDA kernel."""
    torch.manual_seed(seed)
    device = "cuda"

    if input_k_nope is None:
        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if input_k_rope_raw is None:
        input_k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
    if input_v is None:
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)

    gw = gate_weights.to(device=device, dtype=torch.bfloat16)
    pb = positional_bias.to(device=device, dtype=torch.bfloat16)
    cc = compress_cos.to(device=device, dtype=torch.float32)
    cs = compress_sin.to(device=device, dtype=torch.float32)

    import sm120_mla_kernels
    out_k_nope, out_k_rope, out_v = sm120_mla_kernels.v4_csa_compress(
        input_k_nope, input_k_rope_raw, input_v,
        gw, pb, cc, cs,
        HEAD_DIM, QK_ROPE_HEAD_DIM, CSA_WINDOW, CSA_STRIDE,
    )
    return out_k_nope, out_k_rope, out_v, input_k_nope, input_k_rope_raw, input_v


class TestCsaCompressorSmoke(unittest.TestCase):
    """V4K-1a.3: Smoke test — kernel runs without crash or NaN."""

    def test_smoke_16_tokens(self):
        """16 tokens → 2 compressed entries, no crash."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 256)
        gate_weights = torch.randn(CSA_WINDOW)
        positional_bias = torch.randn(CSA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            16, gate_weights, positional_bias, compress_cos, compress_sin)

        num_compressed = max(0, (16 - CSA_WINDOW) // CSA_STRIDE)
        self.assertEqual(out_k_nope.shape, (num_compressed, HEAD_DIM))
        self.assertEqual(out_k_rope.shape, (num_compressed, QK_ROPE_HEAD_DIM))
        self.assertEqual(out_v.shape, (num_compressed, HEAD_DIM))
        self.assertFalse(torch.isnan(out_k_nope).any(), "NaN in out_k_nope")
        self.assertFalse(torch.isnan(out_k_rope).any(), "NaN in out_k_rope")
        self.assertFalse(torch.isnan(out_v).any(), "NaN in out_v")

    def test_smoke_edge_exact_window(self):
        """8 tokens = exactly 1 window → 0 compressed (stride formula)."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 256)
        gate_weights = torch.randn(CSA_WINDOW)
        positional_bias = torch.randn(CSA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            8, gate_weights, positional_bias, compress_cos, compress_sin)

        num_compressed = max(0, (8 - CSA_WINDOW) // CSA_STRIDE)
        self.assertEqual(num_compressed, 0)
        self.assertEqual(out_k_nope.shape[0], 0)

    def test_smoke_large(self):
        """1024 tokens → many compressed entries, no crash."""
        compress_cos, compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 2048)
        gate_weights = torch.randn(CSA_WINDOW)
        positional_bias = torch.randn(CSA_WINDOW)

        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            1024, gate_weights, positional_bias, compress_cos, compress_sin)

        num_compressed = (1024 - CSA_WINDOW) // CSA_STRIDE
        self.assertEqual(out_k_nope.shape[0], num_compressed)
        self.assertFalse(torch.isnan(out_k_nope).any())
        self.assertFalse(torch.isnan(out_v).any())


class TestCsaCompressorAccuracy(unittest.TestCase):
    """V4K-1a.4: Accuracy test — kernel vs ref_csa_compress()."""

    @classmethod
    def setUpClass(cls):
        cls.compress_cos, cls.compress_sin = precompute_rope_freqs(
            COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 4096)
        cls.gate_weights = torch.randn(CSA_WINDOW)
        cls.positional_bias = torch.randn(CSA_WINDOW)

    def _run_and_compare(self, num_tokens, label=""):
        compress_cos = self.compress_cos
        compress_sin = self.compress_sin
        gate_weights = self.gate_weights
        positional_bias = self.positional_bias

        torch.manual_seed(123)
        device = "cuda"
        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
        input_k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)

        # Run CUDA kernel
        out_k_nope, out_k_rope, out_v, _, _, _ = run_kernel(
            num_tokens, gate_weights, positional_bias, compress_cos, compress_sin,
            input_k_nope=input_k_nope, input_k_rope_raw=input_k_rope_raw, input_v=input_v)

        # Run PyTorch reference (on CPU for exact math)
        ref_k_nope, ref_k_rope, ref_v, _ = ref_csa_compress(
            input_k_nope.cpu().float(), input_k_rope_raw.cpu().float(), input_v.cpu().float(),
            gate_weights, positional_bias,
            compress_cos.cpu(), compress_sin.cpu())

        self.assertEqual(out_k_nope.shape[0], ref_k_nope.shape[0],
                         f"num_compressed mismatch: kernel={out_k_nope.shape[0]} ref={ref_k_nope.shape[0]}")

        if ref_k_nope.shape[0] == 0:
            return

        # Compare on CPU in float32
        k_kern = out_k_nope.cpu().float()
        k_ref = ref_k_nope.float()
        cos_k = torch.nn.functional.cosine_similarity(k_kern.flatten().unsqueeze(0),
                                                       k_ref.flatten().unsqueeze(0)).item()

        v_kern = out_v.cpu().float()
        v_ref = ref_v.float()
        cos_v = torch.nn.functional.cosine_similarity(v_kern.flatten().unsqueeze(0),
                                                       v_ref.flatten().unsqueeze(0)).item()

        r_kern = out_k_rope.cpu().float()
        r_ref = ref_k_rope.float()
        cos_r = torch.nn.functional.cosine_similarity(r_kern.flatten().unsqueeze(0),
                                                       r_ref.flatten().unsqueeze(0)).item()

        print(f"  {label} n={num_tokens} compressed={ref_k_nope.shape[0]}: "
              f"K_nope cos={cos_k:.6f}, K_rope cos={cos_r:.6f}, V cos={cos_v:.6f}")

        self.assertGreater(cos_k, 0.999, f"K_nope cosine {cos_k:.6f} < 0.999")
        self.assertGreater(cos_v, 0.999, f"V cosine {cos_v:.6f} < 0.999")
        self.assertGreater(cos_r, 0.999, f"K_rope cosine {cos_r:.6f} < 0.999")

    def test_accuracy_16_tokens(self):
        """16 tokens → 2 compressed: kernel matches reference."""
        self._run_and_compare(16, "16tok")

    def test_accuracy_64_tokens(self):
        """64 tokens → 14 compressed."""
        self._run_and_compare(64, "64tok")

    def test_accuracy_256_tokens(self):
        """256 tokens → 62 compressed."""
        self._run_and_compare(256, "256tok")

    def test_accuracy_1024_tokens(self):
        """1024 tokens → 254 compressed."""
        self._run_and_compare(1024, "1024tok")

    def test_compression_ratio(self):
        """Verify 4:1 compression ratio."""
        num_tokens = 100
        num_compressed = (num_tokens - CSA_WINDOW) // CSA_STRIDE
        self.assertEqual(num_compressed, 23)


if __name__ == "__main__":
    unittest.main()
