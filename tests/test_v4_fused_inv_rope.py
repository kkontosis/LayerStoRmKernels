"""
V4K-6b: Fused Inverse RoPE + FP8 Quantization Tests

Fused kernel must match the unfused pipeline:
  1. Apply inverse RoPE to last 64 dims of attention output
  2. FP8-quantize the full 512-dim output

Usage:
  python tests/test_v4_fused_inv_rope.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, FP8_MAX,
    ref_inverse_rope,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def make_rope_tables(max_pos, rope_dim, theta=10000.0):
    half = rope_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    positions = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles.cos().contiguous(), angles.sin().contiguous()


def unfused_inv_rope_fp8(x_bf16, cos_table, sin_table, positions, rope_dim):
    """Unfused pipeline: inverse RoPE (CPU) then FP8 quant (manual)."""
    import sm120_mla_kernels

    N, hd = x_bf16.shape
    nope_dim = hd - rope_dim

    # Step 1: inverse RoPE on last rope_dim dims
    inv_rope_out = sm120_mla_kernels.v4_inverse_rope(
        x_bf16[:, nope_dim:].cuda().contiguous(),
        cos_table.cuda(), sin_table.cuda(),
        positions.cuda().to(torch.int32),
    ).cpu()

    x_with_inv = torch.cat([x_bf16[:, :nope_dim], inv_rope_out.to(torch.bfloat16)], dim=-1)

    # Step 2: FP8 quantize (CPU reference)
    x_f32 = x_with_inv.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scales = (amax / FP8_MAX).squeeze(-1)
    inv_scales = torch.where(scales > 0, 1.0 / scales, torch.zeros_like(scales)).unsqueeze(-1)
    quantized = (x_f32 * inv_scales).clamp(-FP8_MAX, FP8_MAX)

    return quantized, scales


def run_fused(x_bf16, cos_table, sin_table, positions, rope_dim):
    """Run fused kernel."""
    import sm120_mla_kernels
    out_fp8, out_scales = sm120_mla_kernels.v4_fused_inv_rope_fp8(
        x_bf16.cuda().contiguous(),
        cos_table.cuda(), sin_table.cuda(),
        positions.cuda().to(torch.int32),
        rope_dim,
    )

    # Dequant for comparison
    out_f32 = out_fp8.float().cpu()
    scales_cpu = out_scales.cpu()
    dequant = out_f32 * scales_cpu.unsqueeze(-1)
    return dequant, scales_cpu


class TestFusedInvRopeFp8(unittest.TestCase):
    """V4K-6b: Fused inverse RoPE + FP8 quant matches unfused."""

    def _run(self, N, seed=42):
        torch.manual_seed(seed)
        cos_table, sin_table = make_rope_tables(4096, QK_ROPE_HEAD_DIM)
        x = torch.randn(N, HEAD_DIM, dtype=torch.bfloat16) * 0.5
        positions = torch.randint(0, 1000, (N,), dtype=torch.int32)

        deq_fused, scales_fused = run_fused(x, cos_table, sin_table, positions, QK_ROPE_HEAD_DIM)
        deq_unfused, scales_unfused = unfused_inv_rope_fp8(x, cos_table, sin_table, positions, QK_ROPE_HEAD_DIM)

        return deq_fused, deq_unfused

    def test_small(self):
        """N=8."""
        fused, unfused = self._run(8)
        cos = cosine_sim(fused, unfused)
        print(f"  small: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos}")

    def test_medium(self):
        """N=256."""
        fused, unfused = self._run(256, seed=43)
        cos = cosine_sim(fused, unfused)
        print(f"  medium: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos}")

    def test_large(self):
        """N=4096."""
        fused, unfused = self._run(4096, seed=44)
        cos = cosine_sim(fused, unfused)
        print(f"  large: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99, f"cosine={cos}")

    def test_inverse_rope_correctness(self):
        """Verify the inverse RoPE portion is correct: apply RoPE then fused inverse should recover."""
        torch.manual_seed(50)
        N = 64
        cos_table, sin_table = make_rope_tables(4096, QK_ROPE_HEAD_DIM)
        positions = torch.randint(0, 1000, (N,), dtype=torch.int32)

        # Create an output where rope dims have known RoPE applied
        x_nope = torch.randn(N, HEAD_DIM - QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        x_rope_orig = torch.randn(N, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)

        # Apply forward RoPE to rope dims
        half = QK_ROPE_HEAD_DIM // 2
        x_roped = torch.zeros_like(x_rope_orig)
        for i in range(N):
            pos = positions[i].item()
            c = cos_table[pos]
            s = sin_table[pos]
            for j in range(half):
                xe = x_rope_orig[i, 2*j].float()
                xo = x_rope_orig[i, 2*j+1].float()
                x_roped[i, 2*j] = (xe * c[j] - xo * s[j])
                x_roped[i, 2*j+1] = (xe * s[j] + xo * c[j])
        x_roped = x_roped.to(torch.bfloat16)

        x_full = torch.cat([x_nope, x_roped], dim=-1)

        # Fused: should undo the RoPE, then quantize
        deq_fused, _ = run_fused(x_full, cos_table, sin_table, positions, QK_ROPE_HEAD_DIM)

        # The rope dims should recover the original values (within FP8+BF16 tolerance)
        rope_recovered = deq_fused[:, HEAD_DIM - QK_ROPE_HEAD_DIM:]
        cos = cosine_sim(rope_recovered, x_rope_orig.float())
        print(f"  roundtrip rope: cosine={cos:.6f}")
        self.assertGreater(cos, 0.99, f"Rope roundtrip cosine={cos}")


if __name__ == '__main__':
    unittest.main()
