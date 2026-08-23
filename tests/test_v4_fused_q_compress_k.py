"""
V4K-6c: Fused Q Norm + Compressed K RoPE + K Insert Tests

Single-launch kernel: Q RMSNorm + compressed K RoPE + FP8 cache write.
Must match unfused: separate RMSNorm + separate RoPE + separate k_append.

Usage:
  python tests/test_v4_fused_q_compress_k.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, D_QK, PAGE_SIZE, FP8_MAX,
    V4_FP8_BYTES_PER_ENTRY,
    alloc_v4_fp8_cache, ref_v4_fp8_dequant,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def make_rope_tables(max_pos, rope_dim, theta=160000.0):
    half = rope_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    positions = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles.cos().contiguous(), angles.sin().contiguous()


def ref_rmsnorm(x, eps=1e-6):
    """Pure PyTorch RMSNorm."""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)


class TestFusedQCompressK(unittest.TestCase):
    """V4K-6c: Fused Q norm + K RoPE + cache insert."""

    def test_q_rmsnorm(self):
        """Q RMSNorm portion matches PyTorch reference."""
        torch.manual_seed(42)
        import sm120_mla_kernels

        h_q = 128
        q = torch.randn(h_q, D_QK, dtype=torch.bfloat16, device='cuda')
        q_ref = ref_rmsnorm(q.cpu()).float()

        # Need dummy K/V for the fused call
        k_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        cos_t, sin_t = make_rope_tables(4096, QK_ROPE_HEAD_DIM)
        cache = alloc_v4_fp8_cache(1, PAGE_SIZE).cuda()

        sm120_mla_kernels.v4_fused_q_compress_k(
            q, k_nope, k_rope, v_nope,
            cos_t.cuda(), sin_t.cuda(),
            cache, 0, 100, 1e-6)

        cos = cosine_sim(q.cpu(), q_ref)
        print(f"  Q RMSNorm: cosine={cos:.6f}")
        self.assertGreater(cos, 0.999, f"Q RMSNorm cosine={cos}")

    def test_k_cache_insert(self):
        """K RoPE + cache insert matches unfused k_append."""
        torch.manual_seed(43)
        import sm120_mla_kernels

        h_q = 64
        rope_pos = 47
        cos_t, sin_t = make_rope_tables(4096, QK_ROPE_HEAD_DIM)

        k_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope_raw = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        q = torch.randn(h_q, D_QK, dtype=torch.bfloat16, device='cuda')

        # Fused: writes to cache slot 0
        cache_fused = alloc_v4_fp8_cache(1, PAGE_SIZE).cuda()
        sm120_mla_kernels.v4_fused_q_compress_k(
            q.clone(), k_nope, k_rope_raw, v_nope,
            cos_t.cuda(), sin_t.cuda(),
            cache_fused, 0, rope_pos, 1e-6)

        # Unfused: manually apply RoPE then k_append
        half = QK_ROPE_HEAD_DIM // 2
        k_rope_roped = torch.zeros(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16)
        kr = k_rope_raw.cpu()
        for j in range(half):
            xe = kr[0, 2*j].float()
            xo = kr[0, 2*j+1].float()
            c = cos_t[rope_pos, j]
            s = sin_t[rope_pos, j]
            k_rope_roped[0, 2*j] = (xe * c - xo * s)
            k_rope_roped[0, 2*j+1] = (xe * s + xo * c)

        cache_unfused = alloc_v4_fp8_cache(1, PAGE_SIZE).cuda()
        slot_map = torch.tensor([0], dtype=torch.int32, device='cuda')
        sm120_mla_kernels.v4_fp8_k_append(
            k_nope, k_rope_roped.cuda().to(torch.bfloat16), v_nope,
            cache_unfused, slot_map)

        # Compare via dequant
        k_f, kr_f, v_f = ref_v4_fp8_dequant(cache_fused.cpu(), torch.tensor([0]))
        k_u, kr_u, v_u = ref_v4_fp8_dequant(cache_unfused.cpu(), torch.tensor([0]))

        cos_k = cosine_sim(k_f, k_u)
        cos_kr = cosine_sim(kr_f, kr_u)
        cos_v = cosine_sim(v_f, v_u)
        print(f"  K insert: K={cos_k:.6f} K_rope={cos_kr:.6f} V={cos_v:.6f}")
        self.assertGreater(cos_k, 0.999, f"K cosine={cos_k}")
        self.assertGreater(cos_kr, 0.999, f"K_rope cosine={cos_kr}")
        self.assertGreater(cos_v, 0.999, f"V cosine={cos_v}")

    def test_smoke_h128(self):
        """h_q=128 (V4 Pro) — no crash."""
        torch.manual_seed(44)
        import sm120_mla_kernels

        q = torch.randn(128, D_QK, dtype=torch.bfloat16, device='cuda')
        k_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        v_nope = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        cos_t, sin_t = make_rope_tables(4096, QK_ROPE_HEAD_DIM)
        cache = alloc_v4_fp8_cache(1, PAGE_SIZE).cuda()

        sm120_mla_kernels.v4_fused_q_compress_k(
            q, k_nope, k_rope, v_nope,
            cos_t.cuda(), sin_t.cuda(),
            cache, 0, 500, 1e-6)

        self.assertFalse(torch.isnan(q).any())


if __name__ == '__main__':
    unittest.main()
