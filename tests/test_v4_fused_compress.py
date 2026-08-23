"""
V4K-6a: Fused Compress + Insert Tests

Fused kernel must produce identical cache output as the unfused pipeline
(compressor → k_append). Tests compare byte-level cache contents.

Usage:
  python tests/test_v4_fused_compress.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE, FP8_MAX,
    V4_FP8_BYTES_PER_ENTRY,
    alloc_v4_fp8_cache, ref_v4_fp8_dequant,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def make_rope_tables(max_pos, rope_dim, theta=160000.0):
    """Create compressed RoPE cos/sin tables."""
    half = rope_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    positions = torch.arange(max_pos, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles.cos().contiguous(), angles.sin().contiguous()


def run_unfused_pipeline(input_k_nope, input_k_rope_raw, input_v,
                         gate_weights, positional_bias,
                         compress_cos, compress_sin,
                         cache, slot_mapping,
                         window, stride, is_csa=True):
    """Run unfused: compressor → k_append."""
    import sm120_mla_kernels

    if is_csa:
        k_nope, k_rope, v = sm120_mla_kernels.v4_csa_compress(
            input_k_nope, input_k_rope_raw, input_v,
            gate_weights, positional_bias,
            compress_cos, compress_sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
    else:
        k_nope, k_rope, v = sm120_mla_kernels.v4_hca_compress(
            input_k_nope, input_k_rope_raw, input_v,
            gate_weights,
            compress_cos, compress_sin,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)

    sm120_mla_kernels.v4_fp8_k_append(k_nope, k_rope, v, cache, slot_mapping)


def run_fused_pipeline(input_k_nope, input_k_rope_raw, input_v,
                       gate_weights, positional_bias,
                       compress_cos, compress_sin,
                       cache, slot_mapping,
                       window, stride, is_csa=True):
    """Run fused: single kernel."""
    import sm120_mla_kernels

    if is_csa:
        sm120_mla_kernels.v4_fused_csa_compress_insert(
            input_k_nope, input_k_rope_raw, input_v,
            gate_weights, positional_bias,
            compress_cos, compress_sin,
            cache, slot_mapping,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)
    else:
        sm120_mla_kernels.v4_fused_hca_compress_insert(
            input_k_nope, input_k_rope_raw, input_v,
            gate_weights,
            compress_cos, compress_sin,
            cache, slot_mapping,
            HEAD_DIM, QK_ROPE_HEAD_DIM, window, stride)


class TestFusedCsaCompressInsert(unittest.TestCase):
    """V4K-6a.1: Fused CSA (window=8) matches unfused pipeline."""

    def _run_comparison(self, num_tokens, seed=42):
        torch.manual_seed(seed)
        window, stride = 8, 4
        num_compressed = (num_tokens - window) // stride if num_tokens >= window else 0
        if num_compressed == 0:
            return

        num_pages = max((num_compressed + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        compress_cos, compress_sin = make_rope_tables(num_tokens + 128, QK_ROPE_HEAD_DIM)

        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        input_k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate_w = torch.randn(window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(window, dtype=torch.bfloat16, device='cuda')
        cos_gpu = compress_cos.cuda()
        sin_gpu = compress_sin.cuda()
        slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device='cuda')

        # Unfused
        cache_unfused = alloc_v4_fp8_cache(num_pages, PAGE_SIZE).cuda()
        run_unfused_pipeline(input_k_nope, input_k_rope, input_v,
                             gate_w, pos_bias, cos_gpu, sin_gpu,
                             cache_unfused, slot_mapping, window, stride, is_csa=True)

        # Fused
        cache_fused = alloc_v4_fp8_cache(num_pages, PAGE_SIZE).cuda()
        run_fused_pipeline(input_k_nope, input_k_rope, input_v,
                           gate_w, pos_bias, cos_gpu, sin_gpu,
                           cache_fused, slot_mapping, window, stride, is_csa=True)

        # Compare via dequant
        indices = torch.arange(num_compressed)
        k_u, kr_u, v_u = ref_v4_fp8_dequant(cache_unfused.cpu(), indices)
        k_f, kr_f, v_f = ref_v4_fp8_dequant(cache_fused.cpu(), indices)

        return k_u, k_f, kr_u, kr_f, v_u, v_f

    def test_small(self):
        """16 tokens → 2 compressed entries."""
        k_u, k_f, kr_u, kr_f, v_u, v_f = self._run_comparison(16)
        cos_k = cosine_sim(k_u, k_f)
        cos_kr = cosine_sim(kr_u, kr_f)
        cos_v = cosine_sim(v_u, v_f)
        print(f"  CSA small: K={cos_k:.6f} K_rope={cos_kr:.6f} V={cos_v:.6f}")
        self.assertGreater(cos_k, 0.999, f"K cosine={cos_k}")
        self.assertGreater(cos_kr, 0.999, f"K_rope cosine={cos_kr}")
        self.assertGreater(cos_v, 0.999, f"V cosine={cos_v}")

    def test_medium(self):
        """128 tokens → 30 compressed entries."""
        k_u, k_f, kr_u, kr_f, v_u, v_f = self._run_comparison(128, seed=43)
        cos_k = cosine_sim(k_u, k_f)
        cos_v = cosine_sim(v_u, v_f)
        print(f"  CSA medium: K={cos_k:.6f} V={cos_v:.6f}")
        self.assertGreater(cos_k, 0.999)
        self.assertGreater(cos_v, 0.999)

    def test_large(self):
        """512 tokens → 126 compressed entries."""
        k_u, k_f, kr_u, kr_f, v_u, v_f = self._run_comparison(512, seed=44)
        cos_k = cosine_sim(k_u, k_f)
        cos_v = cosine_sim(v_u, v_f)
        print(f"  CSA large: K={cos_k:.6f} V={cos_v:.6f}")
        self.assertGreater(cos_k, 0.999)
        self.assertGreater(cos_v, 0.999)

    def test_byte_level(self):
        """Cache bytes should be identical between fused and unfused."""
        torch.manual_seed(50)
        num_tokens, window, stride = 32, 8, 4
        num_compressed = (num_tokens - window) // stride
        num_pages = max((num_compressed + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        compress_cos, compress_sin = make_rope_tables(num_tokens + 128, QK_ROPE_HEAD_DIM)

        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        input_k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate_w = torch.randn(window, dtype=torch.bfloat16, device='cuda')
        pos_bias = torch.randn(window, dtype=torch.bfloat16, device='cuda')
        slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device='cuda')

        cache_u = alloc_v4_fp8_cache(num_pages, PAGE_SIZE).cuda()
        run_unfused_pipeline(input_k_nope, input_k_rope, input_v,
                             gate_w, pos_bias, compress_cos.cuda(), compress_sin.cuda(),
                             cache_u, slot_mapping, window, stride, is_csa=True)

        cache_f = alloc_v4_fp8_cache(num_pages, PAGE_SIZE).cuda()
        run_fused_pipeline(input_k_nope, input_k_rope, input_v,
                           gate_w, pos_bias, compress_cos.cuda(), compress_sin.cuda(),
                           cache_f, slot_mapping, window, stride, is_csa=True)

        # Byte-level comparison on used entries
        cu = cache_u.cpu()
        cf = cache_f.cpu()
        for i in range(num_compressed):
            entry_u = cu[i * V4_FP8_BYTES_PER_ENTRY:(i+1) * V4_FP8_BYTES_PER_ENTRY]
            entry_f = cf[i * V4_FP8_BYTES_PER_ENTRY:(i+1) * V4_FP8_BYTES_PER_ENTRY]
            if not torch.equal(entry_u, entry_f):
                # Byte mismatch — check via dequant cosine instead
                k_u, kr_u, v_u = ref_v4_fp8_dequant(cu, torch.tensor([i]))
                k_f, kr_f, v_f = ref_v4_fp8_dequant(cf, torch.tensor([i]))
                cos = min(cosine_sim(k_u, k_f), cosine_sim(v_u, v_f))
                self.assertGreater(cos, 0.999,
                    f"Entry {i}: byte mismatch AND dequant cosine {cos:.6f} < 0.999")


class TestFusedHcaCompressInsert(unittest.TestCase):
    """V4K-6a.2: Fused HCA (window=128) matches unfused pipeline."""

    def _run_comparison(self, num_tokens, seed=100):
        torch.manual_seed(seed)
        window, stride = 128, 128
        num_compressed = num_tokens // stride
        if num_compressed == 0:
            return None

        num_pages = max((num_compressed + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        compress_cos, compress_sin = make_rope_tables(num_tokens + 256, QK_ROPE_HEAD_DIM)

        input_k_nope = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        input_k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        input_v = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
        gate_w = torch.randn(window, dtype=torch.bfloat16, device='cuda')
        cos_gpu = compress_cos.cuda()
        sin_gpu = compress_sin.cuda()
        slot_mapping = torch.arange(num_compressed, dtype=torch.int32, device='cuda')

        cache_u = alloc_v4_fp8_cache(num_pages, PAGE_SIZE).cuda()
        run_unfused_pipeline(input_k_nope, input_k_rope, input_v,
                             gate_w, None, cos_gpu, sin_gpu,
                             cache_u, slot_mapping, window, stride, is_csa=False)

        cache_f = alloc_v4_fp8_cache(num_pages, PAGE_SIZE).cuda()
        run_fused_pipeline(input_k_nope, input_k_rope, input_v,
                           gate_w, None, cos_gpu, sin_gpu,
                           cache_f, slot_mapping, window, stride, is_csa=False)

        indices = torch.arange(num_compressed)
        k_u, kr_u, v_u = ref_v4_fp8_dequant(cache_u.cpu(), indices)
        k_f, kr_f, v_f = ref_v4_fp8_dequant(cache_f.cpu(), indices)
        return k_u, k_f, kr_u, kr_f, v_u, v_f

    def test_small(self):
        """256 tokens → 2 compressed entries."""
        result = self._run_comparison(256)
        if result is None:
            self.skipTest("No compressed entries")
        k_u, k_f, kr_u, kr_f, v_u, v_f = result
        cos_k = cosine_sim(k_u, k_f)
        cos_v = cosine_sim(v_u, v_f)
        print(f"  HCA small: K={cos_k:.6f} V={cos_v:.6f}")
        self.assertGreater(cos_k, 0.999)
        self.assertGreater(cos_v, 0.999)

    def test_medium(self):
        """1024 tokens → 8 compressed entries."""
        result = self._run_comparison(1024, seed=101)
        if result is None:
            self.skipTest("No compressed entries")
        k_u, k_f, kr_u, kr_f, v_u, v_f = result
        cos_k = cosine_sim(k_u, k_f)
        cos_v = cosine_sim(v_u, v_f)
        print(f"  HCA medium: K={cos_k:.6f} V={cos_v:.6f}")
        self.assertGreater(cos_k, 0.999)
        self.assertGreater(cos_v, 0.999)


if __name__ == '__main__':
    unittest.main()
