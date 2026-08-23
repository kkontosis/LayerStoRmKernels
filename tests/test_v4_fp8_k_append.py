"""
V4K-5c: V4 FP8 K-Append Kernel Tests

Smoke test: kernel builds, writes to cache without crash.
Accuracy test: round-trip (write → read back → compare) matches reference.
K NOPE cosine > 0.999, K ROPE BF16 exact, V NOPE cosine > 0.999.

Usage:
  python tests/test_v4_fp8_k_append.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE,
    V4_FP8_BYTES_PER_ENTRY,
    V4_FP8_K_NOPE_OFFSET, V4_FP8_K_NOPE_BYTES,
    V4_FP8_K_SCALE_OFFSET,
    V4_FP8_K_ROPE_OFFSET, V4_FP8_K_ROPE_BYTES,
    V4_FP8_V_NOPE_OFFSET, V4_FP8_V_NOPE_BYTES,
    V4_FP8_V_SCALE_OFFSET,
    alloc_v4_fp8_cache, ref_v4_fp8_k_append, ref_v4_fp8_dequant,
)
import struct


def run_k_append_kernel(k_nope, k_rope, v_nope, kv_cache, slot_mapping):
    """Run the CUDA v4_fp8_k_append kernel."""
    import sm120_mla_kernels

    k_nope_gpu = k_nope.to(device='cuda', dtype=torch.bfloat16).contiguous()
    k_rope_gpu = k_rope.to(device='cuda', dtype=torch.bfloat16).contiguous()
    v_nope_gpu = v_nope.to(device='cuda', dtype=torch.bfloat16).contiguous()
    cache_gpu = kv_cache.to(device='cuda').contiguous()
    slots_gpu = slot_mapping.to(device='cuda', dtype=torch.int32).contiguous()

    sm120_mla_kernels.v4_fp8_k_append(
        k_nope_gpu, k_rope_gpu, v_nope_gpu, cache_gpu, slots_gpu)

    return cache_gpu.cpu()


def read_entry_from_cache(cache, slot):
    """Read one V4 FP8 entry and dequantize."""
    offset = slot * V4_FP8_BYTES_PER_ENTRY

    # K NOPE FP8 → float
    k_fp8_raw = cache[offset:offset + V4_FP8_K_NOPE_BYTES].clone()
    k_fp8 = k_fp8_raw.view(torch.float8_e4m3fn)
    k_scale_bytes = cache[offset + V4_FP8_K_SCALE_OFFSET:
                          offset + V4_FP8_K_SCALE_OFFSET + 4].numpy().tobytes()
    k_scale = struct.unpack('<f', k_scale_bytes)[0]
    k_nope = k_fp8.float() * k_scale

    # K ROPE BF16
    k_rope_raw = cache[offset + V4_FP8_K_ROPE_OFFSET:
                       offset + V4_FP8_K_ROPE_OFFSET + V4_FP8_K_ROPE_BYTES].clone()
    k_rope = k_rope_raw.view(torch.bfloat16).float()

    # V NOPE FP8 → float
    v_fp8_raw = cache[offset + V4_FP8_V_NOPE_OFFSET:
                      offset + V4_FP8_V_NOPE_OFFSET + V4_FP8_V_NOPE_BYTES].clone()
    v_fp8 = v_fp8_raw.view(torch.float8_e4m3fn)
    v_scale_bytes = cache[offset + V4_FP8_V_SCALE_OFFSET:
                          offset + V4_FP8_V_SCALE_OFFSET + 4].numpy().tobytes()
    v_scale = struct.unpack('<f', v_scale_bytes)[0]
    v_nope = v_fp8.float() * v_scale

    return k_nope, k_rope, v_nope


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


class TestV4Fp8KAppendSmoke(unittest.TestCase):
    """V4K-5c.1: Smoke test — kernel writes to cache without crash."""

    def test_smoke_basic(self):
        """8 tokens, sequential slots."""
        torch.manual_seed(42)
        num_tokens = 8
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
        cache = alloc_v4_fp8_cache(1)

        cache_out = run_k_append_kernel(k_nope, k_rope, v_nope, cache, slot_mapping)
        self.assertTrue(cache_out.sum().item() != 0, "Cache should not be all zeros")

    def test_smoke_single_token(self):
        """Single token."""
        k_nope = torch.randn(1, HEAD_DIM)
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(1, HEAD_DIM)
        slot_mapping = torch.tensor([0], dtype=torch.int32)
        cache = alloc_v4_fp8_cache(1)

        cache_out = run_k_append_kernel(k_nope, k_rope, v_nope, cache, slot_mapping)
        self.assertTrue(cache_out.sum().item() != 0)

    def test_smoke_nonsequential_slots(self):
        """Non-sequential slot mapping."""
        torch.manual_seed(55)
        num_tokens = 4
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)
        slot_mapping = torch.tensor([3, 7, 1, 60], dtype=torch.int32)
        cache = alloc_v4_fp8_cache(2)  # 2 pages = 128 slots

        cache_out = run_k_append_kernel(k_nope, k_rope, v_nope, cache, slot_mapping)
        self.assertTrue(cache_out.sum().item() != 0)


class TestV4Fp8KAppendAccuracy(unittest.TestCase):
    """V4K-5c.2: Accuracy test — round-trip matches reference."""

    def test_roundtrip_cosine(self):
        """K NOPE and V NOPE cosine > 0.999, K ROPE BF16 exact."""
        torch.manual_seed(42)
        num_tokens = 8
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
        cache = alloc_v4_fp8_cache(1)

        cache_out = run_k_append_kernel(k_nope, k_rope, v_nope, cache, slot_mapping)

        for i in range(num_tokens):
            k_n, k_r, v_n = read_entry_from_cache(cache_out, i)

            # K NOPE
            cos_k = cosine_sim(k_nope[i], k_n)
            self.assertGreater(cos_k, 0.999,
                f"Token {i} K NOPE cosine {cos_k:.6f} < 0.999")

            # K ROPE (BF16 — should be near-exact vs BF16 input)
            k_rope_bf16 = k_rope[i].to(torch.bfloat16).float()
            max_diff = (k_r - k_rope_bf16).abs().max().item()
            self.assertLess(max_diff, 1e-3,
                f"Token {i} K ROPE diff {max_diff:.2e}")

            # V NOPE
            cos_v = cosine_sim(v_nope[i], v_n)
            self.assertGreater(cos_v, 0.999,
                f"Token {i} V NOPE cosine {cos_v:.6f} < 0.999")

    def test_matches_reference(self):
        """Kernel cache matches ref_v4_fp8_k_append + ref_v4_fp8_dequant."""
        torch.manual_seed(77)
        num_tokens = 16
        k_nope = torch.randn(num_tokens, HEAD_DIM)
        k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(num_tokens, HEAD_DIM)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

        # Reference
        ref_cache = alloc_v4_fp8_cache(1)
        ref_v4_fp8_k_append(k_nope, k_rope, v_nope, ref_cache, slot_mapping)
        ref_kn, ref_kr, ref_vn = ref_v4_fp8_dequant(ref_cache, list(range(num_tokens)))

        # Kernel
        kernel_cache = alloc_v4_fp8_cache(1)
        kernel_cache = run_k_append_kernel(k_nope, k_rope, v_nope, kernel_cache, slot_mapping)

        for i in range(num_tokens):
            k_n, k_r, v_n = read_entry_from_cache(kernel_cache, i)

            # Compare dequantized K NOPE
            cos_k = cosine_sim(ref_kn[i], k_n)
            self.assertGreater(cos_k, 0.999,
                f"Token {i} K NOPE vs ref cosine {cos_k:.6f}")

            # Compare K ROPE (both BF16)
            diff_r = (k_r - ref_kr[i]).abs().max().item()
            self.assertLess(diff_r, 1e-3,
                f"Token {i} K ROPE vs ref diff {diff_r:.2e}")

            # Compare dequantized V NOPE
            cos_v = cosine_sim(ref_vn[i], v_n)
            self.assertGreater(cos_v, 0.999,
                f"Token {i} V NOPE vs ref cosine {cos_v:.6f}")

    def test_scale_correctness(self):
        """Scale values are sensible (amax / 448)."""
        torch.manual_seed(33)
        k_nope = torch.randn(1, HEAD_DIM) * 5.0
        k_rope = torch.randn(1, QK_ROPE_HEAD_DIM)
        v_nope = torch.randn(1, HEAD_DIM) * 2.0
        slot_mapping = torch.tensor([0], dtype=torch.int32)
        cache = alloc_v4_fp8_cache(1)

        cache_out = run_k_append_kernel(k_nope, k_rope, v_nope, cache, slot_mapping)

        # Read K scale
        k_scale_bytes = cache_out[V4_FP8_K_SCALE_OFFSET:
                                  V4_FP8_K_SCALE_OFFSET + 4].numpy().tobytes()
        k_scale = struct.unpack('<f', k_scale_bytes)[0]

        expected_k_amax = k_nope.to(torch.bfloat16).float().abs().max().item()
        expected_k_scale = expected_k_amax / 448.0
        self.assertAlmostEqual(k_scale, expected_k_scale, places=4,
            msg=f"K scale {k_scale:.6f} != expected {expected_k_scale:.6f}")

        # Read V scale
        v_scale_bytes = cache_out[V4_FP8_V_SCALE_OFFSET:
                                  V4_FP8_V_SCALE_OFFSET + 4].numpy().tobytes()
        v_scale = struct.unpack('<f', v_scale_bytes)[0]

        expected_v_amax = v_nope.to(torch.bfloat16).float().abs().max().item()
        expected_v_scale = expected_v_amax / 448.0
        self.assertAlmostEqual(v_scale, expected_v_scale, places=4,
            msg=f"V scale {v_scale:.6f} != expected {expected_v_scale:.6f}")


if __name__ == "__main__":
    unittest.main()
