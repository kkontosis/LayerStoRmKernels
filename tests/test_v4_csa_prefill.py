"""
V4K-3b: CSA FP8 Prefill Kernel Tests

Tests the V4 CSA FP8 prefill attention kernel:
  - Non-absorbed dense attention (separate K and V)
  - Per-query causal masking
  - Single KV head broadcast to all Q heads
  - Chunked prefill with stride-4 alignment

Usage:
  python tests/test_v4_csa_prefill.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, D_QK, SLIDING_WINDOW,
)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def ref_prefill_attention(q, k, v, sm_scale, causal_seqlens=None):
    """Pure PyTorch reference: non-absorbed multi-head attention with causal mask.

    Args:
        q: [s_q, h_q, d_qk] float32
        k: [s_kv, d_qk] float32 (single KV head)
        v: [s_kv, d_v] float32 (single KV head)
        sm_scale: float
        causal_seqlens: [s_q] int, or None

    Returns:
        out: [s_q, h_q, d_v] float32
        lse: [s_q, h_q] float32
    """
    s_q, h_q, d_qk = q.shape
    s_kv = k.shape[0]
    d_v = v.shape[1]

    out = torch.zeros(s_q, h_q, d_v)
    lse = torch.zeros(s_q, h_q)

    for qi in range(s_q):
        s_kv_i = causal_seqlens[qi].item() if causal_seqlens is not None else s_kv
        s_kv_i = min(s_kv_i, s_kv)
        if s_kv_i <= 0:
            lse[qi, :] = float('inf')
            continue

        k_vis = k[:s_kv_i]  # [s_kv_i, d_qk]
        v_vis = v[:s_kv_i]  # [s_kv_i, d_v]

        # [h_q, d_qk] @ [d_qk, s_kv_i] -> [h_q, s_kv_i]
        scores = torch.matmul(q[qi], k_vis.T) * sm_scale
        lse_qi = torch.logsumexp(scores, dim=-1)  # [h_q]
        p = torch.softmax(scores, dim=-1)           # [h_q, s_kv_i]
        out_qi = torch.matmul(p, v_vis)             # [h_q, d_v]

        out[qi] = out_qi
        lse[qi] = lse_qi

    return out, lse


def run_prefill_kernel(q_bf16, k_bf16, v_bf16, sm_scale, causal_seqlens=None):
    """Run the CUDA v4_csa_fp8_prefill kernel."""
    import sm120_mla_kernels

    cs = causal_seqlens.cuda().to(torch.int32).contiguous() if causal_seqlens is not None else None
    out, lse = sm120_mla_kernels.v4_csa_fp8_prefill(
        q_bf16.cuda().contiguous(),
        k_bf16.cuda().contiguous(),
        v_bf16.cuda().contiguous(),
        sm_scale,
        cs,
    )
    return out.cpu(), lse.cpu()


class TestCsaFp8PrefillSmoke(unittest.TestCase):
    """V4K-3b smoke test: kernel runs without crash."""

    def test_small_no_crash(self):
        """s_q=8, s_kv=16, h_q=64 — no crash, no NaN."""
        torch.manual_seed(42)
        s_q, s_kv, h_q = 8, 16, 64
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        out, lse = run_prefill_kernel(q, k, v, sm_scale)
        self.assertEqual(out.shape, (s_q, h_q, HEAD_DIM))
        self.assertEqual(lse.shape, (s_q, h_q))
        self.assertFalse(torch.isnan(out).any(), "NaN in output")
        self.assertFalse(torch.isinf(out).any(), "Inf in output")

    def test_small_causal(self):
        """s_q=8, s_kv=16, causal — no crash."""
        torch.manual_seed(43)
        s_q, s_kv, h_q = 8, 16, 64
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)
        causal_seqlens = torch.arange(1, s_q + 1, dtype=torch.int32)

        out, lse = run_prefill_kernel(q, k, v, sm_scale, causal_seqlens)
        self.assertFalse(torch.isnan(out).any())

    def test_h128_no_crash(self):
        """h_q=128 (V4 Pro) — tests head-group indexing."""
        torch.manual_seed(44)
        s_q, s_kv, h_q = 4, 8, 128
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        out, lse = run_prefill_kernel(q, k, v, sm_scale)
        self.assertEqual(out.shape, (s_q, h_q, HEAD_DIM))
        self.assertFalse(torch.isnan(out).any())


class TestCsaFp8PrefillAccuracy(unittest.TestCase):
    """V4K-3b accuracy test: kernel vs PyTorch reference."""

    def _run_accuracy(self, s_q, s_kv, h_q, causal=False, seed=100):
        torch.manual_seed(seed)
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        if causal:
            # Linear ramp: query i sees i+1 KV tokens (capped at s_kv)
            causal_seqlens = torch.clamp(torch.arange(1, s_q + 1), max=s_kv).to(torch.int32)
        else:
            causal_seqlens = None

        # Kernel
        out_k, lse_k = run_prefill_kernel(q, k, v, sm_scale, causal_seqlens)

        # Reference (float32)
        out_r, lse_r = ref_prefill_attention(
            q.float(), k.float(), v.float(), sm_scale,
            causal_seqlens,
        )

        return out_k, out_r, lse_k, lse_r

    def test_noncausal_h64(self):
        """Non-causal, s_q=32, s_kv=64, h_q=64."""
        out_k, out_r, lse_k, lse_r = self._run_accuracy(32, 64, 64)
        cos = cosine_sim(out_k, out_r)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}, expected > 0.999")
        # LSE: BF16 vs FP32 accumulation causes meaningful absolute diffs;
        # check relative error on finite values instead
        finite = lse_r.isfinite() & lse_k.isfinite()
        if finite.any():
            lse_cos = cosine_sim(lse_k[finite], lse_r[finite])
            self.assertGreater(lse_cos, 0.99, f"LSE cosine={lse_cos:.6f}")

    def test_noncausal_h128(self):
        """Non-causal, h_q=128 (V4 Pro)."""
        out_k, out_r, lse_k, lse_r = self._run_accuracy(16, 32, 128, seed=101)
        cos = cosine_sim(out_k, out_r)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_causal_h64(self):
        """Causal masking, s_q=32, s_kv=64, h_q=64."""
        out_k, out_r, lse_k, lse_r = self._run_accuracy(32, 64, 64, causal=True)
        cos = cosine_sim(out_k, out_r)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_causal_h128(self):
        """Causal masking, h_q=128."""
        out_k, out_r, lse_k, lse_r = self._run_accuracy(16, 32, 128, causal=True, seed=102)
        cos = cosine_sim(out_k, out_r)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_large_context(self):
        """s_kv=1024, s_q=64 — tests multi-block iteration."""
        out_k, out_r, lse_k, lse_r = self._run_accuracy(64, 1024, 64, seed=103)
        cos = cosine_sim(out_k, out_r)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_skv_1(self):
        """Edge case: s_kv=1."""
        out_k, out_r, lse_k, lse_r = self._run_accuracy(4, 1, 64, seed=104)
        cos = cosine_sim(out_k, out_r)
        self.assertGreater(cos, 0.99, f"cosine={cos:.6f}")

    def test_causal_first_query_zero_kv(self):
        """Causal: first query sees 0 KV tokens."""
        torch.manual_seed(105)
        s_q, s_kv, h_q = 8, 16, 64
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)
        causal_seqlens = torch.tensor([0, 1, 2, 4, 8, 12, 15, 16], dtype=torch.int32)

        out_k, lse_k = run_prefill_kernel(q, k, v, sm_scale, causal_seqlens)
        out_r, lse_r = ref_prefill_attention(
            q.float(), k.float(), v.float(), sm_scale, causal_seqlens)

        # First query: causal_seqlens=0, output should be zero
        self.assertTrue(torch.allclose(out_k[0].float(), torch.zeros(h_q, HEAD_DIM), atol=1e-6),
                        "Query with 0 visible KV should produce zero output")

        # Other queries should be accurate
        cos = cosine_sim(out_k[1:], out_r[1:])
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")


class TestCsaFp8PrefillChunked(unittest.TestCase):
    """V4K-3b.2: Chunked prefill alignment — stride-4 boundaries."""

    def test_chunked_matches_full(self):
        """Chunked prefill produces the same result as non-chunked.

        Split s_q into 2 chunks, each seeing the same KV. Since KV is
        identical and causal_seqlens handles masking, both should match.
        """
        torch.manual_seed(200)
        s_q, s_kv, h_q = 32, 128, 64
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        # Full prefill (non-causal)
        out_full, lse_full = run_prefill_kernel(q, k, v, sm_scale)

        # Chunked: split q into 2 halves, each seeing full KV
        mid = s_q // 2
        out_c1, lse_c1 = run_prefill_kernel(q[:mid], k, v, sm_scale)
        out_c2, lse_c2 = run_prefill_kernel(q[mid:], k, v, sm_scale)
        out_chunked = torch.cat([out_c1, out_c2], dim=0)

        cos = cosine_sim(out_full, out_chunked)
        self.assertGreater(cos, 0.9999,
                           f"Chunked vs full cosine={cos:.7f}, expected > 0.9999")

    def test_stride4_alignment(self):
        """Verify stride-4 chunk sizes work correctly.

        CSA compresses every 4 tokens → 1 compressed entry.
        Chunks must be multiples of 4 for correct compression.
        """
        torch.manual_seed(201)
        h_q = 64
        sm_scale = 1.0 / math.sqrt(D_QK)

        # 32 query tokens, 64 KV tokens
        s_q, s_kv = 32, 64
        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        # Process in stride-4 aligned chunks (8, 12, 12 = 32)
        chunks = [8, 12, 12]
        outs = []
        offset = 0
        for sz in chunks:
            assert sz % 4 == 0, "Chunk size must be multiple of 4"
            out_c, _ = run_prefill_kernel(q[offset:offset+sz], k, v, sm_scale)
            outs.append(out_c)
            offset += sz

        out_chunked = torch.cat(outs, dim=0)
        out_full, _ = run_prefill_kernel(q, k, v, sm_scale)
        cos = cosine_sim(out_full, out_chunked)
        self.assertGreater(cos, 0.9999, f"Stride-4 chunked cosine={cos:.7f}")


if __name__ == '__main__':
    unittest.main()
