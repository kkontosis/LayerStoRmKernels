"""
V4K-4b: HCA FP8 Prefill Kernel Tests

HCA prefill reuses the CSA prefill attention kernel (v4_csa_fp8_prefill)
since both are non-absorbed dense attention with causal masking. The only
difference is orchestration: HCA compresses stride-128, chunks align to
128-token boundaries.

Usage:
  python tests/test_v4_hca_prefill.py -v
"""

import torch
import unittest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import HEAD_DIM, QK_ROPE_HEAD_DIM, D_QK


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def ref_prefill_attention(q, k, v, sm_scale, causal_seqlens=None):
    """Pure PyTorch reference: non-absorbed dense attention with causal mask."""
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
        scores = torch.matmul(q[qi], k[:s_kv_i].T) * sm_scale
        lse[qi] = torch.logsumexp(scores, dim=-1)
        p = torch.softmax(scores, dim=-1)
        out[qi] = torch.matmul(p, v[:s_kv_i])
    return out, lse


def run_prefill_kernel(q_bf16, k_bf16, v_bf16, sm_scale, causal_seqlens=None):
    """Run the CUDA v4_csa_fp8_prefill kernel (reused for HCA)."""
    import sm120_mla_kernels
    cs = causal_seqlens.cuda().to(torch.int32).contiguous() if causal_seqlens is not None else None
    out, lse = sm120_mla_kernels.v4_csa_fp8_prefill(
        q_bf16.cuda().contiguous(),
        k_bf16.cuda().contiguous(),
        v_bf16.cuda().contiguous(),
        sm_scale, cs,
    )
    return out.cpu(), lse.cpu()


class TestHcaFp8PrefillSmoke(unittest.TestCase):
    """V4K-4b: Smoke test — kernel runs for HCA-scale inputs."""

    def test_hca_scale(self):
        """s_kv=8 (= 1024 tokens / 128 compression), s_q=16."""
        torch.manual_seed(42)
        s_q, s_kv, h_q = 16, 8, 64
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        out, lse = run_prefill_kernel(q, k, v, sm_scale)
        self.assertFalse(torch.isnan(out).any())


class TestHcaFp8PrefillAccuracy(unittest.TestCase):
    """V4K-4b: Accuracy test — kernel vs PyTorch reference."""

    def _run(self, s_q, s_kv, h_q=64, causal=False, seed=100):
        torch.manual_seed(seed)
        sm_scale = 1.0 / math.sqrt(D_QK)

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        cs = torch.clamp(torch.arange(1, s_q + 1), max=s_kv).to(torch.int32) if causal else None
        out_k, _ = run_prefill_kernel(q, k, v, sm_scale, cs)
        out_r, _ = ref_prefill_attention(q.float(), k.float(), v.float(), sm_scale, cs)
        return cosine_sim(out_k, out_r)

    def test_noncausal(self):
        """Non-causal, s_kv=16 (HCA: ~2K tokens compressed)."""
        cos = self._run(32, 16)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_causal(self):
        """Causal, HCA scale."""
        cos = self._run(32, 16, causal=True, seed=101)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_h128(self):
        """h_q=128 (V4 Pro)."""
        cos = self._run(16, 8, h_q=128, seed=102)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")

    def test_large_compressed(self):
        """s_kv=256 (HCA: ~32K tokens compressed at 128:1)."""
        cos = self._run(32, 256, seed=103)
        self.assertGreater(cos, 0.999, f"cosine={cos:.6f}")


class TestHcaFp8PrefillChunked(unittest.TestCase):
    """V4K-4b: Chunked prefill — 128-token boundary alignment."""

    def test_stride128_alignment(self):
        """Chunks aligned to 128-token boundaries."""
        torch.manual_seed(200)
        h_q = 64
        sm_scale = 1.0 / math.sqrt(D_QK)
        s_q, s_kv = 256, 32

        q = torch.randn(s_q, h_q, D_QK, dtype=torch.bfloat16)
        k = torch.randn(s_kv, D_QK, dtype=torch.bfloat16)
        v = torch.randn(s_kv, HEAD_DIM, dtype=torch.bfloat16)

        out_full, _ = run_prefill_kernel(q, k, v, sm_scale)

        # Split into 128-token aligned chunks
        out_c1, _ = run_prefill_kernel(q[:128], k, v, sm_scale)
        out_c2, _ = run_prefill_kernel(q[128:], k, v, sm_scale)
        out_chunked = torch.cat([out_c1, out_c2], dim=0)

        cos = cosine_sim(out_full, out_chunked)
        self.assertGreater(cos, 0.9999, f"Stride-128 chunked cosine={cos:.7f}")


if __name__ == '__main__':
    unittest.main()
