"""
V4K-2a: Lightning Indexer Scoring Kernel Tests

Smoke test (V4K-2a.3): kernel builds, runs on small input without crash/NaN.
Accuracy test (V4K-2a.3): score ranking matches BF16 reference from V4K-0d.

The kernel accepts FP8 E4M3 K cache with per-block scales, matching the
SGLang/vLLM/TRT-LLM approach (fp8_mqa_logits). Tests quantize BF16 → FP8
before calling the kernel, then compare against the FP32 reference.

Usage:
  python tests/test_v4_lightning_indexer.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v4_reference import (
    INDEX_N_HEADS, INDEX_HEAD_DIM, ref_lightning_score,
)


def quantize_to_fp8(x_bf16):
    """Per-tensor FP8 E4M3 quantization with scale, matching fused_k_append pattern.

    Returns (fp8_tensor, scale) where dequant = fp8_tensor.float() * scale.
    """
    FP8_MAX = 448.0
    amax = x_bf16.float().abs().max().item()
    scale = max(amax, 1e-12) / FP8_MAX
    inv_scale = 1.0 / scale if scale > 0 else 0.0
    scaled = (x_bf16.float() * inv_scale).clamp(-FP8_MAX, FP8_MAX)
    fp8_tensor = scaled.to(torch.float8_e4m3fn)
    return fp8_tensor, scale


def quantize_k_per_block(k_bf16):
    """Per-block FP8 quantization of K cache [num_blocks, n_heads, head_dim].

    Each block gets its own scale factor (matching framework approach).
    Returns (k_fp8, k_scales).
    """
    num_blocks = k_bf16.shape[0]
    k_flat = k_bf16.reshape(num_blocks, -1).float()

    FP8_MAX = 448.0
    amax = k_flat.abs().amax(dim=1)  # [num_blocks]
    scales = torch.clamp(amax, min=1e-12) / FP8_MAX  # [num_blocks]
    inv_scales = 1.0 / scales

    scaled = (k_flat * inv_scales.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX)
    k_fp8 = scaled.reshape_as(k_bf16).to(torch.float8_e4m3fn)
    return k_fp8, scales


def run_score_kernel(q_proj, indexer_k_cache, score_proj):
    """Run the CUDA lightning_score kernel with FP8 K cache."""
    import sm120_mla_kernels
    device = "cuda"

    q_proj_gpu = q_proj.to(device=device, dtype=torch.bfloat16).contiguous()
    score_proj_gpu = score_proj.to(device=device, dtype=torch.float32).contiguous()

    # Quantize K to FP8 per-block (matching framework approach)
    k_fp8, k_scales = quantize_k_per_block(indexer_k_cache)
    k_fp8_gpu = k_fp8.to(device=device).contiguous()
    k_scales_gpu = k_scales.to(device=device, dtype=torch.float32).contiguous()

    scores = sm120_mla_kernels.v4_lightning_score(
        q_proj_gpu, k_fp8_gpu, k_scales_gpu, score_proj_gpu)
    return scores


def ref_lightning_score_fp8(q_proj, indexer_k_cache, score_proj):
    """Reference that matches FP8 quantization: quantize K, dequant, then score.

    This simulates the precision the kernel achieves.
    """
    k_fp8, k_scales = quantize_k_per_block(indexer_k_cache)
    # Dequant: fp8 → float → multiply by per-block scale
    k_dequant = k_fp8.float() * k_scales.view(-1, 1, 1)
    return ref_lightning_score(q_proj, k_dequant, score_proj)


def cosine_similarity(a, b):
    a_f = a.float()
    b_f = b.float()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


class TestLightningScoreSmoke(unittest.TestCase):
    """V4K-2a.3: Smoke test — kernel runs without crash or NaN."""

    def test_smoke_small(self):
        """32 blocks, standard dimensions, no crash."""
        torch.manual_seed(42)
        num_blocks = 32

        q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
        indexer_k_cache = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM)
        score_proj = torch.randn(INDEX_N_HEADS)

        scores = run_score_kernel(q_proj, indexer_k_cache, score_proj)

        self.assertEqual(scores.shape, (num_blocks,))
        self.assertTrue(torch.isfinite(scores).all(), "Scores contain NaN/Inf")

    def test_smoke_single_block(self):
        """1 block edge case."""
        torch.manual_seed(123)
        q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
        indexer_k_cache = torch.randn(1, INDEX_N_HEADS, INDEX_HEAD_DIM)
        score_proj = torch.randn(INDEX_N_HEADS)

        scores = run_score_kernel(q_proj, indexer_k_cache, score_proj)
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_smoke_large(self):
        """4096 blocks — realistic CSA at ~16K context."""
        torch.manual_seed(99)
        num_blocks = 4096

        q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
        indexer_k_cache = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM)
        score_proj = torch.randn(INDEX_N_HEADS)

        scores = run_score_kernel(q_proj, indexer_k_cache, score_proj)
        self.assertEqual(scores.shape, (num_blocks,))
        self.assertTrue(torch.isfinite(scores).all())


class TestLightningScoreAccuracy(unittest.TestCase):
    """V4K-2a.3: Accuracy test — scores match FP8-aware reference."""

    def _run_accuracy(self, num_blocks, seed=42):
        torch.manual_seed(seed)
        q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
        indexer_k_cache = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM)
        score_proj = torch.randn(INDEX_N_HEADS)

        # FP8-aware reference (quantize K → dequant → score in FP32)
        ref_scores = ref_lightning_score_fp8(q_proj, indexer_k_cache, score_proj)

        # CUDA kernel (FP8 K, BF16 Q, FP32 accumulation)
        kernel_scores = run_score_kernel(q_proj, indexer_k_cache, score_proj).cpu()

        return ref_scores, kernel_scores

    def test_accuracy_256_blocks(self):
        """256 blocks: cosine > 0.999 vs FP8-aware reference."""
        ref_scores, kernel_scores = self._run_accuracy(256)

        cos = cosine_similarity(ref_scores, kernel_scores)
        self.assertGreater(cos, 0.999,
            f"Cosine similarity {cos:.6f} below threshold 0.999")

        nrmse = (ref_scores - kernel_scores).norm() / ref_scores.norm()
        self.assertLess(nrmse, 0.05,
            f"NRMSE {nrmse:.4f} above threshold 0.05")

    def test_accuracy_1024_blocks(self):
        """1024 blocks: cosine > 0.999."""
        ref_scores, kernel_scores = self._run_accuracy(1024, seed=77)

        cos = cosine_similarity(ref_scores, kernel_scores)
        self.assertGreater(cos, 0.999,
            f"Cosine similarity {cos:.6f} below threshold 0.999")

    def test_ranking_preserved(self):
        """Top-k ranking order matches reference for top 64 blocks."""
        ref_scores, kernel_scores = self._run_accuracy(512, seed=55)

        _, ref_rank = ref_scores.sort(descending=True)
        _, kernel_rank = kernel_scores.sort(descending=True)

        top_n = 64
        ref_top = set(ref_rank[:top_n].tolist())
        kernel_top = set(kernel_rank[:top_n].tolist())
        overlap = len(ref_top & kernel_top)

        self.assertGreaterEqual(overlap, top_n - 2,
            f"Top-{top_n} overlap: {overlap}/{top_n} (expected >= {top_n - 2})")

    def test_score_values_match(self):
        """Individual score values within FP8 quantization tolerance."""
        ref_scores, kernel_scores = self._run_accuracy(128, seed=33)

        max_diff = (ref_scores - kernel_scores).abs().max().item()
        ref_range = ref_scores.max().item() - ref_scores.min().item()
        rel_diff = max_diff / (ref_range + 1e-8)

        self.assertLess(rel_diff, 0.05,
            f"Max relative diff {rel_diff:.4f} above threshold 0.05")

    def test_fp8_vs_fp32_reference(self):
        """Kernel vs original FP32 reference — looser tolerance (includes quantization)."""
        torch.manual_seed(42)
        num_blocks = 256
        q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
        indexer_k_cache = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM)
        score_proj = torch.randn(INDEX_N_HEADS)

        # Pure FP32 reference (no quantization)
        fp32_scores = ref_lightning_score(q_proj, indexer_k_cache, score_proj)
        kernel_scores = run_score_kernel(q_proj, indexer_k_cache, score_proj).cpu()

        cos = cosine_similarity(fp32_scores, kernel_scores)
        self.assertGreater(cos, 0.99,
            f"FP8 kernel vs FP32 ref cosine {cos:.6f} below 0.99 (includes quantization error)")


if __name__ == "__main__":
    unittest.main()
