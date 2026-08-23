"""
V4 mHC (hyper-connection) kernel tests: mhc_pre / mhc_post / mhc_head.

Golden = torch reference ported from
  ref/vllm/vllm/model_executor/kernels/mhc/torch.py
  (mhc_pre_torch / mhc_post_torch; Apache-2.0, Copyright contributors to the
  vLLM project), extended with the head collapse per
  ref/llama.cpp/src/models/deepseek4.cpp build_hc_head (MIT).

Test grid mirrors ref/vllm/tests/kernels/test_mhc_kernels.py:
  num_tokens in {1, 4, 8, 128}, hidden {4096} (+ 512 fast case), hc_mult 4.

Usage:
  python tests/test_mhc.py -v
"""

import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sm120_mla_kernels as K

HC = 4
RMS_EPS = 1e-6
HC_EPS = 1e-6
POST_MULT = 2.0
SINKHORN_ITERS = 20


# ---------------------------------------------------------------------------
# Torch golden (attributed port — see module docstring)
# ---------------------------------------------------------------------------

def mhc_pre_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                  hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]

    x = residual.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult:].view(num_tokens, hc_mult, hc_mult) * hc_scale[2] \
        + hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    return post_mix, comb_mix, layer_input


def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.unsqueeze(-1).to(torch.float32) \
        * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def mhc_head_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps):
    """build_hc_head: pre branch only, scalar scale, weighted collapse."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]

    x = residual.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre = torch.sigmoid(mixes * hc_scale[0] + hc_base) + hc_pre_eps
    return torch.sum(pre.unsqueeze(-1) * residual.to(torch.float32), dim=1).to(torch.bfloat16)


def make_inputs(num_tokens, hidden, seed=0, head=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    hc_mix = HC if head else (2 + HC) * HC
    residual = torch.randn(num_tokens, HC, hidden, generator=g).to(torch.bfloat16).cuda()
    fn = (torch.randn(hc_mix, HC * hidden, generator=g) * 0.02).cuda()
    scale = (torch.randn(1 if head else 3, generator=g) * 0.5 + 1.0).cuda()
    base = (torch.randn(hc_mix, generator=g) * 0.5).cuda()
    return residual, fn, scale, base


class TestMhcPre(unittest.TestCase):
    def _run(self, num_tokens, hidden):
        residual, fn, scale, base = make_inputs(num_tokens, hidden, seed=num_tokens)
        post_ref, comb_ref, x_ref = mhc_pre_torch(
            residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS)
        post, comb, x = K.mhc_pre(residual, fn, scale, base,
                                  RMS_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS)
        torch.testing.assert_close(post, post_ref, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(comb, comb_ref, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(x.float(), x_ref.float(), atol=2e-2, rtol=2e-2)
        # comb is doubly stochastic up to finite-iteration error: sinkhorn
        # ends on a COLUMN normalization, so col sums are exact (up to +eps)
        # while row sums only converge approximately in 20 iters (extreme
        # random logits can leave a few % — identical in the torch golden,
        # which the assert_close above already pins).
        self.assertTrue(torch.allclose(comb.sum(dim=-1), torch.ones_like(post), atol=0.1))
        self.assertTrue(torch.allclose(comb.sum(dim=-2), torch.ones_like(post), atol=1e-2))

    def test_tokens_1_hidden_4096(self):
        self._run(1, 4096)

    def test_tokens_4_hidden_4096(self):
        self._run(4, 4096)

    def test_tokens_8_hidden_4096(self):
        self._run(8, 4096)

    def test_tokens_128_hidden_4096(self):
        self._run(128, 4096)

    def test_tokens_16_hidden_512(self):
        self._run(16, 512)


class TestMhcPost(unittest.TestCase):
    def _run(self, num_tokens, hidden):
        residual, fn, scale, base = make_inputs(num_tokens, hidden, seed=100 + num_tokens)
        g = torch.Generator(device="cpu").manual_seed(7)
        y = torch.randn(num_tokens, hidden, generator=g).to(torch.bfloat16).cuda()
        post, comb, _ = K.mhc_pre(residual, fn, scale, base,
                                  RMS_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS)
        out_ref = mhc_post_torch(y, residual, post, comb)
        out = K.mhc_post(y, residual, post, comb)
        torch.testing.assert_close(out.float(), out_ref.float(), atol=2e-2, rtol=2e-2)

    def test_tokens_1_hidden_4096(self):
        self._run(1, 4096)

    def test_tokens_8_hidden_4096(self):
        self._run(8, 4096)

    def test_tokens_128_hidden_4096(self):
        self._run(128, 4096)


class TestMhcHead(unittest.TestCase):
    def _run(self, num_tokens, hidden):
        residual, fn, scale, base = make_inputs(num_tokens, hidden,
                                                seed=200 + num_tokens, head=True)
        x_ref = mhc_head_torch(residual, fn, scale, base, RMS_EPS, HC_EPS)
        x = K.mhc_head(residual, fn, scale, base, RMS_EPS, HC_EPS)
        torch.testing.assert_close(x.float(), x_ref.float(), atol=2e-2, rtol=2e-2)

    def test_tokens_1_hidden_4096(self):
        self._run(1, 4096)

    def test_tokens_128_hidden_4096(self):
        self._run(128, 4096)


class TestMhcEmbeddingIdentity(unittest.TestCase):
    """Sanity: after a repeat-expanded embedding, all hc streams are identical,
    so mhc_pre's collapse must equal sum(pre) * embed row."""

    def test_repeat_streams(self):
        num_tokens, hidden = 4, 4096
        g = torch.Generator(device="cpu").manual_seed(3)
        embed = torch.randn(num_tokens, hidden, generator=g).to(torch.bfloat16).cuda()
        residual = embed.unsqueeze(1).repeat(1, HC, 1).contiguous()
        _, fn, scale, base = make_inputs(num_tokens, hidden, seed=3)
        post, comb, x = K.mhc_pre(residual, fn, scale, base,
                                  RMS_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS)
        post_ref, comb_ref, x_ref = mhc_pre_torch(
            residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS)
        torch.testing.assert_close(x.float(), x_ref.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
