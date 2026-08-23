"""
GPU validation for the GGUF weight branch of q_absorb (W_UK query absorption).

kv_b_proj is supplied as a packed GGUF weight; W_UK is dequanted per element
(dequant-only, accuracy-equal to a load-time dequant to BF16). Validated against
a float reference built from the numpy dequant oracle.

Run:  pytest tests/test_q_absorb_gguf.py -v
"""

import numpy as np
import pytest
import torch

import sm120_mla_kernels as K
from tests.test_gguf_gemm_reference import (
    generate_gguf_weight, ref_gguf_dequant, BLOCK_VALUES,
)

# (name, contraction P, lora L, rope R, v_head V). L divisible by the block size.
SHAPES = [
    ("base", 128, 512, 64, 128),
    ("small", 64, 256, 32, 64),
]
TYPES = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0"]


def _cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant_type", TYPES)
@pytest.mark.parametrize("name,P,L,R,V", SHAPES)
def test_q_absorb_gguf(quant_type, name, P, L, R, V):
    if L % BLOCK_VALUES[quant_type] != 0:
        pytest.skip(f"L={L} not divisible by block for {quant_type}")
    s_q, h_q = 2, 4
    kv_row = P + V
    rows = h_q * kv_row
    torch.manual_seed(hash((quant_type, name)) & 0xFFFF)

    packed = generate_gguf_weight(quant_type, rows, L, seed=P + L + V)
    Wf = ref_gguf_dequant(packed, quant_type, rows, L)            # [rows, L] float32
    w_gpu = torch.from_numpy(packed).cuda()

    q_heads = torch.randn(s_q, h_q, P + R, dtype=torch.bfloat16, device="cuda")

    out = K.q_absorb(q_heads, w_gpu, P, L, R, V, gguf_quant_type=quant_type)
    assert out.shape == (s_q, h_q, L + R)

    # Float reference: absorbed content + rope tail (no fused RoPE here).
    qf = q_heads.float().cpu().numpy()
    Wt = torch.from_numpy(Wf)
    ref = np.zeros((s_q, h_q, L + R), np.float32)
    for s in range(s_q):
        for h in range(h_q):
            wuk = Wt[h * kv_row:h * kv_row + P, :].numpy()        # [P, L]
            ref[s, h, :L] = qf[s, h, :P] @ wuk
            ref[s, h, L:] = qf[s, h, P:P + R]

    cos = _cos(out.float().cpu(), torch.from_numpy(ref))
    assert cos > 0.999, f"{quant_type} {name}: cosine={cos:.6f}"
