"""
GPU validation for the GGUF dequant-GEMM kernel (strategy: dequant).

Compares sm120_mla_kernels.gguf_dequant_gemm against the numpy dequant oracles
in test_gguf_gemm_reference.py (and the richer Q4_K oracle in
test_q4k_gemm_reference.py for q4_k), across all 6 weight types and a range of
M (GEMV path M<=8 and tiled path M>8).

Run:  pytest tests/test_gguf_gemm_kernels.py -v
"""

import numpy as np
import pytest
import torch

import sm120_mla_kernels as K

from tests.test_gguf_gemm_reference import (
    generate_gguf_weight, ref_gguf_gemm, bytes_per_row, BLOCK_VALUES,
)

NEW_TYPES = ["q2_k", "q3_k", "q5_k", "q6_k", "q8_0"]

# (M, N, K) cases. K divisible by 256 (k-quants) or 32 (q8_0). Mix GEMV + tiled.
CASES = [
    (1, 128, 256),
    (2, 256, 512),
    (8, 96, 768),
    (16, 128, 512),
    (33, 80, 256),
    (128, 256, 1024),
]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant_type", NEW_TYPES)
@pytest.mark.parametrize("M,N,Kdim", CASES)
def test_gguf_dequant_gemm(quant_type, M, N, Kdim):
    if Kdim % BLOCK_VALUES[quant_type] != 0:
        pytest.skip(f"K={Kdim} not divisible by block for {quant_type}")
    torch.manual_seed(M * 131 + N * 7 + Kdim)
    packed = generate_gguf_weight(quant_type, N, Kdim, seed=M + N + Kdim)
    w_gpu = torch.from_numpy(packed).cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

    out = K.gguf_dequant_gemm(x, w_gpu, quant_type)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16

    ref = ref_gguf_gemm(x, packed, quant_type, N, Kdim)
    cos = _cos(out.float().cpu(), ref)
    assert cos > 0.999, f"{quant_type} M={M} N={N} K={Kdim}: cosine={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("M,N,Kdim", CASES)
def test_gguf_dequant_gemm_q4k(M, N, Kdim):
    """q4_k routed through the unified gguf path, validated vs the Q4_K oracle."""
    from tests.test_q4k_gemm_reference import generate_q4k_weight, ref_q4k_gemm
    torch.manual_seed(M + N + Kdim)
    w = generate_q4k_weight(N, Kdim)            # packed uint8 [N, K*9/16]
    w_gpu = torch.from_numpy(w).cuda() if isinstance(w, np.ndarray) else w.cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

    out = K.gguf_dequant_gemm(x, w_gpu, "q4_k")
    w_np = w if isinstance(w, np.ndarray) else w.cpu().numpy()
    ref = ref_q4k_gemm(x.float().cpu(), w_np)
    cos = _cos(out.float().cpu(), ref)
    assert cos > 0.999, f"q4_k M={M} N={N} K={Kdim}: cosine={cos:.6f}"


# Integer mmvq path is a mat-vec: exercise small M (decode). Carries ~8-bit
# activation-quant error, so a looser cosine bound than the dequant path.
MMVQ_CASES = [
    (1, 128, 256),
    (2, 256, 512),
    (4, 96, 768),
    (8, 128, 1024),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant_type", NEW_TYPES)
@pytest.mark.parametrize("M,N,Kdim", MMVQ_CASES)
def test_gguf_mmvq(quant_type, M, N, Kdim):
    if Kdim % BLOCK_VALUES[quant_type] != 0:
        pytest.skip(f"K={Kdim} not divisible by block for {quant_type}")
    torch.manual_seed(M * 131 + N * 7 + Kdim)
    packed = generate_gguf_weight(quant_type, N, Kdim, seed=M + N + Kdim)
    w_gpu = torch.from_numpy(packed).cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

    out = K.gguf_mmvq(x, w_gpu, quant_type)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16

    ref = ref_gguf_gemm(x, packed, quant_type, N, Kdim)
    cos = _cos(out.float().cpu(), ref)
    assert cos > 0.99, f"{quant_type} mmvq M={M} N={N} K={Kdim}: cosine={cos:.6f}"


# Integer mmq is a tiled mat-mat: exercise larger M (prefill).
MMQ_CASES = [
    (16, 128, 256),
    (64, 96, 512),
    (128, 128, 768),
    (300, 64, 1024),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant_type", NEW_TYPES)
@pytest.mark.parametrize("M,N,Kdim", MMQ_CASES)
def test_gguf_mmq(quant_type, M, N, Kdim):
    if Kdim % BLOCK_VALUES[quant_type] != 0:
        pytest.skip(f"K={Kdim} not divisible by block for {quant_type}")
    torch.manual_seed(M * 17 + N * 3 + Kdim)
    packed = generate_gguf_weight(quant_type, N, Kdim, seed=M * 2 + N + Kdim)
    w_gpu = torch.from_numpy(packed).cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

    out = K.gguf_mmq(x, w_gpu, quant_type)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16

    ref = ref_gguf_gemm(x, packed, quant_type, N, Kdim)
    cos = _cos(out.float().cpu(), ref)
    assert cos > 0.99, f"{quant_type} mmq M={M} N={N} K={Kdim}: cosine={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("quant_type", NEW_TYPES)
@pytest.mark.parametrize("M,N,Kdim", MMQ_CASES)
def test_gguf_mmq_mma(quant_type, M, N, Kdim):
    """int8 tensor-core (CuTe MMA) mmq vs oracle, and matches the dp4a mmq."""
    if Kdim % BLOCK_VALUES[quant_type] != 0:
        pytest.skip(f"K={Kdim} not divisible by block for {quant_type}")
    torch.manual_seed(M * 19 + N * 5 + Kdim)
    packed = generate_gguf_weight(quant_type, N, Kdim, seed=M * 3 + N + Kdim)
    w_gpu = torch.from_numpy(packed).cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

    out = K.gguf_mmq_mma(x, w_gpu, quant_type)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16

    ref = ref_gguf_gemm(x, packed, quant_type, N, Kdim)
    assert _cos(out.float().cpu(), ref) > 0.99, f"{quant_type} mmq_mma vs float"
    # Same int8 math as the dp4a mmq → near-identical.
    dp4a = K.gguf_mmq(x, w_gpu, quant_type)
    assert _cos(out.float().cpu(), dp4a.float().cpu()) > 0.999, f"{quant_type} mma vs dp4a"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguf_int_routes_by_M():
    """strategy='int' uses mmvq at M<=8 and mmq above."""
    packed = generate_gguf_weight("q4_k", 128, 512, seed=3)
    w = torch.from_numpy(packed).cuda()
    x1 = torch.randn(4, 512, dtype=torch.bfloat16, device="cuda")
    x64 = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(K.gguf_mul_mat(x1, w, "q4_k", "int"), K.gguf_mmvq(x1, w, "q4_k"))
    assert torch.equal(K.gguf_mul_mat(x64, w, "q4_k", "int"), K.gguf_mmq_mma(x64, w, "q4_k"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("strategy", ["int", "dequant"])
def test_gguf_mul_mat_strategies(strategy):
    """Unified entry dispatches both strategies and validates each vs oracle."""
    M, N, Kdim = 6, 128, 512
    packed = generate_gguf_weight("q6_k", N, Kdim, seed=5)
    w_gpu = torch.from_numpy(packed).cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")

    out = K.gguf_mul_mat(x, w_gpu, "q6_k", strategy)
    ref = ref_gguf_gemm(x, packed, "q6_k", N, Kdim)
    cos = _cos(out.float().cpu(), ref)
    bound = 0.999 if strategy == "dequant" else 0.99
    assert cos > bound, f"{strategy}: cosine={cos:.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gguf_mul_mat_default_is_int():
    """Default strategy must be 'int' (== explicit mmvq)."""
    M, N, Kdim = 4, 64, 256
    packed = generate_gguf_weight("q6_k", N, Kdim, seed=9)
    w_gpu = torch.from_numpy(packed).cuda()
    x = torch.randn(M, Kdim, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(K.gguf_mul_mat(x, w_gpu, "q6_k"),
                       K.gguf_mmvq(x, w_gpu, "q6_k"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_row_bytes_validation():
    """Wrong packed row width must be rejected."""
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
    bad = torch.zeros(8, 999, dtype=torch.uint8, device="cuda")
    with pytest.raises(Exception):
        K.gguf_dequant_gemm(x, bad, "q6_k")
    with pytest.raises(Exception):
        K.gguf_dequant_gemm(x, bad, "not_a_type")
