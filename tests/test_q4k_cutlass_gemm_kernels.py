"""
Q4_K Tensor-Core GEMM Kernel Tests — validate TC kernel against Phase 2 and Python reference.

Tests sm120_mla_kernels.q4k_cutlass_gemm() (wmma BF16 tensor-core path) across all
target projection shapes and batch sizes. Both the TC kernel and Phase 2 CUDA-core
kernel dequant from identical Q4_K blocks, so output should match near-exactly
(differences only from accumulation order: wmma FP32 vs scalar FP32 FMA).

Usage:
  python tests/test_q4k_cutlass_gemm_kernels.py -v
"""

import torch
import torch.nn.functional as F
import math
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import sm120_mla_kernels

from tests.test_q4k_gemm_reference import (
    ref_q4k_gemm, generate_q4k_weight, quantize_to_q4k,
    compute_metrics, fmt_metrics,
    PROJECTION_SHAPES, BATCH_SIZES,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tc_vs_reference(verbose=False):
    """TC q4k_cutlass_gemm vs Python ref_q4k_gemm for all projection shapes x batch sizes."""
    print("\n=== test_tc_vs_reference ===")

    for name, N, K in PROJECTION_SHAPES:
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        for M in BATCH_SIZES:
            torch.manual_seed(42 + M)
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

            out_tc = sm120_mla_kernels.q4k_cutlass_gemm(x, w_q4k_gpu)
            out_ref = ref_q4k_gemm(x.cpu(), w_q4k)

            assert out_tc.shape == (M, N), \
                f"{name} M={M}: shape {out_tc.shape} != ({M}, {N})"

            m = compute_metrics(out_ref, out_tc.cpu())
            if verbose:
                print(f"  {name} M={M}: {fmt_metrics(m)}")

            assert m["cosine"] > 0.9999, \
                f"{name} M={M} cosine {m['cosine']:.6f} < 0.9999"
            assert m["max_abs_err"] < 2.0, \
                f"{name} M={M} max_abs_err {m['max_abs_err']:.4e} > 2.0"

        print(f"  {name} [{N}, {K}] x M={BATCH_SIZES}: all cosine > 0.9999")

    print("  PASS")


def test_tc_vs_phase2(verbose=False):
    """TC q4k_cutlass_gemm vs Phase 2 q4k_dequant_gemm — near-exact match.

    Threshold is 0.999 (not 0.9999) because Phase 2 dequants to FP16 while
    Phase 3 dequants to BF16 (10 vs 7 mantissa bits), and accumulation order
    differs (scalar FP32 FMA vs wmma tensor-core MMA). Both are correct;
    the primary correctness check is test_tc_vs_reference (vs Python ref).
    """
    print("\n=== test_tc_vs_phase2 ===")

    for name, N, K in PROJECTION_SHAPES:
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        for M in BATCH_SIZES:
            torch.manual_seed(42 + M)
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

            out_tc = sm120_mla_kernels.q4k_cutlass_gemm(x, w_q4k_gpu)
            out_p2 = sm120_mla_kernels.q4k_dequant_gemm(x, w_q4k_gpu)

            m = compute_metrics(out_p2.cpu(), out_tc.cpu())
            if verbose:
                print(f"  {name} M={M}: {fmt_metrics(m)}")

            assert m["cosine"] > 0.999, \
                f"{name} M={M} TC vs Phase2 cosine {m['cosine']:.6f} < 0.999"

        print(f"  {name} [{N}, {K}] x M={BATCH_SIZES}: TC vs Phase2 cosine > 0.999")

    print("  PASS")


def test_tc_quantized_weights(verbose=False):
    """TC GEMM with weights from quantize_to_q4k() round-trip, cross-check vs BF16 truth."""
    print("\n=== test_tc_quantized_weights ===")

    N, K = 1536, 7168  # q_a shape
    torch.manual_seed(99)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    w_q4k = quantize_to_q4k(w_bf16)
    w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

    M = 128
    torch.manual_seed(77)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    out_tc = sm120_mla_kernels.q4k_cutlass_gemm(x, w_q4k_gpu)
    out_ref = ref_q4k_gemm(x.cpu(), w_q4k)
    out_bf16_truth = x.cpu().float() @ w_bf16.float().T

    # TC vs Python reference (lossless dequant, only accumulation order differs)
    m_ref = compute_metrics(out_ref, out_tc.cpu())
    print(f"  vs Q4K reference M={M}: {fmt_metrics(m_ref)}")
    assert m_ref["cosine"] > 0.9999, \
        f"Q4K reference cosine {m_ref['cosine']:.6f} < 0.9999"

    # TC vs BF16 ground truth (includes quantization loss)
    m_bf16 = compute_metrics(out_bf16_truth, out_tc.cpu())
    print(f"  vs BF16 truth    M={M}: {fmt_metrics(m_bf16)}")
    assert m_bf16["cosine"] > 0.995, \
        f"BF16 truth cosine {m_bf16['cosine']:.6f} < 0.995"

    print("  PASS")


def test_tc_m1_decode(verbose=False):
    """M=1 decode path for all projection shapes (delegates to Phase 2 GEMV)."""
    print("\n=== test_tc_m1_decode ===")

    M = 1
    for name, N, K in PROJECTION_SHAPES:
        w_q4k = generate_q4k_weight(N, K)
        w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        out_tc = sm120_mla_kernels.q4k_cutlass_gemm(x, w_q4k_gpu)
        out_ref = ref_q4k_gemm(x.cpu(), w_q4k)

        m = compute_metrics(out_ref, out_tc.cpu())
        print(f"  {name} M=1: {fmt_metrics(m)}")
        assert m["cosine"] > 0.9999, \
            f"{name} M=1 cosine {m['cosine']:.6f} < 0.9999"

    print("  PASS")


def test_tc_determinism(verbose=False):
    """Same TC GEMM called 5 times produces bit-identical output."""
    print("\n=== test_tc_determinism ===")

    N, K = 1536, 7168
    M = 128
    w_q4k = generate_q4k_weight(N, K)
    w_q4k_gpu = torch.from_numpy(w_q4k).cuda()

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    outputs = []
    for i in range(5):
        out = sm120_mla_kernels.q4k_cutlass_gemm(x, w_q4k_gpu)
        outputs.append(out.clone())

    for i in range(1, 5):
        if not torch.equal(outputs[0], outputs[i]):
            diff = (outputs[0].float() - outputs[i].float()).abs().max().item()
            print(f"  WARNING: run 0 vs run {i} differ, max_abs_diff={diff:.4e}")
            assert False, f"Non-deterministic: run 0 vs {i} max diff {diff:.4e}"

    print("  5 runs: bit-identical")
    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_tc_vs_reference,
    test_tc_vs_phase2,
    test_tc_quantized_weights,
    test_tc_m1_decode,
    test_tc_determinism,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("Q4_K Tensor-Core GEMM Kernel Tests")
    print("=" * 55)

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn(verbose=args.verbose)
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
