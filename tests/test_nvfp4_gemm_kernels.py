"""
NVFP4 GEMM CUDA Kernel Tests — validate GPU kernel against Python reference.

Tests the sm120_mla_kernels.nvfp4_gemm() binding across all target projection
shapes and batch sizes. The GPU kernel internally quantizes BF16 activations to
FP4 E2M1, so outputs differ from the BF16-path Python reference by ~1% due to
activation quantization noise.

Usage:
  python tests/test_nvfp4_gemm_kernels.py -v
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

from tests.test_nvfp4_gemm_reference import (
    ref_nvfp4_gemm, generate_nvfp4_weight, quantize_to_nvfp4,
    compute_metrics, fmt_metrics,
    PROJECTION_SHAPES, BATCH_SIZES,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gemm_projection_shapes(verbose=False):
    """GPU nvfp4_gemm vs Python ref_nvfp4_gemm for all projection shapes × batch sizes."""
    print("\n=== test_gemm_projection_shapes ===")

    for name, N, K in PROJECTION_SHAPES:
        w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K)
        w_uint8_gpu = w_uint8.cuda()
        w_scale_gpu = w_scale.cuda()
        scale2_val = w_scale2.item()

        for M in BATCH_SIZES:
            torch.manual_seed(42 + M)
            x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

            out_gpu = sm120_mla_kernels.nvfp4_gemm(x, w_uint8_gpu, w_scale_gpu, scale2_val)
            out_ref = ref_nvfp4_gemm(x.cpu(), w_uint8, w_scale, w_scale2)

            assert out_gpu.shape == (M, N), \
                f"{name} M={M}: shape {out_gpu.shape} != ({M}, {N})"

            m = compute_metrics(out_ref, out_gpu.cpu())
            if verbose:
                print(f"  {name} M={M}: {fmt_metrics(m)}")

            assert m["cosine"] > 0.99, \
                f"{name} M={M} cosine {m['cosine']:.6f} < 0.99"

        print(f"  {name} [{N}, {K}] x M={BATCH_SIZES}: all cosine > 0.99")

    print("  PASS")


def test_gemm_quantized_weights(verbose=False):
    """GPU GEMM with weights from quantize_to_nvfp4() round-trip."""
    print("\n=== test_gemm_quantized_weights ===")

    N, K = 1536, 7168  # q_a shape
    torch.manual_seed(99)
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
    w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w_bf16)

    w_uint8_gpu = w_uint8.cuda()
    w_scale_gpu = w_scale.cuda()
    scale2_val = w_scale2.item()

    M = 128
    torch.manual_seed(77)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    out_gpu = sm120_mla_kernels.nvfp4_gemm(x, w_uint8_gpu, w_scale_gpu, scale2_val)
    out_ref = ref_nvfp4_gemm(x.cpu(), w_uint8, w_scale, w_scale2)

    m = compute_metrics(out_ref, out_gpu.cpu())
    print(f"  quantized q_a M={M}: {fmt_metrics(m)}")
    assert m["cosine"] > 0.99, f"Quantized weight cosine {m['cosine']:.6f} < 0.99"

    print("  PASS")


def test_gemm_m1_decode(verbose=False):
    """M=1 decode path for all projection shapes."""
    print("\n=== test_gemm_m1_decode ===")

    M = 1
    for name, N, K in PROJECTION_SHAPES:
        w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K)
        w_uint8_gpu = w_uint8.cuda()
        w_scale_gpu = w_scale.cuda()
        scale2_val = w_scale2.item()

        torch.manual_seed(55)
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        out_gpu = sm120_mla_kernels.nvfp4_gemm(x, w_uint8_gpu, w_scale_gpu, scale2_val)
        out_ref = ref_nvfp4_gemm(x.cpu(), w_uint8, w_scale, w_scale2)

        m = compute_metrics(out_ref, out_gpu.cpu())
        print(f"  {name} M=1: {fmt_metrics(m)}")
        assert m["cosine"] > 0.99, f"{name} M=1 cosine {m['cosine']:.6f} < 0.99"

    print("  PASS")


def test_gemm_determinism(verbose=False):
    """Same GEMM called 5 times produces bit-identical output."""
    print("\n=== test_gemm_determinism ===")

    N, K = 1536, 7168
    M = 128
    w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K)
    w_uint8_gpu = w_uint8.cuda()
    w_scale_gpu = w_scale.cuda()
    scale2_val = w_scale2.item()

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    outputs = []
    for i in range(5):
        out = sm120_mla_kernels.nvfp4_gemm(x, w_uint8_gpu, w_scale_gpu, scale2_val)
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
    test_gemm_projection_shapes,
    test_gemm_quantized_weights,
    test_gemm_m1_decode,
    test_gemm_determinism,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("NVFP4 GEMM Kernel Tests")
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
