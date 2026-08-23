"""
NVFP4 GEMM Reference Implementation

Golden reference functions for NVFP4 attention weight GEMM kernels. These serve
as correctness oracles for GPU kernel validation (AWD-N3).

NVFP4 format: FP4 E2M1 (2 values/byte) + UE8M0 per-group scales (group_size=16)
+ scalar float32 global scale.

Dequant formula: val = FP4_TABLE[idx] * scale_e4m3[group] * scale_2

Functions:
  ref_nvfp4_dequant()        — dequant packed NVFP4 weight to FP32 matrix
  ref_nvfp4_gemm()           — dequant + matmul (x @ dequant(W).T)
  generate_nvfp4_weight()    — random valid NVFP4 weight for testing
  quantize_to_nvfp4()        — FP32/BF16 weight → NVFP4 packed format

Usage:
  python tests/test_nvfp4_gemm_reference.py -v
"""

import torch
import torch.nn.functional as F
import math
import sys
import argparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FP4 E2M1 lookup table: 4-bit index → float value
# Format: 1 sign bit, 2 exponent bits, 1 mantissa bit (bias=1)
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)

FP4_MAX_ABS = 6.0          # largest representable magnitude in FP4 E2M1
GROUP_SIZE = 16             # NVFP4 per-group scale granularity
FP8_E4M3_MAX = 448.0       # max representable value in FP8 E4M3

# Target projection shapes: (name, out_features, in_features)
PROJECTION_SHAPES = [
    ("q_a",    1536, 7168),   # W_DQ: replicated across TP
    ("q_b",   12288, 1536),   # W_UQ: TP=2 (24576/2)
    ("o_proj",  7168, 8192),  # W_O:  TP=2 (16384/2)
]

BATCH_SIZES = [1, 8, 128, 1024]


# ---------------------------------------------------------------------------
# Reference Functions
# ---------------------------------------------------------------------------

def ref_nvfp4_dequant(
    weight_uint8: torch.Tensor,
    scale_e4m3: torch.Tensor,
    scale_2: torch.Tensor,
) -> torch.Tensor:
    """Dequant NVFP4 packed weight to FP32 matrix.

    Args:
        weight_uint8: [N, K/2] uint8, packed FP4 (low nibble=even, high nibble=odd)
        scale_e4m3:   [N, K/16] per-group scales (FP8 E4M3 or float32)
        scale_2:      scalar float32 global scale

    Returns:
        [N, K] float32 dequantized weight matrix
    """
    out_features, packed_in = weight_uint8.shape
    in_features = packed_in * 2

    # Unpack: low nibble = even index, high nibble = odd index
    lo = (weight_uint8 & 0x0F).to(torch.int64)
    hi = (weight_uint8 >> 4).to(torch.int64)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(out_features, in_features)

    # FP4 → float via lookup
    fp4_float = FP4_E2M1_TABLE[unpacked]

    # Per-group scale (group_size=16) * global scale
    scale = scale_e4m3.float().repeat_interleave(GROUP_SIZE, dim=1)
    return fp4_float * scale * scale_2.float().item()


def ref_nvfp4_gemm(
    x: torch.Tensor,
    weight_uint8: torch.Tensor,
    scale_e4m3: torch.Tensor,
    scale_2: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 dequant + GEMM: x @ dequant(W).T

    Args:
        x:            [M, K] bfloat16 or float32 activation
        weight_uint8: [N, K/2] uint8 packed FP4 weight
        scale_e4m3:   [N, K/16] per-group scales
        scale_2:      scalar float32 global scale

    Returns:
        [M, N] float32 output
    """
    w_float = ref_nvfp4_dequant(weight_uint8, scale_e4m3, scale_2)
    return x.float() @ w_float.T


def generate_nvfp4_weight(
    out_features: int,
    in_features: int,
    seed: int = 42,
) -> tuple:
    """Generate random valid NVFP4 weight for testing.

    Returns:
        (weight_uint8 [N, K/2], scale_e4m3 [N, K/16], scale_2 scalar)
    """
    assert in_features % GROUP_SIZE == 0, \
        f"in_features={in_features} not divisible by GROUP_SIZE={GROUP_SIZE}"

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Random FP4 indices [0, 15] → pack pairs into uint8
    indices = torch.randint(0, 16, (out_features, in_features), generator=rng)
    pairs = indices.reshape(out_features, in_features // 2, 2)
    weight_uint8 = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)

    # Random positive per-group scales
    num_groups = in_features // GROUP_SIZE
    raw_scales = torch.rand(out_features, num_groups, generator=rng) * 0.1 + 0.001
    if hasattr(torch, 'float8_e4m3fn'):
        scale_e4m3 = raw_scales.to(torch.float8_e4m3fn).float()
    else:
        scale_e4m3 = raw_scales

    # Random positive global scale
    scale_2 = torch.tensor(
        torch.rand(1, generator=rng).item() * 0.5 + 0.1, dtype=torch.float32
    )

    return weight_uint8, scale_e4m3, scale_2


def quantize_to_nvfp4(
    weight: torch.Tensor,
) -> tuple:
    """Quantize FP32/BF16 weight matrix to NVFP4 format.

    Two-level quantization: global scale → per-group E4M3 scale → nearest FP4 index.

    Args:
        weight: [N, K] float32 or bfloat16

    Returns:
        (weight_uint8 [N, K/2], scale_e4m3 [N, K/16], scale_2 scalar)
    """
    N, K = weight.shape
    assert K % GROUP_SIZE == 0, f"K={K} not divisible by GROUP_SIZE={GROUP_SIZE}"
    w = weight.float()

    # Step 1: Global scale
    global_amax = w.abs().max()
    scale_2 = global_amax / FP4_MAX_ABS
    scale_2 = torch.clamp(scale_2, min=1e-12)

    # Step 2: Per-group E4M3 scales
    w_groups = w.reshape(N, K // GROUP_SIZE, GROUP_SIZE)  # [N, num_groups, 16]
    group_amax = w_groups.abs().amax(dim=-1)               # [N, num_groups]
    scale_e4m3_raw = group_amax / (scale_2 * FP4_MAX_ABS)
    scale_e4m3_raw = torch.clamp(scale_e4m3_raw, min=1e-12, max=FP8_E4M3_MAX)
    if hasattr(torch, 'float8_e4m3fn'):
        scale_e4m3 = scale_e4m3_raw.to(torch.float8_e4m3fn).float()
    else:
        scale_e4m3 = scale_e4m3_raw
    # Re-clamp after E4M3 rounding (could round to 0 for very small values)
    scale_e4m3 = torch.clamp(scale_e4m3, min=1e-12)

    # Step 3: Nearest FP4 index — process in row chunks to limit memory
    scale_expanded = scale_e4m3.repeat_interleave(GROUP_SIZE, dim=1)  # [N, K]
    chunk_size = 256  # rows per chunk
    all_indices = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        local_vals = w[start:end] / (scale_expanded[start:end] * scale_2.item())
        # [chunk, K, 1] vs [16] → [chunk, K, 16]
        distances = (local_vals.unsqueeze(-1) - FP4_E2M1_TABLE).abs()
        indices = distances.argmin(dim=-1)  # [chunk, K]
        all_indices.append(indices)
    indices = torch.cat(all_indices, dim=0)  # [N, K]

    # Step 4: Pack pairs into uint8
    pairs = indices.reshape(N, K // 2, 2)
    weight_uint8 = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)

    return weight_uint8, scale_e4m3, scale_2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict:
    ref_f = ref.float().flatten()
    test_f = test.float().flatten()
    cosine = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()
    mse = ((ref_f - test_f) ** 2).mean().item()
    nrmse = math.sqrt(mse) / (ref_f.norm().item() / math.sqrt(ref_f.numel()) + 1e-12)
    max_abs = (ref_f - test_f).abs().max().item()
    return {"cosine": cosine, "mse": mse, "nrmse": nrmse, "max_abs_err": max_abs}


def fmt_metrics(m: dict) -> str:
    return f"cos={m['cosine']:.6f} nrmse={m['nrmse']:.4f} max_abs={m['max_abs_err']:.4e}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dequant_known_values(verbose=False):
    """Dequant of hand-constructed NVFP4 weight with known expected output."""
    print("\n=== test_dequant_known_values ===")

    N, K = 4, 32
    num_groups = K // GROUP_SIZE

    # Construct known indices: first group all 0s, second group all index 2 (=1.0)
    indices = torch.zeros(N, K, dtype=torch.int64)
    indices[:, GROUP_SIZE:] = 2  # FP4 index 2 → value 1.0

    # Pack
    pairs = indices.reshape(N, K // 2, 2)
    weight_uint8 = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).to(torch.uint8)

    # Scales: all 1.0
    scale_e4m3 = torch.ones(N, num_groups, dtype=torch.float32)
    scale_2 = torch.tensor(2.0, dtype=torch.float32)

    w = ref_nvfp4_dequant(weight_uint8, scale_e4m3, scale_2)

    # Expected: first 16 cols = 0.0 (FP4[0]=0.0 * 1.0 * 2.0), last 16 cols = 2.0 (FP4[2]=1.0 * 1.0 * 2.0)
    expected = torch.zeros(N, K, dtype=torch.float32)
    expected[:, GROUP_SIZE:] = 2.0

    diff = (w - expected).abs().max().item()
    print(f"  max abs diff from expected: {diff:.2e}")
    assert diff < 1e-6, f"Known-value dequant mismatch: max diff {diff}"

    # Also test negative values: index 10 → -1.0
    indices2 = torch.full((2, 32), 10, dtype=torch.int64)  # all -1.0
    pairs2 = indices2.reshape(2, 16, 2)
    w2_uint8 = (pairs2[:, :, 0] | (pairs2[:, :, 1] << 4)).to(torch.uint8)
    scale2 = torch.ones(2, 2, dtype=torch.float32)
    w2 = ref_nvfp4_dequant(w2_uint8, scale2, torch.tensor(1.0))
    expected2 = torch.full((2, 32), -1.0, dtype=torch.float32)
    diff2 = (w2 - expected2).abs().max().item()
    print(f"  negative values max diff: {diff2:.2e}")
    assert diff2 < 1e-6, f"Negative dequant mismatch: max diff {diff2}"

    print("  PASS")


def test_dequant_gemm_matches_manual(verbose=False):
    """ref_nvfp4_gemm matches manual dequant + matmul."""
    print("\n=== test_dequant_gemm_matches_manual ===")

    N, K, M = 64, 128, 16
    w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K, seed=123)
    x = torch.randn(M, K, dtype=torch.float32)

    # Via ref_nvfp4_gemm
    out_gemm = ref_nvfp4_gemm(x, w_uint8, w_scale, w_scale2)

    # Manual: dequant then matmul
    w_deq = ref_nvfp4_dequant(w_uint8, w_scale, w_scale2)
    out_manual = x @ w_deq.T

    diff = (out_gemm - out_manual).abs().max().item()
    print(f"  max abs diff: {diff:.2e}")
    assert diff < 1e-6, f"GEMM vs manual dequant+matmul mismatch: {diff}"

    # Also test with bfloat16 input
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16)
    out_bf16 = ref_nvfp4_gemm(x_bf16, w_uint8, w_scale, w_scale2)
    out_bf16_manual = x_bf16.float() @ w_deq.T
    diff_bf16 = (out_bf16 - out_bf16_manual).abs().max().item()
    print(f"  bf16 input max abs diff: {diff_bf16:.2e}")
    assert diff_bf16 < 1e-6, f"BF16 GEMM mismatch: {diff_bf16}"

    print("  PASS")


def test_generate_nvfp4_weight_valid(verbose=False):
    """Generated NVFP4 weights have correct shapes, dtypes, and valid data."""
    print("\n=== test_generate_nvfp4_weight_valid ===")

    for name, N, K in PROJECTION_SHAPES:
        w_uint8, w_scale, w_scale2 = generate_nvfp4_weight(N, K)

        # Shape checks
        assert w_uint8.shape == (N, K // 2), \
            f"{name}: weight shape {w_uint8.shape} != ({N}, {K // 2})"
        assert w_scale.shape == (N, K // GROUP_SIZE), \
            f"{name}: scale shape {w_scale.shape} != ({N}, {K // GROUP_SIZE})"
        assert w_scale2.dim() == 0, f"{name}: scale_2 not scalar"

        # Dtype checks
        assert w_uint8.dtype == torch.uint8, f"{name}: weight dtype {w_uint8.dtype}"
        assert w_scale.dtype == torch.float32, f"{name}: scale dtype {w_scale.dtype}"
        assert w_scale2.dtype == torch.float32, f"{name}: scale_2 dtype {w_scale2.dtype}"

        # Valid nibble range: unpack and check [0, 15]
        lo = w_uint8 & 0x0F
        hi = (w_uint8 >> 4) & 0x0F
        assert lo.max() <= 15 and lo.min() >= 0, f"{name}: invalid low nibbles"
        assert hi.max() <= 15 and hi.min() >= 0, f"{name}: invalid high nibbles"

        # Positive scales
        assert (w_scale > 0).all(), f"{name}: non-positive scales found"
        assert w_scale2.item() > 0, f"{name}: non-positive global scale"

        # Dequant doesn't produce NaN/Inf
        w_deq = ref_nvfp4_dequant(w_uint8, w_scale, w_scale2)
        assert not torch.isnan(w_deq).any(), f"{name}: NaN in dequanted weight"
        assert not torch.isinf(w_deq).any(), f"{name}: Inf in dequanted weight"

        print(f"  {name} [{N}, {K}]: OK "
              f"(weight_range=[{w_deq.min():.4f}, {w_deq.max():.4f}])")

    print("  PASS")


def test_quantize_roundtrip_cosine(verbose=False):
    """dequant(quantize(W_bf16)) vs original: cosine > 0.98 for all projection shapes."""
    print("\n=== test_quantize_roundtrip_cosine ===")

    for name, N, K in PROJECTION_SHAPES:
        torch.manual_seed(42)
        w_orig = torch.randn(N, K, dtype=torch.bfloat16)

        w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w_orig)
        w_deq = ref_nvfp4_dequant(w_uint8, w_scale, w_scale2)

        m = compute_metrics(w_orig, w_deq)
        print(f"  {name} [{N}, {K}]: {fmt_metrics(m)}")

        if verbose:
            # Per-row cosine stats
            per_row = F.cosine_similarity(
                w_orig.float(), w_deq, dim=-1)
            print(f"    per-row cosine: mean={per_row.mean():.4f} "
                  f"min={per_row.min():.4f} p5={per_row.quantile(0.05):.4f}")

        assert m["cosine"] > 0.98, \
            f"{name} round-trip cosine {m['cosine']:.6f} < 0.98"

    print("  PASS")


def test_gemm_shapes(verbose=False):
    """GEMM output shapes correct, quantized GEMM vs BF16 ground truth cosine > 0.98."""
    print("\n=== test_gemm_shapes ===")

    for name, N, K in PROJECTION_SHAPES:
        torch.manual_seed(42)
        # Quantize a BF16 weight
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
        w_uint8, w_scale, w_scale2 = quantize_to_nvfp4(w_bf16)

        for M in BATCH_SIZES:
            torch.manual_seed(100 + M)
            x = torch.randn(M, K, dtype=torch.bfloat16)

            # NVFP4 GEMM
            out_nvfp4 = ref_nvfp4_gemm(x, w_uint8, w_scale, w_scale2)

            # Shape check
            assert out_nvfp4.shape == (M, N), \
                f"{name} M={M}: output shape {out_nvfp4.shape} != ({M}, {N})"

            # BF16 ground truth
            out_bf16 = x.float() @ w_bf16.float().T

            m = compute_metrics(out_bf16, out_nvfp4)
            if verbose:
                print(f"  {name} M={M}: {fmt_metrics(m)}")

            assert m["cosine"] > 0.98, \
                f"{name} M={M} GEMM cosine {m['cosine']:.6f} < 0.98"

        print(f"  {name} [{N}, {K}] x M={BATCH_SIZES}: all cosine > 0.98")

    print("  PASS")


def test_quantize_edge_cases(verbose=False):
    """Quantization handles edge cases without NaN/Inf."""
    print("\n=== test_quantize_edge_cases ===")

    # 1. Zero matrix
    w_zero = torch.zeros(32, 64)
    u8, sc, s2 = quantize_to_nvfp4(w_zero)
    w_deq = ref_nvfp4_dequant(u8, sc, s2)
    assert not torch.isnan(w_deq).any(), "NaN from zero matrix"
    assert not torch.isinf(w_deq).any(), "Inf from zero matrix"
    assert w_deq.abs().max() < 1e-6, "Zero matrix should dequant near zero"
    print("  zero matrix: OK")

    # 2. Constant matrix (3.0 is in FP4 table as index 5)
    w_const = torch.full((32, 64), 3.0)
    u8, sc, s2 = quantize_to_nvfp4(w_const)
    w_deq = ref_nvfp4_dequant(u8, sc, s2)
    m = compute_metrics(w_const, w_deq)
    assert m["cosine"] > 0.999, f"Constant matrix cosine {m['cosine']:.6f} < 0.999"
    print(f"  constant 3.0 matrix: {fmt_metrics(m)}")

    # 3. Very small weights
    w_tiny = torch.randn(32, 64) * 1e-8
    u8, sc, s2 = quantize_to_nvfp4(w_tiny)
    w_deq = ref_nvfp4_dequant(u8, sc, s2)
    assert not torch.isnan(w_deq).any(), "NaN from tiny weights"
    assert not torch.isinf(w_deq).any(), "Inf from tiny weights"
    print("  tiny weights (1e-8): OK (no NaN/Inf)")

    # 4. Mixed signs — negative values should be preserved
    torch.manual_seed(42)
    w_mixed = torch.randn(32, 64)
    u8, sc, s2 = quantize_to_nvfp4(w_mixed)
    w_deq = ref_nvfp4_dequant(u8, sc, s2)
    # Check sign agreement on non-tiny values
    mask = w_mixed.abs() > w_mixed.abs().mean() * 0.1
    sign_match = ((w_mixed[mask] > 0) == (w_deq[mask] > 0)).float().mean()
    print(f"  mixed signs: sign agreement = {sign_match:.4f}")
    assert sign_match > 0.95, f"Sign agreement {sign_match:.4f} < 0.95"

    # 5. Large dynamic range across rows
    w_varied = torch.randn(32, 64) * torch.logspace(-3, 3, 32).unsqueeze(1)
    u8, sc, s2 = quantize_to_nvfp4(w_varied)
    w_deq = ref_nvfp4_dequant(u8, sc, s2)
    m = compute_metrics(w_varied, w_deq)
    print(f"  varied magnitude rows: {fmt_metrics(m)}")
    assert m["cosine"] > 0.95, f"Varied rows cosine {m['cosine']:.6f} < 0.95"

    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_dequant_known_values,
    test_dequant_gemm_matches_manual,
    test_generate_nvfp4_weight_valid,
    test_quantize_roundtrip_cosine,
    test_gemm_shapes,
    test_quantize_edge_cases,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("NVFP4 GEMM Reference Tests")
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
