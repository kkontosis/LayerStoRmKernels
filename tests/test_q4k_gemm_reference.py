"""
Q4_K GEMM Reference Implementation

Golden reference functions for Q4_K (GGUF) attention weight GEMM kernels.
These serve as correctness oracles for GPU kernel validation (AWD-G3).

Q4_K format: 256 values per super-block, 8 sub-blocks of 32 values.
144 bytes/block = 4.5 bits/value.
Dequant: y[i] = d * sc[j] * q[i] - dmin * m[j]

Block layout (144 bytes):
  [0:2]    d      FP16 super-block scale
  [2:4]    dmin   FP16 super-block min scale
  [4:16]   scales 12 bytes packed 6-bit sub-block scales + mins
  [16:144] qs     128 bytes = 256 4-bit quants (2 per byte)

Functions:
  unpack_q4k_scales()     — extract 6-bit scales/mins from 12-byte packed array
  dequant_q4k_block()     — dequant one 144-byte super-block to 256 floats
  ref_q4k_dequant()       — dequant entire Q4_K weight matrix
  ref_q4k_gemm()          — dequant + matmul (x @ dequant(W).T)
  quantize_to_q4k()       — float32 weight → Q4_K packed format
  generate_q4k_weight()   — random valid Q4_K weight for testing

Usage:
  python tests/test_q4k_gemm_reference.py -v
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import struct
import sys
import argparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QK_K = 256              # super-block size
BLOCK_Q4K_SIZE = 144    # bytes per block
N_SUB_BLOCKS = 8        # sub-blocks per super-block
SUB_BLOCK_SIZE = 32     # values per sub-block
K_SCALE_SIZE = 12       # packed scales bytes

# Bytes per row = (K / QK_K) * BLOCK_Q4K_SIZE = K * 144 / 256 = K * 9 / 16
def bytes_per_row(K):
    assert K % QK_K == 0
    return K * BLOCK_Q4K_SIZE // QK_K

# Target projection shapes: (name, out_features, in_features)
PROJECTION_SHAPES = [
    ("q_a",    1536, 7168),
    ("q_b",   12288, 1536),
    ("o_proj",  7168, 8192),
]

BATCH_SIZES = [1, 8, 128, 1024]


# ---------------------------------------------------------------------------
# Scale unpacking
# ---------------------------------------------------------------------------

def unpack_q4k_scales(scales_12):
    """Extract 8 scales and 8 mins from 12-byte packed array.

    Args:
        scales_12: numpy uint8 array of shape (..., 12)

    Returns:
        (sc, m): each numpy uint8 array of shape (..., 8), values 0-63
    """
    s = np.asarray(scales_12, dtype=np.uint8)
    orig_shape = s.shape[:-1]
    s = s.reshape(-1, 3, 4)

    d = s[:, 0, :]     # bytes 0-3
    m = s[:, 1, :]     # bytes 4-7
    m_d = s[:, 2, :]   # bytes 8-11

    # Scales: lower 4 from bytes 0-3 (6 bits), upper 4 from bytes 8-11 + bytes 0-3
    sc = np.concatenate([
        (d & 0x3F),
        (m_d & 0x0F) | ((d >> 2) & 0x30),
    ], axis=-1)

    # Mins: lower 4 from bytes 4-7 (6 bits), upper 4 from bytes 8-11 + bytes 4-7
    mn = np.concatenate([
        (m & 0x3F),
        (m_d >> 4) | ((m >> 2) & 0x30),
    ], axis=-1)

    return sc.reshape(*orig_shape, 8), mn.reshape(*orig_shape, 8)


# ---------------------------------------------------------------------------
# Block-level dequantization
# ---------------------------------------------------------------------------

def dequant_q4k_block(block_bytes):
    """Dequant one Q4_K super-block (144 bytes) to 256 float32 values.

    Args:
        block_bytes: numpy uint8 array of shape (144,)

    Returns:
        float32 array of shape (256,)
    """
    b = np.asarray(block_bytes, dtype=np.uint8)
    assert b.shape == (BLOCK_Q4K_SIZE,)

    # Parse fields
    d_fp16 = np.frombuffer(b[0:2], dtype=np.float16).astype(np.float32)[0]
    dmin_fp16 = np.frombuffer(b[2:4], dtype=np.float16).astype(np.float32)[0]
    scales_packed = b[4:16]
    qs = b[16:144]

    sc, mn = unpack_q4k_scales(scales_packed)
    sc = sc.flatten().astype(np.float32)
    mn = mn.flatten().astype(np.float32)

    out = np.zeros(QK_K, dtype=np.float32)

    # Process 4 pairs of sub-blocks (64 values each, 32 bytes of qs each)
    is_idx = 0
    q_offset = 0
    for pair in range(4):
        d1 = d_fp16 * sc[is_idx]
        m1 = dmin_fp16 * mn[is_idx]
        d2 = d_fp16 * sc[is_idx + 1]
        m2 = dmin_fp16 * mn[is_idx + 1]

        base = pair * 64
        for l in range(32):
            qbyte = qs[q_offset + l]
            out[base + l] = d1 * float(qbyte & 0xF) - m1
            out[base + l + 32] = d2 * float(qbyte >> 4) - m2

        q_offset += 32
        is_idx += 2

    return out


# ---------------------------------------------------------------------------
# Vectorized dequantization (numpy, for full matrices)
# ---------------------------------------------------------------------------

def ref_q4k_dequant(weight_q4k):
    """Dequant entire Q4_K weight matrix.

    Args:
        weight_q4k: numpy uint8 array of shape (N, bytes_per_row) or torch uint8 tensor

    Returns:
        torch float32 tensor of shape (N, K)
    """
    if isinstance(weight_q4k, torch.Tensor):
        w = weight_q4k.numpy()
    else:
        w = np.asarray(weight_q4k, dtype=np.uint8)

    N = w.shape[0]
    row_bytes = w.shape[1]
    n_blocks_per_row = row_bytes // BLOCK_Q4K_SIZE
    K = n_blocks_per_row * QK_K

    # Reshape to (N * n_blocks, BLOCK_Q4K_SIZE)
    blocks = w.reshape(N * n_blocks_per_row, BLOCK_Q4K_SIZE)
    n_blocks = blocks.shape[0]

    # Parse fields
    d = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).reshape(n_blocks, 1).astype(np.float32)
    dmin = np.frombuffer(blocks[:, 2:4].tobytes(), dtype=np.float16).reshape(n_blocks, 1).astype(np.float32)
    scales_packed = blocks[:, 4:16]
    qs = blocks[:, 16:144]

    sc, mn = unpack_q4k_scales(scales_packed)
    sc = sc.astype(np.float32)  # (n_blocks, 8)
    mn = mn.astype(np.float32)  # (n_blocks, 8)

    # d * sc for each sub-block → (n_blocks, 8, 1)
    d_sc = (d * sc).reshape(n_blocks, -1, 1)
    # dmin * mn for each sub-block → (n_blocks, 8, 1)
    dm_mn = (dmin * mn).reshape(n_blocks, -1, 1)

    # Unpack 4-bit quants: qs is (n_blocks, 128)
    # Reshape to (n_blocks, 4_pairs, 1, 32) then shift by [0, 4]
    qs_r = qs.reshape(n_blocks, -1, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
    qs_r = (qs_r & np.uint8(0x0F)).reshape(n_blocks, -1, 32).astype(np.float32)
    # Now qs_r is (n_blocks, 8, 32) — 8 sub-blocks of 32 values

    # Dequant: d*sc*q - dmin*m
    result = (d_sc * qs_r - dm_mn).reshape(n_blocks, QK_K)

    # Reshape back to (N, K)
    result = result.reshape(N, K)
    return torch.from_numpy(result)


# ---------------------------------------------------------------------------
# Reference GEMM
# ---------------------------------------------------------------------------

def ref_q4k_gemm(x, weight_q4k):
    """Q4_K dequant + GEMM: x @ dequant(W).T

    Args:
        x: torch tensor [M, K] (any dtype, converted to float32)
        weight_q4k: numpy/torch uint8 [N, bytes_per_row]

    Returns:
        torch float32 tensor [M, N]
    """
    w_float = ref_q4k_dequant(weight_q4k)  # (N, K)
    x_float = x.float()
    return x_float @ w_float.T


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _quantize_sub_block(values):
    """Quantize 32 float values to Q4_K sub-block.

    Returns (scale, min_offset, quants[32]) where quants are 0-15 integers.
    The Q4_K formula: val = scale * q - min_offset, where min_offset >= 0 (unsigned).

    Simple min/max quantization (not the full ggml make_qkx2_quants optimizer).
    """
    v = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(v))
    vmax = float(np.max(v))

    # min_offset is non-negative (unsigned in Q4_K)
    # When vmin >= 0: min_offset = 0, map [0, vmax] to [0, 15]
    # When vmin < 0:  min_offset = -vmin, map [vmin, vmax] to [0, 15]
    min_offset = max(0.0, -vmin)
    range_val = vmax + min_offset

    if range_val < 1e-15:
        return 0.0, min_offset, np.zeros(32, dtype=np.uint8)

    scale = range_val / 15.0
    inv_scale = 1.0 / scale
    quants = np.clip(np.round((v + min_offset) * inv_scale), 0, 15).astype(np.uint8)

    return scale, min_offset, quants


def _pack_q4k_scales(sc_8, mn_8):
    """Pack 8 6-bit scales and 8 6-bit mins into 12 bytes.

    Args:
        sc_8: uint8 array of shape (8,), values 0-63
        mn_8: uint8 array of shape (8,), values 0-63

    Returns:
        uint8 array of shape (12,)
    """
    out = np.zeros(12, dtype=np.uint8)

    for j in range(8):
        ls = int(sc_8[j]) & 63
        lm = int(mn_8[j]) & 63
        if j < 4:
            out[j] = ls
            out[j + 4] = lm
        else:
            out[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4)
            out[j - 4] |= ((ls >> 4) << 6)
            out[j] |= ((lm >> 4) << 6)

    return out


def _pack_q4k_scales_batch(sc_q, mn_q):
    """Vectorized scale packing: [B, 8] uint8 → [B, 12] uint8."""
    B = sc_q.shape[0]
    out = np.zeros((B, 12), dtype=np.uint8)
    # Lower 4 scales (j=0..3): direct 6-bit store
    out[:, 0:4] = sc_q[:, 0:4] & 0x3F
    out[:, 4:8] = mn_q[:, 0:4] & 0x3F
    # Upper 4 scales (j=4..7): split across bytes 8-11 and high bits of 0-3,4-7
    out[:, 8:12] = (sc_q[:, 4:8] & 0x0F) | ((mn_q[:, 4:8] & 0x0F) << 4)
    out[:, 0:4] |= ((sc_q[:, 4:8] >> 4) << 6)
    out[:, 4:8] |= ((mn_q[:, 4:8] >> 4) << 6)
    return out


def quantize_to_q4k(weight):
    """Quantize float weight matrix to Q4_K format (vectorized).

    Args:
        weight: torch tensor [N, K] (float32 or bfloat16)

    Returns:
        numpy uint8 array [N, K*9/16] — contiguous Q4_K blocks
    """
    w = weight.float().numpy()
    N, K = w.shape
    assert K % QK_K == 0, f"K={K} must be divisible by {QK_K}"

    n_blocks = K // QK_K
    bpr = bytes_per_row(K)
    out = np.zeros((N, bpr), dtype=np.uint8)

    # Reshape to [N, n_blocks, 8, 32] — all sub-blocks at once
    blocks = w.reshape(N, n_blocks, N_SUB_BLOCKS, SUB_BLOCK_SIZE)  # [N, nB, 8, 32]

    # ── Sub-block quantization (vectorized) ────────────────────────────────
    # min/max per sub-block: [N, nB, 8]
    sb_min = blocks.min(axis=-1)    # [N, nB, 8]
    sb_max = blocks.max(axis=-1)    # [N, nB, 8]
    min_offset = np.maximum(0.0, -sb_min)  # [N, nB, 8]
    range_val = sb_max + min_offset         # [N, nB, 8]

    scales = np.where(range_val > 1e-15, range_val / 15.0, 0.0)  # [N, nB, 8]
    mins = min_offset  # [N, nB, 8]

    # ── Super-block d, dmin ────────────────────────────────────────────────
    max_scale = scales.max(axis=-1)  # [N, nB]
    max_min = mins.max(axis=-1)      # [N, nB]

    d_f32 = np.where(max_scale > 0, max_scale / 63.0, 0.0)      # [N, nB]
    dmin_f32 = np.where(max_min > 0, max_min / 63.0, 0.0)        # [N, nB]

    # Quantize sub-block scales/mins to 6-bit
    inv_scale = np.where(max_scale > 0, 63.0 / max_scale, 0.0)   # [N, nB]
    inv_min = np.where(max_min > 0, 63.0 / max_min, 0.0)         # [N, nB]

    sc_q = np.clip(np.round(inv_scale[..., None] * scales), 0, 63).astype(np.uint8)  # [N, nB, 8]
    mn_q = np.clip(np.round(inv_min[..., None] * mins), 0, 63).astype(np.uint8)      # [N, nB, 8]

    # ── Pack scales and round-trip to get exact values ─────────────────────
    B_flat = N * n_blocks
    packed_scales = _pack_q4k_scales_batch(
        sc_q.reshape(B_flat, 8), mn_q.reshape(B_flat, 8))  # [B_flat, 12]
    sc_actual, mn_actual = unpack_q4k_scales(
        packed_scales.reshape(B_flat, 12))  # [B_flat, 8] each
    sc_actual = sc_actual.reshape(N, n_blocks, 8).astype(np.float32)
    mn_actual = mn_actual.reshape(N, n_blocks, 8).astype(np.float32)
    packed_scales = packed_scales.reshape(N, n_blocks, 12)

    # FP16 round-trip for d, dmin (matches ggml storage)
    d_fp16 = d_f32.astype(np.float16).astype(np.float32)      # [N, nB]
    dmin_fp16 = dmin_f32.astype(np.float16).astype(np.float32)  # [N, nB]

    # ── Re-quantize L using actual lossy scales (matches ggml) ────────────
    # d_j = d_fp16 * sc_actual, dm_j = dmin_fp16 * mn_actual  per sub-block
    d_j = d_fp16[..., None] * sc_actual        # [N, nB, 8]
    dm_j = dmin_fp16[..., None] * mn_actual    # [N, nB, 8]

    # Expand to per-element: [N, nB, 8, 32]
    d_j_exp = d_j[..., None]    # [N, nB, 8, 1]
    dm_j_exp = dm_j[..., None]  # [N, nB, 8, 1]

    # L = clip(round((val + dm_j) / d_j), 0, 15)
    safe_d_j = np.where(d_j_exp > 0, d_j_exp, 1.0)  # avoid div-by-zero
    L = np.clip(np.round((blocks + dm_j_exp) / safe_d_j), 0, 15).astype(np.uint8)
    L = np.where(d_j_exp > 0, L, 0)  # zero where d_j == 0

    # ── Write output blocks ───────────────────────────────────────────────
    for blk in range(n_blocks):
        bo = blk * BLOCK_Q4K_SIZE

        # d and dmin as FP16 bytes
        d_bytes = d_f32[:, blk].astype(np.float16).view(np.uint8).reshape(N, 2)
        dm_bytes = dmin_f32[:, blk].astype(np.float16).view(np.uint8).reshape(N, 2)
        out[:, bo:bo + 2] = d_bytes
        out[:, bo + 2:bo + 4] = dm_bytes

        # Packed 12-byte scales
        out[:, bo + 4:bo + 16] = packed_scales[:, blk, :]

        # Pack 4-bit quants: pairs of sub-blocks interleaved
        L_blk = L[:, blk, :, :]  # [N, 8, 32]
        q_dst = bo + 16
        for pair in range(4):
            lo = L_blk[:, pair * 2, :]      # [N, 32]
            hi = L_blk[:, pair * 2 + 1, :]  # [N, 32]
            out[:, q_dst:q_dst + 32] = (lo & 0x0F) | ((hi & 0x0F) << 4)
            q_dst += 32

    return out


def generate_q4k_weight(N, K, seed=42):
    """Generate random valid Q4_K weight for testing.

    Args:
        N: number of rows (out_features)
        K: number of columns (in_features), must be divisible by 256

    Returns:
        numpy uint8 array [N, K*9/16]
    """
    rng = np.random.RandomState(seed)
    w = rng.randn(N, K).astype(np.float32)
    return quantize_to_q4k(torch.from_numpy(w))


# ---------------------------------------------------------------------------
# Metrics (reuse same pattern as NVFP4 reference)
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

def test_unpack_scales_known(verbose=False):
    """Hand-crafted 12-byte scales, verify extraction matches manual computation."""
    print("\n=== test_unpack_scales_known ===")

    # Construct known scales: sc[0..7] = [1, 2, 3, 4, 5, 10, 15, 20]
    #                          m[0..7]  = [2, 4, 6, 8, 10, 20, 30, 40]
    # Pack using _pack_q4k_scales
    sc = np.array([1, 2, 3, 4, 5, 10, 15, 20], dtype=np.uint8)
    mn = np.array([2, 4, 6, 8, 10, 20, 30, 40], dtype=np.uint8)

    packed = _pack_q4k_scales(sc, mn)
    assert packed.shape == (12,), f"Packed shape {packed.shape} != (12,)"

    sc_out, mn_out = unpack_q4k_scales(packed)
    sc_out = sc_out.flatten()
    mn_out = mn_out.flatten()

    if verbose:
        print(f"  packed: {list(packed)}")
        print(f"  sc expected: {list(sc)}, got: {list(sc_out)}")
        print(f"  mn expected: {list(mn)}, got: {list(mn_out)}")

    assert np.array_equal(sc_out, sc), f"Scales mismatch: {sc_out} != {sc}"
    assert np.array_equal(mn_out, mn), f"Mins mismatch: {mn_out} != {mn}"

    # Test batch unpacking
    packed_batch = np.stack([packed, packed])
    sc_b, mn_b = unpack_q4k_scales(packed_batch)
    assert sc_b.shape == (2, 8)
    assert np.array_equal(sc_b[0], sc)
    assert np.array_equal(sc_b[1], sc)

    print("  PASS")


def test_dequant_known_values(verbose=False):
    """Hand-crafted Q4_K block with known values, verify dequant output."""
    print("\n=== test_dequant_known_values ===")

    # Create a block where:
    #   d = 1.0 (FP16), dmin = 0.5 (FP16)
    #   All sub-block scales = 1, all mins = 1
    #   All quants = 7 (middle value)
    # Expected: y = 1.0 * 1 * 7 - 0.5 * 1 = 6.5

    block = np.zeros(BLOCK_Q4K_SIZE, dtype=np.uint8)

    # d = 1.0, dmin = 0.5
    d_bytes = np.float16(1.0).tobytes()
    dmin_bytes = np.float16(0.5).tobytes()
    block[0:2] = np.frombuffer(d_bytes, dtype=np.uint8)
    block[2:4] = np.frombuffer(dmin_bytes, dtype=np.uint8)

    # All scales = 1, all mins = 1 (6-bit, packed)
    sc = np.ones(8, dtype=np.uint8)
    mn = np.ones(8, dtype=np.uint8)
    block[4:16] = _pack_q4k_scales(sc, mn)

    # All quants = 7: even and odd nibbles both 7 → byte = 0x77
    block[16:144] = 0x77

    out = dequant_q4k_block(block)
    expected = 1.0 * 1.0 * 7.0 - 0.5 * 1.0  # = 6.5
    assert out.shape == (256,)
    assert np.allclose(out, expected, atol=1e-3), \
        f"Expected all {expected}, got range [{out.min()}, {out.max()}]"

    if verbose:
        print(f"  Expected: {expected}, Got: {out[0]:.4f} (all same: {np.allclose(out, out[0])})")

    # Test with varying quants
    block[16:144] = 0x0F  # low nibble = 15, high nibble = 0
    out2 = dequant_q4k_block(block)
    # Even positions: d*sc*15 - dmin*m = 1*1*15 - 0.5*1 = 14.5
    # Odd positions:  d*sc*0  - dmin*m = 1*1*0  - 0.5*1 = -0.5
    for i in range(0, 256, 64):
        # First 32 of each pair: low nibble = 15
        assert np.allclose(out2[i:i + 32], 14.5, atol=1e-3), \
            f"Even block {i}: expected 14.5, got {out2[i]}"
        # Next 32: high nibble = 0
        assert np.allclose(out2[i + 32:i + 64], -0.5, atol=1e-3), \
            f"Odd block {i+32}: expected -0.5, got {out2[i+32]}"

    print("  PASS")


def test_dequant_roundtrip(verbose=False):
    """quantize_to_q4k() then ref_q4k_dequant() vs original — cosine > 0.995."""
    print("\n=== test_dequant_roundtrip ===")

    torch.manual_seed(42)
    for name, N_full, K in PROJECTION_SHAPES:
        N = min(N_full, 64)  # use small N for speed
        w = torch.randn(N, K)
        w_q4k = quantize_to_q4k(w)
        w_deq = ref_q4k_dequant(w_q4k)

        assert w_deq.shape == (N, K), f"Shape mismatch: {w_deq.shape} != ({N}, {K})"
        assert not torch.isnan(w_deq).any(), "NaN in dequantized output"
        assert not torch.isinf(w_deq).any(), "Inf in dequantized output"

        m = compute_metrics(w, w_deq)
        if verbose:
            print(f"  {name} [{N}, {K}]: {fmt_metrics(m)}")

        assert m["cosine"] > 0.995, \
            f"{name} roundtrip cosine {m['cosine']:.6f} < 0.995"

    print("  PASS")


def test_generate_q4k_weight_valid(verbose=False):
    """Generated weight has correct shape, dtype, dequants without NaN/Inf."""
    print("\n=== test_generate_q4k_weight_valid ===")

    N, K = 64, 256
    w = generate_q4k_weight(N, K)
    assert w.dtype == np.uint8
    assert w.shape == (N, bytes_per_row(K))

    deq = ref_q4k_dequant(w)
    assert deq.shape == (N, K)
    assert not torch.isnan(deq).any()
    assert not torch.isinf(deq).any()

    if verbose:
        print(f"  shape: {w.shape}, dequant range: [{deq.min():.3f}, {deq.max():.3f}]")

    print("  PASS")


def test_gemm_shapes(verbose=False):
    """ref_q4k_gemm produces correct shapes, Q4_K vs BF16 ground truth cosine > 0.995."""
    print("\n=== test_gemm_shapes ===")

    for name, N_full, K in PROJECTION_SHAPES:
        N = min(N_full, 64)  # small N for speed
        torch.manual_seed(42)
        w_bf16 = torch.randn(N, K)
        w_q4k = quantize_to_q4k(w_bf16)

        for M in [1, 128]:
            x = torch.randn(M, K, dtype=torch.bfloat16)

            out_q4k = ref_q4k_gemm(x, w_q4k)
            assert out_q4k.shape == (M, N), \
                f"{name} M={M}: shape {out_q4k.shape} != ({M}, {N})"

            out_bf16 = x.float() @ w_bf16.float().T
            m = compute_metrics(out_bf16, out_q4k)

            if verbose:
                print(f"  {name} M={M}: {fmt_metrics(m)}")

            assert m["cosine"] > 0.995, \
                f"{name} M={M} GEMM cosine {m['cosine']:.6f} < 0.995"

    print("  PASS")


def test_quantize_edge_cases(verbose=False):
    """Quantization handles edge cases without NaN/Inf."""
    print("\n=== test_quantize_edge_cases ===")

    # 1. Zero matrix
    w_zero = torch.zeros(4, 256)
    q = quantize_to_q4k(w_zero)
    deq = ref_q4k_dequant(q)
    assert not torch.isnan(deq).any(), "NaN from zero matrix"
    assert not torch.isinf(deq).any(), "Inf from zero matrix"
    assert deq.abs().max() < 1e-3, "Zero matrix should dequant near zero"
    print("  zero matrix: OK")

    # 2. Constant matrix (cosine undefined for constant vectors — use max_abs_err)
    w_const = torch.full((4, 256), 3.0)
    q = quantize_to_q4k(w_const)
    deq = ref_q4k_dequant(q)
    max_err = (w_const - deq).abs().max().item()
    assert max_err < 0.5, f"Constant max_abs_err {max_err:.4f} >= 0.5"
    print(f"  constant 3.0: max_abs_err={max_err:.4e}")

    # 3. Very small weights
    w_tiny = torch.randn(4, 256) * 1e-8
    q = quantize_to_q4k(w_tiny)
    deq = ref_q4k_dequant(q)
    assert not torch.isnan(deq).any(), "NaN from tiny weights"
    assert not torch.isinf(deq).any(), "Inf from tiny weights"
    print("  tiny weights (1e-8): OK")

    # 4. Mixed signs
    torch.manual_seed(42)
    w_mixed = torch.randn(4, 256)
    q = quantize_to_q4k(w_mixed)
    deq = ref_q4k_dequant(q)
    m = compute_metrics(w_mixed, deq)
    assert m["cosine"] > 0.995, f"Mixed sign cosine {m['cosine']:.6f} < 0.995"
    print(f"  mixed signs: {fmt_metrics(m)}")

    # 5. Large dynamic range
    w_varied = torch.randn(4, 256) * torch.logspace(-2, 2, 4).unsqueeze(1)
    q = quantize_to_q4k(w_varied)
    deq = ref_q4k_dequant(q)
    m = compute_metrics(w_varied, deq)
    print(f"  varied magnitude: {fmt_metrics(m)}")
    assert m["cosine"] > 0.99, f"Varied cosine {m['cosine']:.6f} < 0.99"

    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_unpack_scales_known,
    test_dequant_known_values,
    test_dequant_roundtrip,
    test_generate_q4k_weight_valid,
    test_gemm_shapes,
    test_quantize_edge_cases,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("Q4_K GEMM Reference Tests")
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
