"""
TurboQuant MLA Reference Implementation — Paged Cache Format

Golden reference functions for TQ_MSE CUDA kernels. These operate on the exact
same paged cache byte layout that the CUDA kernels use, serving as correctness
oracles for kernel validation.

TQ KV cache layout per token (V3.2, d_c=512):
  [256B packed_nope (4-bit, 2 indices/byte)] [2B FP16 norm] [128B BF16 rope]
  Total: 386 bytes/token

Functions:
  ref_tq_mse_k_append()          — quantize + pack + write to paged cache
  ref_tq_mse_dequant_indexed()   — gather + unpack + dequant from paged cache
  ref_tq_mse_q_rotate()          — pre-rotate query NOPE by Pi^T
  ref_tq_mse_decode()            — full TQ decode with rotated-space PV
  ref_tq_mse_v_rotate_back()     — epilogue inverse rotation

Usage:
  python tests/test_tq_reference.py -v
"""

import torch
import torch.nn.functional as F
import math
import json
import os
import sys
import argparse
import struct

# ---------------------------------------------------------------------------
# Config (matches test_snapmla_reference.py)
# ---------------------------------------------------------------------------

D_C = 512
D_ROPE = 64
D_QK = D_C + D_ROPE
D_V = 512
H_Q = 64
H_KV = 1
TQ_BITS = 4
TQ_NUM_CENTROIDS = 16
TQ_SEED = 42

# TQ cache layout (V3.2)
TQ_PACKED_NOPE_BYTES = (D_C * TQ_BITS + 7) // 8  # 256
TQ_NORM_BYTES = 2  # FP16
TQ_ROPE_BYTES = D_ROPE * 2  # BF16
TQ_BYTES_PER_TOKEN = TQ_PACKED_NOPE_BYTES + TQ_NORM_BYTES + TQ_ROPE_BYTES  # 386

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEBOOK_DIR = os.path.join(PROJECT_ROOT, "data", "codebooks")


# ---------------------------------------------------------------------------
# Codebook + Rotation helpers
# ---------------------------------------------------------------------------

def load_codebook(d: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pre-computed codebook from data/codebooks/."""
    path = os.path.join(CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if not os.path.exists(path):
        # Fall back to ref/ codebooks
        path = os.path.join(PROJECT_ROOT, "ref", "turboquant", "turboquant",
                            "codebooks", f"codebook_d{d}_b{bits}.json")
    with open(path) as f:
        cb = json.load(f)
    centroids = torch.tensor(cb["centroids"], dtype=torch.float32)
    boundaries = torch.tensor(cb["boundaries"], dtype=torch.float32)
    return centroids, boundaries


def generate_rotation_matrix(d: int, seed: int = TQ_SEED) -> torch.Tensor:
    """Deterministic orthogonal rotation matrix via QR decomposition."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.contiguous()  # Must be contiguous for CUDA kernels


# ---------------------------------------------------------------------------
# Pack / Unpack 4-bit
# ---------------------------------------------------------------------------

def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack indices (values 0-15) into 4-bit packed uint8. 2 values per byte.
    indices: (..., d) int tensor -> (..., d//2) uint8 tensor
    """
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    padded_d = ((d + 1) // 2) * 2
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)
    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, 2)
    packed = reshaped[..., 0] | (reshaped[..., 1] << 4)
    return packed


def unpack_4bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack 4-bit packed uint8 to indices.
    packed: (..., packed_len) uint8 -> (..., d) int64 tensor
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :d].long()


# ---------------------------------------------------------------------------
# Paged cache helpers
# ---------------------------------------------------------------------------

def compute_tq_row_bytes(d_c: int, d_rope: int) -> int:
    packed_nope = (d_c * TQ_BITS + 7) // 8
    return packed_nope + TQ_NORM_BYTES + d_rope * 2


def allocate_tq_cache(num_pages: int, page_size: int, d_c: int = D_C,
                       d_rope: int = D_ROPE) -> torch.Tensor:
    """Allocate paged TQ cache as raw uint8 tensor."""
    row_bytes = compute_tq_row_bytes(d_c, d_rope)
    total_bytes = num_pages * page_size * row_bytes
    return torch.zeros(total_bytes, dtype=torch.uint8)


def write_tq_cache_row(cache: torch.Tensor, slot: int, packed_nope: torch.Tensor,
                         norm_fp16: float, rope_bf16: torch.Tensor,
                         d_c: int = D_C, d_rope: int = D_ROPE, page_size: int = 64):
    """Write one TQ token row to paged cache."""
    packed_nope_bytes = (d_c * TQ_BITS + 7) // 8
    row_bytes = compute_tq_row_bytes(d_c, d_rope)

    page_idx = slot // page_size
    row_in_page = slot % page_size
    offset = (page_idx * page_size + row_in_page) * row_bytes

    # Write packed nope
    cache[offset:offset + packed_nope_bytes] = packed_nope

    # Write FP16 norm (2 bytes)
    norm_offset = offset + packed_nope_bytes
    norm_bytes = struct.pack('<e', norm_fp16)  # little-endian float16
    cache[norm_offset] = norm_bytes[0]
    cache[norm_offset + 1] = norm_bytes[1]

    # Write BF16 rope
    rope_offset = offset + packed_nope_bytes + TQ_NORM_BYTES
    rope_bytes_tensor = rope_bf16.to(torch.bfloat16).view(torch.uint8)
    cache[rope_offset:rope_offset + d_rope * 2] = rope_bytes_tensor


def read_tq_cache_row(cache: torch.Tensor, slot: int, d_c: int = D_C,
                        d_rope: int = D_ROPE, page_size: int = 64):
    """Read one TQ token row from paged cache.
    Returns (packed_nope, norm_fp16, rope_bf16).
    """
    packed_nope_bytes = (d_c * TQ_BITS + 7) // 8
    row_bytes = compute_tq_row_bytes(d_c, d_rope)

    page_idx = slot // page_size
    row_in_page = slot % page_size
    offset = (page_idx * page_size + row_in_page) * row_bytes

    packed_nope = cache[offset:offset + packed_nope_bytes].clone()

    norm_offset = offset + packed_nope_bytes
    norm_bytes = bytes([cache[norm_offset].item(), cache[norm_offset + 1].item()])
    norm_fp16 = struct.unpack('<e', norm_bytes)[0]

    rope_offset = offset + packed_nope_bytes + TQ_NORM_BYTES
    rope_raw = cache[rope_offset:rope_offset + d_rope * 2].clone()
    rope_bf16 = rope_raw.view(torch.bfloat16).clone()

    return packed_nope, norm_fp16, rope_bf16


# ---------------------------------------------------------------------------
# Reference functions (kernel-matching interfaces)
# ---------------------------------------------------------------------------

def ref_tq_mse_k_append(
    c_kv: torch.Tensor,       # [num_tokens, d_c] float32
    k_rope: torch.Tensor,     # [num_tokens, d_rope] float32/bf16
    kv_cache: torch.Tensor,   # [total_bytes] uint8 paged cache
    slot_mapping: torch.Tensor,  # [num_tokens] int
    Pi: torch.Tensor,         # [d_c, d_c] float32 rotation matrix
    centroids: torch.Tensor,  # [16] float32
    boundaries: torch.Tensor, # [17] float32 (includes -1 and 1)
    d_c: int = D_C,
    d_rope: int = D_ROPE,
    page_size: int = 64,
):
    """Reference: TQ fused_k_append kernel.

    Per token: normalize → rotate → quantize → pack → write to cache.
    """
    num_tokens = c_kv.shape[0]
    decision_boundaries = boundaries[1:-1]  # [15] interior boundaries

    for t in range(num_tokens):
        x = c_kv[t].float()  # [d_c]

        # 1. Compute L2 norm
        norm = x.norm().item()

        # 2. Normalize to unit sphere
        x_unit = x / (norm + 1e-10)

        # 3. Rotate: y = x_unit @ Pi^T
        y = x_unit @ Pi.T

        # 4. Quantize via searchsorted
        indices = torch.searchsorted(decision_boundaries, y.contiguous())
        indices = indices.clamp(0, TQ_NUM_CENTROIDS - 1)

        # 5. Pack 4-bit
        packed = pack_4bit(indices)

        # 6. Write to cache
        slot = slot_mapping[t].item()
        write_tq_cache_row(kv_cache, slot, packed, norm,
                            k_rope[t], d_c=d_c, d_rope=d_rope, page_size=page_size)


def ref_tq_mse_dequant_indexed(
    kv_cache: torch.Tensor,    # [total_bytes] uint8
    indices: torch.Tensor,     # [num_fetch] int
    Pi: torch.Tensor,          # [d_c, d_c] float32
    centroids: torch.Tensor,   # [16] float32
    d_c: int = D_C,
    d_rope: int = D_ROPE,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: TQ dequant_ckv_indexed kernel.

    Returns (c_kv_deq, k_rope) as float32 tensors.
    """
    num_fetch = indices.shape[0]
    c_kv_out = torch.zeros(num_fetch, d_c, dtype=torch.float32)
    k_rope_out = torch.zeros(num_fetch, d_rope, dtype=torch.float32)

    for i in range(num_fetch):
        slot = indices[i].item()
        packed_nope, norm_fp16, rope_bf16 = read_tq_cache_row(
            kv_cache, slot, d_c=d_c, d_rope=d_rope, page_size=page_size)

        # Unpack
        idx = unpack_4bit(packed_nope, d_c)

        # Codebook lookup
        y_hat = centroids[idx]

        # Inverse rotation: c_hat = y_hat @ Pi
        c_hat = y_hat @ Pi

        # Scale by norm
        c_kv_out[i] = c_hat * norm_fp16
        k_rope_out[i] = rope_bf16.float()

    return c_kv_out, k_rope_out


def ref_tq_mse_q_rotate(
    q_nope: torch.Tensor,  # [s_q, h_q, d_c] float32/bf16
    Pi: torch.Tensor,      # [d_c, d_c] float32
) -> torch.Tensor:
    """Reference: TQ q_rotate kernel. Returns FP32 q_rot = q_nope @ Pi^T."""
    return torch.matmul(q_nope.float(), Pi.T)


def ref_tq_mse_v_rotate_back(
    out_rotated: torch.Tensor,  # [s_q, h_q, d_c] float32
    Pi: torch.Tensor,           # [d_c, d_c] float32
) -> torch.Tensor:
    """Reference: TQ v_rotate_back epilogue. Returns out = out_rotated @ Pi."""
    return torch.matmul(out_rotated.float(), Pi)


def ref_tq_mse_decode(
    q_rot: torch.Tensor,       # [s_q, h_q, d_c] float32 (pre-rotated)
    q_rope: torch.Tensor,      # [s_q, h_q, d_rope] float32
    kv_cache: torch.Tensor,    # [total_bytes] uint8
    seqlens_k: torch.Tensor,   # [batch] int (here batch=1)
    Pi: torch.Tensor,          # [d_c, d_c] float32
    centroids: torch.Tensor,   # [16] float32
    sm_scale: float,
    d_c: int = D_C,
    d_rope: int = D_ROPE,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: Full TQ MLA decode with rotated-space PV accumulation.

    Score computation:
      score_nope[k] = sum_j(q_rot[j] * centroids[idx_k[j]]) * norm[k]
      score_rope[k] = q_rope @ k_rope[k]^T
      score[k] = (score_nope[k] + score_rope[k]) * sm_scale

    PV accumulation (in rotated space):
      acc_rot[j] = sum_k P[k] * norm[k] * centroids[idx_k[j]]

    Output is in rotated space — needs ref_tq_mse_v_rotate_back() after.

    Returns (out_rotated, lse).
    """
    s_q, h_q, d_c_actual = q_rot.shape
    s_kv = seqlens_k[0].item()

    # Read all KV tokens from cache
    all_packed = []
    all_norms = []
    all_rope = []
    for k in range(s_kv):
        packed_nope, norm_fp16, rope_bf16 = read_tq_cache_row(
            kv_cache, k, d_c=d_c, d_rope=d_rope, page_size=page_size)
        idx = unpack_4bit(packed_nope, d_c)
        all_packed.append(idx)
        all_norms.append(norm_fp16)
        all_rope.append(rope_bf16.float())

    packed_indices = torch.stack(all_packed)   # [s_kv, d_c] int64
    norms = torch.tensor(all_norms)            # [s_kv]
    k_rope_all = torch.stack(all_rope)         # [s_kv, d_rope]

    # Centroid values for each KV token coordinate: [s_kv, d_c]
    centroid_vals = centroids[packed_indices]

    # NOPE scores: q_rot @ centroid_vals^T * norms
    # q_rot: [s_q, h_q, d_c], centroid_vals: [s_kv, d_c]
    scores_nope = torch.einsum('qhd,kd->hqk', q_rot.float(), centroid_vals.float())
    scores_nope = scores_nope * norms.unsqueeze(0).unsqueeze(0)  # [h_q, s_q, s_kv]

    # ROPE scores
    k_rope_exp = k_rope_all.unsqueeze(1).expand(-1, h_q, -1)  # [s_kv, h_q, d_rope]
    scores_rope = torch.einsum('qhd,khd->hqk', q_rope.float(), k_rope_exp.float())

    scores = (scores_nope + scores_rope) * sm_scale

    # Softmax
    lse = torch.logsumexp(scores, dim=-1)  # [h_q, s_q]
    P = torch.softmax(scores, dim=-1)      # [h_q, s_q, s_kv]

    # PV accumulation in ROTATED space
    # acc_rot[j] = sum_k P[k] * norm[k] * centroids[idx_k[j]]
    # = P @ diag(norms) @ centroid_vals
    weighted = P * norms.unsqueeze(0).unsqueeze(0)  # [h_q, s_q, s_kv]
    out_rot = torch.einsum('hqk,kd->qhd', weighted, centroid_vals.float())

    return out_rot, lse.permute(1, 0)  # [s_q, h_q, d_c], [s_q, h_q]


def ref_tq_mse_sparse_decode(
    q_rot: torch.Tensor,       # [s_q, h_q, d_c] float32 (pre-rotated)
    q_rope: torch.Tensor,      # [s_q, h_q, d_rope] float32
    kv_cache: torch.Tensor,    # [total_bytes] uint8
    indices: torch.Tensor,     # [topk] int — which tokens to attend to (-1 = invalid)
    Pi: torch.Tensor,          # [d_c, d_c] float32
    centroids: torch.Tensor,   # [16] float32
    sm_scale: float,
    d_c: int = D_C,
    d_rope: int = D_ROPE,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: Sparse TQ MLA decode — same as dense but attends to indices subset.

    Returns (out_rotated, lse) in rotated space.
    """
    # Filter valid indices
    valid_mask = indices >= 0
    valid_indices = indices[valid_mask]
    s_kv = valid_indices.shape[0]

    s_q, h_q, d_c_actual = q_rot.shape

    # Read selected KV tokens from cache
    all_packed = []
    all_norms = []
    all_rope = []
    for k in range(s_kv):
        slot = valid_indices[k].item()
        packed_nope, norm_fp16, rope_bf16 = read_tq_cache_row(
            kv_cache, slot, d_c=d_c, d_rope=d_rope, page_size=page_size)
        idx = unpack_4bit(packed_nope, d_c)
        all_packed.append(idx)
        all_norms.append(norm_fp16)
        all_rope.append(rope_bf16.float())

    packed_indices = torch.stack(all_packed)   # [s_kv, d_c] int64
    norms = torch.tensor(all_norms)            # [s_kv]
    k_rope_all = torch.stack(all_rope)         # [s_kv, d_rope]

    centroid_vals = centroids[packed_indices]

    # NOPE scores
    scores_nope = torch.einsum('qhd,kd->hqk', q_rot.float(), centroid_vals.float())
    scores_nope = scores_nope * norms.unsqueeze(0).unsqueeze(0)

    # ROPE scores
    k_rope_exp = k_rope_all.unsqueeze(1).expand(-1, h_q, -1)
    scores_rope = torch.einsum('qhd,khd->hqk', q_rope.float(), k_rope_exp.float())

    scores = (scores_nope + scores_rope) * sm_scale

    # Softmax
    lse = torch.logsumexp(scores, dim=-1)
    P = torch.softmax(scores, dim=-1)

    # PV in rotated space
    weighted = P * norms.unsqueeze(0).unsqueeze(0)
    out_rot = torch.einsum('hqk,kd->qhd', weighted, centroid_vals.float())

    return out_rot, lse.permute(1, 0)


def ref_tq_mse_full_pipeline(
    q: torch.Tensor,           # [s_q, h_q, d_qk] float32
    c_kv: torch.Tensor,        # [s_kv, d_c] float32
    k_rope: torch.Tensor,      # [s_kv, d_rope] float32
    sm_scale: float,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    d_c: int = D_C,
    d_rope: int = D_ROPE,
    page_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full TQ pipeline: k_append → q_rotate → decode → v_rotate_back.

    Returns (output, lse) comparable to ref_mla_attention_bf16().
    """
    s_kv = c_kv.shape[0]
    num_pages = (s_kv + page_size - 1) // page_size
    kv_cache = allocate_tq_cache(num_pages, page_size, d_c, d_rope)
    slot_mapping = torch.arange(s_kv, dtype=torch.int64)

    # K append
    ref_tq_mse_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                          Pi, centroids, boundaries, d_c, d_rope, page_size)

    # Q rotate
    q_nope = q[..., :d_c]
    q_rope_q = q[..., d_c:]
    q_rot = ref_tq_mse_q_rotate(q_nope, Pi)

    # Decode
    seqlens_k = torch.tensor([s_kv], dtype=torch.int64)
    out_rot, lse = ref_tq_mse_decode(q_rot, q_rope_q, kv_cache, seqlens_k,
                                      Pi, centroids, sm_scale, d_c, d_rope, page_size)

    # V rotate back
    out = ref_tq_mse_v_rotate_back(out_rot, Pi)

    return out, lse


# ---------------------------------------------------------------------------
# BF16 ground truth (from test_snapmla_reference.py)
# ---------------------------------------------------------------------------

def ref_mla_attention_bf16(
    q: torch.Tensor, c_kv: torch.Tensor, k_rope: torch.Tensor, sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth: BF16 MLA absorbed attention."""
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]

    k_full = torch.cat([c_kv, k_rope], dim=-1)
    k_exp = k_full.unsqueeze(1).expand(-1, h_q, -1)
    v_exp = c_kv.unsqueeze(1).expand(-1, h_q, -1)

    scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * sm_scale
    lse = torch.logsumexp(scores, dim=-1)
    P = torch.softmax(scores, dim=-1)
    out = torch.einsum('hqk,khd->qhd', P, v_exp.float())

    return out, lse.permute(1, 0)


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

def test_k_append_dequant_roundtrip(verbose=False):
    """TQ k_append → dequant round-trip: quantize then read back."""
    print("\n=== test_k_append_dequant_roundtrip ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)

    torch.manual_seed(42)
    n_tokens = 128
    c_kv = torch.randn(n_tokens, D_C)
    k_rope = torch.randn(n_tokens, D_ROPE)

    page_size = 64
    num_pages = (n_tokens + page_size - 1) // page_size
    kv_cache = allocate_tq_cache(num_pages, page_size)
    slot_mapping = torch.arange(n_tokens, dtype=torch.int64)

    ref_tq_mse_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                          Pi, centroids, boundaries, page_size=page_size)

    indices = torch.arange(n_tokens, dtype=torch.int64)
    c_kv_deq, k_rope_deq = ref_tq_mse_dequant_indexed(
        kv_cache, indices, Pi, centroids, page_size=page_size)

    # NOPE round-trip (global cosine across all 128 vectors)
    m = compute_metrics(c_kv, c_kv_deq)
    # Per-vector cosine for better diagnostics
    per_vec_cos = F.cosine_similarity(c_kv, c_kv_deq, dim=-1)
    print(f"  NOPE round-trip: {fmt_metrics(m)}")
    print(f"    per-vector cosine: mean={per_vec_cos.mean():.4f} min={per_vec_cos.min():.4f}")
    # 4-bit at d=512 with FP16 norm storage: per-vec cosine ~0.84
    # (quantization is the dominant error, FP16 norm adds small additional loss)
    assert per_vec_cos.mean() > 0.80, f"NOPE per-vec mean cosine {per_vec_cos.mean():.4f} < 0.80"

    # ROPE round-trip (should be high precision — only FP16 truncation via norm storage)
    rope_cos = F.cosine_similarity(
        k_rope.flatten().unsqueeze(0), k_rope_deq.flatten().unsqueeze(0)).item()
    print(f"  ROPE round-trip cosine: {rope_cos:.6f}")
    assert rope_cos > 0.999, f"ROPE cosine {rope_cos:.4f} < 0.999"

    print("  PASS")


def test_q_rotate(verbose=False):
    """TQ q_rotate: verify rotation matches matmul."""
    print("\n=== test_q_rotate ===")

    Pi = generate_rotation_matrix(D_C)
    torch.manual_seed(123)
    q_nope = torch.randn(1, H_Q, D_C)

    q_rot = ref_tq_mse_q_rotate(q_nope, Pi)
    q_rot_ref = q_nope.float() @ Pi.T

    cos = F.cosine_similarity(
        q_rot.flatten().unsqueeze(0), q_rot_ref.flatten().unsqueeze(0)).item()
    print(f"  q_rotate cosine: {cos:.10f}")
    assert cos > 0.9999, f"q_rotate cosine {cos:.6f} < 0.9999"
    print("  PASS")


def test_decode_short(verbose=False):
    """TQ full pipeline at s_kv=256."""
    print("\n=== test_decode_short (s_kv=256) ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    torch.manual_seed(42)
    s_kv = 256
    q = torch.randn(1, H_Q, D_QK)
    c_kv = torch.randn(s_kv, D_C)
    k_rope = torch.randn(s_kv, D_ROPE)

    # BF16 ground truth
    out_bf16, lse_bf16 = ref_mla_attention_bf16(q, c_kv, k_rope, sm_scale)

    # TQ full pipeline
    out_tq, lse_tq = ref_tq_mse_full_pipeline(
        q, c_kv, k_rope, sm_scale, Pi, centroids, boundaries)

    m = compute_metrics(out_bf16, out_tq)
    print(f"  TQ vs BF16: {fmt_metrics(m)}")
    # At s_kv=256 (short context), TQ 4-bit has noticeable error. Quality improves
    # rapidly with context length (softmax averaging effect).
    assert m["cosine"] > 0.70, f"cosine {m['cosine']:.4f} < 0.70"

    lse_diff = (lse_bf16 - lse_tq).abs().max().item()
    print(f"  LSE max diff: {lse_diff:.4e}")

    print("  PASS")


def test_decode_scaling(verbose=False):
    """TQ decode at increasing context lengths: 256, 1K, 4K."""
    print("\n=== test_decode_scaling ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    for s_kv in [256, 1024, 4096]:
        torch.manual_seed(42)
        q = torch.randn(1, H_Q, D_QK)
        c_kv = torch.randn(s_kv, D_C)
        k_rope = torch.randn(s_kv, D_ROPE)

        out_bf16, _ = ref_mla_attention_bf16(q, c_kv, k_rope, sm_scale)
        out_tq, _ = ref_tq_mse_full_pipeline(
            q, c_kv, k_rope, sm_scale, Pi, centroids, boundaries)

        m = compute_metrics(out_bf16, out_tq)
        print(f"  s_kv={s_kv:5d}: {fmt_metrics(m)}")

        # Quality should improve with context (softmax averaging)
        if s_kv >= 1024:
            assert m["cosine"] > 0.90, f"s_kv={s_kv} cosine {m['cosine']:.4f} < 0.90"

    print("  PASS")


def test_v_rotate_back_identity(verbose=False):
    """Verify rotate_forward then rotate_back is identity."""
    print("\n=== test_v_rotate_back_identity ===")

    Pi = generate_rotation_matrix(D_C)
    torch.manual_seed(99)
    x = torch.randn(1, H_Q, D_C)

    # Forward: x @ Pi^T, Back: result @ Pi
    x_rot = x.float() @ Pi.T
    x_back = ref_tq_mse_v_rotate_back(x_rot, Pi)

    cos = F.cosine_similarity(
        x.flatten().unsqueeze(0), x_back.flatten().unsqueeze(0)).item()
    max_err = (x.float() - x_back).abs().max().item()
    print(f"  Round-trip cosine: {cos:.10f}, max_err: {max_err:.4e}")
    assert cos > 0.99999, f"Rotation round-trip cosine {cos:.6f} < 0.99999"
    print("  PASS")


def test_rotated_space_pv_equivalence(verbose=False):
    """Verify rotated-space PV matches standard dequant-then-matmul PV."""
    print("\n=== test_rotated_space_pv_equivalence ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    torch.manual_seed(42)
    s_kv = 256
    q = torch.randn(1, H_Q, D_QK)
    c_kv = torch.randn(s_kv, D_C)
    k_rope = torch.randn(s_kv, D_ROPE)

    # Method 1: Rotated-space PV (our approach) → then v_rotate_back
    out_tq, lse_tq = ref_tq_mse_full_pipeline(
        q, c_kv, k_rope, sm_scale, Pi, centroids, boundaries)

    # Method 2: Standard approach — dequant all tokens, then matmul
    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    kv_cache = allocate_tq_cache(num_pages, page_size)
    slot_mapping = torch.arange(s_kv, dtype=torch.int64)
    ref_tq_mse_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                          Pi, centroids, boundaries, page_size=page_size)

    # Dequant all
    all_idx = torch.arange(s_kv, dtype=torch.int64)
    c_kv_deq, k_rope_deq = ref_tq_mse_dequant_indexed(
        kv_cache, all_idx, Pi, centroids, page_size=page_size)

    # Standard attention with dequanted values
    k_full = torch.cat([c_kv_deq, k_rope_deq], dim=-1)
    k_exp = k_full.unsqueeze(1).expand(-1, H_Q, -1)
    v_exp = c_kv_deq.unsqueeze(1).expand(-1, H_Q, -1)

    scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * sm_scale
    P = torch.softmax(scores, dim=-1)
    out_standard = torch.einsum('hqk,khd->qhd', P, v_exp.float())

    m = compute_metrics(out_standard, out_tq)
    print(f"  Rotated-space PV vs standard dequant PV: {fmt_metrics(m)}")

    # These should be very close (mathematically equivalent, modulo float precision)
    assert m["cosine"] > 0.999, f"PV equivalence cosine {m['cosine']:.6f} < 0.999"
    print("  PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def test_sparse_decode_equals_dense(verbose=False):
    """Sparse decode with all indices == dense decode."""
    print("\n=== test_sparse_decode_equals_dense ===")

    Pi = generate_rotation_matrix(D_C)
    centroids, boundaries = load_codebook(D_C, TQ_BITS)
    sm_scale = 1.0 / math.sqrt(D_QK)

    torch.manual_seed(42)
    s_kv = 256
    q = torch.randn(1, H_Q, D_QK)
    c_kv = torch.randn(s_kv, D_C)
    k_rope = torch.randn(s_kv, D_ROPE)

    # Build paged cache
    page_size = 64
    num_pages = (s_kv + page_size - 1) // page_size
    kv_cache = allocate_tq_cache(num_pages, page_size)
    slot_mapping = torch.arange(s_kv, dtype=torch.int64)
    ref_tq_mse_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                          Pi, centroids, boundaries, page_size=page_size)

    q_nope = q[..., :D_C]
    q_rope_q = q[..., D_C:]
    q_rot = ref_tq_mse_q_rotate(q_nope, Pi)

    # Dense decode
    seqlens_k = torch.tensor([s_kv], dtype=torch.int64)
    out_dense, lse_dense = ref_tq_mse_decode(
        q_rot, q_rope_q, kv_cache, seqlens_k,
        Pi, centroids, sm_scale, page_size=page_size)

    # Sparse decode with all indices
    all_idx = torch.arange(s_kv, dtype=torch.int64)
    out_sparse, lse_sparse = ref_tq_mse_sparse_decode(
        q_rot, q_rope_q, kv_cache, all_idx,
        Pi, centroids, sm_scale, page_size=page_size)

    m = compute_metrics(out_dense, out_sparse)
    print(f"  Sparse(all) vs Dense: {fmt_metrics(m)}")
    assert m["cosine"] > 0.9999, f"sparse==dense cosine {m['cosine']:.6f} < 0.9999"

    print("  PASS")


ALL_TESTS = [
    test_k_append_dequant_roundtrip,
    test_q_rotate,
    test_v_rotate_back_identity,
    test_rotated_space_pv_equivalence,
    test_decode_short,
    test_decode_scaling,
    test_sparse_decode_equals_dense,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("TurboQuant MLA Reference Tests — Paged Cache Format")
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
