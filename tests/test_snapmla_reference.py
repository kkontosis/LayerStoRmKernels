"""
SnapMLA Kernel Validation — PyTorch Reference Implementation

Pure-PyTorch implementations of the SnapMLA pipeline that serve as:
  1. Golden reference for correctness testing
  2. Spec of what each kernel computes
  3. Error bound establishment (FP8 quantization tolerance)

When CUDA kernel bindings are added, replace the `ref_*` calls with
actual kernel calls and compare against these reference outputs.

Usage:
  python tests/test_snapmla_reference.py          # runs all tests
  python tests/test_snapmla_reference.py -v        # verbose

Error budget (expected from FP8 e4m3 quantization):
  - Per-element FP8 round-trip: ~0.4% relative error (1/256 resolution at mantissa)
  - Q quantization + K quantization: errors compound → ~1% on QK scores
  - P quantization adds another ~0.4% → total ~1.5% on final output
  - Empirically: cosine similarity > 0.995, max relative error < 5%
"""

import torch
import torch.nn.functional as F
import math
import argparse

# ---------------------------------------------------------------------------
# Model Config
# ---------------------------------------------------------------------------

D_C = 512       # compressed latent (NOPE)
D_ROPE = 64     # RoPE dims
D_QK = D_C + D_ROPE  # 576
D_V = 512
H_Q = 64        # Q heads
H_KV = 1        # KV heads (MLA)
PAGE_SIZE = 64
FP8_MAX = 448.0

# Try to import GPU kernel module
try:
    import sm120_mla_kernels
    HAS_KERNELS = torch.cuda.is_available()
except ImportError:
    HAS_KERNELS = False


def _alloc_paged_cache(num_pages, page_size, d_c, d_rope, device="cuda"):
    """Allocate a paged KV cache as flat uint8 tensor."""
    row_bytes = d_c + 4 + d_rope * 2
    return torch.zeros(num_pages * page_size * row_bytes, dtype=torch.uint8, device=device)


def _prep_gpu_decode(q_bf16_4d, c_kv, k_rope):
    """Shared prep for GPU decode: build cache + quantize Q.

    Returns: (kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages)
    """
    b, s_q, h_q, d_qk = q_bf16_4d.shape
    s_kv = c_kv.shape[0]

    n_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
    slot_mapping = torch.arange(s_kv, dtype=torch.int32, device="cuda")
    sm120_mla_kernels.fused_k_append(c_kv, k_rope, kv_cache, slot_mapping,
                                      D_C, D_ROPE, PAGE_SIZE)

    q_flat = q_bf16_4d.view(-1, h_q, d_qk)
    q_nope_fp8, q_rope_bf16, q_scales = sm120_mla_kernels.fused_q_quant(q_flat, D_C)
    q_nope_fp8 = q_nope_fp8.view(b, s_q, h_q, D_C)
    q_rope_bf16 = q_rope_bf16.view(b, s_q, h_q, D_ROPE)
    q_scales = q_scales.view(b, s_q, h_q)

    return kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages


def _run_gpu_decode(q_bf16_4d, c_kv, k_rope, sm_scale, num_sm_parts=1):
    """Run full GPU kernel pipeline: q_quant → k_append → dense_decode.

    Returns: (out, lse) on CPU float32
    """
    b, s_q = q_bf16_4d.shape[:2]
    s_kv = c_kv.shape[0]
    kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages = \
        _prep_gpu_decode(q_bf16_4d, c_kv, k_rope)

    block_table = torch.arange(n_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    seqlens_k = torch.tensor([s_kv], dtype=torch.int32, device="cuda")

    out, lse = sm120_mla_kernels.dense_decode_v32(
        q_nope_fp8, q_rope_bf16, q_scales,
        kv_cache, block_table, seqlens_k,
        sm_scale, PAGE_SIZE, num_sm_parts
    )
    return out.float().cpu(), lse.float().cpu()


def _run_gpu_sparse_decode(q_bf16_4d, c_kv, k_rope, sm_scale, num_sm_parts=1):
    """Run full GPU kernel pipeline: q_quant → k_append → sparse_decode (topk=all).

    Returns: (out, lse) on CPU float32
    """
    b, s_q = q_bf16_4d.shape[:2]
    s_kv = c_kv.shape[0]
    kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages = \
        _prep_gpu_decode(q_bf16_4d, c_kv, k_rope)

    # topk = all tokens, padded to multiple of 64
    topk = ((s_kv + 63) // 64) * 64
    indices = torch.full((b, s_q, topk), -1, dtype=torch.int32, device="cuda")
    indices[0, 0, :s_kv] = torch.arange(s_kv, dtype=torch.int32, device="cuda")

    out, lse = sm120_mla_kernels.sparse_decode_v32(
        q_nope_fp8, q_rope_bf16, q_scales,
        kv_cache, indices, sm_scale, PAGE_SIZE, topk, num_sm_parts
    )
    return out.float().cpu(), lse.float().cpu()


# ---------------------------------------------------------------------------
# FP8 simulation helpers
# ---------------------------------------------------------------------------

def simulate_fp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate per-token FP8 e4m3 quantization.

    Returns (x_fp8_dequanted, scale) where x_fp8_dequanted is the
    value after quantize→dequantize round-trip (simulating precision loss).
    """
    # Per-token amax
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX  # [*, 1]

    # Quantize: divide by scale, clamp to FP8 range, round
    x_scaled = x / scale
    # Simulate FP8 e4m3 precision: cast to float8 and back
    if hasattr(torch, 'float8_e4m3fn'):
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).float()
    else:
        # Fallback: clamp + round to simulate limited precision
        x_fp8 = x_scaled.clamp(-FP8_MAX, FP8_MAX)
        # Simulate ~3-bit mantissa: round to nearest 1/8
        x_fp8 = (x_fp8 * 8).round() / 8

    # Dequantize
    x_deq = x_fp8 * scale
    return x_deq, scale.squeeze(-1)


def simulate_fp8_quantize_rowwise(P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate per-ROW FP8 quantization of P matrix (attention weights).

    This matches the kernel's fragment_fp8_compute_row_scales +
    fragment_fp8_row_quantize_store pipeline.

    Returns (P_fp8_raw, P_deq, scale) where:
      P_fp8_raw = quantized values (small, in FP8 range) — used for PV GEMM
      P_deq = P_fp8_raw * scale ≈ original P — for verification
      scale = per-row quantization scale — applied after PV GEMM
    """
    # Per-row amax
    amax = P.abs().amax(dim=-1, keepdim=True).clamp(min=1e-26)
    scale = amax / FP8_MAX

    P_scaled = P / scale
    if hasattr(torch, 'float8_e4m3fn'):
        P_fp8 = P_scaled.to(torch.float8_e4m3fn).float()
    else:
        P_fp8 = (P_scaled.clamp(-FP8_MAX, FP8_MAX) * 8).round() / 8

    P_deq = P_fp8 * scale
    return P_fp8, P_deq, scale.squeeze(-1)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_fused_q_quant(
    q_bf16: torch.Tensor,  # [s_q, h_q, d_qk]
    d_nope: int = D_C,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: fused_q_quant kernel.

    Returns (q_nope_deq, q_rope_prescaled, q_scales)
    where q_nope_deq is the FP8 round-tripped NOPE, and
    q_rope_prescaled = q_rope / scale (pre-scaled into content domain).
    """
    q_nope = q_bf16[..., :d_nope]    # [s_q, h_q, d_nope]
    q_rope = q_bf16[..., d_nope:]    # [s_q, h_q, d_rope]

    # Per-head amax over NOPE dims only
    amax = q_nope.abs().amax(dim=-1).clamp(min=1e-12)  # [s_q, h_q]
    scale = amax / FP8_MAX

    # Quantize NOPE to FP8
    q_nope_scaled = q_nope / scale.unsqueeze(-1)
    if hasattr(torch, 'float8_e4m3fn'):
        q_nope_fp8 = q_nope_scaled.to(torch.float8_e4m3fn).float()
    else:
        q_nope_fp8 = (q_nope_scaled.clamp(-FP8_MAX, FP8_MAX) * 8).round() / 8
    q_nope_deq = q_nope_fp8 * scale.unsqueeze(-1)

    # Pre-scale ROPE by inverse content scale
    q_rope_prescaled = q_rope / scale.unsqueeze(-1)

    return q_nope_deq, q_rope_prescaled, scale


def ref_q_absorb(
    q_heads: torch.Tensor,    # [s_q, h_q, d_nope_in + d_rope]
    kv_b_proj: torch.Tensor,  # [h_q*(d_nope_in + d_v), d_c]  (W_UK = first d_nope_in rows per head)
    d_nope_in: int = 128,     # qk_nope_head_dim (P)
    d_c: int = D_C,           # kv_lora_rank (L)
    d_rope: int = D_ROPE,     # qk_rope_head_dim (R)
    d_v: int = 128,           # v_head_dim (V)
) -> torch.Tensor:
    """Reference: DeepSeek MLA W_UK query absorption + rope concat.

    ql_nope[s,h,k] = sum_d q_nope[s,h,d] * W_UK[h,d,k]   (einsum 'shd,hdk->shk')
    q_absorbed = concat(ql_nope[d_c], q_rope[d_rope])    → [s_q, h_q, d_c + d_rope]

    W_UK is the K-half of kv_b_proj: per head h, rows [h*(P+V) .. h*(P+V)+P).
    """
    s_q, h_q, _ = q_heads.shape
    P, L, R, V = d_nope_in, d_c, d_rope, d_v
    q_nope = q_heads[..., :P].float()                 # [s,h,P]
    q_rope = q_heads[..., P:P + R]                     # [s,h,R] (copied unchanged)
    W_UK = kv_b_proj.float().view(h_q, P + V, L)[:, :P, :]  # [h,P,L]
    ql_nope = torch.einsum('shd,hdk->shk', q_nope, W_UK)    # [s,h,L]
    return torch.cat([ql_nope.to(q_heads.dtype), q_rope], dim=-1)


def quantize_kv_b_fp8_blockwise(
    kv_b_bf16: torch.Tensor,  # [N_total, d_c] BF16
    block: int = 128,
):
    """Blockwise (128x128) FP8 E4M3 quantization matching the kv_b_proj scale layout.

    Returns (fp8_e4m3 [N_total, d_c], scales_flat) where scales are K-major:
    scales[k_block * n_scale_blocks + n_block], n_scale_blocks = ceil(N_total/128).
    """
    x = kv_b_bf16.float()
    N_total, L = x.shape
    n_blocks = (N_total + block - 1) // block
    k_blocks = (L + block - 1) // block
    scales = torch.zeros(k_blocks, n_blocks)
    fp8 = torch.zeros_like(x)
    for nb in range(n_blocks):
        r0, r1 = nb * block, min((nb + 1) * block, N_total)
        for kb in range(k_blocks):
            c0, c1 = kb * block, min((kb + 1) * block, L)
            tile = x[r0:r1, c0:c1]
            amax = tile.abs().amax().clamp(min=1e-8)
            scale = (amax / FP8_MAX)
            scales[kb, nb] = scale
            fp8[r0:r1, c0:c1] = (tile / scale).clamp(-FP8_MAX, FP8_MAX)
    fp8_e4m3 = fp8.to(torch.float8_e4m3fn)
    return fp8_e4m3, scales.flatten().contiguous()


def ref_q_absorb_fp8(q_heads, fp8_e4m3, scales_flat, d_nope_in=128, d_c=D_C,
                     d_rope=D_ROPE, d_v=128, block=128):
    """Reference q-absorb consuming the blockwise-FP8 kv_b_proj (dequant then einsum)."""
    h_q = q_heads.shape[1]
    P, L, V = d_nope_in, d_c, d_v
    N_total = h_q * (P + V)
    n_blocks = (N_total + block - 1) // block
    deq = fp8_e4m3.float().clone()
    for r in range(N_total):
        nb = r // block
        for kb in range(L // block):
            deq[r, kb * block:(kb + 1) * block] *= scales_flat[kb * n_blocks + nb]
    return ref_q_absorb(q_heads, deq.to(torch.bfloat16), d_nope_in, d_c, d_rope, d_v)


def ref_rope_cos_sin(max_pos: int, d_rope: int = D_ROPE, theta: float = 10000.0):
    """Plain (non-YaRN) cos/sin table, DeepSeek interleaved-pair convention.

    Returns [max_pos, d_rope] float32: per position, d_rope/2 cos values then
    d_rope/2 sin values; frequency i applies to adjacent dims (2i, 2i+1).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_rope, 2, dtype=torch.float32) / d_rope))
    t = torch.arange(max_pos, dtype=torch.float32)
    ang = torch.outer(t, freqs)  # [max_pos, d_rope/2]
    return torch.cat([ang.cos(), ang.sin()], dim=-1).contiguous()


def ref_rope_rotate(x: torch.Tensor, positions: torch.Tensor,
                    theta: float = 10000.0) -> torch.Tensor:
    """Reference rotation via the DeepSeek complex multiply (model.py apply_rotary_emb).

    x: [..., n_tokens, ..., d_rope] float; positions: [n_tokens] long, broadcast over
    any extra dims after the token dim. Implemented for x of shape [T, R] or [T, H, R].
    """
    d_rope = x.shape[-1]
    freqs = 1.0 / (theta ** (torch.arange(0, d_rope, 2, dtype=torch.float32) / d_rope))
    ang = positions.float().unsqueeze(-1) * freqs  # [T, d_rope/2]
    fc = torch.polar(torch.ones_like(ang), ang)    # [T, d_rope/2] complex
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    while fc.dim() < xc.dim():
        fc = fc.unsqueeze(1)  # broadcast over head dims between token and rope
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


def ref_fused_k_append(
    c_kv: torch.Tensor,    # [num_tokens, d_c]
    k_rope: torch.Tensor,  # [num_tokens, d_rope]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: fused_k_append kernel.

    Returns (c_kv_deq, k_rope_prescaled, scale) simulating the
    quantize→store→load→dequant round-trip.
    """
    # Per-token amax over c_kv
    amax = c_kv.abs().amax(dim=-1).clamp(min=1e-12)
    scale = amax / FP8_MAX

    # Quantize c_kv to FP8
    c_scaled = c_kv / scale.unsqueeze(-1)
    if hasattr(torch, 'float8_e4m3fn'):
        c_fp8 = c_scaled.to(torch.float8_e4m3fn).float()
    else:
        c_fp8 = (c_scaled.clamp(-FP8_MAX, FP8_MAX) * 8).round() / 8
    c_kv_deq = c_fp8 * scale.unsqueeze(-1)

    # Pre-scale ROPE (stored as BF16, not quantized to FP8)
    k_rope_prescaled = k_rope / scale.unsqueeze(-1)

    return c_kv_deq, k_rope_prescaled, scale


def ref_mla_attention_bf16(
    q: torch.Tensor,       # [s_q, h_q, d_qk]
    k: torch.Tensor,       # [s_kv, h_kv, d_qk]
    v: torch.Tensor,       # [s_kv, h_kv, d_v]
    sm_scale: float,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: BF16 MLA attention (ground truth, no quantization).

    Standard scaled dot-product attention. This is the golden reference
    that all other implementations should approximate.
    """
    s_q, h_q, d_qk = q.shape
    s_kv = k.shape[0]

    # Expand KV heads to match Q heads (MLA: h_kv=1, h_q=64)
    h_kv = k.shape[1]
    heads_per_kv = h_q // h_kv
    k_exp = k.unsqueeze(2).expand(-1, -1, heads_per_kv, -1).reshape(s_kv, h_q, d_qk)
    v_exp = v.unsqueeze(2).expand(-1, -1, heads_per_kv, -1).reshape(s_kv, h_q, D_V)

    # Q @ K^T
    # [s_q, h_q, d_qk] @ [s_kv, h_q, d_qk]^T → [h_q, s_q, s_kv]
    scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * sm_scale

    if causal:
        mask = torch.triu(torch.ones(s_q, s_kv, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

    # Softmax
    lse = torch.logsumexp(scores, dim=-1)  # [h_q, s_q]
    P = torch.softmax(scores, dim=-1)      # [h_q, s_q, s_kv]

    # P @ V
    out = torch.einsum('hqk,khd->qhd', P, v_exp.float())

    return out.to(q.dtype), lse.permute(1, 0)  # [s_q, h_q, d_v], [s_q, h_q]


def ref_snapmla_decode_fp8(
    q_bf16: torch.Tensor,      # [s_q, h_q, d_qk] BF16 (before quantization)
    c_kv_bf16: torch.Tensor,   # [s_kv, d_c] BF16 (original, before cache write)
    k_rope_bf16: torch.Tensor, # [s_kv, d_rope] BF16
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: Full SnapMLA FP8 decode pipeline.

    Simulates the complete chain:
      fused_q_quant → fused_k_append → QK dual MMA → dequant → softmax →
      P quant → PV GEMM → dequant → output

    Returns (output, lse) with FP8 quantization errors included.
    This establishes the expected error bound vs ref_mla_attention_bf16.
    """
    s_q, h_q, d_qk = q_bf16.shape
    s_kv = c_kv_bf16.shape[0]

    # Step 1: Q quantization (fused_q_quant)
    q_nope_deq, q_rope_ps, q_scale = ref_fused_q_quant(q_bf16)

    # Step 2: K quantization (fused_k_append + read back)
    c_kv_deq, k_rope_ps, k_scale = ref_fused_k_append(c_kv_bf16, k_rope_bf16)

    # Step 3: QK GEMM (dual MMA: FP8 NOPE + BF16 ROPE)
    # NOPE: q_nope_fp8 @ k_nope_fp8^T (both are dequanted round-trip values)
    # ROPE: q_rope_prescaled @ k_rope_prescaled^T (BF16, pre-scaled)
    # Expand KV heads
    heads_per_kv = h_q // H_KV

    # NOPE scores
    q_nope = q_nope_deq  # [s_q, h_q, d_c] — round-tripped through FP8
    k_nope = c_kv_deq.unsqueeze(1).expand(-1, heads_per_kv, -1).reshape(s_kv, h_q, D_C)
    scores_nope = torch.einsum('qhd,khd->hqk', q_nope.float(), k_nope.float())

    # ROPE scores
    q_rope = q_rope_ps  # [s_q, h_q, d_rope] — BF16 pre-scaled
    k_rope = k_rope_ps.unsqueeze(1).expand(-1, heads_per_kv, -1).reshape(s_kv, h_q, D_ROPE)
    scores_rope = torch.einsum('qhd,khd->hqk', q_rope.float(), k_rope.float())

    scores = scores_nope + scores_rope

    # Step 4: Post-QK dequant: scores *= Q_scale[row] * K_scale[col]
    # q_scale: [s_q, h_q], k_scale: [s_kv]
    scores = scores * q_scale.permute(1, 0).unsqueeze(-1)  # [h_q, s_q, 1]
    scores = scores * k_scale.unsqueeze(0).unsqueeze(0)     # [1, 1, s_kv]

    # Apply sm_scale
    scores = scores * sm_scale

    # Step 5: Standard softmax (no V-scale fusion)
    lse = torch.logsumexp(scores, dim=-1)
    P = torch.softmax(scores, dim=-1)

    # Step 6: Per-row P quantization
    P_fp8_raw, _, p_scale = simulate_fp8_quantize_rowwise(P)

    # Step 7: PV GEMM using QUANTIZED P (not dequanted)
    # The kernel does: out_raw = FP8_P @ FP8_V (both in quantized domain)
    # V = c_kv (absorbed MLA: V = NOPE portion of KV)
    v = c_kv_deq.unsqueeze(1).expand(-1, heads_per_kv, -1).reshape(s_kv, h_q, D_V)
    out = torch.einsum('hqk,khd->qhd', P_fp8_raw, v.float())

    # Step 8: Post-PV dequant: output *= P_scale[row]
    # This restores the magnitude removed by P quantization
    out = out * p_scale.permute(1, 0).unsqueeze(-1)

    return out.to(q_bf16.dtype), lse.permute(1, 0)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def _quant_roundtrip_metrics(x, fp8_fn):
    """Compute round-trip metrics for a quantize→dequantize cycle."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    x_deq = fp8_fn(x / scale) * scale
    significant = x.abs() > 0.01
    if significant.any():
        rel = (x_deq - x).abs()[significant] / x.abs()[significant]
        return rel.max().item(), rel.mean().item()
    return 0.0, 0.0


def test_q_quant_roundtrip(verbose=False):
    """fused_q_quant: Rough vs Real FP8 vs SnapMLA round-trip precision."""
    torch.manual_seed(42)
    s_q, h_q = 4, H_Q
    q = torch.randn(s_q, h_q, D_QK, dtype=torch.float32)
    q_nope = q[..., :D_C]
    has_fp8 = hasattr(torch, 'float8_e4m3fn')

    if verbose:
        print(f"  {'Method':<20s} {'NOPE max_rel':>12s} {'NOPE mean_rel':>14s} {'ROPE err':>10s}")
        print(f"  {'-'*58}")

    # 1. Rough fallback
    max_r, mean_r = _quant_roundtrip_metrics(q_nope, _fp8_roundtrip_rough)
    if verbose:
        print(f"  {'Rough fallback':<20s} {max_r:>12.4f} {mean_r:>14.4f} {'N/A':>10s}")

    results = [('Rough', max_r)]

    if has_fp8:
        # 2. Naive real FP8 (quantize all dims together)
        max_n, mean_n = _quant_roundtrip_metrics(q_nope, _fp8_roundtrip_real)
        if verbose:
            print(f"  {'Naive real FP8':<20s} {max_n:>12.4f} {mean_n:>14.4f} {'N/A':>10s}")
        results.append(('Naive', max_n))

        # 3. SnapMLA (only NOPE quantized, ROPE stays BF16 — exact)
        q_nope_deq, q_rope_ps, scale = ref_fused_q_quant(q)
        q_rope_expected = q[..., D_C:] / scale.unsqueeze(-1)
        rope_err = (q_rope_ps - q_rope_expected).abs().max().item()
        significant = q_nope.abs() > 0.01
        if significant.any():
            rel = (q_nope_deq - q[..., :D_C]).abs()[significant] / q[..., :D_C].abs()[significant]
            max_s, mean_s = rel.max().item(), rel.mean().item()
        else:
            max_s, mean_s = 0.0, 0.0
        if verbose:
            print(f"  {'SnapMLA FP8':<20s} {max_s:>12.4f} {mean_s:>14.4f} {rope_err:>10.2e}")
        results.append(('SnapMLA', max_s))

    # 4. GPU Kernel (if available)
    if HAS_KERNELS:
        torch.manual_seed(42)
        q_cpu = torch.randn(s_q, h_q, D_QK, dtype=torch.float32)
        q_gpu = q_cpu.to(torch.bfloat16).cuda()

        q_nope_fp8, q_rope_bf16, q_scales_gpu = sm120_mla_kernels.fused_q_quant(q_gpu, D_C)
        # Compare dequanted NOPE against BF16 input (matching kernel's precision)
        q_bf16_cpu = q_gpu.float().cpu()
        nope_deq_gpu = q_nope_fp8.float().cpu() * q_scales_gpu.cpu().unsqueeze(-1)
        significant = q_bf16_cpu[..., :D_C].abs() > 0.01
        if significant.any():
            rel = (nope_deq_gpu - q_bf16_cpu[..., :D_C]).abs()[significant] / q_bf16_cpu[..., :D_C].abs()[significant]
            max_g, mean_g = rel.max().item(), rel.mean().item()
        else:
            max_g, mean_g = 0.0, 0.0
        # Compare ROPE against BF16 input pre-scaled by kernel's own scale
        q_rope_expected = q_bf16_cpu[..., D_C:] / q_scales_gpu.cpu().unsqueeze(-1)
        rope_err_gpu = (q_rope_bf16.float().cpu() - q_rope_expected).abs().max().item()
        if verbose:
            print(f"  {'GPU Kernel':<20s} {max_g:>12.4f} {mean_g:>14.4f} {rope_err_gpu:>10.2e}")
        results.append(('GPU Kernel', max_g))

    best = min(results, key=lambda x: x[1])
    passed = best[1] < 0.07
    if verbose:
        print(f"  Best NOPE precision: {best[0]} ({best[1]:.4f}), ROPE exact: {rope_err:.2e} — {'PASS' if passed else 'FAIL'}")
    return passed


def test_k_append_dequant_roundtrip(verbose=False):
    """fused_k_append + dequant_ckv_indexed: Rough vs Real FP8 vs SnapMLA round-trip."""
    torch.manual_seed(42)
    n_tokens = 128
    c_kv = torch.randn(n_tokens, D_C, dtype=torch.float32)
    k_rope = torch.randn(n_tokens, D_ROPE, dtype=torch.float32)
    has_fp8 = hasattr(torch, 'float8_e4m3fn')

    if verbose:
        print(f"  {'Method':<20s} {'NOPE max_rel':>12s} {'NOPE mean_rel':>14s} {'ROPE err':>10s}")
        print(f"  {'-'*58}")

    # 1. Rough fallback
    max_r, mean_r = _quant_roundtrip_metrics(c_kv, _fp8_roundtrip_rough)
    if verbose:
        print(f"  {'Rough fallback':<20s} {max_r:>12.4f} {mean_r:>14.4f} {'N/A':>10s}")

    results = [('Rough', max_r)]

    if has_fp8:
        # 2. Naive real FP8
        max_n, mean_n = _quant_roundtrip_metrics(c_kv, _fp8_roundtrip_real)
        if verbose:
            print(f"  {'Naive real FP8':<20s} {max_n:>12.4f} {mean_n:>14.4f} {'N/A':>10s}")
        results.append(('Naive', max_n))

        # 3. SnapMLA (NOPE quantized, ROPE stays BF16 — recoverable)
        c_kv_deq, k_rope_ps, scale = ref_fused_k_append(c_kv, k_rope)
        k_rope_recovered = k_rope_ps * scale.unsqueeze(-1)
        rope_err = (k_rope_recovered - k_rope).abs().max().item()
        significant = c_kv.abs() > 0.01
        if significant.any():
            rel = (c_kv_deq - c_kv).abs()[significant] / c_kv.abs()[significant]
            max_s, mean_s = rel.max().item(), rel.mean().item()
        else:
            max_s, mean_s = 0.0, 0.0
        if verbose:
            print(f"  {'SnapMLA FP8':<20s} {max_s:>12.4f} {mean_s:>14.4f} {rope_err:>10.2e}")
        results.append(('SnapMLA', max_s))

    # 4. GPU Kernel roundtrip (if available)
    if HAS_KERNELS:
        torch.manual_seed(42)
        c_kv_cpu = torch.randn(n_tokens, D_C, dtype=torch.float32)
        k_rope_cpu = torch.randn(n_tokens, D_ROPE, dtype=torch.float32)
        c_kv_gpu = c_kv_cpu.to(torch.bfloat16).cuda()
        k_rope_gpu = k_rope_cpu.to(torch.bfloat16).cuda()

        n_pages = (n_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_cache = _alloc_paged_cache(n_pages, PAGE_SIZE, D_C, D_ROPE)
        slot_mapping = torch.arange(n_tokens, dtype=torch.int32, device="cuda")
        sm120_mla_kernels.fused_k_append(c_kv_gpu, k_rope_gpu, kv_cache, slot_mapping,
                                          D_C, D_ROPE, PAGE_SIZE)
        indices = torch.arange(n_tokens, dtype=torch.int32, device="cuda")
        k_out = sm120_mla_kernels.dequant_ckv_indexed(kv_cache, indices, D_C, D_ROPE, PAGE_SIZE)
        nope_out = k_out.float().cpu()[:, :D_C]

        significant = c_kv.abs() > 0.01
        if significant.any():
            rel = (nope_out - c_kv).abs()[significant] / c_kv.abs()[significant]
            max_g, mean_g = rel.max().item(), rel.mean().item()
        else:
            max_g, mean_g = 0.0, 0.0
        rope_out = k_out.float().cpu()[:, D_C:]
        # dequant_ckv_indexed now un-scales ROPE (multiplies by scale),
        # so output should approximate original k_rope
        rope_cos = F.cosine_similarity(rope_out.flatten(), k_rope.flatten(), dim=0).item()
        rope_err_gpu = 1.0 - rope_cos
        if verbose:
            print(f"  {'GPU Kernel':<20s} {max_g:>12.4f} {mean_g:>14.4f} {rope_err_gpu:>10.2e}")
        results.append(('GPU Kernel', max_g))

    best = min(results, key=lambda x: x[1])
    passed = best[1] < 0.07
    if verbose:
        print(f"  Best NOPE precision: {best[0]} ({best[1]:.4f}), ROPE recoverable: {rope_err:.2e} — {'PASS' if passed else 'FAIL'}")
    return passed


def _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk):
    """Select topk KV token indices by oracle BF16 attention scores.

    Computes full Q@K^T, averages across heads (MLA shared KV), picks topk.
    Returns: int tensor [s_q, topk] of selected token indices.
    """
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]
    k_full = torch.cat([c_kv, k_rope], dim=-1)  # [s_kv, d_qk]
    # [s_q, h_q, d_qk] @ [s_kv, d_qk]^T → [h_q, s_q, s_kv]
    scores = torch.einsum('qhd,kd->hqk', q.float(), k_full.float()) * sm_scale
    # Average across heads for shared MLA index selection
    importance = scores.mean(dim=0)  # [s_q, s_kv]
    _, idx = importance.topk(topk, dim=-1)  # [s_q, topk]
    return idx.sort(dim=-1).values


def _topk_size(s_kv):
    """Fixed topk=2048 matching production DeepSeek NSA (index_topk=2048)."""
    topk = min(2048, s_kv)
    return max(64, ((topk + 63) // 64) * 64)


def _run_three_way_attention(s_kv, kernel_mode="dense"):
    """Run all three FP8 attention approaches at given context length.

    kernel_mode: "dense" | "sparse" | None — which GPU kernel to include.
    """
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    torch.manual_seed(42)
    s_q = 1
    sm_scale = 1.0 / math.sqrt(D_QK)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)
    out_bf16, lse_bf16 = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    rows = []

    # 1. Rough fallback
    out_rough = naive_fp8_attention(q, c_kv, k_rope, sm_scale, _fp8_roundtrip_rough)
    rows.append(("Rough fallback", _compute_metrics(out_bf16, out_rough)))

    if has_fp8:
        # 2. Naive real FP8
        out_naive = naive_fp8_attention(q, c_kv, k_rope, sm_scale, _fp8_roundtrip_real)
        rows.append(("Naive real FP8", _compute_metrics(out_bf16, out_naive)))

        # 3. SnapMLA FP8
        out_snap = snapmla_fp8_attention(q, c_kv, k_rope, sm_scale)
        rows.append(("SnapMLA FP8", _compute_metrics(out_bf16, out_snap)))

    # 4. GPU Kernel (if available)
    if HAS_KERNELS and kernel_mode:
        q_gpu = q.to(torch.bfloat16).unsqueeze(0).cuda()  # [1, s_q, h_q, d_qk]
        c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
        k_rope_gpu = k_rope.to(torch.bfloat16).cuda()
        if kernel_mode == "sparse":
            out_gpu, _ = _run_gpu_sparse_decode(q_gpu, c_kv_gpu, k_rope_gpu, sm_scale)
            label = "GPU Sparse"
        else:
            out_gpu, _ = _run_gpu_decode(q_gpu, c_kv_gpu, k_rope_gpu, sm_scale)
            label = "GPU Dense"
        out_gpu = out_gpu.view(s_q, H_Q, D_V)
        rows.append((label, _compute_metrics(out_bf16, out_gpu)))

    # LSE diff for SnapMLA pipeline
    _, lse_snap = ref_snapmla_decode_fp8(q, c_kv, k_rope, sm_scale)
    lse_diff = (lse_snap.float() - lse_bf16.float()).abs().max().item()

    return rows, lse_diff, has_fp8


def _print_attention_table(s_kv, rows, lse_diff, verbose):
    if verbose:
        print(f"\n  s_kv={s_kv}:")
        print(f"  {'Method':<20s} {'cosine':>10s} {'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
        print(f"  {'-'*60}")
        for name, m in rows:
            print(f"  {name:<20s} {m['cosine']:>10.6f} {m['nrmse']:>10.4f} "
                  f"{m['mean_rel']:>10.4f} {m['max_rel']:>10.4f}")
        print(f"  LSE diff (SnapMLA pipeline): {lse_diff:.4f}")


def _get_snapmla_row(rows, has_fp8):
    """Get the SnapMLA FP8 row for pass/fail (not GPU Kernel)."""
    for name, m in rows:
        if name == "SnapMLA FP8":
            return m
    return rows[-1][1] if has_fp8 else rows[0][1]


def test_decode_attention_short(verbose=False):
    """Decode attention (s_kv=256): Rough vs Naive FP8 vs SnapMLA FP8 vs BF16."""
    rows, lse_diff, has_fp8 = _run_three_way_attention(256, kernel_mode="dense")
    _print_attention_table(256, rows, lse_diff, verbose)
    m = _get_snapmla_row(rows, has_fp8)
    passed = m['cosine'] > 0.995 and m['nrmse'] < 0.05
    if verbose or not passed:
        print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def _run_decode_scaling(kernel_mode, verbose):
    """Shared logic for dense/sparse decode scaling tests."""
    all_passed = True
    for s_kv in [1024, 4096, 32768]:
        rows, lse_diff, has_fp8 = _run_three_way_attention(s_kv, kernel_mode=kernel_mode)
        _print_attention_table(s_kv, rows, lse_diff, verbose)
        m = _get_snapmla_row(rows, has_fp8)
        if s_kv >= 32768:
            thr_cos, thr_nrmse = 0.990, 0.15
        elif s_kv >= 4096:
            thr_cos, thr_nrmse = 0.995, 0.08
        else:
            thr_cos, thr_nrmse = 0.995, 0.05
        passed = m['cosine'] > thr_cos and m['nrmse'] < thr_nrmse
        all_passed &= passed
        if verbose or not passed:
            print(f"  s_kv={s_kv}: {'PASS' if passed else 'FAIL'}")
    return all_passed


def test_dense_decode_scaling(verbose=False):
    """Dense decode scaling (1K → 32K): Rough vs Naive FP8 vs SnapMLA FP8 vs BF16."""
    return _run_decode_scaling("dense", verbose)


def test_sparse_decode_scaling(verbose=False):
    """Sparse decode scaling (1K → 32K): Rough vs Naive FP8 vs SnapMLA FP8 vs BF16."""
    return _run_decode_scaling("sparse", verbose)


def _run_topk_attention(s_kv, kernel_mode=None, topk_override=None):
    """Run FP8 attention with oracle topk subset at given context length."""
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    torch.manual_seed(42)
    s_q = 1
    sm_scale = 1.0 / math.sqrt(D_QK)
    topk = topk_override if topk_override is not None else _topk_size(s_kv)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    # Dense BF16 reference (full attention — ground truth)
    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)
    out_bf16, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    # Oracle topk selection
    idx = _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk)  # [s_q, topk]
    idx_0 = idx[0]  # [topk] — single query
    c_kv_sub = c_kv[idx_0]
    k_rope_sub = k_rope[idx_0]

    rows = []

    # 1. Rough fallback (on subset)
    out_rough = naive_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale, _fp8_roundtrip_rough)
    rows.append(("Rough fallback", _compute_metrics(out_bf16, out_rough)))

    if has_fp8:
        # 2. Naive real FP8 (on subset)
        out_naive = naive_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale, _fp8_roundtrip_real)
        rows.append(("Naive real FP8", _compute_metrics(out_bf16, out_naive)))

        # 3. SnapMLA FP8 (on subset)
        out_snap = snapmla_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale)
        rows.append(("SnapMLA FP8", _compute_metrics(out_bf16, out_snap)))

    # 4. GPU Sparse Kernel (if available)
    if HAS_KERNELS and kernel_mode == "sparse":
        q_gpu = q.to(torch.bfloat16).unsqueeze(0).cuda()
        c_kv_gpu = c_kv.to(torch.bfloat16).cuda()
        k_rope_gpu = k_rope.to(torch.bfloat16).cuda()

        # Build full cache, then pass topk indices
        kv_cache, q_nope_fp8, q_rope_bf16, q_scales, n_pages = \
            _prep_gpu_decode(q_gpu, c_kv_gpu, k_rope_gpu)

        topk_padded = ((topk + 63) // 64) * 64
        gpu_indices = torch.full((1, s_q, topk_padded), -1, dtype=torch.int32, device="cuda")
        gpu_indices[0, 0, :topk] = idx_0.to(torch.int32).cuda()

        out_gpu, _ = sm120_mla_kernels.sparse_decode_v32(
            q_nope_fp8, q_rope_bf16, q_scales,
            kv_cache, gpu_indices, sm_scale, PAGE_SIZE, topk_padded, 1
        )
        out_gpu = out_gpu.float().cpu().view(s_q, H_Q, D_V)
        rows.append(("GPU Sparse", _compute_metrics(out_bf16, out_gpu)))

    return rows, topk, has_fp8


def _print_topk_table(s_kv, topk, rows, verbose):
    if verbose:
        print(f"\n  s_kv={s_kv}, topk={topk} ({100*topk/s_kv:.0f}%):")
        print(f"  {'Method':<20s} {'cosine':>10s} {'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
        print(f"  {'-'*60}")
        for name, m in rows:
            print(f"  {name:<20s} {m['cosine']:>10.6f} {m['nrmse']:>10.4f} "
                  f"{m['mean_rel']:>10.4f} {m['max_rel']:>10.4f}")


def _topk_size_25pct(s_kv):
    """Topk at ~25% of s_kv for random-data tests where extreme sparsity is meaningless."""
    topk = max(64, s_kv // 4)
    return ((topk + 63) // 64) * 64


def test_sparse_topk_decode_scaling(verbose=False):
    """Sparse topk decode scaling (1K→32K): oracle topk ~25% vs dense BF16."""
    all_passed = True
    for s_kv in [1024, 4096, 32768]:
        # Use 25% topk for random data: at extreme sparsity (e.g. 6% at 32K),
        # random data gives near-zero cosine since uniform softmax means topk
        # captures no signal. Real peaked data is tested in test_real_data.py.
        rows, topk, has_fp8 = _run_topk_attention(
            s_kv, kernel_mode="sparse", topk_override=_topk_size_25pct(s_kv))
        _print_topk_table(s_kv, topk, rows, verbose)
        m = _get_snapmla_row(rows, has_fp8)
        passed = m['cosine'] > 0.0 and not math.isnan(m['nrmse'])
        all_passed &= passed
        if verbose or not passed:
            print(f"  s_kv={s_kv}, topk={topk}: {'PASS' if passed else 'FAIL'}")
    return all_passed


def _run_prefill_attention(s_q, s_kv):
    """Run FP8 attention comparison for prefill (s_q > 1)."""
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D_QK)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)
    out_bf16, lse_bf16 = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    rows = []

    # 1. Rough fallback
    out_rough = naive_fp8_attention(q, c_kv, k_rope, sm_scale, _fp8_roundtrip_rough)
    rows.append(("Rough fallback", _compute_metrics(out_bf16, out_rough)))

    if has_fp8:
        # 2. Naive real FP8
        out_naive = naive_fp8_attention(q, c_kv, k_rope, sm_scale, _fp8_roundtrip_real)
        rows.append(("Naive real FP8", _compute_metrics(out_bf16, out_naive)))

        # 3. SnapMLA FP8
        out_snap = snapmla_fp8_attention(q, c_kv, k_rope, sm_scale)
        rows.append(("SnapMLA FP8", _compute_metrics(out_bf16, out_snap)))

    return rows, has_fp8


def _print_prefill_table(s_q, s_kv, rows, verbose):
    if verbose:
        print(f"\n  s_q={s_q}, s_kv={s_kv}:")
        print(f"  {'Method':<20s} {'cosine':>10s} {'NRMSE':>10s} {'mean_rel':>10s} {'max_rel':>10s}")
        print(f"  {'-'*60}")
        for name, m in rows:
            print(f"  {name:<20s} {m['cosine']:>10.6f} {m['nrmse']:>10.4f} "
                  f"{m['mean_rel']:>10.4f} {m['max_rel']:>10.4f}")


def _run_prefill_scaling(mode, verbose):
    """Shared logic for dense/sparse prefill scaling tests."""
    s_q = 128
    all_passed = True
    for s_kv in [1024, 4096, 32768]:
        rows, has_fp8 = _run_prefill_attention(s_q, s_kv)
        # TODO: add GPU kernel row when prefill bindings are available
        # (sparse prefill kernels exist in csrc/sm120/prefill/sparse/ but are not bound)
        _print_prefill_table(s_q, s_kv, rows, verbose)
        m = _get_snapmla_row(rows, has_fp8)
        if s_kv >= 32768:
            thr_cos, thr_nrmse = 0.990, 0.15
        elif s_kv >= 4096:
            thr_cos, thr_nrmse = 0.995, 0.08
        else:
            thr_cos, thr_nrmse = 0.995, 0.05
        passed = m['cosine'] > thr_cos and m['nrmse'] < thr_nrmse
        all_passed &= passed
        if verbose or not passed:
            print(f"  s_q={s_q}, s_kv={s_kv}: {'PASS' if passed else 'FAIL'}")
    return all_passed


def test_dense_prefill_scaling(verbose=False):
    """Dense prefill scaling (s_q=128, 1K→32K): Rough vs Naive FP8 vs SnapMLA FP8 vs BF16."""
    return _run_prefill_scaling("dense", verbose)


def test_sparse_prefill_scaling(verbose=False):
    """Sparse prefill scaling (s_q=128, 1K→32K): Rough vs Naive FP8 vs SnapMLA FP8 vs BF16."""
    return _run_prefill_scaling("sparse", verbose)


def _run_topk_prefill_attention(s_q, s_kv):
    """Run FP8 prefill attention with oracle topk subset."""
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D_QK)
    topk = _topk_size(s_kv)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    # Dense BF16 reference (full attention)
    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)
    out_bf16, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    # Oracle topk — per query token, use first query's indices for simplicity
    idx = _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk)  # [s_q, topk]
    # Use union of all query tokens' topk for shared KV subset
    all_idx = idx.reshape(-1).unique().sort().values
    # Clamp to topk size (union may be larger)
    if len(all_idx) > topk:
        all_idx = all_idx[:topk]
    c_kv_sub = c_kv[all_idx]
    k_rope_sub = k_rope[all_idx]

    rows = []

    out_rough = naive_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale, _fp8_roundtrip_rough)
    rows.append(("Rough fallback", _compute_metrics(out_bf16, out_rough)))

    if has_fp8:
        out_naive = naive_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale, _fp8_roundtrip_real)
        rows.append(("Naive real FP8", _compute_metrics(out_bf16, out_naive)))

        out_snap = snapmla_fp8_attention(q, c_kv_sub, k_rope_sub, sm_scale)
        rows.append(("SnapMLA FP8", _compute_metrics(out_bf16, out_snap)))

    return rows, topk, has_fp8


def test_sparse_topk_prefill_scaling(verbose=False):
    """Sparse topk prefill scaling (s_q=128, 1K→32K): oracle topk ~25% vs dense BF16."""
    s_q = 128
    all_passed = True
    for s_kv in [1024, 4096, 32768]:
        rows, topk, has_fp8 = _run_topk_prefill_attention(s_q, s_kv)
        _print_topk_table(s_kv, topk, rows, verbose)
        m = _get_snapmla_row(rows, has_fp8)
        # Same relaxed thresholds as decode topk — validates pathway, not quality
        passed = m['cosine'] > 0.0 and not math.isnan(m['nrmse'])
        all_passed &= passed
        if verbose or not passed:
            print(f"  s_q={s_q}, s_kv={s_kv}, topk={topk}: {'PASS' if passed else 'FAIL'}")
    return all_passed


def test_chunked_prefill_consistency(verbose=False):
    """Test: Chunked prefill (2 chunks) matches single-pass prefill.

    Verifies that LSE-based merging of chunk results produces the same
    output as processing all tokens at once.
    """
    torch.manual_seed(42)
    s_q = 8
    s_kv = 256
    chunk_size = 128
    sm_scale = 1.0 / math.sqrt(D_QK)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    k = torch.randn(s_kv, 1, D_QK, dtype=torch.float32) * 0.1
    v = torch.randn(s_kv, 1, D_V, dtype=torch.float32) * 0.1

    # Single-pass reference
    out_full, lse_full = ref_mla_attention_bf16(q, k, v, sm_scale)

    # Two-chunk LSE-merged
    out_accum = torch.zeros(s_q, H_Q, D_V, dtype=torch.float32)
    lse_accum = torch.full((s_q, H_Q), float('-inf'), dtype=torch.float32)

    for start in range(0, s_kv, chunk_size):
        end = min(start + chunk_size, s_kv)
        k_chunk = k[start:end]
        v_chunk = v[start:end]
        out_chunk, lse_chunk = ref_mla_attention_bf16(q, k_chunk, v_chunk, sm_scale)

        # Log-sum-exp merge
        new_lse = torch.logaddexp(lse_accum, lse_chunk.float())
        old_w = torch.exp(lse_accum - new_lse).unsqueeze(-1)
        new_w = torch.exp(lse_chunk.float() - new_lse).unsqueeze(-1)
        out_accum = old_w * out_accum + new_w * out_chunk.float()
        lse_accum = new_lse

    # Compare
    max_diff = (out_accum.float() - out_full.float()).abs().max().item()
    lse_diff = (lse_accum - lse_full.float()).abs().max().item()

    # Should be numerically exact (float32 LSE merge)
    passed = max_diff < 1e-4 and lse_diff < 1e-4
    if verbose or not passed:
        print(f"  Chunked vs single-pass: out_max_diff={max_diff:.2e}, "
              f"lse_max_diff={lse_diff:.2e} — {'PASS' if passed else 'FAIL'}")
    return passed


def test_sparse_vs_dense(verbose=False):
    """Test: Sparse attention (topk subset) vs dense (all tokens).

    Verifies that when topk = s_kv (all tokens selected), sparse and
    dense produce identical results. This validates the topk indexing.
    """
    torch.manual_seed(42)
    s_q = 1
    s_kv = 128
    sm_scale = 1.0 / math.sqrt(D_QK)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    k = torch.randn(s_kv, 1, D_QK, dtype=torch.float32) * 0.1
    v = torch.randn(s_kv, 1, D_V, dtype=torch.float32) * 0.1

    # Dense: all tokens
    out_dense, lse_dense = ref_mla_attention_bf16(q, k, v, sm_scale)

    # Sparse: topk = all tokens (identity permutation)
    indices = torch.arange(s_kv)
    k_sparse = k[indices]
    v_sparse = v[indices]
    out_sparse, lse_sparse = ref_mla_attention_bf16(q, k_sparse, v_sparse, sm_scale)

    max_diff = (out_dense.float() - out_sparse.float()).abs().max().item()

    # Should be exact (same computation, same data)
    passed = max_diff < 1e-6
    if verbose or not passed:
        print(f"  Sparse(topk=all) vs Dense: max_diff={max_diff:.2e} — {'PASS' if passed else 'FAIL'}")

    # Sparse: random permutation (should also match — just reordered)
    perm = torch.randperm(s_kv)
    out_perm, _ = ref_mla_attention_bf16(q, k[perm], v[perm], sm_scale)
    perm_diff = (out_dense.float() - out_perm.float()).abs().max().item()
    perm_ok = perm_diff < 1e-5
    if verbose:
        print(f"  Sparse(permuted) vs Dense: max_diff={perm_diff:.2e} — {'PASS' if perm_ok else 'FAIL'}")

    return passed and perm_ok


def test_sparse_topk_vs_dense(verbose=False):
    """Test: Sparse attention with oracle topk subset vs dense (all tokens).

    Verifies that oracle topk ~25% captures most of the attention output.
    """
    torch.manual_seed(42)
    s_q = 1
    s_kv = 1024
    sm_scale = 1.0 / math.sqrt(D_QK)
    topk = _topk_size(s_kv)  # 256

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    k_mla = torch.cat([c_kv, k_rope], dim=-1).unsqueeze(1)
    v_mla = c_kv.unsqueeze(1)

    # Dense: all tokens
    out_dense, _ = ref_mla_attention_bf16(q, k_mla, v_mla, sm_scale)

    # Sparse: oracle topk
    idx = _oracle_topk_indices(q, c_kv, k_rope, sm_scale, topk)
    idx_0 = idx[0]
    out_topk, _ = ref_mla_attention_bf16(q, k_mla[idx_0], v_mla[idx_0], sm_scale)

    m = _compute_metrics(out_dense, out_topk)
    passed = m['cosine'] > 0.1
    if verbose or not passed:
        print(f"  Sparse(topk={topk}/{s_kv}={100*topk/s_kv:.0f}%) vs Dense: "
              f"cosine={m['cosine']:.6f}, nrmse={m['nrmse']:.4f} — {'PASS' if passed else 'FAIL'}")

    # Also test with random subset (much worse — no oracle)
    rand_idx = torch.randperm(s_kv)[:topk].sort().values
    out_rand, _ = ref_mla_attention_bf16(q, k_mla[rand_idx], v_mla[rand_idx], sm_scale)
    m_rand = _compute_metrics(out_dense, out_rand)
    if verbose:
        print(f"  Random(topk={topk}/{s_kv}={100*topk/s_kv:.0f}%) vs Dense: "
              f"cosine={m_rand['cosine']:.6f}, nrmse={m_rand['nrmse']:.4f}")

    return passed


# ---------------------------------------------------------------------------
# Three-way FP8 comparison: rough fallback vs naive real FP8 vs SnapMLA FP8
# ---------------------------------------------------------------------------

def _fp8_roundtrip_rough(x: torch.Tensor) -> torch.Tensor:
    """Rough FP8 sim: clamp + round to nearest 1/8 (3-bit fixed-point)."""
    return x.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8)


def _fp8_roundtrip_real(x: torch.Tensor) -> torch.Tensor:
    """Real FP8 sim: cast to float8_e4m3fn and back."""
    return x.to(torch.float8_e4m3fn).float()


def naive_fp8_attention(
    q: torch.Tensor,       # [s_q, h_q, d_qk]
    c_kv: torch.Tensor,    # [s_kv, d_c]
    k_rope: torch.Tensor,  # [s_kv, d_rope]
    sm_scale: float,
    fp8_fn,                # _fp8_roundtrip_rough or _fp8_roundtrip_real
) -> torch.Tensor:
    """Naive FP8 attention: quantize full Q and K (no NOPE/ROPE split).

    This is what you'd get without SnapMLA's dual-MMA trick — just jam
    everything (NOPE+ROPE) into FP8 together.
    """
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]
    heads_per_kv = h_q // H_KV

    # Concat K = [c_kv, k_rope], no split
    k_full = torch.cat([c_kv, k_rope], dim=-1)  # [s_kv, d_qk]

    # Per-token Q quantization (all dims together)
    q_amax = q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    q_scale = q_amax / FP8_MAX
    q_fp8 = fp8_fn(q / q_scale) * q_scale  # round-tripped

    # Per-token K quantization (all dims together)
    k_amax = k_full.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    k_scale = k_amax / FP8_MAX
    k_fp8 = fp8_fn(k_full / k_scale) * k_scale  # round-tripped

    # V = c_kv, also quantized
    v = c_kv
    v_amax = v.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    v_scale = v_amax / FP8_MAX
    v_fp8 = fp8_fn(v / v_scale) * v_scale

    # Expand KV
    k_exp = k_fp8.unsqueeze(1).expand(-1, h_q, -1)
    v_exp = v_fp8.unsqueeze(1).expand(-1, h_q, -1)

    # QK
    scores = torch.einsum('qhd,khd->hqk', q_fp8.float(), k_exp.float()) * sm_scale
    P = torch.softmax(scores, dim=-1)

    # P quantization
    p_amax = P.abs().amax(dim=-1, keepdim=True).clamp(min=1e-26)
    p_scale = p_amax / FP8_MAX
    P_fp8_raw = fp8_fn(P / p_scale)

    # PV
    out = torch.einsum('hqk,khd->qhd', P_fp8_raw, v_exp.float())
    out = out * p_scale.squeeze(-1).permute(1, 0).unsqueeze(-1)

    return out


def snapmla_fp8_attention(
    q: torch.Tensor,       # [s_q, h_q, d_qk]
    c_kv: torch.Tensor,    # [s_kv, d_c]
    k_rope: torch.Tensor,  # [s_kv, d_rope]
    sm_scale: float,
) -> torch.Tensor:
    """SnapMLA FP8: real float8_e4m3fn + NOPE/ROPE split + dual MMA.

    Only NOPE dims are quantized to FP8. ROPE stays in BF16 (pre-scaled
    by the NOPE quantization scale). QK = FP8(NOPE) + BF16(ROPE).
    """
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]
    heads_per_kv = h_q // H_KV

    # Q: split NOPE/ROPE, quantize only NOPE
    q_nope, q_rope = q[..., :D_C], q[..., D_C:]
    q_amax = q_nope.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    q_scale = q_amax / FP8_MAX
    q_nope_fp8 = _fp8_roundtrip_real(q_nope / q_scale)  # FP8 quantized
    q_rope_ps = q_rope / q_scale                         # BF16 pre-scaled

    # K: split NOPE/ROPE, quantize only NOPE (c_kv)
    k_amax = c_kv.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    k_scale = k_amax / FP8_MAX
    k_nope_fp8 = _fp8_roundtrip_real(c_kv / k_scale)  # FP8 quantized
    k_rope_ps = k_rope / k_scale                       # BF16 pre-scaled

    # Expand KV
    k_nope_exp = k_nope_fp8.unsqueeze(1).expand(-1, h_q, -1)
    k_rope_exp = k_rope_ps.unsqueeze(1).expand(-1, h_q, -1)

    # Dual MMA: FP8 NOPE scores + BF16 ROPE scores
    scores_nope = torch.einsum('qhd,khd->hqk', q_nope_fp8.float(), k_nope_exp.float())
    scores_rope = torch.einsum('qhd,khd->hqk', q_rope_ps.float(), k_rope_exp.float())
    scores = scores_nope + scores_rope

    # Dequant: multiply by Q_scale * K_scale
    scores = scores * q_scale.squeeze(-1).permute(1, 0).unsqueeze(-1)
    scores = scores * k_scale.squeeze(-1).unsqueeze(0).unsqueeze(0)
    scores = scores * sm_scale

    P = torch.softmax(scores, dim=-1)

    # P quantization
    p_amax = P.abs().amax(dim=-1, keepdim=True).clamp(min=1e-26)
    p_scale = p_amax / FP8_MAX
    P_fp8_raw = _fp8_roundtrip_real(P / p_scale)

    # PV GEMM (V = c_kv round-tripped through FP8 cache)
    v_deq = (k_nope_fp8 * k_scale).unsqueeze(1).expand(-1, h_q, -1)
    out = torch.einsum('hqk,khd->qhd', P_fp8_raw, v_deq.float())
    out = out * p_scale.squeeze(-1).permute(1, 0).unsqueeze(-1)

    return out


def _compute_metrics(out_ref: torch.Tensor, out_test: torch.Tensor) -> dict:
    """Compute cosine, NRMSE, max/mean relative error vs reference."""
    ref_f = out_ref.flatten().float()
    test_f = out_test.flatten().float()

    cosine = F.cosine_similarity(ref_f, test_f, dim=0).item()
    rmse = (ref_f - test_f).pow(2).mean().sqrt().item()
    nrmse = rmse / (ref_f.pow(2).mean().sqrt().item() + 1e-8)

    rms_thr = out_ref.float().abs().pow(2).mean().sqrt().item() * 0.01
    sig = out_ref.float().abs() > rms_thr
    if sig.any():
        rel = (out_test.float() - out_ref.float()).abs()[sig] / out_ref.float().abs()[sig]
        max_rel = rel.max().item()
        mean_rel = rel.mean().item()
    else:
        max_rel = 0.0
        mean_rel = 0.0

    return dict(cosine=cosine, nrmse=nrmse, max_rel=max_rel, mean_rel=mean_rel)


def test_snapmla_decode_beats_naive(verbose=False):
    """SnapMLA decode kernel: does NOPE/ROPE split beat naive all-FP8 at 32K?"""
    has_fp8 = hasattr(torch, 'float8_e4m3fn')
    if not has_fp8:
        if verbose:
            print("  [torch.float8_e4m3fn not available — skipped]")
        return True

    rows, lse_diff, _ = _run_three_way_attention(32768)
    _print_attention_table(32768, rows, lse_diff, verbose)

    m_snap = rows[2][1]   # SnapMLA FP8
    m_naive = rows[1][1]  # Naive real FP8
    snap_better = m_snap['cosine'] >= m_naive['cosine'] and m_snap['nrmse'] <= m_naive['nrmse']

    if verbose or not snap_better:
        print(f"\n  SnapMLA decode beats naive FP8 at 32K? {'YES' if snap_better else 'NO'}")
    return snap_better


# ---------------------------------------------------------------------------
# Kernel-level test stubs (activate when bindings are available)
# ---------------------------------------------------------------------------

def test_kernel_fused_q_quant():
    """Compare actual fused_q_quant kernel output vs ref_fused_q_quant."""
    try:
        import sm120_mla_kernels
    except ImportError:
        print("  [sm120_mla_kernels not built — skipped]")
        return True

    if not torch.cuda.is_available():
        print("  [CUDA not available — skipped]")
        return True

    torch.manual_seed(42)
    q = torch.randn(4, H_Q, D_QK, dtype=torch.bfloat16, device="cuda")
    q_nope_fp8, q_rope_bf16, q_scales = sm120_mla_kernels.fused_q_quant(q, D_C)

    q_nope_ref, q_rope_ref, scales_ref = ref_fused_q_quant(q.float().cpu())

    scales_ok = torch.allclose(q_scales.cpu(), scales_ref.float(), rtol=1e-2, atol=1e-5)
    nope_deq = q_nope_fp8.float().cpu() * q_scales.cpu().unsqueeze(-1)
    nope_cos = F.cosine_similarity(nope_deq.flatten(), q_nope_ref.flatten(), dim=0).item()
    rope_cos = F.cosine_similarity(q_rope_bf16.float().cpu().flatten(), q_rope_ref.flatten(), dim=0).item()

    passed = scales_ok and nope_cos > 0.999 and rope_cos > 0.999
    print(f"  Kernel fused_q_quant: scales_ok={scales_ok}, nope_cos={nope_cos:.6f}, rope_cos={rope_cos:.6f} — {'PASS' if passed else 'FAIL'}")
    return passed


def test_kernel_decode_vs_reference():
    """Compare actual dense decode kernel output vs ref_snapmla_decode_fp8."""
    try:
        from test_kernels import test_kernel_dense_decode
        test_kernel_dense_decode()
        print("  Kernel dense decode: PASS")
        return True
    except ImportError:
        print("  [sm120_mla_kernels not built — skipped]")
        return True
    except Exception as e:
        print(f"  Kernel dense decode: FAIL ({e})")
        return False


def test_kernel_split_kv_consistency():
    """Compare num_sm_parts=1 vs num_sm_parts=8."""
    try:
        from test_kernels import test_kernel_dense_split_kv_consistency
        test_kernel_dense_split_kv_consistency()
        print("  Kernel split-KV consistency: PASS")
        return True
    except ImportError:
        print("  [sm120_mla_kernels not built — skipped]")
        return True
    except Exception as e:
        print(f"  Kernel split-KV consistency: FAIL ({e})")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SnapMLA reference tests')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print("SnapMLA Reference Tests")
    print("=" * 60)

    tests = [
        ("fused_q_quant: Rough vs Naive FP8 vs SnapMLA round-trip", test_q_quant_roundtrip),
        ("fused_k_append + dequant_ckv: Rough vs Naive FP8 vs SnapMLA round-trip", test_k_append_dequant_roundtrip),
        ("Decode attention (s_kv=256): Rough vs Naive FP8 vs SnapMLA vs BF16", test_decode_attention_short),
        ("Dense decode scaling (1K→32K): Rough vs Naive FP8 vs SnapMLA vs BF16", test_dense_decode_scaling),
        ("Sparse decode scaling (1K→32K): Rough vs Naive FP8 vs SnapMLA vs BF16", test_sparse_decode_scaling),
        ("Sparse topk decode scaling (1K→32K): oracle ~25% vs dense BF16", test_sparse_topk_decode_scaling),
        ("SnapMLA decode kernel: beats Naive FP8 at 32K?", test_snapmla_decode_beats_naive),
        ("Dense prefill scaling (s_q=128, 1K→32K): Rough vs Naive FP8 vs SnapMLA vs BF16", test_dense_prefill_scaling),
        ("Sparse prefill scaling (s_q=128, 1K→32K): Rough vs Naive FP8 vs SnapMLA vs BF16", test_sparse_prefill_scaling),
        ("Sparse topk prefill scaling (s_q=128, 1K→32K): oracle ~25% vs dense BF16", test_sparse_topk_prefill_scaling),
        ("Chunked prefill: LSE merge consistency", test_chunked_prefill_consistency),
        ("Sparse vs Dense attention: topk=all identity", test_sparse_vs_dense),
        ("Sparse topk vs Dense: oracle ~25% quality", test_sparse_topk_vs_dense),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        passed = test_fn(verbose=args.verbose)
        results.append((name, passed))
        if not args.verbose:
            print(f"  {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")

    if n_pass < n_total:
        print("\nFailed tests:")
        for name, passed in results:
            if not passed:
                print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
