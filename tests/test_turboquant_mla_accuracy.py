"""
TurboQuant MLA Accuracy Validation — PyTorch Reference Implementation

Measures the accuracy of applying TurboQuant to MLA's compressed latent cache.
The key question: MLA's latent c_t (kv_lora_rank=512) serves as BOTH key and
value. TurboQuant was designed for standard MHA where K and V are separate.
This test validates whether TQ compression of the shared latent preserves
attention quality for both the score path (inner product) and the value
reconstruction path (MSE).

Two TQ variants tested:
  1. TQ_Prod 4-bit: 3-bit MSE (8 centroids) + 1-bit QJL signs — unbiased IP
  2. TQ_MSE 4-bit: 4-bit MSE (16 centroids, no QJL) — lower reconstruction MSE

Both compared against:
  - BF16 ground truth (ref_mla_attention_bf16)
  - SnapMLA FP8 baseline (ref_snapmla_decode_fp8)

Usage:
  python tests/test_turboquant_mla_accuracy.py          # runs all tests
  python tests/test_turboquant_mla_accuracy.py -v        # verbose
  python tests/test_turboquant_mla_accuracy.py -v --long # include 80K context

Context lengths tested: 256, 1K, 4K, 16K, 32K, (80K with --long)
"""

import torch
import torch.nn.functional as F
import math
import argparse
import time
from typing import NamedTuple, Optional

# ---------------------------------------------------------------------------
# Model Config (matches test_snapmla_reference.py)
# ---------------------------------------------------------------------------

D_C = 512       # compressed latent (NOPE) = kv_lora_rank
D_ROPE = 64     # RoPE dims = qk_rope_head_dim
D_QK = D_C + D_ROPE  # 576
D_V = 512       # value dim (= kv_lora_rank in MLA absorbed mode)
H_Q = 64        # Q heads
H_KV = 1        # KV heads (MLA: single shared latent)
FP8_MAX = 448.0

# ---------------------------------------------------------------------------
# TurboQuant Implementation (self-contained, no external deps)
# ---------------------------------------------------------------------------


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""
    indices: torch.Tensor       # (..., packed_len) uint8 bit-packed
    norms: torch.Tensor         # (...,) original L2 norms
    bits: int


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization."""
    mse_indices: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    norms: torch.Tensor
    mse_bits: int


def _generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Random orthogonal matrix via QR decomposition (Algorithm 1, step 2)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


def _generate_qjl_matrix(d: int, seed: int = 12345) -> torch.Tensor:
    """Random projection matrix S for QJL (i.i.d. N(0,1) entries)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return torch.randn(d, d, generator=rng, dtype=torch.float32)


def _lloyd_max_codebook(d: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Lloyd-Max optimal scalar quantizer for Beta distribution on [-1,1].

    After random rotation, each coordinate of a unit vector follows:
      f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    For large d this converges to N(0, 1/d). We use numerical optimization
    via iterative Lloyd-Max on a fine grid.

    Returns: (centroids, boundaries) of shape (2^bits,) and (2^bits+1,)
    """
    n_clusters = 2 ** bits
    # Sample from the distribution using the Gaussian approximation (valid for d>=64)
    sigma = 1.0 / math.sqrt(d)

    # Use a fine grid for numerical Lloyd-Max
    n_samples = 200_000
    rng = torch.Generator()
    rng.manual_seed(42)
    samples = torch.randn(n_samples, generator=rng) * sigma
    samples = samples.clamp(-1.0, 1.0)

    # Initialize centroids uniformly
    centroids = torch.linspace(-3 * sigma, 3 * sigma, n_clusters)

    # Lloyd-Max iterations
    for _ in range(100):
        # Assign samples to nearest centroid
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = samples[mask].mean()
            else:
                new_centroids[k] = centroids[k]

        if (new_centroids - centroids).abs().max() < 1e-8:
            break
        centroids = new_centroids

    centroids = centroids.sort().values

    # Boundaries: midpoints between consecutive centroids
    boundaries = torch.zeros(n_clusters + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(1, n_clusters):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return centroids, boundaries


# Cache codebooks to avoid recomputation
_codebook_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_codebook(d: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (d, bits)
    if key not in _codebook_cache:
        _codebook_cache[key] = _lloyd_max_codebook(d, bits)
    return _codebook_cache[key]


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices into uint8 bytes."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    if bits <= 2:
        vals_per_byte = 8 // bits
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(torch.uint8)

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, dtype=torch.uint8) * bits
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack bit-packed indices."""
    batch_shape = packed.shape[:-1]
    if bits <= 2:
        vals_per_byte = 8 // bits
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, dtype=torch.uint8)
    shifts = shifts * bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


class TurboQuantMSE:
    """TurboQuant optimized for MSE (Algorithm 1)."""

    def __init__(self, dim: int, bits: int = 4, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.Pi = _generate_rotation_matrix(dim, seed=seed)
        centroids, boundaries = _get_codebook(dim, bits)
        self.centroids = centroids
        self.decision_boundaries = boundaries[1:-1].contiguous()

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        norms = x.norm(dim=-1)
        x_unit = x / (norms.unsqueeze(-1) + 1e-10)
        y = torch.matmul(x_unit.float(), self.Pi.T)
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
        packed = _pack_indices(indices, self.bits)
        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = torch.matmul(y_hat, self.Pi)
        return x_hat * q.norms.unsqueeze(-1)


class TurboQuantProd:
    """TurboQuant optimized for inner products (Algorithm 2)."""

    def __init__(self, dim: int, bits: int = 4, seed: int = 42):
        self.dim = dim
        self.bits = bits
        assert bits >= 2
        self.mse_quantizer = TurboQuantMSE(dim=dim, bits=bits - 1, seed=seed)
        self.S = _generate_qjl_matrix(dim, seed=seed + 1000)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def _pack_signs(self, projected: torch.Tensor) -> torch.Tensor:
        signs = (projected > 0).to(torch.uint8)
        d = signs.shape[-1]
        if d % 8 != 0:
            signs = F.pad(signs, (0, 8 - d % 8), value=0)
        reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        return (reshaped * powers).sum(dim=-1, dtype=torch.uint8)

    def _unpack_signs(self, packed: torch.Tensor) -> torch.Tensor:
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
        signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :self.dim]
        return 2.0 * signs - 1.0

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        mse_q = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_q)
        residual = x - x_hat
        residual_norms = residual.norm(dim=-1)
        projected = torch.matmul(residual.float(), self.S.T)
        packed_signs = self._pack_signs(projected)
        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)
        signs = self._unpack_signs(q.qjl_signs)
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))
        return x_mse + x_qjl

    def attention_score(self, query: torch.Tensor, quantized_key: ProdQuantized) -> torch.Tensor:
        """Compute <query, key> using TQ compressed keys (unbiased estimator)."""
        mse_q = MSEQuantized(indices=quantized_key.mse_indices,
                             norms=quantized_key.norms, bits=quantized_key.mse_bits)
        k_mse = self.mse_quantizer.dequantize(mse_q)
        scores_mse = torch.matmul(query.float(), k_mse.float().transpose(-2, -1))

        q_sketched = torch.matmul(query.float(), self.S.T)
        signs = self._unpack_signs(quantized_key.qjl_signs)
        scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1))
        scores_qjl = scores_qjl * (self.qjl_scale * quantized_key.residual_norms.unsqueeze(-2))

        return scores_mse + scores_qjl.to(scores_mse.dtype)


# ---------------------------------------------------------------------------
# Storage calculation helpers
# ---------------------------------------------------------------------------

def tq_mse_bytes_per_token(d: int, bits: int) -> int:
    """Storage for TQ_MSE compressed nope + BF16 rope."""
    packed_bytes = (d * bits + 7) // 8  # bit-packed indices
    norm_bytes = 2  # FP16 L2 norm
    rope_bytes = D_ROPE * 2  # BF16 rope
    return packed_bytes + norm_bytes + rope_bytes


def tq_prod_bytes_per_token(d: int, bits: int) -> int:
    """Storage for TQ_Prod compressed nope + BF16 rope."""
    mse_bits = bits - 1
    mse_bytes = (d * mse_bits + 7) // 8
    qjl_bytes = (d + 7) // 8  # 1 bit per dim
    norms_bytes = 4  # FP16 key_norm + FP16 residual_norm
    rope_bytes = D_ROPE * 2
    return mse_bytes + qjl_bytes + norms_bytes + rope_bytes


# ---------------------------------------------------------------------------
# SnapMLA FP8 reference (from test_snapmla_reference.py)
# ---------------------------------------------------------------------------

def _simulate_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 e4m3 quantize→dequantize."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    x_scaled = x / scale
    if hasattr(torch, 'float8_e4m3fn'):
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).float()
    else:
        x_fp8 = (x_scaled.clamp(-FP8_MAX, FP8_MAX) * 8).round() / 8
    return x_fp8 * scale, scale.squeeze(-1)


def ref_mla_attention_bf16(
    q: torch.Tensor, c_kv: torch.Tensor, k_rope: torch.Tensor, sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth: BF16 MLA absorbed attention. No quantization."""
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]
    heads_per_kv = h_q // H_KV

    # Build full K: [c_kv | k_rope]
    k_full = torch.cat([c_kv, k_rope], dim=-1)  # [s_kv, d_qk]
    k_exp = k_full.unsqueeze(1).expand(-1, h_q, -1)  # [s_kv, h_q, d_qk]

    # V = c_kv (absorbed MLA)
    v_exp = c_kv.unsqueeze(1).expand(-1, h_q, -1)  # [s_kv, h_q, d_v]

    scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * sm_scale
    lse = torch.logsumexp(scores, dim=-1)
    P = torch.softmax(scores, dim=-1)
    out = torch.einsum('hqk,khd->qhd', P, v_exp.float())

    return out, lse.permute(1, 0)


def ref_snapmla_fp8_attention(
    q: torch.Tensor, c_kv: torch.Tensor, k_rope: torch.Tensor, sm_scale: float,
) -> torch.Tensor:
    """SnapMLA FP8 attention (simplified — key + value from FP8 round-tripped c_kv)."""
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]

    # FP8 round-trip on nope (c_kv)
    c_kv_deq, k_scale = _simulate_fp8(c_kv)

    # Build K and V from FP8-dequanted c_kv + original rope
    k_full = torch.cat([c_kv_deq, k_rope], dim=-1)
    k_exp = k_full.unsqueeze(1).expand(-1, h_q, -1)
    v_exp = c_kv_deq.unsqueeze(1).expand(-1, h_q, -1)

    scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * sm_scale
    P = torch.softmax(scores, dim=-1)
    out = torch.einsum('hqk,khd->qhd', P, v_exp.float())
    return out


# ---------------------------------------------------------------------------
# TurboQuant MLA attention
# ---------------------------------------------------------------------------

def tq_prod_mla_attention(
    q: torch.Tensor, c_kv: torch.Tensor, k_rope: torch.Tensor,
    sm_scale: float, tq: TurboQuantProd,
) -> torch.Tensor:
    """MLA attention using TQ_Prod compressed nope.

    Score path: uses TQ_Prod's unbiased attention_score() for NOPE
                + exact BF16 matmul for ROPE
    Value path: dequantizes c_kv via TQ_Prod inverse for weighted sum
    """
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]

    # Quantize c_kv nope
    tq_compressed = tq.quantize(c_kv)  # [s_kv, ...]

    # --- Score computation ---
    q_nope = q[..., :D_C]   # [s_q, h_q, d_c]
    q_rope = q[..., D_C:]   # [s_q, h_q, d_rope]

    # NOPE scores via TQ unbiased estimator
    # Need c_kv in shape [1, s_kv, d_c] for TQ attention_score batch dim
    # q_nope in shape [h_q, s_q, d_c]
    scores_nope = torch.zeros(h_q, s_q, s_kv)
    for h in range(h_q):
        q_h = q_nope[:, h, :].unsqueeze(0)  # [1, s_q, d_c]
        # TQ attention_score expects query [*, n_q, d], key as ProdQuantized [*, n_k, ...]
        scores_h = tq.attention_score(q_h, tq_compressed)  # [1, s_q, s_kv]
        scores_nope[h] = scores_h.squeeze(0)

    # ROPE scores (exact, no quantization)
    k_rope_exp = k_rope.unsqueeze(1).expand(-1, h_q, -1)  # [s_kv, h_q, d_rope]
    scores_rope = torch.einsum('qhd,khd->hqk', q_rope.float(), k_rope_exp.float())

    scores = (scores_nope + scores_rope) * sm_scale

    # Softmax
    P = torch.softmax(scores, dim=-1)  # [h_q, s_q, s_kv]

    # --- Value reconstruction via TQ dequant ---
    c_kv_deq = tq.dequantize(tq_compressed)  # [s_kv, d_v]
    v_exp = c_kv_deq.unsqueeze(1).expand(-1, h_q, -1)  # [s_kv, h_q, d_v]
    out = torch.einsum('hqk,khd->qhd', P, v_exp.float())

    return out


def tq_mse_mla_attention(
    q: torch.Tensor, c_kv: torch.Tensor, k_rope: torch.Tensor,
    sm_scale: float, tq: TurboQuantMSE,
) -> torch.Tensor:
    """MLA attention using TQ_MSE compressed nope.

    Score path: dequantizes c_kv via TQ_MSE, then standard matmul
                (slightly biased at low bit-widths, negligible at 4-bit)
    Value path: same dequantized c_kv for weighted sum (lower MSE than Prod)
    """
    s_q, h_q, d_qk = q.shape
    s_kv = c_kv.shape[0]

    # Quantize + dequantize nope
    tq_compressed = tq.quantize(c_kv)
    c_kv_deq = tq.dequantize(tq_compressed)  # [s_kv, d_v]

    # Build K from dequanted nope + exact rope
    k_full = torch.cat([c_kv_deq, k_rope], dim=-1)  # [s_kv, d_qk]
    k_exp = k_full.unsqueeze(1).expand(-1, h_q, -1)
    v_exp = c_kv_deq.unsqueeze(1).expand(-1, h_q, -1)

    scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * sm_scale
    P = torch.softmax(scores, dim=-1)
    out = torch.einsum('hqk,khd->qhd', P, v_exp.float())

    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict:
    """Compute accuracy metrics between reference and test outputs."""
    ref_f = ref.float().flatten()
    test_f = test.float().flatten()
    cosine = F.cosine_similarity(ref_f.unsqueeze(0), test_f.unsqueeze(0)).item()
    mse = ((ref_f - test_f) ** 2).mean().item()
    nrmse = math.sqrt(mse) / (ref_f.norm().item() / math.sqrt(ref_f.numel()) + 1e-12)
    max_abs = (ref_f - test_f).abs().max().item()
    return {"cosine": cosine, "mse": mse, "nrmse": nrmse, "max_abs_err": max_abs}


def compute_nope_roundtrip_metrics(c_kv: torch.Tensor, c_kv_deq: torch.Tensor) -> dict:
    """Metrics for nope quantize→dequantize round-trip."""
    cos = F.cosine_similarity(c_kv.flatten().unsqueeze(0),
                              c_kv_deq.flatten().unsqueeze(0)).item()
    mse = ((c_kv - c_kv_deq) ** 2).mean().item()
    significant = c_kv.abs() > 0.01
    if significant.any():
        rel = (c_kv_deq - c_kv).abs()[significant] / c_kv.abs()[significant]
        max_rel = rel.max().item()
        mean_rel = rel.mean().item()
    else:
        max_rel, mean_rel = 0.0, 0.0
    return {"cosine": cos, "mse": mse, "max_rel": max_rel, "mean_rel": mean_rel}


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_nope_roundtrip(verbose=False):
    """Compare nope quantize→dequantize fidelity: FP8 vs TQ_MSE vs TQ_Prod."""
    torch.manual_seed(42)
    n_tokens = 1024
    c_kv = torch.randn(n_tokens, D_C, dtype=torch.float32) * 0.1

    methods = {}

    # FP8 baseline
    c_fp8, _ = _simulate_fp8(c_kv)
    methods["SnapMLA FP8 (8-bit)"] = compute_nope_roundtrip_metrics(c_kv, c_fp8)

    # TQ_MSE 4-bit
    tq_mse = TurboQuantMSE(dim=D_C, bits=4, seed=42)
    q_mse = tq_mse.quantize(c_kv)
    c_mse = tq_mse.dequantize(q_mse)
    methods["TQ_MSE 4-bit"] = compute_nope_roundtrip_metrics(c_kv, c_mse)

    # TQ_MSE 3-bit
    tq_mse3 = TurboQuantMSE(dim=D_C, bits=3, seed=42)
    q_mse3 = tq_mse3.quantize(c_kv)
    c_mse3 = tq_mse3.dequantize(q_mse3)
    methods["TQ_MSE 3-bit"] = compute_nope_roundtrip_metrics(c_kv, c_mse3)

    # TQ_Prod 4-bit (3-bit MSE + 1-bit QJL)
    tq_prod = TurboQuantProd(dim=D_C, bits=4, seed=42)
    q_prod = tq_prod.quantize(c_kv)
    c_prod = tq_prod.dequantize(q_prod)
    methods["TQ_Prod 4-bit (3+1)"] = compute_nope_roundtrip_metrics(c_kv, c_prod)

    # TQ_Prod 3-bit (2-bit MSE + 1-bit QJL)
    tq_prod3 = TurboQuantProd(dim=D_C, bits=3, seed=42)
    q_prod3 = tq_prod3.quantize(c_kv)
    c_prod3 = tq_prod3.dequantize(q_prod3)
    methods["TQ_Prod 3-bit (2+1)"] = compute_nope_roundtrip_metrics(c_kv, c_prod3)

    # TQ_MSE 5-bit
    tq_mse5 = TurboQuantMSE(dim=D_C, bits=5, seed=42)
    c_mse5 = tq_mse5.dequantize(tq_mse5.quantize(c_kv))
    methods["TQ_MSE 5-bit"] = compute_nope_roundtrip_metrics(c_kv, c_mse5)

    # TQ_MSE 6-bit
    tq_mse6 = TurboQuantMSE(dim=D_C, bits=6, seed=42)
    c_mse6 = tq_mse6.dequantize(tq_mse6.quantize(c_kv))
    methods["TQ_MSE 6-bit"] = compute_nope_roundtrip_metrics(c_kv, c_mse6)

    # TQ_Prod 5-bit (4-bit MSE + 1-bit QJL)
    tq_prod5 = TurboQuantProd(dim=D_C, bits=5, seed=42)
    c_prod5 = tq_prod5.dequantize(tq_prod5.quantize(c_kv))
    methods["TQ_Prod 5-bit (4+1)"] = compute_nope_roundtrip_metrics(c_kv, c_prod5)

    if verbose:
        print(f"\n  Nope round-trip quality ({n_tokens} tokens, d={D_C}):")
        print(f"  {'Method':<24s} {'cosine':>10s} {'MSE':>12s} {'max_rel':>10s} {'mean_rel':>10s}")
        print(f"  {'-'*68}")
        for name, m in methods.items():
            print(f"  {name:<24s} {m['cosine']:>10.6f} {m['mse']:>12.6f} "
                  f"{m['max_rel']:>10.4f} {m['mean_rel']:>10.4f}")

    # FP8 is far better than TQ at per-element reconstruction.
    # MLA latent (d=512) is already compressed — less redundancy than standard KV.
    # TQ quality thresholds reflect this reality.
    assert methods["SnapMLA FP8 (8-bit)"]["cosine"] > 0.999
    assert methods["TQ_MSE 4-bit"]["cosine"] > 0.90
    assert methods["TQ_Prod 4-bit (3+1)"]["cosine"] > 0.85

    if verbose:
        # Storage comparison
        print(f"\n  Storage per token:")
        print(f"    SnapMLA FP8:        {644:>4d} B")
        print(f"    TQ_MSE 4-bit:       {tq_mse_bytes_per_token(D_C, 4):>4d} B "
              f"({644 / tq_mse_bytes_per_token(D_C, 4):.2f}x compression)")
        print(f"    TQ_Prod 4-bit (3+1):{tq_prod_bytes_per_token(D_C, 4):>4d} B "
              f"({644 / tq_prod_bytes_per_token(D_C, 4):.2f}x compression)")
        print(f"    TQ_MSE 3-bit:       {tq_mse_bytes_per_token(D_C, 3):>4d} B "
              f"({644 / tq_mse_bytes_per_token(D_C, 3):.2f}x compression)")
        print(f"    TQ_Prod 3-bit (2+1):{tq_prod_bytes_per_token(D_C, 3):>4d} B "
              f"({644 / tq_prod_bytes_per_token(D_C, 3):.2f}x compression)")
        print(f"    TQ_MSE 5-bit:       {tq_mse_bytes_per_token(D_C, 5):>4d} B "
              f"({644 / tq_mse_bytes_per_token(D_C, 5):.2f}x compression)")
        print(f"    TQ_MSE 6-bit:       {tq_mse_bytes_per_token(D_C, 6):>4d} B "
              f"({644 / tq_mse_bytes_per_token(D_C, 6):.2f}x compression)")
        print(f"    TQ_Prod 5-bit (4+1):{tq_prod_bytes_per_token(D_C, 5):>4d} B "
              f"({644 / tq_prod_bytes_per_token(D_C, 5):.2f}x compression)")

    return True


def test_inner_product_bias(verbose=False):
    """Verify TQ_Prod gives unbiased inner products, TQ_MSE is biased at low bits."""
    torch.manual_seed(42)
    n_keys = 512
    n_queries = 64

    c_kv = torch.randn(n_keys, D_C, dtype=torch.float32) * 0.1
    queries = torch.randn(n_queries, D_C, dtype=torch.float32) * 0.1

    # Ground truth inner products
    true_ip = torch.matmul(queries, c_kv.T)  # [n_queries, n_keys]

    results = {}
    for bits in [3, 4]:
        # TQ_MSE
        tq_mse = TurboQuantMSE(dim=D_C, bits=bits, seed=42)
        c_mse = tq_mse.dequantize(tq_mse.quantize(c_kv))
        ip_mse = torch.matmul(queries, c_mse.T)
        bias_mse = (ip_mse - true_ip).mean().item()
        var_mse = ((ip_mse - true_ip) ** 2).mean().item()
        results[f"TQ_MSE {bits}b"] = {"bias": bias_mse, "variance": var_mse}

        # TQ_Prod
        tq_prod = TurboQuantProd(dim=D_C, bits=bits, seed=42)
        compressed = tq_prod.quantize(c_kv)
        c_prod = tq_prod.dequantize(compressed)
        ip_prod = torch.matmul(queries, c_prod.T)
        bias_prod = (ip_prod - true_ip).mean().item()
        var_prod = ((ip_prod - true_ip) ** 2).mean().item()
        results[f"TQ_Prod {bits}b"] = {"bias": bias_prod, "variance": var_prod}

        # TQ_Prod via attention_score (unbiased estimator, no full dequant)
        ip_tq = tq_prod.attention_score(
            queries.unsqueeze(0), compressed  # queries: [1, n_q, d]
        ).squeeze(0)
        bias_tq = (ip_tq - true_ip).mean().item()
        var_tq = ((ip_tq - true_ip) ** 2).mean().item()
        results[f"TQ_Prod {bits}b (attn_score)"] = {"bias": bias_tq, "variance": var_tq}

    if verbose:
        print(f"\n  Inner product bias analysis (d={D_C}, {n_queries} queries × {n_keys} keys):")
        print(f"  {'Method':<30s} {'bias':>12s} {'variance':>12s} {'|bias|/var':>12s}")
        print(f"  {'-'*68}")
        for name, m in results.items():
            ratio = abs(m["bias"]) / (m["variance"] + 1e-15)
            print(f"  {name:<30s} {m['bias']:>12.6f} {m['variance']:>12.6f} {ratio:>12.4f}")

    # TQ_Prod attention_score should have near-zero bias
    assert abs(results["TQ_Prod 4b (attn_score)"]["bias"]) < 0.01
    return True


def _run_accuracy_at_context(s_kv: int, verbose: bool = False) -> dict:
    """Run all attention variants at given context length, return metrics."""
    torch.manual_seed(42)
    s_q = 1
    sm_scale = 1.0 / math.sqrt(D_QK)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    # Ground truth
    out_bf16, lse_bf16 = ref_mla_attention_bf16(q, c_kv, k_rope, sm_scale)

    results = {}

    # SnapMLA FP8
    out_fp8 = ref_snapmla_fp8_attention(q, c_kv, k_rope, sm_scale)
    results["SnapMLA FP8"] = compute_metrics(out_bf16, out_fp8)

    # TQ_MSE 4-bit
    tq_mse4 = TurboQuantMSE(dim=D_C, bits=4, seed=42)
    out_mse4 = tq_mse_mla_attention(q, c_kv, k_rope, sm_scale, tq_mse4)
    results["TQ_MSE 4-bit"] = compute_metrics(out_bf16, out_mse4)

    # TQ_Prod 4-bit
    tq_prod4 = TurboQuantProd(dim=D_C, bits=4, seed=42)
    out_prod4 = tq_prod_mla_attention(q, c_kv, k_rope, sm_scale, tq_prod4)
    results["TQ_Prod 4-bit (3+1)"] = compute_metrics(out_bf16, out_prod4)

    # TQ_MSE 3-bit
    tq_mse3 = TurboQuantMSE(dim=D_C, bits=3, seed=42)
    out_mse3 = tq_mse_mla_attention(q, c_kv, k_rope, sm_scale, tq_mse3)
    results["TQ_MSE 3-bit"] = compute_metrics(out_bf16, out_mse3)

    # TQ_Prod 3-bit
    tq_prod3 = TurboQuantProd(dim=D_C, bits=3, seed=42)
    out_prod3 = tq_prod_mla_attention(q, c_kv, k_rope, sm_scale, tq_prod3)
    results["TQ_Prod 3-bit (2+1)"] = compute_metrics(out_bf16, out_prod3)

    # TQ_MSE 5-bit
    tq_mse5 = TurboQuantMSE(dim=D_C, bits=5, seed=42)
    out_mse5 = tq_mse_mla_attention(q, c_kv, k_rope, sm_scale, tq_mse5)
    results["TQ_MSE 5-bit"] = compute_metrics(out_bf16, out_mse5)

    # TQ_MSE 6-bit
    tq_mse6 = TurboQuantMSE(dim=D_C, bits=6, seed=42)
    out_mse6 = tq_mse_mla_attention(q, c_kv, k_rope, sm_scale, tq_mse6)
    results["TQ_MSE 6-bit"] = compute_metrics(out_bf16, out_mse6)

    # TQ_Prod 5-bit (4-bit MSE + 1-bit QJL)
    tq_prod5 = TurboQuantProd(dim=D_C, bits=5, seed=42)
    out_prod5 = tq_prod_mla_attention(q, c_kv, k_rope, sm_scale, tq_prod5)
    results["TQ_Prod 5-bit (4+1)"] = compute_metrics(out_bf16, out_prod5)

    return results


def test_decode_attention_short(verbose=False):
    """MLA decode attention quality at 256 tokens — all variants vs BF16."""
    results = _run_accuracy_at_context(256, verbose)

    if verbose:
        print(f"\n  Decode attention quality (s_kv=256):")
        _print_results_table(results)

    # At short context (256 tokens), TQ quality is lower because softmax
    # concentrates on fewer tokens — individual quantization errors matter more.
    # Quality improves dramatically at longer contexts (see scaling test).
    assert results["SnapMLA FP8"]["cosine"] > 0.99
    assert results["TQ_MSE 4-bit"]["cosine"] > 0.80
    assert results["TQ_Prod 4-bit (3+1)"]["cosine"] > 0.78
    return True


def test_decode_attention_scaling(verbose=False):
    """MLA decode attention quality vs context length: 256 → 32K."""
    context_lengths = [256, 1024, 4096, 16384, 32768]

    methods_short = ["SnapMLA FP8", "TQ_MSE 6b", "TQ_MSE 5b", "TQ_MSE 4b",
                      "TQ_Prod 5b", "TQ_Prod 4b", "TQ_MSE 3b", "TQ_Prod 3b"]
    methods_full = ["SnapMLA FP8", "TQ_MSE 6-bit", "TQ_MSE 5-bit", "TQ_MSE 4-bit",
                    "TQ_Prod 5-bit (4+1)", "TQ_Prod 4-bit (3+1)",
                    "TQ_MSE 3-bit", "TQ_Prod 3-bit (2+1)"]

    if verbose:
        print(f"\n  Decode attention cosine similarity vs context length:")
        header = f"  {'s_kv':>8s}"
        for m in methods_short:
            header += f"  {m:>14s}"
        print(header)
        print(f"  {'-' * (8 + len(methods_short) * 16)}")

    all_results = {}
    for s_kv in context_lengths:
        t0 = time.time()
        results = _run_accuracy_at_context(s_kv)
        dt = time.time() - t0
        all_results[s_kv] = results

        if verbose:
            row = f"  {s_kv:>8d}"
            for method in methods_full:
                row += f"  {results[method]['cosine']:>14.6f}"
            row += f"  ({dt:.1f}s)"
            print(row)

    # TQ quality improves with context length (softmax averaging).
    # At 4K+ tokens, TQ_MSE 4-bit reaches >0.96 cosine.
    for s_kv, results in all_results.items():
        if s_kv >= 4096:
            assert results["TQ_MSE 4-bit"]["cosine"] > 0.96, \
                f"TQ_MSE 4-bit cosine {results['TQ_MSE 4-bit']['cosine']:.4f} < 0.96 at s_kv={s_kv}"
        else:
            # Short context: lower bar
            assert results["TQ_MSE 4-bit"]["cosine"] > 0.80, \
                f"TQ_MSE 4-bit cosine {results['TQ_MSE 4-bit']['cosine']:.4f} < 0.80 at s_kv={s_kv}"

    return True


def test_decode_attention_long(verbose=False):
    """MLA decode attention quality at 80K context — stress test.

    This is the expensive test. Run with --long flag.
    """
    s_kv = 80_000
    t0 = time.time()

    if verbose:
        print(f"\n  Long context test (s_kv={s_kv:,d})...")

    results = _run_accuracy_at_context(s_kv)
    dt = time.time() - t0

    if verbose:
        print(f"  Completed in {dt:.1f}s")
        _print_results_table(results)

    # At 80K, softmax averaging should give strong results
    assert results["TQ_MSE 4-bit"]["cosine"] > 0.98, \
        f"TQ_MSE 4-bit cosine {results['TQ_MSE 4-bit']['cosine']:.4f} < 0.98 at s_kv={s_kv}"
    return True


def test_sparse_attention_accuracy(verbose=False):
    """Sparse MLA attention: TQ on top-K selected tokens vs BF16 on same tokens."""
    torch.manual_seed(42)
    s_kv = 8192
    s_q = 1
    topk = 2048
    sm_scale = 1.0 / math.sqrt(D_QK)

    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    c_kv = torch.randn(s_kv, D_C, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    # Oracle top-K selection (BF16 scores, averaged across heads)
    k_full = torch.cat([c_kv, k_rope], dim=-1)
    scores_full = torch.einsum('qhd,kd->hqk', q.float(), k_full.float()) * sm_scale
    importance = scores_full.mean(dim=0).squeeze(0)  # [s_kv]
    _, topk_idx = importance.topk(topk)
    topk_idx = topk_idx.sort().values

    # Extract top-K tokens
    c_kv_topk = c_kv[topk_idx]
    k_rope_topk = k_rope[topk_idx]

    # BF16 ground truth on top-K
    out_bf16, _ = ref_mla_attention_bf16(q, c_kv_topk, k_rope_topk, sm_scale)

    results = {}

    # SnapMLA FP8 on top-K
    out_fp8 = ref_snapmla_fp8_attention(q, c_kv_topk, k_rope_topk, sm_scale)
    results["SnapMLA FP8 (sparse)"] = compute_metrics(out_bf16, out_fp8)

    # TQ_MSE 4-bit on top-K
    tq_mse4 = TurboQuantMSE(dim=D_C, bits=4, seed=42)
    out_mse4 = tq_mse_mla_attention(q, c_kv_topk, k_rope_topk, sm_scale, tq_mse4)
    results["TQ_MSE 4-bit (sparse)"] = compute_metrics(out_bf16, out_mse4)

    # TQ_Prod 4-bit on top-K
    tq_prod4 = TurboQuantProd(dim=D_C, bits=4, seed=42)
    out_prod4 = tq_prod_mla_attention(q, c_kv_topk, k_rope_topk, sm_scale, tq_prod4)
    results["TQ_Prod 4-bit (sparse)"] = compute_metrics(out_bf16, out_prod4)

    if verbose:
        print(f"\n  Sparse attention quality (s_kv={s_kv}, topk={topk}):")
        _print_results_table(results)

    # Sparse attention is TQ's sweet spot — fewer tokens, more relevant ones
    assert results["TQ_MSE 4-bit (sparse)"]["cosine"] > 0.99
    return True


def test_value_reconstruction_through_projection(verbose=False):
    """Test value quality through W_UV projection (simulates MLA output path).

    In MLA, the value path is: softmax_weights @ c @ W_UV^T
    where W_UV is a (d_c × v_head_dim) projection per head.
    We test whether TQ reconstruction errors are amplified or attenuated
    by this projection.
    """
    torch.manual_seed(42)
    n_tokens = 4096
    v_head_dim = 128  # typical per-head value dimension

    c_kv = torch.randn(n_tokens, D_C, dtype=torch.float32) * 0.1

    # Simulate random W_UV projection (one head)
    W_UV = torch.randn(D_C, v_head_dim, dtype=torch.float32) * (1.0 / math.sqrt(D_C))

    # Ground truth projected values
    v_true = torch.matmul(c_kv, W_UV)  # [n_tokens, v_head_dim]

    results = {}

    # FP8
    c_fp8, _ = _simulate_fp8(c_kv)
    v_fp8 = torch.matmul(c_fp8, W_UV)
    cos_fp8 = F.cosine_similarity(v_true.flatten().unsqueeze(0),
                                   v_fp8.flatten().unsqueeze(0)).item()
    results["SnapMLA FP8"] = cos_fp8

    # TQ_MSE 4-bit
    tq_mse4 = TurboQuantMSE(dim=D_C, bits=4, seed=42)
    c_mse4 = tq_mse4.dequantize(tq_mse4.quantize(c_kv))
    v_mse4 = torch.matmul(c_mse4, W_UV)
    cos_mse4 = F.cosine_similarity(v_true.flatten().unsqueeze(0),
                                    v_mse4.flatten().unsqueeze(0)).item()
    results["TQ_MSE 4-bit"] = cos_mse4

    # TQ_Prod 4-bit
    tq_prod4 = TurboQuantProd(dim=D_C, bits=4, seed=42)
    c_prod4 = tq_prod4.dequantize(tq_prod4.quantize(c_kv))
    v_prod4 = torch.matmul(c_prod4, W_UV)
    cos_prod4 = F.cosine_similarity(v_true.flatten().unsqueeze(0),
                                     v_prod4.flatten().unsqueeze(0)).item()
    results["TQ_Prod 4-bit (3+1)"] = cos_prod4

    # TQ_MSE 3-bit
    tq_mse3 = TurboQuantMSE(dim=D_C, bits=3, seed=42)
    c_mse3 = tq_mse3.dequantize(tq_mse3.quantize(c_kv))
    v_mse3 = torch.matmul(c_mse3, W_UV)
    cos_mse3 = F.cosine_similarity(v_true.flatten().unsqueeze(0),
                                    v_mse3.flatten().unsqueeze(0)).item()
    results["TQ_MSE 3-bit"] = cos_mse3

    # TQ_Prod 3-bit
    tq_prod3 = TurboQuantProd(dim=D_C, bits=3, seed=42)
    c_prod3 = tq_prod3.dequantize(tq_prod3.quantize(c_kv))
    v_prod3 = torch.matmul(c_prod3, W_UV)
    cos_prod3 = F.cosine_similarity(v_true.flatten().unsqueeze(0),
                                     v_prod3.flatten().unsqueeze(0)).item()
    results["TQ_Prod 3-bit (2+1)"] = cos_prod3

    if verbose:
        print(f"\n  Value reconstruction through W_UV projection ({n_tokens} tokens, "
              f"d_c={D_C} → v_head_dim={v_head_dim}):")
        print(f"  {'Method':<24s} {'projected cosine':>16s}")
        print(f"  {'-'*42}")
        for name, cos in results.items():
            print(f"  {name:<24s} {cos:>16.6f}")

    # W_UV projection (512→128) averages errors — quality should be decent
    assert results["TQ_MSE 4-bit"] > 0.97
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_results_table(results: dict):
    print(f"  {'Method':<28s} {'cosine':>10s} {'NRMSE':>10s} {'max_abs':>10s}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        print(f"  {name:<28s} {m['cosine']:>10.6f} {m['nrmse']:>10.4f} "
              f"{m['max_abs_err']:>10.6f}")


# ---------------------------------------------------------------------------
# V4 Compressed Distribution Tests (V4K-10a: TQ-FP8-Mix accuracy gate)
# ---------------------------------------------------------------------------


def _make_csa_vectors(n_entries: int, d: int = D_C, tokens_per_sum: int = 8) -> torch.Tensor:
    """Generate CSA-compressed vectors: softmax-weighted sums of 8 random tokens."""
    entries = []
    for _ in range(n_entries):
        raw = torch.randn(tokens_per_sum, d) * 0.1
        weights = torch.softmax(torch.randn(tokens_per_sum), dim=0)
        entries.append((raw * weights.unsqueeze(-1)).sum(dim=0))
    return torch.stack(entries)


def _make_hca_vectors(n_entries: int, d: int = D_C, tokens_per_avg: int = 128) -> torch.Tensor:
    """Generate HCA-compressed vectors: averages of 128 random tokens."""
    entries = []
    for _ in range(n_entries):
        raw = torch.randn(tokens_per_avg, d) * 0.1
        entries.append(raw.mean(dim=0))
    return torch.stack(entries)


def test_tq_mse_csa_compressed_accuracy(verbose=False):
    """TQ_MSE 4-bit round-trip on CSA-compressed vectors (8-token softmax sums).

    CSA entries are ~96.9% of V4 compressed tokens. This validates TQ
    accuracy on the dominant distribution.
    """
    torch.manual_seed(42)
    n_entries = 4096
    tq = TurboQuantMSE(dim=D_C, bits=4, seed=42)

    csa = _make_csa_vectors(n_entries)
    q_csa = tq.quantize(csa)
    csa_deq = tq.dequantize(q_csa)

    csa_metrics = compute_nope_roundtrip_metrics(csa, csa_deq)

    # Also test attention quality with CSA vectors
    s_kv = n_entries
    s_q = 1
    sm_scale = 1.0 / math.sqrt(D_QK)
    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    out_bf16, _ = ref_mla_attention_bf16(q, csa, k_rope, sm_scale)
    out_tq = tq_mse_mla_attention(q, csa, k_rope, sm_scale, tq)
    attn_metrics = compute_metrics(out_bf16, out_tq)

    if verbose:
        print(f"\n  TQ_MSE 4-bit on CSA-compressed vectors ({n_entries} entries, 8-token sums):")
        print(f"    Round-trip cosine:   {csa_metrics['cosine']:.6f}")
        print(f"    Round-trip MSE:      {csa_metrics['mse']:.8f}")
        print(f"    Attention cosine:    {attn_metrics['cosine']:.6f}")

    assert csa_metrics["cosine"] > 0.90, \
        f"CSA round-trip cosine {csa_metrics['cosine']:.4f} < 0.90"
    assert attn_metrics["cosine"] > 0.96, \
        f"CSA attention cosine {attn_metrics['cosine']:.4f} < 0.96"
    return True


def test_tq_mse_hca_compressed_accuracy(verbose=False):
    """TQ_MSE 4-bit round-trip on HCA-compressed vectors (128-token averages).

    HCA entries are ~3.1% of V4 compressed tokens. 128-token averages are
    smoother (CLT → more Gaussian), fitting the codebook well. This validates
    that TQ works on HCA, though FP8 is preferred for the Mix path.
    """
    torch.manual_seed(42)
    n_entries = 1024
    tq = TurboQuantMSE(dim=D_C, bits=4, seed=42)

    hca = _make_hca_vectors(n_entries)
    q_hca = tq.quantize(hca)
    hca_deq = tq.dequantize(q_hca)

    hca_metrics = compute_nope_roundtrip_metrics(hca, hca_deq)

    # Attention quality with HCA vectors
    s_kv = n_entries
    s_q = 1
    sm_scale = 1.0 / math.sqrt(D_QK)
    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.float32) * 0.1
    k_rope = torch.randn(s_kv, D_ROPE, dtype=torch.float32) * 0.1

    out_bf16, _ = ref_mla_attention_bf16(q, hca, k_rope, sm_scale)
    out_tq = tq_mse_mla_attention(q, hca, k_rope, sm_scale, tq)
    attn_metrics = compute_metrics(out_bf16, out_tq)

    if verbose:
        print(f"\n  TQ_MSE 4-bit on HCA-compressed vectors ({n_entries} entries, 128-token avgs):")
        print(f"    Round-trip cosine:   {hca_metrics['cosine']:.6f}")
        print(f"    Round-trip MSE:      {hca_metrics['mse']:.8f}")
        print(f"    Attention cosine:    {attn_metrics['cosine']:.6f}")

        # Compare CSA vs HCA
        csa = _make_csa_vectors(n_entries)
        csa_deq = tq.dequantize(tq.quantize(csa))
        csa_cos = compute_nope_roundtrip_metrics(csa, csa_deq)["cosine"]
        print(f"    CSA round-trip (for comparison): {csa_cos:.6f}")
        print(f"    HCA quantizes {'better' if hca_metrics['cosine'] > csa_cos else 'worse'} "
              f"than CSA ({hca_metrics['cosine']:.4f} vs {csa_cos:.4f})")

    assert hca_metrics["cosine"] > 0.90, \
        f"HCA round-trip cosine {hca_metrics['cosine']:.4f} < 0.90"
    assert attn_metrics["cosine"] > 0.96, \
        f"HCA attention cosine {attn_metrics['cosine']:.4f} < 0.96"
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("Nope round-trip quality", test_nope_roundtrip),
    ("Inner product bias", test_inner_product_bias),
    ("Decode attention (256 tokens)", test_decode_attention_short),
    ("Decode attention scaling (256→32K)", test_decode_attention_scaling),
    ("Sparse attention (8K, topk=2048)", test_sparse_attention_accuracy),
    ("Value reconstruction through W_UV", test_value_reconstruction_through_projection),
    ("TQ_MSE on CSA-compressed vectors", test_tq_mse_csa_compressed_accuracy),
    ("TQ_MSE on HCA-compressed vectors", test_tq_mse_hca_compressed_accuracy),
]

LONG_TESTS = [
    ("Decode attention (80K context)", test_decode_attention_long),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant MLA accuracy validation")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--long", action="store_true", help="Include 80K context test")
    args = parser.parse_args()

    tests = ALL_TESTS + (LONG_TESTS if args.long else [])

    print("=" * 70)
    print("TurboQuant MLA Accuracy Validation")
    print("=" * 70)

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            t0 = time.time()
            ok = fn(verbose=args.verbose)
            dt = time.time() - t0
            status = "PASS" if ok else "FAIL"
            if not ok:
                failed += 1
            else:
                passed += 1
            print(f"  [{status}] {name} ({dt:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
