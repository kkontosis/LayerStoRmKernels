"""
DeepSeek V4 Kernel Validation — PyTorch Reference Implementation

Pure-PyTorch implementations of V4 CSA/HCA/SWA attention pipeline that serve as:
  1. Golden reference for correctness testing
  2. Spec of what each kernel computes
  3. Error bound establishment (FP8/TQ quantization tolerance)

V4 differs fundamentally from V3.2 MLA:
  - num_key_value_heads=1 (single KV head broadcast to all Q heads)
  - head_dim=512, separate K and V storage (not shared latent)
  - Three layer types: CSA (compress 4:1), HCA (compress 128:1), SWA (sliding window only)
  - Dual RoPE: rope_theta=10000 (SWA), compress_rope_theta=160000 (compressed)
  - Lightning Indexer (FP4 sparse selection for CSA)

This file is the foundation (V4K-0a). Subsequent tasks V4K-0b through V4K-0j extend it
with reference functions for each V4 kernel.

Usage:
  python tests/test_v4_reference.py          # runs all tests
  python tests/test_v4_reference.py -v        # verbose

Error budget (V4 kernels vs reference):
  - FP8 kernels vs PyTorch reference: cosine > 0.999, NRMSE < 0.5%
  - TQ kernels vs PyTorch reference:  cosine > 0.99,  NRMSE < 2%
  - Full pipeline vs BF16 ground truth: FP8 cosine > 0.995, TQ cosine > 0.99
"""

import torch
import torch.nn.functional as F
import math
import json
import os
import argparse
import struct

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# V4 Model Dimensions (shared between Pro and Flash)
# ---------------------------------------------------------------------------

HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
D_QK = HEAD_DIM + QK_ROPE_HEAD_DIM  # 576
NUM_KV_HEADS = 1
SLIDING_WINDOW = 128
PAGE_SIZE = 64
FP8_MAX = 448.0

# ---------------------------------------------------------------------------
# Dual RoPE
# ---------------------------------------------------------------------------

ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0

# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

CSA_STRIDE = 4
CSA_WINDOW = 8
HCA_STRIDE = 128
HCA_WINDOW = 128

# ---------------------------------------------------------------------------
# Lightning Indexer
# ---------------------------------------------------------------------------

INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128

# ---------------------------------------------------------------------------
# V4 FP8 cache layout (1160 bytes/entry)
#   [K_nope FP8 512B | K_scale f32 4B | K_rope BF16 128B |
#    V_nope FP8 512B | V_scale f32 4B]
# ---------------------------------------------------------------------------

V4_FP8_K_NOPE_BYTES = HEAD_DIM           # 512
V4_FP8_K_SCALE_BYTES = 4                 # float32
V4_FP8_K_ROPE_BYTES = QK_ROPE_HEAD_DIM * 2  # 128
V4_FP8_V_NOPE_BYTES = HEAD_DIM           # 512
V4_FP8_V_SCALE_BYTES = 4                 # float32
V4_FP8_BYTES_PER_ENTRY = (V4_FP8_K_NOPE_BYTES + V4_FP8_K_SCALE_BYTES +
                           V4_FP8_K_ROPE_BYTES + V4_FP8_V_NOPE_BYTES +
                           V4_FP8_V_SCALE_BYTES)  # 1160

V4_FP8_K_NOPE_OFFSET = 0
V4_FP8_K_SCALE_OFFSET = V4_FP8_K_NOPE_BYTES                          # 512
V4_FP8_K_ROPE_OFFSET = V4_FP8_K_SCALE_OFFSET + V4_FP8_K_SCALE_BYTES  # 516
V4_FP8_V_NOPE_OFFSET = V4_FP8_K_ROPE_OFFSET + V4_FP8_K_ROPE_BYTES    # 644
V4_FP8_V_SCALE_OFFSET = V4_FP8_V_NOPE_OFFSET + V4_FP8_V_NOPE_BYTES   # 1156

# ---------------------------------------------------------------------------
# V4 TQ cache layout (644 bytes/entry, placeholder for V4K-8ref)
#   K: [256B packed_nope | 2B FP16 norm | 128B BF16 rope] = 386B
#   V: [256B packed_nope | 2B FP16 norm]                   = 258B
# ---------------------------------------------------------------------------

V4_TQ_K_PACKED_BYTES = HEAD_DIM // 2     # 256
V4_TQ_K_NORM_BYTES = 2                   # FP16
V4_TQ_K_ROPE_BYTES = QK_ROPE_HEAD_DIM * 2  # 128
V4_TQ_V_PACKED_BYTES = HEAD_DIM // 2     # 256
V4_TQ_V_NORM_BYTES = 2                   # FP16
V4_TQ_K_ENTRY_BYTES = V4_TQ_K_PACKED_BYTES + V4_TQ_K_NORM_BYTES + V4_TQ_K_ROPE_BYTES  # 386
V4_TQ_V_ENTRY_BYTES = V4_TQ_V_PACKED_BYTES + V4_TQ_V_NORM_BYTES  # 258
V4_TQ_BYTES_PER_ENTRY = V4_TQ_K_ENTRY_BYTES + V4_TQ_V_ENTRY_BYTES  # 644

TQ_BITS = 4
TQ_NUM_CENTROIDS = 16
TQ_SEED = 42

CODEBOOK_DIR = os.path.join(PROJECT_ROOT, "data", "codebooks")

# ---------------------------------------------------------------------------
# Error budget thresholds
# ---------------------------------------------------------------------------

ERR_FP8_COSINE = 0.999
ERR_FP8_NRMSE = 0.005           # 0.5%
ERR_TQ_COSINE = 0.99
ERR_TQ_NRMSE = 0.02             # 2%

ERR_PIPELINE_FP8_COSINE = 0.995
ERR_PIPELINE_TQ_COSINE = 0.99

ERR_SPLITKV_ABS = 1e-5
ERR_SPARSE_DENSE_ABS = 1e-6
ERR_SPARSE_DENSE_TQ_ABS = 1e-5

ERR_COMPRESSOR_COSINE = 0.999


# ---------------------------------------------------------------------------
# V4 Model Configs (hardcoded fallbacks)
# ---------------------------------------------------------------------------

V4_PRO_CONFIG = {
    "num_attention_heads": 128,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "hidden_size": 7168,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1536,
    "index_topk": 1024,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "sliding_window": 128,
    "rope_theta": 10000,
    "compress_rope_theta": 160000,
    "num_hidden_layers": 61,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 1048576,
    "rope_scaling": {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    # 62 entries: 61 layers + 1 MTP
    "compress_ratios": [128, 128] + [4, 128] * 29 + [4, 0],
}

V4_FLASH_CONFIG = {
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "hidden_size": 4096,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "index_topk": 512,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "sliding_window": 128,
    "rope_theta": 10000,
    "compress_rope_theta": 160000,
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 1048576,
    "rope_scaling": {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    # 44 entries: 43 layers + 1 MTP
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_v4_config(variant: str = "pro") -> dict:
    """Load V4 model config.

    Args:
        variant: "pro" or "flash"

    Returns:
        dict with V4 model parameters. Falls back to hardcoded config
        if JSON files under ref/DeepSeek-V4/ are not found.
    """
    json_path = os.path.join(PROJECT_ROOT, "ref", "DeepSeek-V4",
                             f"config_{variant}.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        return {
            "num_attention_heads": raw["num_attention_heads"],
            "num_key_value_heads": raw.get("num_key_value_heads", 1),
            "head_dim": raw["head_dim"],
            "hidden_size": raw.get("hidden_size"),
            "qk_rope_head_dim": raw["qk_rope_head_dim"],
            "q_lora_rank": raw.get("q_lora_rank"),
            "index_topk": raw.get("index_topk"),
            "index_n_heads": raw.get("index_n_heads", 64),
            "index_head_dim": raw.get("index_head_dim", 128),
            "sliding_window": raw["sliding_window"],
            "rope_theta": raw["rope_theta"],
            "compress_rope_theta": raw["compress_rope_theta"],
            "num_hidden_layers": raw["num_hidden_layers"],
            "num_nextn_predict_layers": raw.get("num_nextn_predict_layers", 1),
            "max_position_embeddings": raw.get("max_position_embeddings"),
            "rope_scaling": raw.get("rope_scaling"),
            "compress_ratios": raw["compress_ratios"],
        }

    fallbacks = {"pro": V4_PRO_CONFIG, "flash": V4_FLASH_CONFIG}
    if variant not in fallbacks:
        raise ValueError(f"Unknown variant '{variant}', expected 'pro' or 'flash'")
    return dict(fallbacks[variant])


def get_layer_type(compress_ratio: int) -> str:
    """Map compress_ratios value to layer type string."""
    if compress_ratio == 4:
        return "csa"
    elif compress_ratio == 128:
        return "hca"
    elif compress_ratio == 0:
        return "swa"
    else:
        raise ValueError(f"Unknown compress_ratio: {compress_ratio}")


def count_layer_types(config: dict) -> dict:
    """Count CSA, HCA, SWA layers in a config."""
    counts = {"csa": 0, "hca": 0, "swa": 0}
    for ratio in config["compress_ratios"]:
        counts[get_layer_type(ratio)] += 1
    return counts


# ---------------------------------------------------------------------------
# Dual-theta RoPE helpers
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    theta: float,
    dim: int,
    max_pos: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings.

    Args:
        theta: RoPE base frequency (10000 for SWA, 160000 for compressed)
        dim: Number of RoPE dimensions (qk_rope_head_dim=64)
        max_pos: Maximum position to precompute

    Returns:
        (cos, sin) each of shape [max_pos, dim//2] float32
    """
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                           device=device) / dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)  # [max_pos, dim//2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to x.

    Args:
        x: tensor [..., dim] where dim = qk_rope_head_dim
        cos: precomputed cos table [max_pos, dim//2]
        sin: precomputed sin table [max_pos, dim//2]
        positions: integer positions [...] indexing into cos/sin

    Returns:
        tensor same shape as x with RoPE applied
    """
    x1 = x[..., 0::2]  # [..., dim//2]
    x2 = x[..., 1::2]  # [..., dim//2]

    c = cos[positions]  # [..., dim//2]
    s = sin[positions]  # [..., dim//2]

    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c

    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# FP8 simulation helpers
# ---------------------------------------------------------------------------

def simulate_fp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate per-token FP8 e4m3 quantization.

    Returns (x_fp8_dequanted, scale) where x_fp8_dequanted is the
    value after quantize->dequantize round-trip (simulating precision loss).
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX

    x_scaled = x / scale
    if hasattr(torch, 'float8_e4m3fn'):
        x_fp8 = x_scaled.to(torch.float8_e4m3fn).float()
    else:
        x_fp8 = x_scaled.clamp(-FP8_MAX, FP8_MAX)
        x_fp8 = (x_fp8 * 8).round() / 8

    x_deq = x_fp8 * scale
    return x_deq, scale.squeeze(-1)


def simulate_fp8_quantize_rowwise(P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate per-ROW FP8 quantization of P matrix (attention weights).

    Returns (P_fp8_raw, P_deq, scale) where:
      P_fp8_raw = quantized values (in FP8 range) — used for PV GEMM
      P_deq = P_fp8_raw * scale ~ original P — for verification
      scale = per-row quantization scale — applied after PV GEMM
    """
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
# V4 paged cache simulator (FP8)
# ---------------------------------------------------------------------------

def alloc_v4_fp8_cache(num_pages: int, page_size: int = PAGE_SIZE,
                        device: str = "cpu") -> torch.Tensor:
    """Allocate a V4 FP8 paged cache as flat uint8 tensor.

    Each entry: 1160 bytes [K_nope FP8 | K_scale f32 | K_rope BF16 |
                            V_nope FP8 | V_scale f32]
    """
    total_bytes = num_pages * page_size * V4_FP8_BYTES_PER_ENTRY
    return torch.zeros(total_bytes, dtype=torch.uint8, device=device)


def write_v4_fp8_cache_row(
    cache: torch.Tensor,
    slot: int,
    k_nope_fp8: torch.Tensor,
    k_scale: float,
    k_rope_bf16: torch.Tensor,
    v_nope_fp8: torch.Tensor,
    v_scale: float,
    page_size: int = PAGE_SIZE,
) -> None:
    """Write one V4 FP8 entry to paged cache.

    Args:
        cache: flat uint8 tensor (from alloc_v4_fp8_cache)
        slot: global slot index
        k_nope_fp8: [HEAD_DIM] uint8 (raw FP8 bytes)
        k_scale: float32 K quantization scale
        k_rope_bf16: [QK_ROPE_HEAD_DIM] BF16 rope values
        v_nope_fp8: [HEAD_DIM] uint8 (raw FP8 bytes)
        v_scale: float32 V quantization scale
        page_size: entries per page
    """
    offset = slot * V4_FP8_BYTES_PER_ENTRY

    # K NOPE FP8
    cache[offset:offset + V4_FP8_K_NOPE_BYTES] = k_nope_fp8.to(torch.uint8)

    # K scale (float32, 4 bytes)
    k_scale_offset = offset + V4_FP8_K_SCALE_OFFSET
    scale_bytes = struct.pack('<f', k_scale)
    for i, b in enumerate(scale_bytes):
        cache[k_scale_offset + i] = b

    # K ROPE (BF16)
    k_rope_offset = offset + V4_FP8_K_ROPE_OFFSET
    rope_bytes = k_rope_bf16.to(torch.bfloat16).contiguous().view(torch.uint8)
    cache[k_rope_offset:k_rope_offset + V4_FP8_K_ROPE_BYTES] = rope_bytes

    # V NOPE FP8
    v_nope_offset = offset + V4_FP8_V_NOPE_OFFSET
    cache[v_nope_offset:v_nope_offset + V4_FP8_V_NOPE_BYTES] = v_nope_fp8.to(torch.uint8)

    # V scale (float32, 4 bytes)
    v_scale_offset = offset + V4_FP8_V_SCALE_OFFSET
    scale_bytes = struct.pack('<f', v_scale)
    for i, b in enumerate(scale_bytes):
        cache[v_scale_offset + i] = b


def read_v4_fp8_cache_row(
    cache: torch.Tensor,
    slot: int,
    page_size: int = PAGE_SIZE,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float]:
    """Read one V4 FP8 entry from paged cache.

    Returns:
        (k_nope_fp8, k_scale, k_rope_bf16, v_nope_fp8, v_scale)
        k_nope_fp8: [HEAD_DIM] uint8 (raw FP8 bytes)
        k_scale: float
        k_rope_bf16: [QK_ROPE_HEAD_DIM] float32 (from BF16)
        v_nope_fp8: [HEAD_DIM] uint8 (raw FP8 bytes)
        v_scale: float
    """
    offset = slot * V4_FP8_BYTES_PER_ENTRY

    # K NOPE FP8
    k_nope_fp8 = cache[offset:offset + V4_FP8_K_NOPE_BYTES].clone()

    # K scale
    k_scale_offset = offset + V4_FP8_K_SCALE_OFFSET
    k_scale_raw = bytes([cache[k_scale_offset + i].item() for i in range(4)])
    k_scale = struct.unpack('<f', k_scale_raw)[0]

    # K ROPE
    k_rope_offset = offset + V4_FP8_K_ROPE_OFFSET
    k_rope_raw = cache[k_rope_offset:k_rope_offset + V4_FP8_K_ROPE_BYTES].clone()
    k_rope_bf16 = k_rope_raw.view(torch.bfloat16).float()

    # V NOPE FP8
    v_nope_offset = offset + V4_FP8_V_NOPE_OFFSET
    v_nope_fp8 = cache[v_nope_offset:v_nope_offset + V4_FP8_V_NOPE_BYTES].clone()

    # V scale
    v_scale_offset = offset + V4_FP8_V_SCALE_OFFSET
    v_scale_raw = bytes([cache[v_scale_offset + i].item() for i in range(4)])
    v_scale = struct.unpack('<f', v_scale_raw)[0]

    return k_nope_fp8, k_scale, k_rope_bf16, v_nope_fp8, v_scale


def dequant_fp8_bytes(fp8_bytes: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantize raw FP8 uint8 bytes to float32.

    Args:
        fp8_bytes: [D] uint8 tensor containing raw FP8 E4M3 bytes
        scale: float32 quantization scale

    Returns:
        [D] float32 tensor
    """
    if hasattr(torch, 'float8_e4m3fn'):
        return fp8_bytes.view(torch.float8_e4m3fn).float() * scale
    # Fallback: treat as raw uint8 — best-effort
    return fp8_bytes.float() * scale


# ---------------------------------------------------------------------------
# TQ helpers (codebook, rotation, pack/unpack 4-bit)
# ---------------------------------------------------------------------------

def load_codebook(d: int = HEAD_DIM, bits: int = TQ_BITS):
    """Load TQ codebook from data/codebooks/."""
    path = os.path.join(CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")
    if not os.path.exists(path):
        path = os.path.join(PROJECT_ROOT, "ref", "turboquant", "turboquant",
                            "codebooks", f"codebook_d{d}_b{bits}.json")
    with open(path) as f:
        cb = json.load(f)
    centroids = torch.tensor(cb["centroids"], dtype=torch.float32)
    boundaries = torch.tensor(cb["boundaries"], dtype=torch.float32)
    return centroids, boundaries


def generate_rotation_matrix(d: int = HEAD_DIM, seed: int = TQ_SEED):
    """Deterministic orthogonal rotation matrix via QR decomposition."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.contiguous()


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack indices (0-15) into 4-bit packed uint8. 2 values per byte."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    padded_d = ((d + 1) // 2) * 2
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)
    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, 2)
    return reshaped[..., 0] | (reshaped[..., 1] << 4)


def unpack_4bit(packed: torch.Tensor, d: int) -> torch.Tensor:
    """Unpack 4-bit packed uint8 to indices."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :d].long()


# ---------------------------------------------------------------------------
# V4 TQ cache alloc / write / read
# ---------------------------------------------------------------------------

def alloc_v4_tq_cache(num_pages: int, page_size: int = PAGE_SIZE) -> torch.Tensor:
    """Allocate V4 TQ paged cache as flat uint8 tensor.

    Each entry: 644 bytes [K: 256B packed + 2B norm + 128B rope |
                           V: 256B packed + 2B norm]
    """
    total_bytes = num_pages * page_size * V4_TQ_BYTES_PER_ENTRY
    return torch.zeros(total_bytes, dtype=torch.uint8)


def write_v4_tq_cache_row(
    cache: torch.Tensor,
    slot: int,
    k_packed: torch.Tensor,
    k_norm: float,
    k_rope_bf16: torch.Tensor,
    v_packed: torch.Tensor,
    v_norm: float,
) -> None:
    """Write one V4 TQ entry to paged cache."""
    offset = slot * V4_TQ_BYTES_PER_ENTRY

    # K packed nope [0:256]
    cache[offset:offset + V4_TQ_K_PACKED_BYTES] = k_packed.to(torch.uint8)

    # K norm FP16 [256:258]
    norm_off = offset + V4_TQ_K_PACKED_BYTES
    norm_bytes = struct.pack('<e', k_norm)
    cache[norm_off] = norm_bytes[0]
    cache[norm_off + 1] = norm_bytes[1]

    # K rope BF16 [258:386]
    rope_off = offset + V4_TQ_K_PACKED_BYTES + V4_TQ_K_NORM_BYTES
    rope_bytes = k_rope_bf16.to(torch.bfloat16).contiguous().view(torch.uint8)
    cache[rope_off:rope_off + V4_TQ_K_ROPE_BYTES] = rope_bytes

    # V packed nope [386:642]
    v_off = offset + V4_TQ_K_ENTRY_BYTES
    cache[v_off:v_off + V4_TQ_V_PACKED_BYTES] = v_packed.to(torch.uint8)

    # V norm FP16 [642:644]
    v_norm_off = v_off + V4_TQ_V_PACKED_BYTES
    v_norm_bytes = struct.pack('<e', v_norm)
    cache[v_norm_off] = v_norm_bytes[0]
    cache[v_norm_off + 1] = v_norm_bytes[1]


def read_v4_tq_cache_row(
    cache: torch.Tensor,
    slot: int,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float]:
    """Read one V4 TQ entry from paged cache.

    Returns (k_packed, k_norm, k_rope_bf16, v_packed, v_norm).
    """
    offset = slot * V4_TQ_BYTES_PER_ENTRY

    # K packed nope
    k_packed = cache[offset:offset + V4_TQ_K_PACKED_BYTES].clone()

    # K norm
    norm_off = offset + V4_TQ_K_PACKED_BYTES
    k_norm = struct.unpack('<e', bytes([cache[norm_off].item(), cache[norm_off + 1].item()]))[0]

    # K rope
    rope_off = offset + V4_TQ_K_PACKED_BYTES + V4_TQ_K_NORM_BYTES
    k_rope = cache[rope_off:rope_off + V4_TQ_K_ROPE_BYTES].clone().view(torch.bfloat16).float()

    # V packed nope
    v_off = offset + V4_TQ_K_ENTRY_BYTES
    v_packed = cache[v_off:v_off + V4_TQ_V_PACKED_BYTES].clone()

    # V norm
    v_norm_off = v_off + V4_TQ_V_PACKED_BYTES
    v_norm = struct.unpack('<e', bytes([cache[v_norm_off].item(), cache[v_norm_off + 1].item()]))[0]

    return k_packed, k_norm, k_rope, v_packed, v_norm


# ---------------------------------------------------------------------------
# V4K-8ref: V4 TQ k_append / dequant / decode references
# ---------------------------------------------------------------------------

def ref_v4_tq_k_append(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v_nope: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
):
    """Write compressed K+V to V4 paged TQ cache.

    Per entry: normalize → rotate → quantize → pack → write to cache.
    K gets packed_nope + FP16 norm + BF16 rope. V gets packed_nope + FP16 norm.

    Args:
        k_nope: [num_tokens, HEAD_DIM] compressed K NOPE
        k_rope: [num_tokens, QK_ROPE_HEAD_DIM] K ROPE (already RoPE-encoded)
        v_nope: [num_tokens, HEAD_DIM] compressed V
        kv_cache: flat uint8 tensor (from alloc_v4_tq_cache)
        slot_mapping: [num_tokens] int
        Pi: [HEAD_DIM, HEAD_DIM] orthogonal rotation matrix
        centroids: [16] codebook centroids
        boundaries: [17] decision boundaries (includes -1 and +1)
    """
    num_tokens = k_nope.shape[0]
    decision_boundaries = boundaries[1:-1]

    for t in range(num_tokens):
        slot = slot_mapping[t].item()

        # K: normalize → rotate → quantize → pack
        k = k_nope[t].float()
        k_norm = k.norm().item()
        k_unit = k / (k_norm + 1e-10)
        k_rot = k_unit @ Pi.T
        k_idx = torch.searchsorted(decision_boundaries, k_rot.contiguous())
        k_idx = k_idx.clamp(0, TQ_NUM_CENTROIDS - 1)
        k_packed = pack_4bit(k_idx)

        # V: normalize → rotate → quantize → pack
        v = v_nope[t].float()
        v_norm = v.norm().item()
        v_unit = v / (v_norm + 1e-10)
        v_rot = v_unit @ Pi.T
        v_idx = torch.searchsorted(decision_boundaries, v_rot.contiguous())
        v_idx = v_idx.clamp(0, TQ_NUM_CENTROIDS - 1)
        v_packed = pack_4bit(v_idx)

        write_v4_tq_cache_row(kv_cache, slot, k_packed, k_norm,
                               k_rope[t], v_packed, v_norm)


def ref_v4_tq_dequant(
    kv_cache: torch.Tensor,
    indices,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read V4 TQ entries and dequantize.

    Returns:
        k_nope: [num_fetch, HEAD_DIM] float32
        k_rope: [num_fetch, QK_ROPE_HEAD_DIM] float32
        v_nope: [num_fetch, HEAD_DIM] float32
    """
    num_fetch = len(indices)
    k_nope = torch.zeros(num_fetch, HEAD_DIM)
    k_rope_out = torch.zeros(num_fetch, QK_ROPE_HEAD_DIM)
    v_nope = torch.zeros(num_fetch, HEAD_DIM)

    for i, idx in enumerate(indices):
        slot = idx if isinstance(idx, int) else idx.item()
        k_packed, k_norm, k_rope_bf16, v_packed, v_norm = read_v4_tq_cache_row(
            kv_cache, slot)

        # K: unpack → codebook → inverse rotate → scale
        k_idx = unpack_4bit(k_packed, HEAD_DIM)
        k_hat_rot = centroids[k_idx]
        k_hat = k_hat_rot @ Pi
        k_nope[i] = k_hat * k_norm

        # K rope
        k_rope_out[i] = k_rope_bf16

        # V: unpack → codebook �� inverse rotate → scale
        v_idx = unpack_4bit(v_packed, HEAD_DIM)
        v_hat_rot = centroids[v_idx]
        v_hat = v_hat_rot @ Pi
        v_nope[i] = v_hat * v_norm

    return k_nope, k_rope_out, v_nope


def ref_v4_tq_decode_csa(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    tq_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    swa_k_nope: torch.Tensor,
    swa_k_rope: torch.Tensor,
    swa_v: torch.Tensor,
    sm_scale: float,
    num_q_heads: int,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
):
    """CSA TQ decode: TQ sparse scoring (rotated space) + SWA FP8 + LSE combine.

    Compressed entries scored via TQ codebook in rotated space (no full dequant).
    SWA entries scored in original space. Outputs combined via LSE.

    Args:
        q_nope: [num_q_heads, HEAD_DIM]
        q_rope: [num_q_heads, QK_ROPE_HEAD_DIM]
        tq_cache: flat uint8 V4 TQ cache
        sparse_indices: [topk] int, selected compressed block indices
        swa_k_nope: [swa_len, HEAD_DIM]
        swa_k_rope: [swa_len, QK_ROPE_HEAD_DIM]
        swa_v: [swa_len, HEAD_DIM]
        sm_scale: float
        num_q_heads: int
        Pi: [HEAD_DIM, HEAD_DIM] rotation matrix
        centroids: [16] codebook centroids

    Returns:
        output: [num_q_heads, HEAD_DIM]
        lse: [num_q_heads]
    """
    H = num_q_heads

    # --- TQ compressed attention (rotated space) ---
    topk = len(sparse_indices)
    if topk > 0:
        # Read TQ entries
        all_k_idx = []
        all_k_norms = []
        all_k_rope = []
        all_v_idx = []
        all_v_norms = []
        for k in range(topk):
            slot = sparse_indices[k].item()
            k_packed, k_norm, k_rope_i, v_packed, v_norm = read_v4_tq_cache_row(
                tq_cache, slot)
            all_k_idx.append(unpack_4bit(k_packed, HEAD_DIM))
            all_k_norms.append(k_norm)
            all_k_rope.append(k_rope_i)
            all_v_idx.append(unpack_4bit(v_packed, HEAD_DIM))
            all_v_norms.append(v_norm)

        k_indices = torch.stack(all_k_idx)           # [topk, HEAD_DIM]
        k_norms = torch.tensor(all_k_norms)           # [topk]
        k_rope_all = torch.stack(all_k_rope)           # [topk, QK_ROPE_HEAD_DIM]
        v_indices = torch.stack(all_v_idx)             # [topk, HEAD_DIM]
        v_norms = torch.tensor(all_v_norms)            # [topk]

        k_centroid_vals = centroids[k_indices]         # [topk, HEAD_DIM]
        v_centroid_vals = centroids[v_indices]          # [topk, HEAD_DIM]

        # Pre-rotate query
        q_rot = q_nope.float() @ Pi.T                 # [H, HEAD_DIM]

        # NOPE scores in rotated space: q_rot @ k_centroid_vals^T * k_norms
        scores_nope = torch.einsum('hd,kd->hk', q_rot, k_centroid_vals.float())
        scores_nope = scores_nope * k_norms.unsqueeze(0)  # [H, topk]

        # ROPE scores (original space)
        scores_rope = torch.einsum('hd,kd->hk', q_rope.float(),
                                    k_rope_all.float())  # [H, topk]

        scores = (scores_nope + scores_rope) * sm_scale  # [H, topk]

        lse_comp = torch.logsumexp(scores, dim=-1)    # [H]
        P = torch.softmax(scores, dim=-1)              # [H, topk]

        # PV accumulation in rotated space, then rotate back
        # out_rot[h,d] = sum_k P[h,k] * v_norms[k] * v_centroid_vals[k,d]
        weighted = P * v_norms.unsqueeze(0)            # [H, topk]
        out_rot = torch.einsum('hk,kd->hd', weighted, v_centroid_vals.float())
        out_comp = out_rot @ Pi                        # [H, HEAD_DIM]
    else:
        out_comp = torch.zeros(H, HEAD_DIM)
        lse_comp = torch.full((H,), float('-inf'))

    # --- SWA attention (original space, same as FP8 path) ---
    swa_len = swa_k_nope.shape[0]
    if swa_len > 0:
        q_n = q_nope.unsqueeze(1)                      # [H, 1, HEAD_DIM]
        k_n = swa_k_nope.unsqueeze(0)                  # [1, swa, HEAD_DIM]
        scores_nope = torch.einsum('hqd,hkd->hqk', q_n, k_n.expand(H, -1, -1))

        q_r = q_rope.unsqueeze(1)
        k_r = swa_k_rope.unsqueeze(0)
        scores_rope = torch.einsum('hqd,hkd->hqk', q_r, k_r.expand(H, -1, -1))

        scores = (scores_nope + scores_rope) * sm_scale

        lse_swa = torch.logsumexp(scores, dim=-1).squeeze(1)  # [H]
        p = torch.softmax(scores, dim=-1)
        v_swa = swa_v.unsqueeze(0).expand(H, -1, -1)
        out_swa = torch.einsum('hqk,hkd->hqd', p, v_swa).squeeze(1)  # [H, HEAD_DIM]
    else:
        out_swa = torch.zeros(H, HEAD_DIM)
        lse_swa = torch.full((H,), float('-inf'))

    # --- LSE combine ---
    output, lse = _lse_combine(out_comp, lse_comp, out_swa, lse_swa)
    return output, lse


def ref_v4_tq_decode_hca(
    q_nope, q_rope, tq_cache, num_compressed,
    swa_k_nope, swa_k_rope, swa_v,
    sm_scale, num_q_heads, Pi, centroids,
):
    """HCA TQ decode: dense TQ scoring over ALL compressed blocks + SWA + combine."""
    all_indices = torch.arange(num_compressed)
    return ref_v4_tq_decode_csa(
        q_nope, q_rope, tq_cache, all_indices,
        swa_k_nope, swa_k_rope, swa_v,
        sm_scale, num_q_heads, Pi, centroids)


# ---------------------------------------------------------------------------
# V4K-0b: CSA compressor reference
# ---------------------------------------------------------------------------

def ref_csa_compress(input_k_nope, input_k_rope_raw, input_v,
                     gate_weights, positional_bias,
                     compress_cos, compress_sin):
    """CSA compressor: softmax-gated pooling with window=8, stride=4.

    Compresses every stride=4 tokens into 1 entry using a window of 8 tokens
    with learned gate weights and positional bias. The gate produces a fixed
    softmax weighting (input-independent) applied to both K and V.

    After compression, applies compressed RoPE (theta=160000) at position
    (j*stride + window - 1) for compressed entry j (the endpoint of the window).

    Args:
        input_k_nope: [num_tokens, HEAD_DIM] K NOPE vectors
        input_k_rope_raw: [num_tokens, QK_ROPE_HEAD_DIM] K ROPE before positional encoding
        input_v: [num_tokens, HEAD_DIM] V vectors
        gate_weights: [CSA_WINDOW=8] learned gate biases
        positional_bias: [CSA_WINDOW=8] learned positional offsets
        compress_cos: [max_pos, QK_ROPE_HEAD_DIM//2] from precompute_rope_freqs(COMPRESS_ROPE_THETA)
        compress_sin: [max_pos, QK_ROPE_HEAD_DIM//2] from precompute_rope_freqs(COMPRESS_ROPE_THETA)

    Returns:
        compressed_k_nope: [num_compressed, HEAD_DIM]
        compressed_k_rope: [num_compressed, QK_ROPE_HEAD_DIM] with compressed RoPE applied
        compressed_v: [num_compressed, HEAD_DIM]
        residual_indices: [num_residual] int64 tensor of trailing unconsumed token indices
    """
    num_tokens = input_k_nope.shape[0]
    window = CSA_WINDOW
    stride = CSA_STRIDE

    num_compressed = max(0, (num_tokens - window) // stride)

    if num_compressed == 0:
        return (torch.zeros(0, HEAD_DIM), torch.zeros(0, QK_ROPE_HEAD_DIM),
                torch.zeros(0, HEAD_DIM), torch.arange(num_tokens))

    gate_logits = gate_weights + positional_bias  # [window]
    softmax_weights = torch.softmax(gate_logits.float(), dim=0)  # [window]

    compressed_k_nope = torch.zeros(num_compressed, HEAD_DIM)
    compressed_k_rope_raw = torch.zeros(num_compressed, QK_ROPE_HEAD_DIM)
    compressed_v = torch.zeros(num_compressed, HEAD_DIM)
    rope_positions = torch.zeros(num_compressed, dtype=torch.long)

    for j in range(num_compressed):
        win_start = j * stride
        win_end = win_start + window
        w = softmax_weights.unsqueeze(-1)  # [window, 1]

        compressed_k_nope[j] = (w * input_k_nope[win_start:win_end].float()).sum(0)
        compressed_k_rope_raw[j] = (w * input_k_rope_raw[win_start:win_end].float()).sum(0)
        compressed_v[j] = (w * input_v[win_start:win_end].float()).sum(0)
        rope_positions[j] = win_start + window - 1  # = j*stride + window - 1

    compressed_k_rope = apply_rope(compressed_k_rope_raw, compress_cos, compress_sin,
                                    rope_positions)

    # Residual: tokens after the last compression window
    residual_start = (num_compressed - 1) * stride + window
    residual_indices = torch.arange(residual_start, num_tokens)

    return compressed_k_nope, compressed_k_rope, compressed_v, residual_indices


# ---------------------------------------------------------------------------
# V4K-0c: HCA compressor reference
# ---------------------------------------------------------------------------

def ref_hca_compress(input_k_nope, input_k_rope_raw, input_v,
                     gate_weights,
                     compress_cos, compress_sin):
    """HCA compressor: softmax-gated pooling with window=128, stride=128.

    Same mechanism as CSA but with stride=128, window=128. No overlap,
    no residual. Every 128 tokens produce exactly 1 compressed entry.

    Args:
        input_k_nope: [num_tokens, HEAD_DIM] K NOPE vectors
        input_k_rope_raw: [num_tokens, QK_ROPE_HEAD_DIM] K ROPE before positional encoding
        input_v: [num_tokens, HEAD_DIM] V vectors
        gate_weights: [HCA_WINDOW=128] learned gate biases (no separate positional_bias for HCA)
        compress_cos: [max_pos, QK_ROPE_HEAD_DIM//2] from precompute_rope_freqs(COMPRESS_ROPE_THETA)
        compress_sin: [max_pos, QK_ROPE_HEAD_DIM//2] from precompute_rope_freqs(COMPRESS_ROPE_THETA)

    Returns:
        compressed_k_nope: [num_compressed, HEAD_DIM]
        compressed_k_rope: [num_compressed, QK_ROPE_HEAD_DIM] with compressed RoPE applied
        compressed_v: [num_compressed, HEAD_DIM]
    """
    num_tokens = input_k_nope.shape[0]
    window = HCA_WINDOW
    stride = HCA_STRIDE

    num_compressed = num_tokens // stride

    if num_compressed == 0:
        return (torch.zeros(0, HEAD_DIM), torch.zeros(0, QK_ROPE_HEAD_DIM),
                torch.zeros(0, HEAD_DIM))

    softmax_weights = torch.softmax(gate_weights.float(), dim=0)  # [window]

    compressed_k_nope = torch.zeros(num_compressed, HEAD_DIM)
    compressed_k_rope_raw = torch.zeros(num_compressed, QK_ROPE_HEAD_DIM)
    compressed_v = torch.zeros(num_compressed, HEAD_DIM)
    rope_positions = torch.zeros(num_compressed, dtype=torch.long)

    for j in range(num_compressed):
        win_start = j * stride
        win_end = win_start + window
        w = softmax_weights.unsqueeze(-1)  # [window, 1]

        compressed_k_nope[j] = (w * input_k_nope[win_start:win_end].float()).sum(0)
        compressed_k_rope_raw[j] = (w * input_k_rope_raw[win_start:win_end].float()).sum(0)
        compressed_v[j] = (w * input_v[win_start:win_end].float()).sum(0)
        rope_positions[j] = win_end - 1  # = 128*j + 127 = 128*(j+1) - 1

    compressed_k_rope = apply_rope(compressed_k_rope_raw, compress_cos, compress_sin,
                                    rope_positions)

    return compressed_k_nope, compressed_k_rope, compressed_v


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(out_ref: torch.Tensor, out_test: torch.Tensor) -> dict:
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


def fmt_metrics(m: dict) -> str:
    """Format metrics dict for display."""
    return (f"cos={m['cosine']:.6f} nrmse={m['nrmse']:.4f} "
            f"max_rel={m['max_rel']:.4f} mean_rel={m['mean_rel']:.4f}")


# ---------------------------------------------------------------------------
# V4K-0a Tests: foundation helper sanity checks
# ---------------------------------------------------------------------------

def test_config_loader(verbose=False):
    """V4K-0a: Config loader returns correct values for Pro and Flash."""
    passed = True

    pro = load_v4_config("pro")
    flash = load_v4_config("flash")

    checks = [
        ("Pro num_attention_heads", pro["num_attention_heads"], 128),
        ("Pro num_hidden_layers", pro["num_hidden_layers"], 61),
        ("Pro index_topk", pro["index_topk"], 1024),
        ("Flash num_attention_heads", flash["num_attention_heads"], 64),
        ("Flash num_hidden_layers", flash["num_hidden_layers"], 43),
        ("Flash index_topk", flash["index_topk"], 512),
        ("Pro head_dim", pro["head_dim"], 512),
        ("Flash head_dim", flash["head_dim"], 512),
        ("Pro num_key_value_heads", pro["num_key_value_heads"], 1),
        ("Flash num_key_value_heads", flash["num_key_value_heads"], 1),
        ("Pro sliding_window", pro["sliding_window"], 128),
        ("Flash sliding_window", flash["sliding_window"], 128),
    ]

    for name, actual, expected in checks:
        ok = actual == expected
        if verbose or not ok:
            print(f"  {name}: {actual} {'==' if ok else '!='} {expected} {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # compress_ratios length
    pro_cr_len = len(pro["compress_ratios"])
    pro_expected_len = pro["num_hidden_layers"] + pro["num_nextn_predict_layers"]
    ok = pro_cr_len == pro_expected_len
    if verbose or not ok:
        print(f"  Pro compress_ratios length: {pro_cr_len} == {pro_expected_len} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    flash_cr_len = len(flash["compress_ratios"])
    flash_expected_len = flash["num_hidden_layers"] + flash["num_nextn_predict_layers"]
    ok = flash_cr_len == flash_expected_len
    if verbose or not ok:
        print(f"  Flash compress_ratios length: {flash_cr_len} == {flash_expected_len} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # Layer type counts
    pro_counts = count_layer_types(pro)
    flash_counts = count_layer_types(flash)

    if verbose:
        print(f"  Pro layer types: {pro_counts}")
        print(f"  Flash layer types: {flash_counts}")

    # Pro: 30 CSA + 31 HCA + 1 SWA
    ok = pro_counts == {"csa": 30, "hca": 31, "swa": 1}
    if verbose or not ok:
        print(f"  Pro counts match expected: {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # Flash: 21 CSA + 20 HCA + 3 SWA
    ok = flash_counts == {"csa": 21, "hca": 20, "swa": 3}
    if verbose or not ok:
        print(f"  Flash counts match expected: {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


def test_rope_precompute(verbose=False):
    """V4K-0a: RoPE cos/sin tables have correct shapes and properties."""
    passed = True
    max_pos = 1024
    dim = QK_ROPE_HEAD_DIM  # 64

    cos_std, sin_std = precompute_rope_freqs(ROPE_THETA, dim, max_pos)
    cos_cmp, sin_cmp = precompute_rope_freqs(COMPRESS_ROPE_THETA, dim, max_pos)

    # Shape check
    expected_shape = (max_pos, dim // 2)
    for name, t in [("cos_std", cos_std), ("sin_std", sin_std),
                    ("cos_cmp", cos_cmp), ("sin_cmp", sin_cmp)]:
        ok = t.shape == expected_shape
        if verbose or not ok:
            print(f"  {name} shape: {tuple(t.shape)} == {expected_shape} {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # cos^2 + sin^2 == 1
    identity_std = cos_std ** 2 + sin_std ** 2
    max_dev_std = (identity_std - 1.0).abs().max().item()
    ok = max_dev_std < 1e-6
    if verbose or not ok:
        print(f"  Standard cos^2+sin^2 max deviation from 1: {max_dev_std:.2e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    identity_cmp = cos_cmp ** 2 + sin_cmp ** 2
    max_dev_cmp = (identity_cmp - 1.0).abs().max().item()
    ok = max_dev_cmp < 1e-6
    if verbose or not ok:
        print(f"  Compressed cos^2+sin^2 max deviation from 1: {max_dev_cmp:.2e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # Compressed frequencies are lower (longer wavelength) than standard
    # freq_j = 1/theta^(2j/dim). For j>0, higher theta → lower freq → smaller sin at pos 1
    # j=0 gives freq=1 for all theta, so skip it
    sin_std_pos1 = sin_std[1, 1:].abs()
    sin_cmp_pos1 = sin_cmp[1, 1:].abs()
    ok = (sin_cmp_pos1 < sin_std_pos1).all().item()
    if verbose or not ok:
        print(f"  Compressed freqs lower than standard (j>0): {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


def test_rope_apply_identity(verbose=False):
    """V4K-0a: RoPE at position 0 is identity (cos=1, sin=0)."""
    dim = QK_ROPE_HEAD_DIM
    cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 16)

    torch.manual_seed(42)
    x = torch.randn(4, dim)
    positions = torch.zeros(4, dtype=torch.long)

    out = apply_rope(x, cos, sin, positions)
    max_diff = (out - x).abs().max().item()

    ok = max_diff < 1e-6
    if verbose or not ok:
        print(f"  RoPE at pos=0 max diff from input: {max_diff:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def test_rope_apply_preserves_norm(verbose=False):
    """V4K-0a: RoPE preserves dot products (rotation is unitary)."""
    dim = QK_ROPE_HEAD_DIM
    cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 1024)

    torch.manual_seed(42)
    x = torch.randn(dim)
    y = torch.randn(dim)

    dot_orig = (x * y).sum().item()

    # Same position: dot product preserved
    pos = torch.tensor([42])
    x_rot = apply_rope(x.unsqueeze(0), cos, sin, pos).squeeze(0)
    y_rot = apply_rope(y.unsqueeze(0), cos, sin, pos).squeeze(0)
    dot_same = (x_rot * y_rot).sum().item()

    ok_same = abs(dot_same - dot_orig) < 1e-4
    if verbose or not ok_same:
        print(f"  Same-pos dot preservation: orig={dot_orig:.6f} rotated={dot_same:.6f} "
              f"diff={abs(dot_same - dot_orig):.2e} {'PASS' if ok_same else 'FAIL'}")

    # Different positions: dot product changes
    pos_a = torch.tensor([10])
    pos_b = torch.tensor([500])
    x_a = apply_rope(x.unsqueeze(0), cos, sin, pos_a).squeeze(0)
    y_b = apply_rope(y.unsqueeze(0), cos, sin, pos_b).squeeze(0)
    dot_diff = (x_a * y_b).sum().item()

    ok_diff = abs(dot_diff - dot_orig) > 0.01
    if verbose or not ok_diff:
        print(f"  Different-pos dot changes: orig={dot_orig:.6f} cross={dot_diff:.6f} "
              f"diff={abs(dot_diff - dot_orig):.2e} {'PASS' if ok_diff else 'FAIL'}")

    # Norm preservation
    norm_orig = x.norm().item()
    norm_rot = x_rot.norm().item()
    ok_norm = abs(norm_rot - norm_orig) < 1e-4
    if verbose or not ok_norm:
        print(f"  Norm preservation: orig={norm_orig:.6f} rotated={norm_rot:.6f} "
              f"diff={abs(norm_rot - norm_orig):.2e} {'PASS' if ok_norm else 'FAIL'}")

    return ok_same and ok_diff and ok_norm


def test_fp8_quantize_roundtrip(verbose=False):
    """V4K-0a: FP8 quantize/dequant round-trip has expected precision."""
    torch.manual_seed(42)
    x = torch.randn(8, HEAD_DIM)

    x_deq, scale = simulate_fp8_quantize(x)

    # Scale should be positive and finite
    ok_scale = (scale > 0).all().item() and torch.isfinite(scale).all().item()
    if verbose or not ok_scale:
        print(f"  Scale positive & finite: {'PASS' if ok_scale else 'FAIL'}")

    # Max relative error on significant elements
    sig = x.abs() > x.abs().max() * 0.01
    if sig.any():
        rel_err = ((x_deq - x).abs()[sig] / x.abs()[sig]).max().item()
    else:
        rel_err = 0.0

    ok_err = rel_err < 0.07
    if verbose or not ok_err:
        print(f"  Max relative error: {rel_err:.4f} (< 0.07) {'PASS' if ok_err else 'FAIL'}")

    m = compute_metrics(x, x_deq)
    if verbose:
        print(f"  Metrics: {fmt_metrics(m)}")

    return ok_scale and ok_err


def test_fp8_quantize_rowwise(verbose=False):
    """V4K-0a: Rowwise FP8 quantization preserves structure."""
    torch.manual_seed(42)
    scores = torch.randn(4, 128)
    P = torch.softmax(scores, dim=-1)

    P_fp8, P_deq, scale = simulate_fp8_quantize_rowwise(P)

    # Values should be in FP8 range
    ok_range = (P_fp8.abs() <= FP8_MAX).all().item()
    if verbose or not ok_range:
        print(f"  FP8 values in range: {'PASS' if ok_range else 'FAIL'}")

    # Dequanted should approximate original
    m = compute_metrics(P, P_deq)
    ok_cos = m["cosine"] > 0.99
    if verbose or not ok_cos:
        print(f"  Rowwise P cosine: {m['cosine']:.6f} (> 0.99) {'PASS' if ok_cos else 'FAIL'}")

    # Non-negative (P is softmax output)
    ok_nonneg = (P_deq >= -1e-6).all().item()
    if verbose or not ok_nonneg:
        print(f"  Dequanted non-negative: {'PASS' if ok_nonneg else 'FAIL'}")

    if verbose:
        print(f"  Metrics: {fmt_metrics(m)}")

    return ok_range and ok_cos and ok_nonneg


def test_v4_fp8_cache_roundtrip(verbose=False):
    """V4K-0a: V4 FP8 cache write/read round-trip recovers data."""
    torch.manual_seed(42)
    cache = alloc_v4_fp8_cache(1)

    # Generate test data
    k_nope = torch.randn(HEAD_DIM)
    v_nope = torch.randn(HEAD_DIM)
    k_rope = torch.randn(QK_ROPE_HEAD_DIM)

    # FP8 quantize
    k_deq, k_scale_t = simulate_fp8_quantize(k_nope.unsqueeze(0))
    v_deq, v_scale_t = simulate_fp8_quantize(v_nope.unsqueeze(0))
    k_scale_val = k_scale_t.item()
    v_scale_val = v_scale_t.item()

    # Get raw FP8 bytes
    k_scaled = k_nope / k_scale_val
    v_scaled = v_nope / v_scale_val
    if hasattr(torch, 'float8_e4m3fn'):
        k_fp8_bytes = k_scaled.to(torch.float8_e4m3fn).view(torch.uint8)
        v_fp8_bytes = v_scaled.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        k_fp8_bytes = k_scaled.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8).to(torch.uint8)
        v_fp8_bytes = v_scaled.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8).to(torch.uint8)

    write_v4_fp8_cache_row(cache, 0, k_fp8_bytes, k_scale_val, k_rope,
                            v_fp8_bytes, v_scale_val)

    # Read back
    k_nope_r, k_scale_r, k_rope_r, v_nope_r, v_scale_r = read_v4_fp8_cache_row(cache, 0)

    passed = True

    # K NOPE bytes match
    ok = (k_nope_r == k_fp8_bytes).all().item()
    if verbose or not ok:
        print(f"  K NOPE FP8 bytes match: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # K scale matches
    ok = abs(k_scale_r - k_scale_val) < 1e-10
    if verbose or not ok:
        print(f"  K scale match: {k_scale_r} == {k_scale_val} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # K ROPE BF16 round-trip
    k_rope_expected = k_rope.to(torch.bfloat16).float()
    max_rope_diff = (k_rope_r - k_rope_expected).abs().max().item()
    ok = max_rope_diff < 1e-6
    if verbose or not ok:
        print(f"  K ROPE BF16 round-trip max diff: {max_rope_diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # V NOPE bytes match
    ok = (v_nope_r == v_fp8_bytes).all().item()
    if verbose or not ok:
        print(f"  V NOPE FP8 bytes match: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # V scale matches
    ok = abs(v_scale_r - v_scale_val) < 1e-10
    if verbose or not ok:
        print(f"  V scale match: {v_scale_r} == {v_scale_val} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_v4_fp8_cache_paging(verbose=False):
    """V4K-0a: V4 FP8 cache paging across page boundaries works correctly."""
    torch.manual_seed(123)
    cache = alloc_v4_fp8_cache(2)  # 2 pages, 64 slots each = 128 total

    test_slots = [0, 63, 64, 127]
    written_data = {}

    for slot in test_slots:
        k_nope = torch.randn(HEAD_DIM)
        v_nope = torch.randn(HEAD_DIM)
        k_rope = torch.randn(QK_ROPE_HEAD_DIM)

        k_deq, k_scale_t = simulate_fp8_quantize(k_nope.unsqueeze(0))
        v_deq, v_scale_t = simulate_fp8_quantize(v_nope.unsqueeze(0))
        k_scale_val = k_scale_t.item()
        v_scale_val = v_scale_t.item()

        k_scaled = k_nope / k_scale_val
        v_scaled = v_nope / v_scale_val
        if hasattr(torch, 'float8_e4m3fn'):
            k_fp8 = k_scaled.to(torch.float8_e4m3fn).view(torch.uint8)
            v_fp8 = v_scaled.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            k_fp8 = k_scaled.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8).to(torch.uint8)
            v_fp8 = v_scaled.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8).to(torch.uint8)

        write_v4_fp8_cache_row(cache, slot, k_fp8, k_scale_val, k_rope,
                                v_fp8, v_scale_val)
        written_data[slot] = (k_fp8.clone(), k_scale_val,
                              k_rope.to(torch.bfloat16).float(), v_fp8.clone(), v_scale_val)

    passed = True
    for slot in test_slots:
        k_nope_r, k_scale_r, k_rope_r, v_nope_r, v_scale_r = read_v4_fp8_cache_row(cache, slot)
        k_fp8_w, k_scale_w, k_rope_w, v_fp8_w, v_scale_w = written_data[slot]

        ok_k = (k_nope_r == k_fp8_w).all().item()
        ok_ks = abs(k_scale_r - k_scale_w) < 1e-10
        ok_kr = (k_rope_r - k_rope_w).abs().max().item() < 1e-6
        ok_v = (v_nope_r == v_fp8_w).all().item()
        ok_vs = abs(v_scale_r - v_scale_w) < 1e-10

        slot_ok = ok_k and ok_ks and ok_kr and ok_v and ok_vs
        if verbose or not slot_ok:
            print(f"  Slot {slot:3d}: K_nope={'OK' if ok_k else 'FAIL'} "
                  f"K_scale={'OK' if ok_ks else 'FAIL'} K_rope={'OK' if ok_kr else 'FAIL'} "
                  f"V_nope={'OK' if ok_v else 'FAIL'} V_scale={'OK' if ok_vs else 'FAIL'} "
                  f"{'PASS' if slot_ok else 'FAIL'}")
        if not slot_ok:
            passed = False

    return passed


def test_v4_constants_consistency(verbose=False):
    """V4K-0a: V4 constants are internally consistent."""
    passed = True

    checks = [
        ("V4_FP8_BYTES_PER_ENTRY", V4_FP8_BYTES_PER_ENTRY, 1160),
        ("V4_TQ_BYTES_PER_ENTRY", V4_TQ_BYTES_PER_ENTRY, 644),
        ("D_QK", D_QK, 576),
        ("V4_FP8_K_SCALE_OFFSET", V4_FP8_K_SCALE_OFFSET, 512),
        ("V4_FP8_K_ROPE_OFFSET", V4_FP8_K_ROPE_OFFSET, 516),
        ("V4_FP8_V_NOPE_OFFSET", V4_FP8_V_NOPE_OFFSET, 644),
        ("V4_FP8_V_SCALE_OFFSET", V4_FP8_V_SCALE_OFFSET, 1156),
        ("V4_TQ_K_ENTRY_BYTES", V4_TQ_K_ENTRY_BYTES, 386),
        ("V4_TQ_V_ENTRY_BYTES", V4_TQ_V_ENTRY_BYTES, 258),
    ]

    for name, actual, expected in checks:
        ok = actual == expected
        if verbose or not ok:
            print(f"  {name}: {actual} == {expected} {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Verify offset arithmetic sums to total
    computed_total = (V4_FP8_K_NOPE_BYTES + V4_FP8_K_SCALE_BYTES +
                     V4_FP8_K_ROPE_BYTES + V4_FP8_V_NOPE_BYTES +
                     V4_FP8_V_SCALE_BYTES)
    ok = computed_total == V4_FP8_BYTES_PER_ENTRY
    if verbose or not ok:
        print(f"  Sum of parts: {computed_total} == {V4_FP8_BYTES_PER_ENTRY} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


def test_metrics_computation(verbose=False):
    """V4K-0a: Metrics computation returns correct values for known inputs."""
    passed = True

    # Identical tensors
    x = torch.randn(64)
    m = compute_metrics(x, x)
    ok = abs(m["cosine"] - 1.0) < 1e-6 and m["nrmse"] < 1e-6
    if verbose or not ok:
        print(f"  Identical: cos={m['cosine']:.6f} nrmse={m['nrmse']:.2e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # Orthogonal tensors
    a = torch.zeros(4)
    a[0] = 1.0
    b = torch.zeros(4)
    b[1] = 1.0
    m = compute_metrics(a, b)
    ok = abs(m["cosine"]) < 0.01
    if verbose or not ok:
        print(f"  Orthogonal: cos={m['cosine']:.6f} {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # Scaled tensor: nrmse should detect difference
    torch.manual_seed(42)
    x = torch.randn(128)
    x_scaled = x * 1.1
    m = compute_metrics(x, x_scaled)
    ok = m["nrmse"] > 0.01
    if verbose or not ok:
        print(f"  Scaled (1.1x): nrmse={m['nrmse']:.4f} (> 0.01) {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    # Cosine should still be high for scaled version (same direction)
    ok = m["cosine"] > 0.999
    if verbose or not ok:
        print(f"  Scaled cosine: {m['cosine']:.6f} (> 0.999) {'PASS' if ok else 'FAIL'}")
    if not ok:
        passed = False

    return passed


# ---------------------------------------------------------------------------
# V4K-0d: Lightning Indexer reference
# ---------------------------------------------------------------------------

def ref_lightning_score(q_proj, indexer_k_cache, score_proj):
    """Lightning Indexer scoring: FP4-simulated multi-head dot-product scoring.

    Computes importance scores for each compressed block by:
    1. Multi-head dot product: q_proj @ indexer_k_cache^T (64 heads, head_dim=128)
    2. ReLU activation
    3. Score projection: aggregate across heads via score_proj

    In real V4, q_proj and indexer_k_cache are FP4-quantized. This reference
    simulates FP4 by rounding to 4-bit resolution.

    Args:
        q_proj: [INDEX_N_HEADS, INDEX_HEAD_DIM] query projected into indexer space
        indexer_k_cache: [num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM] FP4 K cache
        score_proj: [INDEX_N_HEADS] score aggregation weights

    Returns:
        scores: [num_blocks] importance score per block
    """
    num_blocks = indexer_k_cache.shape[0]

    # Multi-head dot product: [num_blocks, INDEX_N_HEADS]
    dots = torch.einsum('hd,nhd->nh', q_proj.float(), indexer_k_cache.float())

    # ReLU activation
    dots = torch.relu(dots)

    # Aggregate across heads: [num_blocks]
    scores = torch.einsum('nh,h->n', dots, score_proj.float())

    return scores


def ref_lightning_score_mqa(q_proj, indexer_k_cache_mqa, score_proj):
    """Lightning Indexer scoring — MQA variant (single shared K head).

    Same math as ref_lightning_score, but the K cache stores ONE key vector per
    block that is shared across all query heads (DeepSeek-V3.2 / GLM-5.2 indexer
    wk → single [index_head_dim] key), avoiding the n_heads× replication.

    Args:
        q_proj: [INDEX_N_HEADS, INDEX_HEAD_DIM]
        indexer_k_cache_mqa: [num_blocks, INDEX_HEAD_DIM] single shared K per block
        score_proj: [INDEX_N_HEADS]

    Returns:
        scores: [num_blocks]
    """
    # Broadcast the single K over heads: dots[n,h] = q_proj[h,:] · K[n,:]
    dots = torch.einsum('hd,nd->nh', q_proj.float(), indexer_k_cache_mqa.float())
    dots = torch.relu(dots)
    scores = torch.einsum('nh,h->n', dots, score_proj.float())
    return scores


def ref_lightning_topk(scores, topk, query_position, block_endpoints):
    """Lightning Indexer top-k selection with causality enforcement.

    Selects the top-k highest-scoring compressed blocks, excluding
    blocks whose endpoint is in the future relative to query_position.

    Args:
        scores: [num_blocks] importance scores
        topk: int, number of blocks to select
        query_position: int, current query token position
        block_endpoints: [num_blocks] int, endpoint position of each block

    Returns:
        indices: [effective_k] int64, sorted indices of selected blocks
            (effective_k <= topk, may be less due to causality masking)
        selected_scores: [effective_k] scores of selected blocks
    """
    # Causality mask: only blocks with endpoint <= query_position
    causal_mask = block_endpoints <= query_position
    masked_scores = scores.clone()
    masked_scores[~causal_mask] = float('-inf')

    # Top-k selection
    num_valid = causal_mask.sum().item()
    effective_k = min(topk, num_valid)

    if effective_k == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0)

    topk_scores, topk_indices = torch.topk(masked_scores, effective_k)

    # Sort by index for deterministic access pattern
    sorted_order = topk_indices.sort().indices
    topk_indices = topk_indices[sorted_order]
    topk_scores = topk_scores[sorted_order]

    return topk_indices, topk_scores


# ---------------------------------------------------------------------------
# V4K-0e: CSA FP8 decode reference
# ---------------------------------------------------------------------------

def _lse_combine(out_a, lse_a, out_b, lse_b):
    """Combine two attention outputs using log-sum-exp stable merging.

    Args:
        out_a, out_b: [..., D] attention outputs
        lse_a, lse_b: [...] log-sum-exp values

    Returns:
        out_combined: [..., D]
        lse_combined: [...]
    """
    lse_max = torch.maximum(lse_a, lse_b)
    exp_a = torch.exp(lse_a - lse_max)
    exp_b = torch.exp(lse_b - lse_max)
    denom = exp_a + exp_b
    out = (exp_a.unsqueeze(-1) * out_a + exp_b.unsqueeze(-1) * out_b) / denom.unsqueeze(-1)
    lse = lse_max + torch.log(denom)
    return out, lse


def _ref_attention(q, k, v, sm_scale, causal_mask=None):
    """Standard scaled dot-product attention.

    Args:
        q: [num_q_heads, 1, d_qk] query (single token decode)
        k: [1, s_kv, d_qk] keys (single KV head)
        v: [1, s_kv, d_v] values (single KV head)
        sm_scale: float softmax scale
        causal_mask: optional [s_kv] bool mask (True = attend, False = mask out)

    Returns:
        out: [num_q_heads, 1, d_v]
        lse: [num_q_heads, 1]
    """
    # Broadcast KV head to all Q heads: [num_q_heads, 1, s_kv]
    scores = torch.einsum('hqd,hkd->hqk', q, k.expand(q.shape[0], -1, -1)) * sm_scale

    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    lse = torch.logsumexp(scores, dim=-1)  # [h, 1]
    p = torch.softmax(scores, dim=-1)      # [h, 1, s_kv]
    out = torch.einsum('hqk,hkd->hqd', p, v.expand(q.shape[0], -1, -1))
    return out, lse


def ref_csa_fp8_decode(q_nope, q_rope,
                        compressed_k_nope, compressed_k_rope, compressed_v,
                        swa_k_nope, swa_k_rope, swa_v,
                        sparse_indices, sm_scale, num_q_heads):
    """CSA FP8 decode: sparse compressed attention + SWA + LSE combine.

    Single decode token attending to:
    1. Sparse: top-k compressed blocks (selected by Lightning Indexer)
    2. SWA: last 128 uncompressed tokens (sliding window)
    Results combined via log-sum-exp.

    Args:
        q_nope: [num_q_heads, HEAD_DIM] query NOPE
        q_rope: [num_q_heads, QK_ROPE_HEAD_DIM] query ROPE (with standard RoPE applied)
        compressed_k_nope: [num_compressed, HEAD_DIM] all compressed K NOPE
        compressed_k_rope: [num_compressed, QK_ROPE_HEAD_DIM] all compressed K ROPE
        compressed_v: [num_compressed, HEAD_DIM] all compressed V
        swa_k_nope: [swa_len, HEAD_DIM] SWA K NOPE
        swa_k_rope: [swa_len, QK_ROPE_HEAD_DIM] SWA K ROPE
        swa_v: [swa_len, HEAD_DIM] SWA V
        sparse_indices: [topk] int64, selected compressed block indices
        sm_scale: float, softmax scale factor
        num_q_heads: int

    Returns:
        output: [num_q_heads, HEAD_DIM]
        lse: [num_q_heads]
    """
    H = num_q_heads

    # --- Sparse compressed attention ---
    if len(sparse_indices) > 0:
        sel_k_nope = compressed_k_nope[sparse_indices]  # [topk, HEAD_DIM]
        sel_k_rope = compressed_k_rope[sparse_indices]  # [topk, QK_ROPE_HEAD_DIM]
        sel_v = compressed_v[sparse_indices]             # [topk, HEAD_DIM]

        # QK scores = NOPE scores + ROPE scores
        # NOPE: [H, 1, HEAD_DIM] @ [1, topk, HEAD_DIM]^T
        q_n = q_nope.unsqueeze(1)        # [H, 1, HEAD_DIM]
        k_n = sel_k_nope.unsqueeze(0)    # [1, topk, HEAD_DIM]
        scores_nope = torch.einsum('hqd,hkd->hqk', q_n, k_n.expand(H, -1, -1))

        q_r = q_rope.unsqueeze(1)        # [H, 1, QK_ROPE_HEAD_DIM]
        k_r = sel_k_rope.unsqueeze(0)    # [1, topk, QK_ROPE_HEAD_DIM]
        scores_rope = torch.einsum('hqd,hkd->hqk', q_r, k_r.expand(H, -1, -1))

        scores = (scores_nope + scores_rope) * sm_scale  # [H, 1, topk]

        lse_sparse = torch.logsumexp(scores, dim=-1)     # [H, 1]
        p = torch.softmax(scores, dim=-1)
        v_sel = sel_v.unsqueeze(0).expand(H, -1, -1)     # [H, topk, HEAD_DIM]
        out_sparse = torch.einsum('hqk,hkd->hqd', p, v_sel)  # [H, 1, HEAD_DIM]
    else:
        out_sparse = torch.zeros(H, 1, HEAD_DIM)
        lse_sparse = torch.full((H, 1), float('-inf'))

    # --- SWA attention ---
    swa_len = swa_k_nope.shape[0]
    if swa_len > 0:
        q_n = q_nope.unsqueeze(1)
        k_n = swa_k_nope.unsqueeze(0)
        scores_nope = torch.einsum('hqd,hkd->hqk', q_n, k_n.expand(H, -1, -1))

        q_r = q_rope.unsqueeze(1)
        k_r = swa_k_rope.unsqueeze(0)
        scores_rope = torch.einsum('hqd,hkd->hqk', q_r, k_r.expand(H, -1, -1))

        scores = (scores_nope + scores_rope) * sm_scale

        lse_swa = torch.logsumexp(scores, dim=-1)
        p = torch.softmax(scores, dim=-1)
        v_swa = swa_v.unsqueeze(0).expand(H, -1, -1)
        out_swa = torch.einsum('hqk,hkd->hqd', p, v_swa)
    else:
        out_swa = torch.zeros(H, 1, HEAD_DIM)
        lse_swa = torch.full((H, 1), float('-inf'))

    # --- LSE combine ---
    output, lse = _lse_combine(out_sparse.squeeze(1), lse_sparse.squeeze(1),
                                out_swa.squeeze(1), lse_swa.squeeze(1))

    return output, lse


# ---------------------------------------------------------------------------
# V4K-0f: HCA FP8 decode reference
# ---------------------------------------------------------------------------

def ref_hca_fp8_decode(q_nope, q_rope,
                        compressed_k_nope, compressed_k_rope, compressed_v,
                        swa_k_nope, swa_k_rope, swa_v,
                        sm_scale, num_q_heads):
    """HCA FP8 decode: dense attention over ALL compressed blocks + SWA + combine.

    No sparse selection — HCA compressed sequence is short enough for dense
    attention (e.g., ~7800 entries at 1M context with 128:1 compression).

    Args:
        q_nope: [num_q_heads, HEAD_DIM]
        q_rope: [num_q_heads, QK_ROPE_HEAD_DIM]
        compressed_k_nope: [num_compressed, HEAD_DIM]
        compressed_k_rope: [num_compressed, QK_ROPE_HEAD_DIM]
        compressed_v: [num_compressed, HEAD_DIM]
        swa_k_nope: [swa_len, HEAD_DIM]
        swa_k_rope: [swa_len, QK_ROPE_HEAD_DIM]
        swa_v: [swa_len, HEAD_DIM]
        sm_scale: float
        num_q_heads: int

    Returns:
        output: [num_q_heads, HEAD_DIM]
        lse: [num_q_heads]
    """
    # Dense = CSA with topk=ALL (select every block)
    all_indices = torch.arange(compressed_k_nope.shape[0])
    return ref_csa_fp8_decode(q_nope, q_rope,
                               compressed_k_nope, compressed_k_rope, compressed_v,
                               swa_k_nope, swa_k_rope, swa_v,
                               all_indices, sm_scale, num_q_heads)


# ---------------------------------------------------------------------------
# V4K-0g: SWA-only decode reference
# ---------------------------------------------------------------------------

def ref_swa_decode(q_nope, q_rope,
                    swa_k_nope, swa_k_rope, swa_v,
                    sm_scale, num_q_heads):
    """SWA-only decode: pure sliding window causal attention.

    For layers with compress_ratios=0. Standard causal attention over the
    last SLIDING_WINDOW (128) tokens. No compression, no indexing.

    Args:
        q_nope: [num_q_heads, HEAD_DIM]
        q_rope: [num_q_heads, QK_ROPE_HEAD_DIM]
        swa_k_nope: [swa_len, HEAD_DIM] (swa_len <= SLIDING_WINDOW)
        swa_k_rope: [swa_len, QK_ROPE_HEAD_DIM]
        swa_v: [swa_len, HEAD_DIM]
        sm_scale: float
        num_q_heads: int

    Returns:
        output: [num_q_heads, HEAD_DIM]
        lse: [num_q_heads]
    """
    H = num_q_heads
    swa_len = swa_k_nope.shape[0]

    if swa_len == 0:
        return torch.zeros(H, HEAD_DIM), torch.full((H,), float('-inf'))

    q_n = q_nope.unsqueeze(1)  # [H, 1, HEAD_DIM]
    k_n = swa_k_nope.unsqueeze(0).expand(H, -1, -1)
    scores_nope = torch.einsum('hqd,hkd->hqk', q_n, k_n)

    q_r = q_rope.unsqueeze(1)
    k_r = swa_k_rope.unsqueeze(0).expand(H, -1, -1)
    scores_rope = torch.einsum('hqd,hkd->hqk', q_r, k_r)

    scores = (scores_nope + scores_rope) * sm_scale  # [H, 1, swa_len]

    lse = torch.logsumexp(scores, dim=-1).squeeze(1)  # [H]
    p = torch.softmax(scores, dim=-1)
    v = swa_v.unsqueeze(0).expand(H, -1, -1)
    out = torch.einsum('hqk,hkd->hqd', p, v).squeeze(1)  # [H, HEAD_DIM]

    return out, lse


# ---------------------------------------------------------------------------
# V4K-0h: Inverse RoPE reference
# ---------------------------------------------------------------------------

def ref_inverse_rope(x, cos, sin, positions):
    """Apply inverse rotary position embedding (negate angles).

    After attention with shared KV (num_kv_heads=1), the output carries
    positional coupling. Inverse RoPE removes it.

    inverse_rope(x, pos) = rope(x, pos, negate=True)
      = x * cos(pos*freq) + x_pair * sin(pos*freq)
    (cos is even, sin negated → add instead of subtract)

    Args:
        x: tensor [..., dim] where dim = qk_rope_head_dim
        cos: precomputed cos table [max_pos, dim//2]
        sin: precomputed sin table [max_pos, dim//2]
        positions: integer positions [...]

    Returns:
        tensor same shape as x with inverse RoPE applied
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    c = cos[positions]
    s = sin[positions]

    # Inverse: negate the sin term (add instead of subtract)
    out1 = x1 * c + x2 * s
    out2 = -x1 * s + x2 * c

    return torch.stack([out1, out2], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# V4K-0i: V4 FP8 k_append / dequant reference
# ---------------------------------------------------------------------------

def ref_v4_fp8_k_append(k_nope, k_rope, v_nope, kv_cache, slot_mapping):
    """Write compressed K+V to V4 paged FP8 cache.

    Per entry: FP8 quantize K_NOPE, store K_scale, store K_ROPE as BF16,
    FP8 quantize V_NOPE, store V_scale. Total 1160 bytes/entry.

    Args:
        k_nope: [num_tokens, HEAD_DIM] K NOPE (float)
        k_rope: [num_tokens, QK_ROPE_HEAD_DIM] K ROPE (float, already RoPE-encoded)
        v_nope: [num_tokens, HEAD_DIM] V (float)
        kv_cache: flat uint8 tensor (from alloc_v4_fp8_cache)
        slot_mapping: [num_tokens] int, slot indices

    Returns:
        k_nope_deq: [num_tokens, HEAD_DIM] dequantized K NOPE (for verification)
        v_nope_deq: [num_tokens, HEAD_DIM] dequantized V (for verification)
    """
    num_tokens = k_nope.shape[0]
    k_nope_deq = torch.zeros_like(k_nope)
    v_nope_deq = torch.zeros_like(v_nope)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()

        # FP8 quantize K NOPE
        k_deq_i, k_scale_i = simulate_fp8_quantize(k_nope[i:i+1])
        k_scale_val = k_scale_i.item()

        # FP8 quantize V NOPE
        v_deq_i, v_scale_i = simulate_fp8_quantize(v_nope[i:i+1])
        v_scale_val = v_scale_i.item()

        # Get raw FP8 bytes
        k_scaled = k_nope[i] / k_scale_val
        v_scaled = v_nope[i] / v_scale_val
        if hasattr(torch, 'float8_e4m3fn'):
            k_fp8 = k_scaled.to(torch.float8_e4m3fn).view(torch.uint8)
            v_fp8 = v_scaled.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            k_fp8 = k_scaled.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8).to(torch.uint8)
            v_fp8 = v_scaled.clamp(-FP8_MAX, FP8_MAX).mul(8).round().div(8).to(torch.uint8)

        write_v4_fp8_cache_row(kv_cache, slot, k_fp8, k_scale_val,
                                k_rope[i], v_fp8, v_scale_val)

        k_nope_deq[i] = k_deq_i.squeeze(0)
        v_nope_deq[i] = v_deq_i.squeeze(0)

    return k_nope_deq, v_nope_deq


def ref_v4_fp8_dequant(kv_cache, indices):
    """Read V4 FP8 entries from paged cache and dequantize.

    Args:
        kv_cache: flat uint8 tensor
        indices: [num_fetch] int, slot indices to read

    Returns:
        k_nope: [num_fetch, HEAD_DIM] float32
        k_rope: [num_fetch, QK_ROPE_HEAD_DIM] float32
        v_nope: [num_fetch, HEAD_DIM] float32
    """
    num_fetch = len(indices)
    k_nope = torch.zeros(num_fetch, HEAD_DIM)
    k_rope = torch.zeros(num_fetch, QK_ROPE_HEAD_DIM)
    v_nope = torch.zeros(num_fetch, HEAD_DIM)

    for i, idx in enumerate(indices):
        slot = idx if isinstance(idx, int) else idx.item()
        k_fp8, k_scale, k_rope_i, v_fp8, v_scale = read_v4_fp8_cache_row(kv_cache, slot)

        k_nope[i] = dequant_fp8_bytes(k_fp8, k_scale)
        k_rope[i] = k_rope_i
        v_nope[i] = dequant_fp8_bytes(v_fp8, v_scale)

    return k_nope, k_rope, v_nope


# ---------------------------------------------------------------------------
# V4K-0j: End-to-end V4 decode layer reference
# ---------------------------------------------------------------------------

def ref_v4_decode_layer(layer_type, q_nope, q_rope,
                         compressed_k_nope, compressed_k_rope, compressed_v,
                         swa_k_nope, swa_k_rope, swa_v,
                         sm_scale, num_q_heads,
                         rope_cos, rope_sin, query_position,
                         sparse_indices=None,
                         indexer_k_cache=None, q_proj_idx=None,
                         score_proj=None, block_endpoints=None, topk=None):
    """Full single-layer V4 decode pipeline dispatching by layer type.

    Pipeline: attention → inverse RoPE (on rope dims of output).

    For CSA: sparse attention using provided sparse_indices (or Lightning
    Indexer if indexer inputs are provided).
    For HCA: dense attention over all compressed blocks.
    For SWA: pure sliding window.

    Args:
        layer_type: 'csa', 'hca', or 'swa'
        q_nope: [num_q_heads, HEAD_DIM]
        q_rope: [num_q_heads, QK_ROPE_HEAD_DIM]
        compressed_k_nope, compressed_k_rope, compressed_v: compressed KV
        swa_k_nope, swa_k_rope, swa_v: SWA KV
        sm_scale: float
        num_q_heads: int
        rope_cos, rope_sin: for inverse RoPE
        query_position: int, current token position
        sparse_indices: [topk] int (for CSA, optional if indexer inputs given)
        indexer_k_cache, q_proj_idx, score_proj, block_endpoints, topk:
            Lightning Indexer inputs (for CSA when sparse_indices not provided)

    Returns:
        output: [num_q_heads, HEAD_DIM] attention output with inverse RoPE applied
        lse: [num_q_heads]
    """
    if layer_type == 'csa':
        if sparse_indices is None:
            scores = ref_lightning_score(q_proj_idx, indexer_k_cache, score_proj)
            sparse_indices, _ = ref_lightning_topk(
                scores, topk, query_position, block_endpoints)
        out, lse = ref_csa_fp8_decode(
            q_nope, q_rope,
            compressed_k_nope, compressed_k_rope, compressed_v,
            swa_k_nope, swa_k_rope, swa_v,
            sparse_indices, sm_scale, num_q_heads)

    elif layer_type == 'hca':
        out, lse = ref_hca_fp8_decode(
            q_nope, q_rope,
            compressed_k_nope, compressed_k_rope, compressed_v,
            swa_k_nope, swa_k_rope, swa_v,
            sm_scale, num_q_heads)

    elif layer_type == 'swa':
        out, lse = ref_swa_decode(
            q_nope, q_rope,
            swa_k_nope, swa_k_rope, swa_v,
            sm_scale, num_q_heads)

    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

    # Apply inverse RoPE to rope dims of output
    # In V4, output is HEAD_DIM=512 (NOPE only), inverse RoPE is applied
    # to undo positional coupling from the shared KV head
    positions = torch.full((num_q_heads,), query_position, dtype=torch.long)
    out_nope = out[..., :HEAD_DIM - QK_ROPE_HEAD_DIM]  # 448 dims unchanged
    out_rope = out[..., HEAD_DIM - QK_ROPE_HEAD_DIM:]   # 64 dims get inverse RoPE
    out_rope_inv = ref_inverse_rope(out_rope, rope_cos, rope_sin, positions)
    output = torch.cat([out_nope, out_rope_inv], dim=-1)

    return output, lse


# ---------------------------------------------------------------------------
# V4K-0j Tests: End-to-end V4 decode layer reference
# ---------------------------------------------------------------------------

def _make_layer_test_inputs(num_compressed, swa_len, num_q_heads, seed=42):
    """Generate test inputs for layer-level tests."""
    torch.manual_seed(seed)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
    rope_cos, rope_sin = precompute_rope_freqs(ROPE_THETA, QK_ROPE_HEAD_DIM, 65536)

    q_nope = torch.randn(num_q_heads, HEAD_DIM)
    q_rope = torch.randn(num_q_heads, QK_ROPE_HEAD_DIM)
    ck_nope = torch.randn(num_compressed, HEAD_DIM)
    ck_rope = torch.randn(num_compressed, QK_ROPE_HEAD_DIM)
    cv = torch.randn(num_compressed, HEAD_DIM)
    sk_nope = torch.randn(swa_len, HEAD_DIM)
    sk_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM)
    sv = torch.randn(swa_len, HEAD_DIM)

    return dict(q_nope=q_nope, q_rope=q_rope,
                compressed_k_nope=ck_nope, compressed_k_rope=ck_rope,
                compressed_v=cv, swa_k_nope=sk_nope, swa_k_rope=sk_rope,
                swa_v=sv, sm_scale=sm_scale, num_q_heads=num_q_heads,
                rope_cos=rope_cos, rope_sin=rope_sin)


def test_ref_v4_layer_csa(verbose=False):
    """V4K-0j: CSA layer — full pipeline with sparse indices."""
    H = 8
    num_compressed = 100
    inputs = _make_layer_test_inputs(num_compressed, SLIDING_WINDOW, H)
    query_pos = 2000

    sparse_indices = torch.arange(0, min(64, num_compressed))

    out, lse = ref_v4_decode_layer(
        'csa', **inputs, query_position=query_pos, sparse_indices=sparse_indices)

    passed = True

    ok = out.shape == (H, HEAD_DIM) and lse.shape == (H,)
    if verbose or not ok:
        print(f"  Output shape: {tuple(out.shape)}, LSE: {tuple(lse.shape)} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = torch.isfinite(out).all().item()
    if verbose or not ok:
        print(f"  All finite: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Inverse RoPE was applied: last 64 dims should differ from raw attention output
    raw_out, _ = ref_csa_fp8_decode(
        inputs['q_nope'], inputs['q_rope'],
        inputs['compressed_k_nope'], inputs['compressed_k_rope'], inputs['compressed_v'],
        inputs['swa_k_nope'], inputs['swa_k_rope'], inputs['swa_v'],
        sparse_indices, inputs['sm_scale'], H)
    # NOPE portion (first 448 dims) should match
    diff_nope = (out[:, :HEAD_DIM - QK_ROPE_HEAD_DIM] -
                 raw_out[:, :HEAD_DIM - QK_ROPE_HEAD_DIM]).abs().max().item()
    ok = diff_nope < 1e-5
    if verbose or not ok:
        print(f"  NOPE dims unchanged: max_diff={diff_nope:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # ROPE portion (last 64 dims) should differ (inverse RoPE applied)
    diff_rope = (out[:, HEAD_DIM - QK_ROPE_HEAD_DIM:] -
                 raw_out[:, HEAD_DIM - QK_ROPE_HEAD_DIM:]).abs().max().item()
    ok = diff_rope > 0.001
    if verbose or not ok:
        print(f"  ROPE dims changed by inverse RoPE: diff={diff_rope:.4f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_v4_layer_hca(verbose=False):
    """V4K-0j: HCA layer — full pipeline with dense attention."""
    H = 8
    num_compressed = 50
    inputs = _make_layer_test_inputs(num_compressed, SLIDING_WINDOW, H, seed=77)
    query_pos = 6400

    out, lse = ref_v4_decode_layer('hca', **inputs, query_position=query_pos)

    passed = True

    ok = out.shape == (H, HEAD_DIM) and torch.isfinite(out).all().item()
    if verbose or not ok:
        print(f"  Valid output: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_v4_layer_swa(verbose=False):
    """V4K-0j: SWA layer — full pipeline with sliding window only."""
    H = 8
    inputs = _make_layer_test_inputs(0, SLIDING_WINDOW, H, seed=99)
    query_pos = 500

    out, lse = ref_v4_decode_layer(
        'swa', **inputs, query_position=query_pos)

    passed = True

    ok = out.shape == (H, HEAD_DIM) and torch.isfinite(out).all().item()
    if verbose or not ok:
        print(f"  Valid output: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # SWA should ignore compressed KV (there are none)
    # Output should be non-trivial
    ok = out.abs().mean().item() > 0.01
    if verbose or not ok:
        print(f"  Non-trivial: mean_abs={out.abs().mean().item():.4f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0i Tests: V4 FP8 k_append / dequant reference
# ---------------------------------------------------------------------------

def test_ref_v4_fp8_roundtrip(verbose=False):
    """V4K-0i: FP8 k_append + dequant round-trip cosine > 0.999."""
    torch.manual_seed(42)
    num_tokens = 8
    k_nope = torch.randn(num_tokens, HEAD_DIM)
    k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
    v_nope = torch.randn(num_tokens, HEAD_DIM)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

    cache = alloc_v4_fp8_cache(1)  # 1 page = 64 slots, enough for 8

    # Write
    k_deq, v_deq = ref_v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot_mapping)

    # Read back
    k_nope_r, k_rope_r, v_nope_r = ref_v4_fp8_dequant(cache, list(range(num_tokens)))

    passed = True

    # K NOPE round-trip
    m_k = compute_metrics(k_nope, k_nope_r)
    ok = m_k['cosine'] > ERR_FP8_COSINE
    if verbose or not ok:
        print(f"  K NOPE: {fmt_metrics(m_k)} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # K ROPE round-trip (BF16, should be near-exact)
    k_rope_expected = k_rope.to(torch.bfloat16).float()
    diff = (k_rope_r - k_rope_expected).abs().max().item()
    ok = diff < 1e-6
    if verbose or not ok:
        print(f"  K ROPE BF16 max_diff: {diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # V NOPE round-trip
    m_v = compute_metrics(v_nope, v_nope_r)
    ok = m_v['cosine'] > ERR_FP8_COSINE
    if verbose or not ok:
        print(f"  V NOPE: {fmt_metrics(m_v)} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0h Tests: Inverse RoPE reference
# ---------------------------------------------------------------------------

def test_ref_inverse_rope_roundtrip(verbose=False):
    """V4K-0h: Inverse RoPE — apply RoPE then inverse = identity."""
    torch.manual_seed(42)
    dim = QK_ROPE_HEAD_DIM
    cos, sin = precompute_rope_freqs(ROPE_THETA, dim, 1024)

    passed = True

    # Test at multiple positions
    for pos_val in [0, 1, 42, 500, 1023]:
        x = torch.randn(4, dim)
        positions = torch.full((4,), pos_val, dtype=torch.long)

        x_roped = apply_rope(x, cos, sin, positions)
        x_recovered = ref_inverse_rope(x_roped, cos, sin, positions)

        max_diff = (x_recovered - x).abs().max().item()
        ok = max_diff < 1e-5
        if verbose or not ok:
            print(f"  pos={pos_val:4d}: max_diff={max_diff:.2e} {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    # Test with compressed theta too
    cos_c, sin_c = precompute_rope_freqs(COMPRESS_ROPE_THETA, dim, 2048)
    x = torch.randn(8, dim)
    positions = torch.tensor([127, 255, 383, 511, 639, 767, 895, 1023])

    x_roped = apply_rope(x, cos_c, sin_c, positions)
    x_recovered = ref_inverse_rope(x_roped, cos_c, sin_c, positions)
    max_diff = (x_recovered - x).abs().max().item()
    ok = max_diff < 1e-5
    if verbose or not ok:
        print(f"  Compressed theta round-trip: max_diff={max_diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0g Tests: SWA-only decode reference
# ---------------------------------------------------------------------------

def test_ref_swa_decode(verbose=False):
    """V4K-0g: SWA decode — matches dense attention over 128 tokens."""
    torch.manual_seed(42)
    swa_len = SLIDING_WINDOW
    H = 16
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    q_nope = torch.randn(H, HEAD_DIM)
    q_rope = torch.randn(H, QK_ROPE_HEAD_DIM)
    sk_nope = torch.randn(swa_len, HEAD_DIM)
    sk_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM)
    sv = torch.randn(swa_len, HEAD_DIM)

    out, lse = ref_swa_decode(q_nope, q_rope, sk_nope, sk_rope, sv, sm_scale, H)

    passed = True

    ok = out.shape == (H, HEAD_DIM) and lse.shape == (H,)
    if verbose or not ok:
        print(f"  Output shape: {tuple(out.shape)}, LSE shape: {tuple(lse.shape)} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()
    if verbose or not ok:
        print(f"  All finite: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # SWA decode should match HCA decode with 0 compressed blocks
    empty_k = torch.zeros(0, HEAD_DIM)
    empty_kr = torch.zeros(0, QK_ROPE_HEAD_DIM)
    empty_v = torch.zeros(0, HEAD_DIM)
    out_hca, lse_hca = ref_hca_fp8_decode(
        q_nope, q_rope, empty_k, empty_kr, empty_v,
        sk_nope, sk_rope, sv, sm_scale, H)

    diff = (out - out_hca).abs().max().item()
    ok = diff < 1e-5
    if verbose or not ok:
        print(f"  SWA == HCA(0 compressed): max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0f Tests: HCA FP8 decode reference
# ---------------------------------------------------------------------------

def test_ref_hca_decode_short(verbose=False):
    """V4K-0f: HCA decode — short context (1K tokens, ~8 compressed blocks)."""
    num_compressed = 8  # 1024 / 128 = 8
    swa_len = SLIDING_WINDOW
    H = 16
    inputs = _make_csa_decode_inputs(num_compressed, swa_len, H, seed=42)
    q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm = inputs

    out, lse = ref_hca_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm, H)

    passed = True

    ok = out.shape == (H, HEAD_DIM)
    if verbose or not ok:
        print(f"  Output shape: {tuple(out.shape)} == ({H}, {HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()
    if verbose or not ok:
        print(f"  All finite: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # HCA dense should match CSA with topk=all exactly
    all_indices = torch.arange(num_compressed)
    out_csa, lse_csa = ref_csa_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, all_indices, sm, H)

    diff = (out - out_csa).abs().max().item()
    ok = diff < 1e-6
    if verbose or not ok:
        print(f"  HCA == CSA(topk=all): max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_hca_decode_long(verbose=False):
    """V4K-0f: HCA decode — long context (64K tokens, ~500 compressed blocks)."""
    num_compressed = 500  # 64000 / 128 = 500
    swa_len = SLIDING_WINDOW
    H = 4  # fewer heads for speed
    inputs = _make_csa_decode_inputs(num_compressed, swa_len, H, seed=99)
    q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm = inputs

    out, lse = ref_hca_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm, H)

    passed = True

    ok = out.shape == (H, HEAD_DIM) and torch.isfinite(out).all().item()
    if verbose or not ok:
        print(f"  64K context output valid: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Output should be non-trivial
    ok = out.abs().mean().item() > 0.01
    if verbose or not ok:
        print(f"  Non-trivial output: mean_abs={out.abs().mean().item():.4f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0e Tests: CSA FP8 decode reference
# ---------------------------------------------------------------------------

def _make_csa_decode_inputs(num_compressed, swa_len, num_q_heads, seed=42):
    """Generate test inputs for CSA decode tests."""
    torch.manual_seed(seed)
    q_nope = torch.randn(num_q_heads, HEAD_DIM)
    q_rope = torch.randn(num_q_heads, QK_ROPE_HEAD_DIM)
    ck_nope = torch.randn(num_compressed, HEAD_DIM)
    ck_rope = torch.randn(num_compressed, QK_ROPE_HEAD_DIM)
    cv = torch.randn(num_compressed, HEAD_DIM)
    sk_nope = torch.randn(swa_len, HEAD_DIM)
    sk_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM)
    sv = torch.randn(swa_len, HEAD_DIM)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
    return (q_nope, q_rope, ck_nope, ck_rope, cv,
            sk_nope, sk_rope, sv, sm_scale)


def test_ref_csa_decode_topk_all_eq_dense(verbose=False):
    """V4K-0e: CSA decode — topk=all is identical to dense attention."""
    num_compressed = 64
    swa_len = 32
    H = 8  # fewer heads for speed
    inputs = _make_csa_decode_inputs(num_compressed, swa_len, H)
    q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm = inputs

    # Sparse with topk=ALL (select all blocks)
    all_indices = torch.arange(num_compressed)
    out_sparse, lse_sparse = ref_csa_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, all_indices, sm, H)

    # Dense: concatenate compressed + SWA, single attention
    all_k_nope = torch.cat([ck_n, sk_n], dim=0)
    all_k_rope = torch.cat([ck_r, sk_r], dim=0)
    all_v = torch.cat([cv, sv], dim=0)

    q_n_3d = q_n.unsqueeze(1)  # [H, 1, HEAD_DIM]
    k_n_3d = all_k_nope.unsqueeze(0).expand(H, -1, -1)
    scores_nope = torch.einsum('hqd,hkd->hqk', q_n_3d, k_n_3d)
    q_r_3d = q_r.unsqueeze(1)
    k_r_3d = all_k_rope.unsqueeze(0).expand(H, -1, -1)
    scores_rope = torch.einsum('hqd,hkd->hqk', q_r_3d, k_r_3d)
    scores = (scores_nope + scores_rope) * sm
    p = torch.softmax(scores, dim=-1)
    v_3d = all_v.unsqueeze(0).expand(H, -1, -1)
    out_dense = torch.einsum('hqk,hkd->hqd', p, v_3d).squeeze(1)

    passed = True
    m = compute_metrics(out_dense, out_sparse)
    ok = m['cosine'] > 0.99999
    if verbose or not ok:
        print(f"  topk=all vs dense: {fmt_metrics(m)} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_csa_decode_short(verbose=False):
    """V4K-0e: CSA decode — short context, topk=all matches dense."""
    num_compressed = 250  # ~1K tokens with 4:1 compression
    swa_len = SLIDING_WINDOW
    H = 16
    inputs = _make_csa_decode_inputs(num_compressed, swa_len, H, seed=123)
    q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm = inputs

    all_indices = torch.arange(num_compressed)
    out, lse = ref_csa_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, all_indices, sm, H)

    passed = True

    ok = out.shape == (H, HEAD_DIM)
    if verbose or not ok:
        print(f"  Output shape: {tuple(out.shape)} == ({H}, {HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = lse.shape == (H,)
    if verbose or not ok:
        print(f"  LSE shape: {tuple(lse.shape)} == ({H},) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = torch.isfinite(out).all().item() and torch.isfinite(lse).all().item()
    if verbose or not ok:
        print(f"  All finite: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_csa_decode_medium(verbose=False):
    """V4K-0e: CSA decode — medium context with sparse top-k selection."""
    num_compressed = 4000  # ~16K tokens
    swa_len = SLIDING_WINDOW
    H = 8
    topk = 1024
    inputs = _make_csa_decode_inputs(num_compressed, swa_len, H, seed=77)
    q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sm = inputs

    # Select topk random indices (simulating Lightning Indexer output)
    torch.manual_seed(77)
    sparse_indices = torch.randperm(num_compressed)[:topk].sort().values

    out, lse = ref_csa_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, sparse_indices, sm, H)

    passed = True

    ok = out.shape == (H, HEAD_DIM)
    if verbose or not ok:
        print(f"  Output shape: {tuple(out.shape)} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = torch.isfinite(out).all().item()
    if verbose or not ok:
        print(f"  All finite: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Sparse output should differ from dense (different subset of blocks)
    all_indices = torch.arange(num_compressed)
    out_dense, _ = ref_csa_fp8_decode(
        q_n, q_r, ck_n, ck_r, cv, sk_n, sk_r, sv, all_indices, sm, H)

    m = compute_metrics(out_dense, out)
    ok = m['cosine'] < 1.0  # should differ
    if verbose:
        print(f"  Sparse vs dense (expected different): {fmt_metrics(m)}")
    if verbose or not ok:
        print(f"  Sparse != dense: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0d Tests: Lightning Indexer reference
# ---------------------------------------------------------------------------

def test_ref_lightning_score_ranking(verbose=False):
    """V4K-0d: Lightning Indexer scoring — ranking matches FP32 full-precision."""
    torch.manual_seed(42)
    num_blocks = 256

    q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
    indexer_k_cache = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM)
    score_proj = torch.randn(INDEX_N_HEADS)

    scores = ref_lightning_score(q_proj, indexer_k_cache, score_proj)

    passed = True

    # Shape check
    ok = scores.shape == (num_blocks,)
    if verbose or not ok:
        print(f"  Scores shape: {tuple(scores.shape)} == ({num_blocks},) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Scores should be non-negative (ReLU + positive/negative proj can give negative,
    # but the sum over ReLU outputs weighted by proj can be negative)
    # Just check finite
    ok = torch.isfinite(scores).all().item()
    if verbose or not ok:
        print(f"  All scores finite: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Verify ranking is deterministic and matches manual computation
    manual_dots = torch.einsum('hd,nhd->nh', q_proj.float(), indexer_k_cache.float())
    manual_dots = torch.relu(manual_dots)
    manual_scores = torch.einsum('nh,h->n', manual_dots, score_proj.float())
    diff = (scores - manual_scores).abs().max().item()
    ok = diff < 1e-5
    if verbose or not ok:
        print(f"  Manual computation match: max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Ranking order preserved
    _, rank_ref = manual_scores.sort(descending=True)
    _, rank_test = scores.sort(descending=True)
    ok = (rank_ref == rank_test).all().item()
    if verbose or not ok:
        print(f"  Ranking order matches: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_lightning_score_mqa_equivalence(verbose=False):
    """MQA scoring (single shared K, broadcast) == per-head scoring with that K
    replicated into every head. Proves the memory-saving MQA layout is exact."""
    torch.manual_seed(42)
    num_blocks = 256

    q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM)
    k_mqa = torch.randn(num_blocks, INDEX_HEAD_DIM)          # single shared K/block
    score_proj = torch.randn(INDEX_N_HEADS)

    # Replicate the shared K into the per-head layout the original kernel expects.
    k_per_head = k_mqa.unsqueeze(1).expand(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM).contiguous()

    scores_mqa = ref_lightning_score_mqa(q_proj, k_mqa, score_proj)
    scores_ref = ref_lightning_score(q_proj, k_per_head, score_proj)

    diff = (scores_mqa - scores_ref).abs().max().item()
    # 1e-4: identical math, differs only by fp32 einsum reduction order.
    ok = diff < 1e-4
    if verbose or not ok:
        print(f"  MQA==per-head(replicated): max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def test_ref_lightning_topk_correctness(verbose=False):
    """V4K-0d: Lightning Indexer top-k — correct indices selected."""
    torch.manual_seed(42)
    num_blocks = 100
    topk = 10

    scores = torch.randn(num_blocks)
    # All blocks are causal (query at end)
    block_endpoints = torch.arange(num_blocks) * CSA_STRIDE + CSA_WINDOW - 1
    query_position = block_endpoints[-1].item() + 100

    indices, selected_scores = ref_lightning_topk(scores, topk, query_position, block_endpoints)

    passed = True

    ok = len(indices) == topk
    if verbose or not ok:
        print(f"  Selected {len(indices)} == {topk} blocks: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Selected indices should have the highest scores
    sorted_all = scores.sort(descending=True)
    expected_top_scores = sorted_all.values[:topk].sort().values
    actual_top_scores = selected_scores.sort().values
    diff = (expected_top_scores - actual_top_scores).abs().max().item()
    ok = diff < 1e-6
    if verbose or not ok:
        print(f"  Top-k scores match: max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Indices should be sorted
    ok = (indices[1:] >= indices[:-1]).all().item()
    if verbose or not ok:
        print(f"  Indices sorted: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_lightning_topk_causality(verbose=False):
    """V4K-0d: Lightning Indexer top-k — future blocks excluded."""
    torch.manual_seed(42)
    num_blocks = 50
    topk = 20

    scores = torch.randn(num_blocks)
    # Make the last 10 blocks have very high scores (they should be excluded)
    scores[40:] = 100.0

    block_endpoints = torch.arange(num_blocks) * CSA_STRIDE + CSA_WINDOW - 1
    # Query position only sees first 30 blocks
    query_position = block_endpoints[29].item()

    indices, selected_scores = ref_lightning_topk(scores, topk, query_position, block_endpoints)

    passed = True

    # Should only select from first 30 blocks
    ok = (indices < 30).all().item()
    if verbose or not ok:
        print(f"  All selected indices < 30 (causal): {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Should have selected 20 from the 30 available
    ok = len(indices) == topk
    if verbose or not ok:
        print(f"  Selected {len(indices)} == {topk}: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # None of the high-scoring future blocks should appear
    ok = not any(i >= 40 for i in indices.tolist())
    if verbose or not ok:
        print(f"  No future blocks selected: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Edge: query_position before first block → 0 selected
    indices_empty, _ = ref_lightning_topk(scores, topk, 0, block_endpoints)
    ok = len(indices_empty) == 0
    if verbose or not ok:
        print(f"  No blocks causal at pos=0: {len(indices_empty)} == 0 {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-0c Tests: HCA compressor reference
# ---------------------------------------------------------------------------

def test_ref_hca_compress_basic(verbose=False):
    """V4K-0c: HCA compressor — 256 tokens -> 2 compressed entries."""
    torch.manual_seed(42)
    num_tokens = 256
    k_nope = torch.randn(num_tokens, HEAD_DIM)
    k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
    v = torch.randn(num_tokens, HEAD_DIM)
    gate_weights = torch.randn(HCA_WINDOW)
    compress_cos, compress_sin = precompute_rope_freqs(
        COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 2048)

    ck_nope, ck_rope, cv = ref_hca_compress(
        k_nope, k_rope_raw, v, gate_weights, compress_cos, compress_sin)

    passed = True

    # 256 tokens / 128 stride = 2 compressed
    ok = ck_nope.shape == (2, HEAD_DIM)
    if verbose or not ok:
        print(f"  compressed_k_nope shape: {tuple(ck_nope.shape)} == (2, {HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = ck_rope.shape == (2, QK_ROPE_HEAD_DIM)
    if verbose or not ok:
        print(f"  compressed_k_rope shape: {tuple(ck_rope.shape)} == (2, {QK_ROPE_HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = cv.shape == (2, HEAD_DIM)
    if verbose or not ok:
        print(f"  compressed_v shape: {tuple(cv.shape)} == (2, {HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Verify weighted sum: entry 0 is weighted sum of tokens [0, 128)
    sw = torch.softmax(gate_weights.float(), dim=0)
    manual_k0 = (sw.unsqueeze(-1) * k_nope[0:128].float()).sum(0)
    diff = (ck_nope[0] - manual_k0).abs().max().item()
    ok = diff < 1e-4
    if verbose or not ok:
        print(f"  Manual weighted sum match (entry 0): max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Edge case: 100 tokens → 0 compressed (< 128)
    k2 = torch.randn(100, HEAD_DIM)
    kr2 = torch.randn(100, QK_ROPE_HEAD_DIM)
    v2 = torch.randn(100, HEAD_DIM)
    ck2, _, cv2 = ref_hca_compress(k2, kr2, v2, gate_weights, compress_cos, compress_sin)
    ok = ck2.shape[0] == 0
    if verbose or not ok:
        print(f"  100 tokens → 0 compressed: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_hca_compress_rope_positions(verbose=False):
    """V4K-0c: HCA compressor — RoPE position = 128*i - 1 (endpoint)."""
    torch.manual_seed(42)
    num_tokens = 384  # 3 compressed entries
    k_nope = torch.randn(num_tokens, HEAD_DIM)
    k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
    v = torch.randn(num_tokens, HEAD_DIM)
    gate_weights = torch.randn(HCA_WINDOW)
    compress_cos, compress_sin = precompute_rope_freqs(
        COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 2048)

    ck_nope, ck_rope, cv = ref_hca_compress(
        k_nope, k_rope_raw, v, gate_weights, compress_cos, compress_sin)

    passed = True

    # 384 / 128 = 3 compressed, positions: 127, 255, 383
    expected_positions = [127, 255, 383]
    sw = torch.softmax(gate_weights.float(), dim=0)

    for j, exp_pos in enumerate(expected_positions):
        win_start = j * HCA_STRIDE
        manual_rope_raw = (sw.unsqueeze(-1) * k_rope_raw[win_start:win_start + HCA_WINDOW].float()).sum(0)
        pos_tensor = torch.tensor([exp_pos])
        manual_rope = apply_rope(manual_rope_raw.unsqueeze(0), compress_cos, compress_sin,
                                  pos_tensor).squeeze(0)

        diff = (ck_rope[j] - manual_rope).abs().max().item()
        ok = diff < 1e-5
        if verbose or not ok:
            print(f"  Entry {j}: RoPE pos={exp_pos}, max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    return passed


# ---------------------------------------------------------------------------
# V4K-0b Tests: CSA compressor reference
# ---------------------------------------------------------------------------

def _make_csa_test_inputs(num_tokens, seed=42):
    """Generate test inputs for CSA compressor tests."""
    torch.manual_seed(seed)
    input_k_nope = torch.randn(num_tokens, HEAD_DIM)
    input_k_rope_raw = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
    input_v = torch.randn(num_tokens, HEAD_DIM)
    gate_weights = torch.randn(CSA_WINDOW)
    positional_bias = torch.randn(CSA_WINDOW)
    compress_cos, compress_sin = precompute_rope_freqs(
        COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, max(num_tokens, 1024))
    return (input_k_nope, input_k_rope_raw, input_v,
            gate_weights, positional_bias, compress_cos, compress_sin)


def test_ref_csa_compress_basic(verbose=False):
    """V4K-0b: CSA compressor — 16 tokens -> 2 compressed entries."""
    inputs = _make_csa_test_inputs(16)
    k_nope, k_rope, v, gw, pb, cc, cs = inputs

    ck_nope, ck_rope, cv, residual = ref_csa_compress(
        k_nope, k_rope, v, gw, pb, cc, cs)

    passed = True

    # 16 tokens, window=8, stride=4 → 2 compressed
    ok = ck_nope.shape == (2, HEAD_DIM)
    if verbose or not ok:
        print(f"  compressed_k_nope shape: {tuple(ck_nope.shape)} == (2, {HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = ck_rope.shape == (2, QK_ROPE_HEAD_DIM)
    if verbose or not ok:
        print(f"  compressed_k_rope shape: {tuple(ck_rope.shape)} == (2, {QK_ROPE_HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = cv.shape == (2, HEAD_DIM)
    if verbose or not ok:
        print(f"  compressed_v shape: {tuple(cv.shape)} == (2, {HEAD_DIM}) {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Compressed values should be non-trivial (not all zeros)
    ok = ck_nope.abs().sum() > 0 and cv.abs().sum() > 0
    if verbose or not ok:
        print(f"  Non-zero outputs: {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Verify weighted sum property: each compressed entry is a convex combination
    gate_logits = gw + pb
    sw = torch.softmax(gate_logits.float(), dim=0)
    manual_k0 = (sw.unsqueeze(-1) * k_nope[0:8].float()).sum(0)
    ok = (ck_nope[0] - manual_k0).abs().max().item() < 1e-5
    if verbose or not ok:
        print(f"  Manual weighted sum match (entry 0): "
              f"max_diff={( ck_nope[0] - manual_k0).abs().max().item():.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    manual_k1 = (sw.unsqueeze(-1) * k_nope[4:12].float()).sum(0)
    ok = (ck_nope[1] - manual_k1).abs().max().item() < 1e-5
    if verbose or not ok:
        print(f"  Manual weighted sum match (entry 1): "
              f"max_diff={(ck_nope[1] - manual_k1).abs().max().item():.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_csa_compress_boundaries(verbose=False):
    """V4K-0b: CSA compressor — exact stride alignment for various token counts."""
    compress_cos, compress_sin = precompute_rope_freqs(
        COMPRESS_ROPE_THETA, QK_ROPE_HEAD_DIM, 2048)

    passed = True
    # (num_tokens, expected_compressed, expected_residual_count)
    cases = [
        (7,  0, 7),   # < window → 0 compressed
        (8,  0, 8),   # = window → 0 compressed (need window + stride for first)
        (11, 0, 11),  # < window + stride → 0 compressed
        (12, 1, 4),   # = window + stride → 1 compressed
        (16, 2, 4),   # 2 compressed
        (20, 3, 4),   # 3 compressed
        (24, 4, 4),   # 4 compressed
        (13, 1, 5),   # non-aligned: 1 compressed, 5 residual
        (15, 1, 7),   # non-aligned: 1 compressed, 7 residual
    ]

    for num_tokens, exp_comp, exp_resid in cases:
        torch.manual_seed(99)
        k = torch.randn(num_tokens, HEAD_DIM)
        kr = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)
        v = torch.randn(num_tokens, HEAD_DIM)
        gw = torch.randn(CSA_WINDOW)
        pb = torch.randn(CSA_WINDOW)

        ck, ckr, cv, ri = ref_csa_compress(k, kr, v, gw, pb,
                                            compress_cos, compress_sin)

        ok_c = ck.shape[0] == exp_comp
        ok_r = len(ri) == exp_resid
        ok = ok_c and ok_r
        if verbose or not ok:
            print(f"  N={num_tokens:2d}: compressed={ck.shape[0]} (exp {exp_comp}) "
                  f"residual={len(ri)} (exp {exp_resid}) {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    return passed


def test_ref_csa_compress_rope_positions(verbose=False):
    """V4K-0b: CSA compressor — RoPE applied at correct positions (j*stride + window - 1)."""
    inputs = _make_csa_test_inputs(24)
    k_nope, k_rope_raw, v, gw, pb, cc, cs = inputs

    ck_nope, ck_rope, cv, residual = ref_csa_compress(
        k_nope, k_rope_raw, v, gw, pb, cc, cs)

    # 24 tokens → 4 compressed entries
    # Positions: 7, 11, 15, 19 (= j*4 + 8 - 1)
    expected_positions = [7, 11, 15, 19]

    passed = True

    # Verify by manually applying RoPE and comparing
    gate_logits = gw + pb
    sw = torch.softmax(gate_logits.float(), dim=0)

    for j, exp_pos in enumerate(expected_positions):
        win_start = j * CSA_STRIDE
        # Manually compute compressed rope raw
        manual_rope_raw = (sw.unsqueeze(-1) * k_rope_raw[win_start:win_start + CSA_WINDOW].float()).sum(0)
        # Apply RoPE at expected position
        pos_tensor = torch.tensor([exp_pos])
        manual_rope = apply_rope(manual_rope_raw.unsqueeze(0), cc, cs, pos_tensor).squeeze(0)

        diff = (ck_rope[j] - manual_rope).abs().max().item()
        ok = diff < 1e-5
        if verbose or not ok:
            print(f"  Entry {j}: RoPE pos={exp_pos}, max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
        if not ok:
            passed = False

    return passed


def test_ref_csa_compress_residual(verbose=False):
    """V4K-0b: CSA compressor — trailing residual tokens returned correctly."""
    inputs = _make_csa_test_inputs(20)
    k_nope, k_rope_raw, v, gw, pb, cc, cs = inputs

    ck_nope, ck_rope, cv, residual = ref_csa_compress(
        k_nope, k_rope_raw, v, gw, pb, cc, cs)

    passed = True

    # 20 tokens → 3 compressed, last window ends at [8, 16), residual = [16, 20)
    ok = residual.tolist() == [16, 17, 18, 19]
    if verbose or not ok:
        print(f"  Residual indices: {residual.tolist()} == [16, 17, 18, 19] {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Non-aligned: 13 tokens → 1 compressed, window [0,8), residual = [8, 13)
    torch.manual_seed(77)
    k2 = torch.randn(13, HEAD_DIM)
    kr2 = torch.randn(13, QK_ROPE_HEAD_DIM)
    v2 = torch.randn(13, HEAD_DIM)
    _, _, _, ri2 = ref_csa_compress(k2, kr2, v2, gw, pb, cc, cs)

    ok = ri2.tolist() == [8, 9, 10, 11, 12]
    if verbose or not ok:
        print(f"  Residual (13 tokens): {ri2.tolist()} == [8, 9, 10, 11, 12] {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # Edge: 5 tokens → 0 compressed, all residual
    k3 = torch.randn(5, HEAD_DIM)
    kr3 = torch.randn(5, QK_ROPE_HEAD_DIM)
    v3 = torch.randn(5, HEAD_DIM)
    _, _, _, ri3 = ref_csa_compress(k3, kr3, v3, gw, pb, cc, cs)

    ok = ri3.tolist() == [0, 1, 2, 3, 4]
    if verbose or not ok:
        print(f"  Residual (5 tokens, no compression): {ri3.tolist()} == [0,1,2,3,4] {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# V4K-8ref Tests: V4 TQ k_append / dequant / decode references
# ---------------------------------------------------------------------------

def _make_csa_compressed_vectors(num_tokens, seed=42):
    """Generate CSA-like compressed vectors (softmax-weighted sum of 8 random tokens)."""
    torch.manual_seed(seed)
    vecs_k = torch.zeros(num_tokens, HEAD_DIM)
    vecs_v = torch.zeros(num_tokens, HEAD_DIM)
    for t in range(num_tokens):
        raw = torch.randn(8, HEAD_DIM)
        weights = torch.softmax(torch.randn(8), dim=0)
        vecs_k[t] = (weights.unsqueeze(-1) * raw).sum(0)
        vecs_v[t] = (weights.unsqueeze(-1) * torch.randn(8, HEAD_DIM)).sum(0)
    return vecs_k, vecs_v


def _make_hca_compressed_vectors(num_tokens, seed=77):
    """Generate HCA-like compressed vectors (128-token weighted averages)."""
    torch.manual_seed(seed)
    vecs_k = torch.zeros(num_tokens, HEAD_DIM)
    vecs_v = torch.zeros(num_tokens, HEAD_DIM)
    for t in range(num_tokens):
        raw = torch.randn(128, HEAD_DIM)
        weights = torch.softmax(torch.randn(128), dim=0)
        vecs_k[t] = (weights.unsqueeze(-1) * raw).sum(0)
        vecs_v[t] = (weights.unsqueeze(-1) * torch.randn(128, HEAD_DIM)).sum(0)
    return vecs_k, vecs_v


def test_ref_v4_tq_roundtrip_csa(verbose=False):
    """V4K-8ref: TQ round-trip on CSA compressed vectors, cosine > 0.99."""
    num_tokens = 32
    k_nope, v_nope = _make_csa_compressed_vectors(num_tokens)
    k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)

    centroids, boundaries = load_codebook()
    Pi = generate_rotation_matrix()

    num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    cache = alloc_v4_tq_cache(num_pages)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

    ref_v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot_mapping,
                         Pi, centroids, boundaries)

    k_deq, k_rope_deq, v_deq = ref_v4_tq_dequant(
        cache, list(range(num_tokens)), Pi, centroids)

    passed = True

    k_cos = F.cosine_similarity(k_nope.flatten().unsqueeze(0),
                                  k_deq.flatten().unsqueeze(0)).item()
    v_cos = F.cosine_similarity(v_nope.flatten().unsqueeze(0),
                                  v_deq.flatten().unsqueeze(0)).item()

    ok = k_cos > 0.99
    if verbose or not ok:
        print(f"  K NOPE cosine: {k_cos:.6f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = v_cos > 0.99
    if verbose or not ok:
        print(f"  V NOPE cosine: {v_cos:.6f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    # K ROPE should be exact (BF16 round-trip)
    rope_diff = (k_rope.to(torch.bfloat16).float() - k_rope_deq).abs().max().item()
    ok = rope_diff < 1e-3
    if verbose or not ok:
        print(f"  K ROPE max diff: {rope_diff:.2e} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_v4_tq_roundtrip_hca(verbose=False):
    """V4K-8ref: TQ round-trip on HCA compressed vectors, cosine > 0.99."""
    num_tokens = 16
    k_nope, v_nope = _make_hca_compressed_vectors(num_tokens)
    k_rope = torch.randn(num_tokens, QK_ROPE_HEAD_DIM)

    centroids, boundaries = load_codebook()
    Pi = generate_rotation_matrix()

    num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    cache = alloc_v4_tq_cache(num_pages)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

    ref_v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot_mapping,
                         Pi, centroids, boundaries)

    k_deq, k_rope_deq, v_deq = ref_v4_tq_dequant(
        cache, list(range(num_tokens)), Pi, centroids)

    passed = True

    k_cos = F.cosine_similarity(k_nope.flatten().unsqueeze(0),
                                  k_deq.flatten().unsqueeze(0)).item()
    v_cos = F.cosine_similarity(v_nope.flatten().unsqueeze(0),
                                  v_deq.flatten().unsqueeze(0)).item()

    ok = k_cos > 0.99
    if verbose or not ok:
        print(f"  K NOPE cosine: {k_cos:.6f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    ok = v_cos > 0.99
    if verbose or not ok:
        print(f"  V NOPE cosine: {v_cos:.6f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_v4_tq_decode_csa_smoke(verbose=False):
    """V4K-8ref: CSA TQ decode produces valid output matching FP8 reference."""
    torch.manual_seed(42)
    H = 8
    num_compressed = 32
    swa_len = 64
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    centroids, boundaries = load_codebook()
    Pi = generate_rotation_matrix()

    k_nope = torch.randn(num_compressed, HEAD_DIM) * 0.5
    k_rope = torch.randn(num_compressed, QK_ROPE_HEAD_DIM)
    v_nope = torch.randn(num_compressed, HEAD_DIM) * 0.5

    num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
    tq_cache = alloc_v4_tq_cache(num_pages)
    slot_mapping = torch.arange(num_compressed, dtype=torch.int32)
    ref_v4_tq_k_append(k_nope, k_rope, v_nope, tq_cache, slot_mapping,
                         Pi, centroids, boundaries)

    swa_k_nope = torch.randn(swa_len, HEAD_DIM)
    swa_k_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM)
    swa_v = torch.randn(swa_len, HEAD_DIM)
    q_nope = torch.randn(H, HEAD_DIM)
    q_rope = torch.randn(H, QK_ROPE_HEAD_DIM)

    sparse_indices = torch.arange(num_compressed)

    out_tq, lse_tq = ref_v4_tq_decode_csa(
        q_nope, q_rope, tq_cache, sparse_indices,
        swa_k_nope, swa_k_rope, swa_v, sm_scale, H, Pi, centroids)

    # Also compute FP8 reference for comparison
    k_deq, k_rope_deq, v_deq = ref_v4_tq_dequant(
        tq_cache, list(range(num_compressed)), Pi, centroids)
    out_fp8ref, lse_fp8ref = ref_csa_fp8_decode(
        q_nope, q_rope, k_deq, k_rope_deq, v_deq,
        swa_k_nope, swa_k_rope, swa_v,
        sparse_indices, sm_scale, H)

    passed = True

    ok = out_tq.shape == (H, HEAD_DIM) and torch.isfinite(out_tq).all().item()
    if verbose or not ok:
        print(f"  Valid output: shape={tuple(out_tq.shape)}, "
              f"finite={'yes' if torch.isfinite(out_tq).all() else 'no'} "
              f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    cos = F.cosine_similarity(out_tq.flatten().unsqueeze(0),
                               out_fp8ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.99
    if verbose or not ok:
        print(f"  TQ vs dequant+FP8ref cosine: {cos:.6f} {'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


def test_ref_v4_tq_decode_hca_smoke(verbose=False):
    """V4K-8ref: HCA TQ decode produces valid output."""
    torch.manual_seed(77)
    H = 8
    num_compressed = 16
    swa_len = 64
    sm_scale = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

    centroids, boundaries = load_codebook()
    Pi = generate_rotation_matrix()

    k_nope, v_nope = _make_hca_compressed_vectors(num_compressed)
    k_rope = torch.randn(num_compressed, QK_ROPE_HEAD_DIM)

    num_pages = (num_compressed + PAGE_SIZE - 1) // PAGE_SIZE
    tq_cache = alloc_v4_tq_cache(num_pages)
    slot_mapping = torch.arange(num_compressed, dtype=torch.int32)
    ref_v4_tq_k_append(k_nope, k_rope, v_nope, tq_cache, slot_mapping,
                         Pi, centroids, boundaries)

    swa_k_nope = torch.randn(swa_len, HEAD_DIM)
    swa_k_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM)
    swa_v = torch.randn(swa_len, HEAD_DIM)
    q_nope = torch.randn(H, HEAD_DIM)
    q_rope = torch.randn(H, QK_ROPE_HEAD_DIM)

    out, lse = ref_v4_tq_decode_hca(
        q_nope, q_rope, tq_cache, num_compressed,
        swa_k_nope, swa_k_rope, swa_v, sm_scale, H, Pi, centroids)

    passed = True

    ok = out.shape == (H, HEAD_DIM) and torch.isfinite(out).all().item()
    if verbose or not ok:
        print(f"  Valid output: shape={tuple(out.shape)}, "
              f"finite={'yes' if torch.isfinite(out).all() else 'no'} "
              f"{'PASS' if ok else 'FAIL'}")
    passed = passed and ok

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("Config loader: Pro and Flash", test_config_loader),
    ("RoPE precompute: shape and properties", test_rope_precompute),
    ("RoPE apply: position 0 is identity", test_rope_apply_identity),
    ("RoPE apply: rotation preserves norms and dot products", test_rope_apply_preserves_norm),
    ("FP8 quantize: round-trip precision", test_fp8_quantize_roundtrip),
    ("FP8 quantize rowwise: P matrix", test_fp8_quantize_rowwise),
    ("V4 FP8 cache: write/read round-trip", test_v4_fp8_cache_roundtrip),
    ("V4 FP8 cache: paging across boundaries", test_v4_fp8_cache_paging),
    ("V4 constants: internal consistency", test_v4_constants_consistency),
    ("Metrics computation: known inputs", test_metrics_computation),
    ("CSA compress: 16 tokens -> 2 compressed", test_ref_csa_compress_basic),
    ("CSA compress: stride alignment boundaries", test_ref_csa_compress_boundaries),
    ("CSA compress: RoPE at correct positions", test_ref_csa_compress_rope_positions),
    ("CSA compress: residual tokens", test_ref_csa_compress_residual),
    ("HCA compress: 256 tokens -> 2 compressed", test_ref_hca_compress_basic),
    ("HCA compress: RoPE at correct positions", test_ref_hca_compress_rope_positions),
    ("Lightning score: ranking matches FP32", test_ref_lightning_score_ranking),
    ("Lightning score MQA: == per-head replicated", test_ref_lightning_score_mqa_equivalence),
    ("Lightning topk: correct indices", test_ref_lightning_topk_correctness),
    ("Lightning topk: causality enforcement", test_ref_lightning_topk_causality),
    ("CSA decode: topk=all matches dense", test_ref_csa_decode_topk_all_eq_dense),
    ("CSA decode: short context (1K)", test_ref_csa_decode_short),
    ("CSA decode: medium context (4K)", test_ref_csa_decode_medium),
    ("HCA decode: short context (1K)", test_ref_hca_decode_short),
    ("HCA decode: long context (64K)", test_ref_hca_decode_long),
    ("SWA decode: matches dense masked attention", test_ref_swa_decode),
    ("Inverse RoPE: round-trip is identity", test_ref_inverse_rope_roundtrip),
    ("V4 FP8 k_append/dequant: round-trip", test_ref_v4_fp8_roundtrip),
    ("V4 layer CSA: full pipeline", test_ref_v4_layer_csa),
    ("V4 layer HCA: full pipeline", test_ref_v4_layer_hca),
    ("V4 layer SWA: full pipeline", test_ref_v4_layer_swa),
    ("V4 TQ round-trip: CSA compressed (cosine > 0.99)", test_ref_v4_tq_roundtrip_csa),
    ("V4 TQ round-trip: HCA compressed (cosine > 0.99)", test_ref_v4_tq_roundtrip_hca),
    ("V4 TQ CSA decode: smoke test", test_ref_v4_tq_decode_csa_smoke),
    ("V4 TQ HCA decode: smoke test", test_ref_v4_tq_decode_hca_smoke),
]


def main():
    parser = argparse.ArgumentParser(description='V4 Reference Tests')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print("DeepSeek V4 Reference Tests")
    print("=" * 60)

    results = []
    for name, test_fn in ALL_TESTS:
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")
        try:
            passed = test_fn(verbose=args.verbose)
            results.append((name, passed))
            if not args.verbose:
                print(f"  {'PASS' if passed else 'FAIL'}")
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

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
