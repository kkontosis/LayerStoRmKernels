"""
GGUF GEMM Reference Implementation (Q2_K/Q3_K/Q5_K/Q6_K/Q8_0)

Golden numpy oracles for the GGUF dequant-GEMM kernels. Each dequant is an
independent, vectorized reimplementation of ggml's canonical CPU dequant
(ggml-quants.c `dequantize_row_*`) — deliberately NOT a transcription of the
CUDA-kernel decomposition, so it independently validates kernel ordering+math.

(Q4_K has its own richer reference in test_q4k_gemm_reference.py, including a
float→Q4_K quantizer; here we validate the GPU kernel against dequant of
randomly generated packed blocks, which exercises both kernel dequant and GEMM.)

Block layouts (binary-compatible with ggml-common.h):
  Q8_0  34 B  : d(f16) | qs[32] int8
  Q2_K  84 B  : scales[16] | qs[64] | d(f16) dmin(f16)
  Q3_K 110 B  : hmask[32] | qs[64] | scales[12] | d(f16)
  Q5_K 176 B  : d(f16) dmin(f16) | scales[12] | qh[32] | qs[128]
  Q6_K 210 B  : ql[128] | qh[64] | scales[16] int8 | d(f16)

Usage:  python tests/test_gguf_gemm_reference.py -v
"""

import numpy as np
import torch

QK_K = 256

BLOCK_BYTES = {
    "q2_k": 84, "q3_k": 110, "q4_k": 144, "q5_k": 176, "q6_k": 210, "q8_0": 34,
}
BLOCK_VALUES = {  # weights per packed block
    "q2_k": 256, "q3_k": 256, "q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32,
}

# DeepSeek MLA projection shapes (out_features, in_features); K divisible by 256.
PROJECTION_SHAPES = [
    ("q_a",    1536, 7168),
    ("q_b",   12288, 1536),
    ("o_proj",  7168, 8192),
]
BATCH_SIZES = [1, 8, 128, 1024]


def bytes_per_row(quant_type: str, K: int) -> int:
    qk = BLOCK_VALUES[quant_type]
    assert K % qk == 0, f"K={K} not divisible by {qk} for {quant_type}"
    return (K // qk) * BLOCK_BYTES[quant_type]


def _f16(buf: np.ndarray) -> np.ndarray:
    """View trailing 2-byte pairs as little-endian float16 -> float32."""
    return buf.view(np.float16).astype(np.float32)


def _get_scale_min_k4(j: int, scales: np.ndarray):
    """Vectorized ggml get_scale_min_k4 over [nb, 12] uint8. Returns (d, m) [nb]."""
    s = scales.astype(np.int32)
    if j < 4:
        d = s[:, j] & 63
        m = s[:, j + 4] & 63
    else:
        d = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
        m = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-type vectorized dequant: packed [nb, block_bytes] uint8 -> [nb, VALS] f32
# ---------------------------------------------------------------------------

def _dequant_q8_0(b: np.ndarray) -> np.ndarray:
    d = _f16(b[:, 0:2])                       # [nb,1]
    qs = b[:, 2:34].view(np.int8).astype(np.float32)
    return d * qs


def _dequant_q2k(b: np.ndarray) -> np.ndarray:
    nb = b.shape[0]
    scales = b[:, 0:16].astype(np.int32)
    qs = b[:, 16:80].astype(np.int32)
    d = _f16(b[:, 80:82])[:, 0:1]
    dmin = _f16(b[:, 82:84])[:, 0:1]
    out = np.empty((nb, 256), np.float32)
    for ni in range(2):
        qn = qs[:, 32 * ni:32 * ni + 32]
        for j in range(4):
            shift = 2 * j
            o = 128 * ni + 32 * j
            sca = scales[:, 8 * ni + 2 * j]
            scb = scales[:, 8 * ni + 2 * j + 1]
            dla = d[:, 0] * (sca & 0xF); mla = dmin[:, 0] * (sca >> 4)
            dlb = d[:, 0] * (scb & 0xF); mlb = dmin[:, 0] * (scb >> 4)
            out[:, o:o + 16] = dla[:, None] * ((qn[:, 0:16] >> shift) & 3) - mla[:, None]
            out[:, o + 16:o + 32] = dlb[:, None] * ((qn[:, 16:32] >> shift) & 3) - mlb[:, None]
    return out


def _dequant_q3k(b: np.ndarray) -> np.ndarray:
    nb = b.shape[0]
    hmask = b[:, 0:32].astype(np.int32)
    qs = b[:, 32:96].astype(np.int32)
    raw = b[:, 96:108].astype(np.uint32)
    d = _f16(b[:, 108:110])[:, 0]
    # Unpack the 16 6-bit scales via ggml's aux/kmask scheme (independent of the
    # per-element formula used in the device kernel).
    kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
    aux = np.zeros((nb, 4), np.uint32)
    aux[:, 0] = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16) | (raw[:, 3] << 24)
    aux[:, 1] = raw[:, 4] | (raw[:, 5] << 8) | (raw[:, 6] << 16) | (raw[:, 7] << 24)
    aux[:, 2] = raw[:, 8] | (raw[:, 9] << 8) | (raw[:, 10] << 16) | (raw[:, 11] << 24)
    tmp = aux[:, 2].copy()
    a0, a1 = aux[:, 0].copy(), aux[:, 1].copy()
    aux[:, 2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    aux[:, 3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux[:, 0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    aux[:, 1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
    scales = np.empty((nb, 16), np.int32)
    for w in range(4):
        for k in range(4):
            scales[:, w * 4 + k] = ((aux[:, w] >> (8 * k)) & 0xFF).astype(np.int32)
    scales -= 32
    out = np.empty((nb, 256), np.float32)
    for ni in range(2):
        qn = qs[:, 32 * ni:32 * ni + 32]
        hn = hmask  # hmask indexed [l] / [l+16] over full 32
        for j in range(4):
            shift = 2 * j
            o = 128 * ni + 32 * j
            m = 1 << (4 * ni + j)
            dl_a = d * scales[:, 8 * ni + 2 * j]
            dl_b = d * scales[:, 8 * ni + 2 * j + 1]
            qa = ((qn[:, 0:16] >> shift) & 3) - np.where((hn[:, 0:16] & m) != 0, 0, 4)
            qb = ((qn[:, 16:32] >> shift) & 3) - np.where((hn[:, 16:32] & m) != 0, 0, 4)
            out[:, o:o + 16] = dl_a[:, None] * qa
            out[:, o + 16:o + 32] = dl_b[:, None] * qb
    return out


def _dequant_q4k(b: np.ndarray) -> np.ndarray:
    nb = b.shape[0]
    d = _f16(b[:, 0:2])[:, 0]
    dmin = _f16(b[:, 2:4])[:, 0]
    scales = b[:, 4:16]
    qs = b[:, 16:144].astype(np.int32)
    out = np.empty((nb, 256), np.float32)
    for grp in range(4):
        d1f, m1f = _get_scale_min_k4(2 * grp, scales)
        d2f, m2f = _get_scale_min_k4(2 * grp + 1, scales)
        d1 = d * d1f; m1 = dmin * m1f
        d2 = d * d2f; m2 = dmin * m2f
        ql = qs[:, 32 * grp:32 * grp + 32]
        o = 64 * grp
        out[:, o:o + 32] = d1[:, None] * (ql & 0xF) - m1[:, None]
        out[:, o + 32:o + 64] = d2[:, None] * (ql >> 4) - m2[:, None]
    return out


def _dequant_q5k(b: np.ndarray) -> np.ndarray:
    nb = b.shape[0]
    d = _f16(b[:, 0:2])[:, 0]
    dmin = _f16(b[:, 2:4])[:, 0]
    scales = b[:, 4:16]
    qh = b[:, 16:48].astype(np.int32)
    ql = b[:, 48:176].astype(np.int32)
    out = np.empty((nb, 256), np.float32)
    for grp in range(4):
        d1f, m1f = _get_scale_min_k4(2 * grp, scales)
        d2f, m2f = _get_scale_min_k4(2 * grp + 1, scales)
        d1 = d * d1f; m1 = dmin * m1f
        d2 = d * d2f; m2 = dmin * m2f
        qlg = ql[:, 32 * grp:32 * grp + 32]
        u1 = 1 << (2 * grp)
        u2 = 1 << (2 * grp + 1)
        o = 64 * grp
        hi1 = np.where((qh & u1) != 0, 16, 0)
        hi2 = np.where((qh & u2) != 0, 16, 0)
        out[:, o:o + 32] = d1[:, None] * ((qlg & 0xF) + hi1) - m1[:, None]
        out[:, o + 32:o + 64] = d2[:, None] * ((qlg >> 4) + hi2) - m2[:, None]
    return out


def _dequant_q6k(b: np.ndarray) -> np.ndarray:
    nb = b.shape[0]
    ql = b[:, 0:128].astype(np.int32)
    qh = b[:, 128:192].astype(np.int32)
    scales = b[:, 192:208].view(np.int8).astype(np.int32)
    d = _f16(b[:, 208:210])[:, 0]
    out = np.empty((nb, 256), np.float32)
    l = np.arange(32)
    is_base = (l // 16)  # [32], 0 for l<16 else 1
    for half in range(2):
        o = 128 * half
        qlh = ql[:, 64 * half:64 * half + 64]
        H = qh[:, 32 * half:32 * half + 32]
        sch = scales[:, 8 * half:8 * half + 8]
        qlo = qlh[:, 0:32]; qhi = qlh[:, 32:64]
        sA = sch[:, is_base + 0]; sB = sch[:, is_base + 2]
        sC = sch[:, is_base + 4]; sD = sch[:, is_base + 6]
        q1 = ((qlo & 0xF) | (((H >> 0) & 3) << 4)) - 32
        q2 = ((qhi & 0xF) | (((H >> 2) & 3) << 4)) - 32
        q3 = ((qlo >> 4) | (((H >> 4) & 3) << 4)) - 32
        q4 = ((qhi >> 4) | (((H >> 6) & 3) << 4)) - 32
        out[:, o + 0:o + 32] = d[:, None] * sA * q1
        out[:, o + 32:o + 64] = d[:, None] * sB * q2
        out[:, o + 64:o + 96] = d[:, None] * sC * q3
        out[:, o + 96:o + 128] = d[:, None] * sD * q4
    return out


_DEQUANT = {
    "q2_k": _dequant_q2k, "q3_k": _dequant_q3k, "q4_k": _dequant_q4k,
    "q5_k": _dequant_q5k, "q6_k": _dequant_q6k, "q8_0": _dequant_q8_0,
}


def ref_gguf_dequant(packed: np.ndarray, quant_type: str, N: int, K: int) -> np.ndarray:
    """Packed [N, bytes_per_row] uint8 -> dequantized weight [N, K] float32."""
    qk = BLOCK_VALUES[quant_type]
    bb = BLOCK_BYTES[quant_type]
    bpr = K // qk
    blocks = packed.reshape(N * bpr, bb)
    vals = _DEQUANT[quant_type](blocks)               # [N*bpr, qk]
    return vals.reshape(N, K)


def ref_gguf_gemm(x: torch.Tensor, packed: np.ndarray, quant_type: str, N: int, K: int) -> torch.Tensor:
    """x[M,K] @ dequant(W)[N,K]^T -> [M,N], float32 reference."""
    W = torch.from_numpy(ref_gguf_dequant(packed, quant_type, N, K))
    return x.float().cpu() @ W.t()


# ---------------------------------------------------------------------------
# Random packed-weight generator (modest fp16 scales to keep magnitudes sane)
# ---------------------------------------------------------------------------

def _set_f16(block: np.ndarray, off: int, value: float):
    h = np.array([value], np.float16).view(np.uint8)
    block[off] = h[0]; block[off + 1] = h[1]


def generate_gguf_weight(quant_type: str, N: int, K: int, seed: int = 0) -> np.ndarray:
    """Random valid packed GGUF weight [N, bytes_per_row] uint8."""
    rng = np.random.default_rng(seed)
    qk = BLOCK_VALUES[quant_type]
    bb = BLOCK_BYTES[quant_type]
    bpr = K // qk
    nb = N * bpr
    blocks = rng.integers(0, 256, size=(nb, bb), dtype=np.uint8)
    # Pin the fp16 super-block scale(s) to a modest value per block.
    for i in range(nb):
        if quant_type == "q8_0":
            _set_f16(blocks[i], 0, 0.05)
        elif quant_type == "q2_k":
            _set_f16(blocks[i], 80, 0.06); _set_f16(blocks[i], 82, 0.04)
        elif quant_type == "q3_k":
            _set_f16(blocks[i], 108, 0.05)
        elif quant_type == "q4_k":
            _set_f16(blocks[i], 0, 0.06); _set_f16(blocks[i], 2, 0.04)
        elif quant_type == "q5_k":
            _set_f16(blocks[i], 0, 0.06); _set_f16(blocks[i], 2, 0.04)
        elif quant_type == "q6_k":
            _set_f16(blocks[i], 208, 0.05)
    return blocks.reshape(N, bpr * bb)


# ---------------------------------------------------------------------------
# Self-tests: sanity-check the references against direct scalar computation.
# ---------------------------------------------------------------------------

def _selftest():
    # Q8_0: y = d * qs, trivially checkable.
    b = np.zeros((1, 34), np.uint8)
    _set_f16(b[0], 0, 0.5)
    b[0, 2:34] = np.frombuffer(np.arange(-16, 16, dtype=np.int8).tobytes(), np.uint8)
    y = _dequant_q8_0(b)[0]
    exp = 0.5 * np.arange(-16, 16)
    assert np.allclose(y, exp, atol=1e-2), (y[:4], exp[:4])
    print("  q8_0 scalar check: OK")

    # Each dequant produces finite, correctly-shaped output on random blocks.
    for qt in ["q2_k", "q3_k", "q5_k", "q6_k", "q8_0"]:
        w = generate_gguf_weight(qt, 4, 512, seed=1)
        W = ref_gguf_dequant(w, qt, 4, 512)
        assert W.shape == (4, 512)
        assert np.isfinite(W).all(), qt
        assert np.abs(W).max() > 0, qt
        print(f"  {qt} dequant finite/shape: OK  (range {W.min():.3f}..{W.max():.3f})")
    print("PASS")


if __name__ == "__main__":
    _selftest()
