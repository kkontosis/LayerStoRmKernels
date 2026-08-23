#pragma once
/***************************************************************************************************
 * RoPE-Rotate: in-place interleaved-pair rotary position embedding (DeepSeek MLA)
 *
 * Rotates the decoupled-RoPE slice of q_pe / k_pe by each token's absolute position,
 * using the DeepSeek complex-multiply (interleaved adjacent-pair) convention
 * (ref/DeepSeek-V3 inference/model.py apply_rotary_emb, view_as_complex):
 *
 *     for pair i in [0, d_rope/2):
 *         o[2i]   = x[2i]·cos_i − x[2i+1]·sin_i
 *         o[2i+1] = x[2i]·sin_i + x[2i+1]·cos_i
 *
 * NOT NeoX half-split. cos/sin are PURE (no YaRN mscale folded in — mscale belongs in
 * the softmax scale only, per DeepSeek model.py:434-437). YaRN enters only through the
 * frequencies baked into the cos_sin table by the host builder.
 *
 * The SnapMLA pipeline expects rope ALREADY rotated at kernel input (paper Eq. 2:
 * k_R = RoPE(W_KR·h_t)); fused_q_quant / fused_k_append then pre-scale the rotated
 * BF16 rope by the inverse content scale (paper Eq. 6). So this kernel runs upstream,
 * before those kernels. TurboQuant likewise carries rope as BF16 pass-through.
 *
 * Position source: the engine batch model is one new token per sequence per step, so
 * the absolute position of token t is seqlens_k[t] − 1 (seqlens_k = tokens-to-attend
 * including the new one). rows_per_token generalizes the layout:
 *   - k_pe:  [num_tokens, row_stride] with the 64-dim rope at some offset → rows_per_token=1
 *   - q_pe:  [num_tokens, h_q, row_stride] → rows_per_token=h_q (all heads share the position)
 * `x` points at the FIRST rope element of row 0; row r's rope starts at x + r*row_stride.
 *
 * cos_sin layout: [max_pos][d_rope] float32 — per position: d_rope/2 cos values then
 * d_rope/2 sin values (one coalesced row read).
 *
 * Grid: (num_tokens * rows_per_token, 1, 1) — one CTA per rope row. Block: 64 threads
 * (d_rope=64: thread d handles element d; pairs resolved via __shfl_sync).
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct RopeRotateParams {
    __nv_bfloat16* __restrict__ x;        // first rope element of row 0 (in-place)
    const int*     __restrict__ seqlens_k; // [num_tokens]; pos = seqlens_k[t] − 1
    const float*   __restrict__ cos_sin;   // [max_pos][d_rope]: d_rope/2 cos | d_rope/2 sin
    int     num_tokens;
    int     rows_per_token;   // 1 for k_pe, h_q for q_pe
    int64_t row_stride;       // elements between consecutive rope rows
    int     d_rope;           // rope dim (64), must be even
    int     max_pos;          // cos_sin rows (positions are clamped to max_pos−1)
};

void run_rope_rotate(const RopeRotateParams& params, cudaStream_t stream);

}  // namespace sm120::prep
