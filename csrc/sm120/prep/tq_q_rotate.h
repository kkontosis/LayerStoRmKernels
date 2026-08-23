#pragma once
/***************************************************************************************************
 * TQ Q-Rotate: Pre-rotate query NOPE by Pi^T for TQ score computation
 *
 * q_rot = q_nope @ Pi^T  (BF16 input → FP32 output)
 *
 * Each head's d_c-dimensional NOPE vector is multiplied by the d_c × d_c
 * rotation matrix Pi^T. Done once per decode step per layer (not per KV token).
 *
 * Grid: (batch_heads, 1, 1) — one CTA per (batch * s_q * h_q) element
 * Block: 256 threads
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct TqQRotateParams {
    const __nv_bfloat16* __restrict__ q_nope;  // [batch_heads, d_c] BF16
    const float* __restrict__ Pi;               // [d_c, d_c] float32 rotation matrix
    float* __restrict__ q_rot;                  // [batch_heads, d_c] FP32 output
    int batch_heads;  // b * s_q * h_q
    int d_c;
    // Elements between consecutive head rows of q_nope (0 → d_c, contiguous).
    // Lets the rotate consume an interleaved [nope|rope] per-head layout
    // (row stride d_c + d_rope) without a repack (TD-TQ-SPARSE-DECODE-UNWIRED).
    int q_row_stride = 0;
};

void run_tq_q_rotate(const TqQRotateParams& params, cudaStream_t stream);

// §12l warp-per-output row-GEMV: out[h][j] = Σ_k in[h][k]·M[j][k].
// Forward rotation: M = Pi. Inverse rotation: M = Pi^T (precomputed).
void run_tq_rotate_rows_bf16_to_f32(const __nv_bfloat16* in, const float* M,
                                    float* out, int batch_heads, int d_c,
                                    int in_row_stride, cudaStream_t stream);
void run_tq_rotate_rows_f32_to_bf16(const float* in, const float* M,
                                    __nv_bfloat16* out, int batch_heads,
                                    int d_c, cudaStream_t stream);

}  // namespace sm120::prep
