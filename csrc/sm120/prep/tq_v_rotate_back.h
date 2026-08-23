#pragma once
/***************************************************************************************************
 * TQ V-Rotate-Back: Apply inverse rotation to decode output (rotated space → original space)
 *
 * out_final = out_rotated @ Pi
 *
 * Applied once after mla_combine to convert the rotated-space accumulator
 * back to the original latent space. This avoids a d_c × d_c matmul per KV token
 * during decode (the rotation factors out of the weighted sum).
 *
 * Grid: (batch_heads, 1, 1)
 * Block: 256 threads
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct TqVRotateBackParams {
    const float* __restrict__ out_rotated;     // [batch_heads, d_c] FP32 rotated-space output
    const float* __restrict__ Pi;               // [d_c, d_c] rotation matrix
    __nv_bfloat16* __restrict__ out_final;     // [batch_heads, d_c] BF16 final output
    // §12l: precomputed Pi^T (rows = Pi columns). When non-null and d_c<=512
    // the inverse rotation runs the warp-per-output row-GEMV (fast path).
    const float* Pi_t = nullptr;
    int batch_heads;
    int d_c;
};

void run_tq_v_rotate_back(const TqVRotateBackParams& params, cudaStream_t stream);

}  // namespace sm120::prep
