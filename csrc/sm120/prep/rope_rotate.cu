#pragma once
#include "rope_rotate.h"

namespace sm120::prep {

// One CTA per rope row (token×head); thread i rotates adjacent pair (2i, 2i+1).
// FP32 math, BF16 in/out, in-place. Interleaved (complex-multiply) convention.
__global__ void __launch_bounds__(64)
rope_rotate_kernel(const RopeRotateParams params) {
    const int row = blockIdx.x;
    const int total_rows = params.num_tokens * params.rows_per_token;
    if (row >= total_rows) return;

    const int token = row / params.rows_per_token;
    int pos = __ldg(params.seqlens_k + token) - 1;
    if (pos < 0) pos = 0;
    if (pos >= params.max_pos) pos = params.max_pos - 1;

    const int half = params.d_rope / 2;
    const float* cs = params.cos_sin + static_cast<int64_t>(pos) * params.d_rope;
    __nv_bfloat16* x = params.x + static_cast<int64_t>(row) * params.row_stride;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = __ldg(cs + i);
        const float s = __ldg(cs + half + i);
        const float x0 = __bfloat162float(x[2 * i]);
        const float x1 = __bfloat162float(x[2 * i + 1]);
        x[2 * i]     = __float2bfloat16_rn(x0 * c - x1 * s);
        x[2 * i + 1] = __float2bfloat16_rn(x0 * s + x1 * c);
    }
}

void run_rope_rotate(const RopeRotateParams& params, cudaStream_t stream) {
    if (params.num_tokens <= 0 || params.rows_per_token <= 0 || params.d_rope <= 0) return;
    const int total_rows = params.num_tokens * params.rows_per_token;
    rope_rotate_kernel<<<total_rows, 64, 0, stream>>>(params);
}

}  // namespace sm120::prep
