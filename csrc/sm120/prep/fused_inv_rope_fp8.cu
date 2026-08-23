#pragma once
#include "fused_inv_rope_fp8.h"

namespace sm120::prep {

static constexpr float INV_ROPE_FP8_MAX = 448.0f;

__global__ void __launch_bounds__(256)
fused_inv_rope_fp8_kernel(const FusedInvRopeFp8Params params) {
    const int row = blockIdx.x;
    if (row >= params.N) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;
    const int nope_dim = hd - rd;  // 448
    const int half_rope = rd / 2;  // 32
    const int base = tid * 2;

    const __nv_bfloat16* x_row = params.x + (int64_t)row * hd;
    float v0, v1;

    if (base < nope_dim) {
        // NOPE dims: pass through
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(x_row) + tid);
        v0 = __low2float(pair);
        v1 = __high2float(pair);
    } else if (base < hd) {
        // ROPE dims: apply inverse RoPE
        int rope_idx = (base - nope_dim) / 2;  // which pair (0..31)
        int pos = __ldg(params.positions + row);
        float c = __ldg(params.cos_table + pos * half_rope + rope_idx);
        float s = __ldg(params.sin_table + pos * half_rope + rope_idx);

        float x_even = __bfloat162float(x_row[base]);
        float x_odd  = __bfloat162float(x_row[base + 1]);

        // Inverse RoPE: negate sin in even component
        v0 = x_even * c + x_odd * s;
        v1 = -x_even * s + x_odd * c;
    } else {
        v0 = 0.0f; v1 = 0.0f;
    }

    // Per-row amax reduction
    float local_amax = (base < hd) ? fmaxf(fabsf(v0), fabsf(v1)) : 0.0f;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));

    __shared__ float sWarpMax[8];
    if (tid % 32 == 0) sWarpMax[tid / 32] = local_amax;
    __syncthreads();
    if (tid == 0) {
        float m = sWarpMax[0];
        for (int w = 1; w < 8; ++w) m = fmaxf(m, sWarpMax[w]);
        sWarpMax[0] = m;
    }
    __syncthreads();

    float scale = sWarpMax[0] / INV_ROPE_FP8_MAX;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Write FP8 + scale
    __nv_fp8_e4m3* out = params.out_fp8 + (int64_t)row * hd;
    if (base < hd) {
        out[base]     = __nv_fp8_e4m3(fmaxf(-INV_ROPE_FP8_MAX, fminf(INV_ROPE_FP8_MAX, v0 * inv_scale)));
        out[base + 1] = __nv_fp8_e4m3(fmaxf(-INV_ROPE_FP8_MAX, fminf(INV_ROPE_FP8_MAX, v1 * inv_scale)));
    }
    if (tid == 0) params.out_scales[row] = scale;
}

void run_fused_inv_rope_fp8(const FusedInvRopeFp8Params& params, cudaStream_t stream) {
    if (params.N == 0) return;
    fused_inv_rope_fp8_kernel<<<params.N, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
