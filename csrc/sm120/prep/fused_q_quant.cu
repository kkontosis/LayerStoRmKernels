#pragma once
#include "fused_q_quant.h"

namespace sm120::prep {

__global__ void __launch_bounds__(256)
fused_q_quant_kernel(const FusedQQuantParams params) {
    const int head_idx = blockIdx.x;  // Flat index into [s_q * h_q]
    const int thread_idx = threadIdx.x;
    const int num_threads = blockDim.x;

    if (head_idx >= params.s_q * params.h_q) return;

    const int d_qk = params.d_qk;
    const int d_nope = params.d_nope;
    const int d_rope = d_qk - d_nope;

    const __nv_bfloat16* q_in = params.q_bf16 + head_idx * d_qk;
    __nv_fp8_e4m3* nope_out = params.q_nope_fp8 + head_idx * d_nope;
    __nv_bfloat16* rope_out = params.q_rope_bf16 + head_idx * d_rope;

    // =========================================================================
    // Single-pass: read NOPE data once, compute amax AND cache for quantization.
    // Each thread handles 2 consecutive BF16 values via vectorized bfloat162 load.
    // d_nope=512 → 256 pairs → exactly 1 pair per thread (256 threads).
    // d_nope=448 → 224 pairs → 224 threads active, 32 idle.
    // =========================================================================
    const int pair_count = d_nope / 2;
    const int base = thread_idx * 2;

    float cached_v0 = 0.0f, cached_v1 = 0.0f;
    float local_amax = 0.0f;

    if (base < d_nope) {
        // Vectorized load: 2 BF16 values in one 32-bit transaction
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(q_in) + thread_idx);
        cached_v0 = __low2float(pair);
        cached_v1 = __high2float(pair);
        local_amax = fmaxf(fabsf(cached_v0), fabsf(cached_v1));
    }
    // Handle odd d_nope (shouldn't happen for 512/448 but be safe)
    if (d_nope & 1) {
        if (thread_idx == pair_count) {
            cached_v0 = __bfloat162float(__ldg(q_in + d_nope - 1));
            local_amax = fabsf(cached_v0);
        }
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));

    // Cross-warp reduce via shared memory (no atomics)
    __shared__ float sWarpMax[8];
    if (thread_idx % 32 == 0)
        sWarpMax[thread_idx / 32] = local_amax;
    __syncthreads();
    if (thread_idx == 0) {
        float final_amax = sWarpMax[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            final_amax = fmaxf(final_amax, sWarpMax[w]);
        sWarpMax[0] = final_amax;
    }
    __syncthreads();

    float scale = sWarpMax[0] / FP8_E4M3_MAX;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Quantize NOPE from cached registers (no re-read from global memory)
    if (base < d_nope) {
        float q0 = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, cached_v0 * inv_scale));
        float q1 = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, cached_v1 * inv_scale));
        nope_out[base]     = __nv_fp8_e4m3(q0);
        nope_out[base + 1] = __nv_fp8_e4m3(q1);
    }
    if ((d_nope & 1) && thread_idx == pair_count) {
        float q0 = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, cached_v0 * inv_scale));
        nope_out[d_nope - 1] = __nv_fp8_e4m3(q0);
    }

    // Pre-scale ROPE dims by inverse content scale, store as BF16
    for (int d = thread_idx; d < d_rope; d += num_threads) {
        float val = __bfloat162float(__ldg(q_in + d_nope + d)) * inv_scale;
        rope_out[d] = __float2bfloat16_rn(val);
    }

    // Store scale
    if (thread_idx == 0) {
        params.q_scales[head_idx] = scale;
    }
}

void run_fused_q_quant(const FusedQQuantParams& params, cudaStream_t stream) {
    int grid = params.s_q * params.h_q;
    fused_q_quant_kernel<<<grid, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
