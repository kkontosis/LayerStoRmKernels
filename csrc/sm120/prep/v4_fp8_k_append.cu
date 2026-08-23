#pragma once
#include "v4_fp8_k_append.h"

namespace sm120::prep {

static constexpr float V4_FP8_MAX = 448.0f;

// Quantize a BF16 vector to FP8 and write to cache.
// Returns the scale used. head_dim elements, 256 threads → 2 elements per thread.
__device__ float v4_quantize_and_write(
    const __nv_bfloat16* __restrict__ src, int head_dim,
    uint8_t* __restrict__ dst, int tid
) {
    // Single-pass: read, compute amax, cache in registers
    const int base = tid * 2;
    float cached_v0 = 0.0f, cached_v1 = 0.0f;
    float local_amax = 0.0f;

    if (base < head_dim) {
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(src) + tid);
        cached_v0 = __low2float(pair);
        cached_v1 = __high2float(pair);
        local_amax = fmaxf(fabsf(cached_v0), fabsf(cached_v1));
    }

    // Warp reduce amax
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));

    // Cross-warp reduce
    __shared__ float sWarpMax[8];
    if (tid % 32 == 0)
        sWarpMax[tid / 32] = local_amax;
    __syncthreads();
    if (tid == 0) {
        float final_amax = sWarpMax[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            final_amax = fmaxf(final_amax, sWarpMax[w]);
        sWarpMax[0] = final_amax;
    }
    __syncthreads();

    float scale = sWarpMax[0] / V4_FP8_MAX;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    // Quantize and write FP8
    __nv_fp8_e4m3* out = reinterpret_cast<__nv_fp8_e4m3*>(dst);
    if (base < head_dim) {
        float q0 = fmaxf(-V4_FP8_MAX, fminf(V4_FP8_MAX, cached_v0 * inv_scale));
        float q1 = fmaxf(-V4_FP8_MAX, fminf(V4_FP8_MAX, cached_v1 * inv_scale));
        out[base]     = __nv_fp8_e4m3(q0);
        out[base + 1] = __nv_fp8_e4m3(q1);
    }

    return scale;
}


__global__ void __launch_bounds__(256)
v4_fp8_k_append_kernel(const V4Fp8KAppendParams params) {
    const int token_idx = blockIdx.x;
    if (token_idx >= params.num_tokens) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;

    // Entry layout: [K_NOPE FP8 | K_scale f32 | K_ROPE BF16 | V_NOPE FP8 | V_scale f32]
    const int k_scale_off = hd;
    const int k_rope_off  = hd + 4;
    const int v_nope_off  = hd + 4 + rd * 2;
    const int v_scale_off = v_nope_off + hd;
    const int entry_bytes = v_scale_off + 4;

    int slot = __ldg(params.slot_mapping + token_idx);
    uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    // --- K NOPE: quantize to FP8 ---
    const __nv_bfloat16* k_nope_in = params.k_nope + (int64_t)token_idx * hd;
    float k_scale = v4_quantize_and_write(k_nope_in, hd, entry, tid);

    if (tid == 0)
        *reinterpret_cast<float*>(entry + k_scale_off) = k_scale;

    // --- K ROPE: copy as BF16 ---
    const __nv_bfloat16* k_rope_in = params.k_rope + (int64_t)token_idx * rd;
    __nv_bfloat16* k_rope_out = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
    for (int d = tid; d < rd; d += 256)
        k_rope_out[d] = __ldg(k_rope_in + d);

    __syncthreads();

    // --- V NOPE: quantize to FP8 ---
    const __nv_bfloat16* v_nope_in = params.v_nope + (int64_t)token_idx * hd;
    float v_scale = v4_quantize_and_write(v_nope_in, hd, entry + v_nope_off, tid);

    if (tid == 0)
        *reinterpret_cast<float*>(entry + v_scale_off) = v_scale;
}

void run_v4_fp8_k_append(const V4Fp8KAppendParams& params, cudaStream_t stream) {
    if (params.num_tokens == 0) return;
    v4_fp8_k_append_kernel<<<params.num_tokens, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
