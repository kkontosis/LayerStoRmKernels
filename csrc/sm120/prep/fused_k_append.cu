#pragma once
#include "fused_k_append.h"

namespace sm120::prep {

__global__ void __launch_bounds__(256)
fused_k_append_kernel(const FusedKAppendParams params) {
    const int token_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int num_threads = blockDim.x;

    if (token_idx >= params.num_tokens) return;

    const int d_c = params.d_c;
    const int d_rope = params.d_rope;

    // Strided source rows (0 → tight). Resolved here so zero-initialized
    // params keep the legacy tight-layout semantics.
    const int stride_ckv  = params.src_stride_ckv  > 0 ? params.src_stride_ckv  : d_c;
    const int stride_rope = params.src_stride_rope > 0 ? params.src_stride_rope : d_rope;
    const __nv_bfloat16* ckv_in = params.c_kv + (int64_t)token_idx * stride_ckv;
    const __nv_bfloat16* rope_in = params.k_rope + (int64_t)token_idx * stride_rope;

    // Resolve paged cache destination
    int slot = __ldg(params.slot_mapping + token_idx);
    int page_idx = slot / params.page_size;
    int row_in_page = slot % params.page_size;

    __nv_fp8_e4m3* cache_row = params.kv_cache +
                                (int64_t)page_idx * params.cache_stride_block +
                                (int64_t)row_in_page * params.cache_stride_row;

    // =========================================================================
    // Single-pass: read c_KV once, compute amax AND cache for quantization.
    // Each thread handles 2 consecutive BF16 values via vectorized bfloat162 load.
    // d_c=512 → 256 pairs → exactly 1 pair per thread.
    // =========================================================================
    const int pair_count = d_c / 2;
    const int base = thread_idx * 2;

    float cached_v0 = 0.0f, cached_v1 = 0.0f;
    float local_amax = 0.0f;

    if (base < d_c) {
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(ckv_in) + thread_idx);
        cached_v0 = __low2float(pair);
        cached_v1 = __high2float(pair);
        local_amax = fmaxf(fabsf(cached_v0), fabsf(cached_v1));
    }
    if (d_c & 1) {
        if (thread_idx == pair_count) {
            cached_v0 = __bfloat162float(__ldg(ckv_in + d_c - 1));
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

    // Quantize c_KV from cached registers (no re-read from global memory)
    if (base < d_c) {
        float q0 = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, cached_v0 * inv_scale));
        float q1 = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, cached_v1 * inv_scale));
        cache_row[base]     = __nv_fp8_e4m3(q0);
        cache_row[base + 1] = __nv_fp8_e4m3(q1);
    }
    if ((d_c & 1) && thread_idx == pair_count) {
        float q0 = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, cached_v0 * inv_scale));
        cache_row[d_c - 1] = __nv_fp8_e4m3(q0);
    }

    // Store scale
    if (thread_idx == 0) {
        *reinterpret_cast<float*>(cache_row + d_c) = scale;
    }

    // Pre-scale ROPE by inverse content scale, store as BF16
    __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(cache_row + d_c + 4);
    for (int d = thread_idx; d < d_rope; d += num_threads) {
        float val = __bfloat162float(__ldg(rope_in + d)) * inv_scale;
        rope_out[d] = __float2bfloat16_rn(val);
    }
}

void run_fused_k_append(const FusedKAppendParams& params, cudaStream_t stream) {
    fused_k_append_kernel<<<params.num_tokens, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
