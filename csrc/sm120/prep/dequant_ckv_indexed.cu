#pragma once
#include "dequant_ckv_indexed.h"

namespace sm120::prep {

__global__ void __launch_bounds__(128)
dequant_ckv_fused_indexed_kernel(const DequantCKVIndexedParams params) {
    const int fetch_idx = blockIdx.x;
    if (fetch_idx >= params.num_fetch) return;

    const int thread_idx = threadIdx.x;
    const int lane_id = thread_idx % 32;
    const int d_c = params.d_c;
    const int d_rope = params.d_rope;
    const int d_qk = d_c + d_rope;

    // Resolve paged cache address for this token
    const int slot = __ldg(params.indices + fetch_idx);
    const int page_idx = slot / params.page_size;
    const int row_in_page = slot % params.page_size;

    const __nv_fp8_e4m3* cache_row = params.kv_cache +
        (int64_t)page_idx * params.cache_stride_block +
        (int64_t)row_in_page * params.cache_stride_row;

    // Read per-token scale (lane 0 loads, broadcast within warp)
    float scale;
    if (lane_id == 0)
        scale = __ldg(reinterpret_cast<const float*>(cache_row + d_c));
    scale = __shfl_sync(0xffffffff, scale, 0);

    __nv_bfloat16* out = params.k_out + (int64_t)fetch_idx * d_qk;

    // Dequant NOPE: d_c FP8 elements → BF16
    // 128 threads × 4 elements = 512 capacity (fits d_c=512, covers d_c=448)
    {
        constexpr int VEC = 4;  // 4 FP8 bytes = 1 uint load
        const int base = thread_idx * VEC;
        if (base < d_c) {
            uint fp8_vec = __ldg(reinterpret_cast<const uint*>(cache_row + base));
            const __nv_fp8_e4m3* fp8_arr = reinterpret_cast<const __nv_fp8_e4m3*>(&fp8_vec);
            __nv_bfloat16 bf16_out[VEC];
            #pragma unroll
            for (int i = 0; i < VEC; ++i)
                bf16_out[i] = __float2bfloat16_rn((float)fp8_arr[i] * scale);
            *reinterpret_cast<uint2*>(out + base) = *reinterpret_cast<const uint2*>(bf16_out);
        }
    }

    // Un-scale ROPE: BF16 in cache was pre-scaled (k_rope / scale) by fused_k_append.
    // Multiply by scale to recover original k_rope values for BF16 prefill.
    // 64 BF16 elements / 2 per thread = 32 active threads (uint load = 2 BF16)
    {
        constexpr int VEC = 2;  // 2 BF16 per uint
        const int base = thread_idx * VEC;
        if (base < d_rope) {
            const __nv_bfloat16* rope_src = reinterpret_cast<const __nv_bfloat16*>(cache_row + d_c + 4);
            __nv_bfloat16 bf16_out[VEC];
            #pragma unroll
            for (int i = 0; i < VEC; ++i)
                bf16_out[i] = __float2bfloat16_rn(__bfloat162float(rope_src[base + i]) * scale);
            *reinterpret_cast<uint*>(out + d_c + base) = *reinterpret_cast<const uint*>(bf16_out);
        }
    }
}

void run_dequant_ckv_fused_indexed(const DequantCKVIndexedParams& params, cudaStream_t stream) {
    if (params.num_fetch == 0) return;
    dequant_ckv_fused_indexed_kernel<<<params.num_fetch, 128, 0, stream>>>(params);
}

}  // namespace sm120::prep
