#pragma once
#include "v4_fp8_dequant_indexed.h"

namespace sm120::prep {

__global__ void __launch_bounds__(256)
v4_fp8_dequant_indexed_kernel(const V4Fp8DequantIndexedParams params) {
    const int fetch_idx = blockIdx.x;
    if (fetch_idx >= params.num_fetch) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;

    // Entry layout offsets
    const int k_scale_off = hd;
    const int k_rope_off  = hd + 4;
    const int v_nope_off  = hd + 4 + rd * 2;
    const int v_scale_off = v_nope_off + hd;
    const int entry_bytes = v_scale_off + 4;

    const int slot = __ldg(params.indices + fetch_idx);
    const uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    // Read K and V scales — broadcast via shared memory to all 256 threads
    __shared__ float s_scales[2];
    if (tid == 0) {
        s_scales[0] = __ldg(reinterpret_cast<const float*>(entry + k_scale_off));
        s_scales[1] = __ldg(reinterpret_cast<const float*>(entry + v_scale_off));
    }
    __syncthreads();
    float k_scale = s_scales[0];
    float v_scale = s_scales[1];

    // Dequant K_NOPE: FP8 * k_scale → BF16
    // 256 threads × 2 elements = 512 capacity (exactly head_dim=512)
    {
        const __nv_fp8_e4m3* k_fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(entry);
        __nv_bfloat16* k_out = params.k_nope_out + (int64_t)fetch_idx * hd;
        const int base = tid * 2;
        if (base < hd) {
            // Vectorized 2-byte load (2 FP8 values)
            uint16_t packed = __ldg(reinterpret_cast<const uint16_t*>(k_fp8 + base));
            const __nv_fp8_e4m3* vals = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);
            k_out[base]     = __float2bfloat16_rn((float)vals[0] * k_scale);
            k_out[base + 1] = __float2bfloat16_rn((float)vals[1] * k_scale);
        }
    }

    // Copy K_ROPE: BF16 → BF16 (no scaling)
    {
        const __nv_bfloat16* rope_src = reinterpret_cast<const __nv_bfloat16*>(entry + k_rope_off);
        __nv_bfloat16* rope_out = params.k_rope_out + (int64_t)fetch_idx * rd;
        for (int d = tid; d < rd; d += 256)
            rope_out[d] = __ldg(rope_src + d);
    }

    // Dequant V_NOPE: FP8 * v_scale → BF16
    {
        const __nv_fp8_e4m3* v_fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(entry + v_nope_off);
        __nv_bfloat16* v_out = params.v_nope_out + (int64_t)fetch_idx * hd;
        const int base = tid * 2;
        if (base < hd) {
            uint16_t packed = __ldg(reinterpret_cast<const uint16_t*>(v_fp8 + base));
            const __nv_fp8_e4m3* vals = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);
            v_out[base]     = __float2bfloat16_rn((float)vals[0] * v_scale);
            v_out[base + 1] = __float2bfloat16_rn((float)vals[1] * v_scale);
        }
    }
}

void run_v4_fp8_dequant_indexed(const V4Fp8DequantIndexedParams& params, cudaStream_t stream) {
    if (params.num_fetch == 0) return;
    v4_fp8_dequant_indexed_kernel<<<params.num_fetch, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
