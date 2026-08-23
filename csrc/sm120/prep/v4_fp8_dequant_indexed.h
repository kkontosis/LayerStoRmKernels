#pragma once
/***************************************************************************************************
 * V4 FP8 Dequant Indexed: Gather + dequant from V4 paged FP8 cache → BF16
 *
 * Reads selected entries from the V4 FP8 cache, dequantizes K_NOPE and V_NOPE,
 * copies K_ROPE as-is. Outputs three separate BF16 tensors.
 *
 * V4 cache entry layout (1160 bytes):
 *   [K_NOPE FP8(512) | K_scale f32(4) | K_ROPE BF16(128) |
 *    V_NOPE FP8(512) | V_scale f32(4)]
 *
 * Unlike MLA dequant (single output, pre-scaled RoPE), V4 outputs K_NOPE, K_ROPE,
 * V_NOPE separately. K_ROPE is not pre-scaled, so no un-scaling needed.
 *
 * Grid:  (num_fetch, 1, 1) — one CTA per fetched entry
 * Block: 256 threads
 *
 * Adapted from: csrc/sm120/prep/dequant_ckv_indexed.cu (MLA variant)
 * Reference: tests/test_v4_reference.py::ref_v4_fp8_dequant()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct V4Fp8DequantIndexedParams {
    const uint8_t* __restrict__ kv_cache;         // flat paged cache
    const int* __restrict__ indices;               // [num_fetch] slot indices

    __nv_bfloat16* __restrict__ k_nope_out;       // [num_fetch, head_dim]
    __nv_bfloat16* __restrict__ k_rope_out;       // [num_fetch, qk_rope_head_dim]
    __nv_bfloat16* __restrict__ v_nope_out;       // [num_fetch, head_dim]

    int num_fetch;
    int head_dim;          // 512
    int qk_rope_head_dim;  // 64
};

void run_v4_fp8_dequant_indexed(const V4Fp8DequantIndexedParams& params, cudaStream_t stream);

}  // namespace sm120::prep
