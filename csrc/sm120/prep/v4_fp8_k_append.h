#pragma once
/***************************************************************************************************
 * V4 FP8 K-Append: Dual K+V quantization + paged cache write
 *
 * Per entry (1160 bytes):
 *   [K_NOPE FP8 (512B) | K_scale f32 (4B) | K_ROPE BF16 (128B) |
 *    V_NOPE FP8 (512B) | V_scale f32 (4B)]
 *
 * Unlike MLA k_append (single latent + pre-scaled RoPE), V4 stores K and V
 * separately with independent scales. K_ROPE is stored as-is in BF16 (no
 * pre-scaling needed since NOPE/ROPE dot products are computed separately).
 *
 * Grid: (num_tokens, 1, 1) — one CTA per compressed token
 * Block: 256 threads
 *
 * Adapted from: csrc/sm120/prep/fused_k_append.cu (MLA variant)
 * Reference: tests/test_v4_reference.py::ref_v4_fp8_k_append()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct V4Fp8KAppendParams {
    const __nv_bfloat16* __restrict__ k_nope;   // [num_tokens, head_dim]
    const __nv_bfloat16* __restrict__ k_rope;    // [num_tokens, qk_rope_head_dim]
    const __nv_bfloat16* __restrict__ v_nope;    // [num_tokens, head_dim]

    uint8_t* __restrict__ kv_cache;              // flat paged cache
    const int* __restrict__ slot_mapping;         // [num_tokens] → global slot index

    int num_tokens;
    int head_dim;          // 512
    int qk_rope_head_dim;  // 64
};

void run_v4_fp8_k_append(const V4Fp8KAppendParams& params, cudaStream_t stream);

}  // namespace sm120::prep
