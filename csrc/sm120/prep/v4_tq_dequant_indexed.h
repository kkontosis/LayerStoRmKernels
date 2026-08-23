#pragma once
/***************************************************************************************************
 * V4 TQ Dequant Indexed: Gather + dequant from V4 paged TQ cache → BF16
 *
 * Reads selected entries, dequantizes K_NOPE and V_NOPE via codebook + inverse
 * rotation, copies K_ROPE as-is. Outputs three separate BF16 tensors.
 *
 * V4 TQ cache entry layout (644 bytes):
 *   K: [256B packed_nope | 2B FP16 norm | 128B BF16 rope] = 386B
 *   V: [256B packed_nope | 2B FP16 norm]                   = 258B
 *
 * Grid:  (num_fetch, 1, 1) — one CTA per fetched entry
 * Block: 256 threads
 * Shared memory: ~2.1KB (y_hat[512] + centroids[16])
 *
 * Adapted from: csrc/sm120/prep/tq_dequant_ckv_indexed.cu (V3.2 TQ)
 * Reference: tests/test_v4_reference.py::ref_v4_tq_dequant()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct V4TqDequantIndexedParams {
    const uint8_t* __restrict__ kv_cache;         // flat paged TQ cache
    const int* __restrict__ indices;               // [num_fetch] slot indices

    __nv_bfloat16* __restrict__ k_nope_out;       // [num_fetch, head_dim]
    __nv_bfloat16* __restrict__ k_rope_out;       // [num_fetch, qk_rope_head_dim]
    __nv_bfloat16* __restrict__ v_nope_out;       // [num_fetch, head_dim]

    const float* __restrict__ Pi;                  // [head_dim, head_dim] rotation
    const float* __restrict__ centroids;           // [16] codebook centroids

    int num_fetch;
    int head_dim;           // 512
    int qk_rope_head_dim;   // 64
};

void run_v4_tq_dequant_indexed(const V4TqDequantIndexedParams& params, cudaStream_t stream);

}  // namespace sm120::prep
