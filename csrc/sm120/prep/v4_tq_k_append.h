#pragma once
/***************************************************************************************************
 * V4 TQ K-Append: TurboQuant 4-bit quantization of compressed K+V → paged cache write
 *
 * Adapts tq_fused_k_append.cu (V3.2 single latent) to V4's separate K+V storage.
 * Per entry (644 bytes):
 *   K: [256B packed_nope | 2B FP16 norm | 128B BF16 rope] = 386B
 *   V: [256B packed_nope | 2B FP16 norm]                   = 258B
 *
 * Both CSA and HCA compressed vectors are quantized identically.
 * K and V are independently normalized, rotated, and quantized.
 * K_ROPE is stored as-is in BF16 (no rotation/quantization).
 *
 * Grid: (num_tokens, 1, 1) — one CTA per compressed token
 * Block: 256 threads
 * Shared memory: ~2.2KB (unit vector + centroids + boundaries)
 *
 * Adapted from: csrc/sm120/prep/tq_fused_k_append.cu (V3.2 TQ)
 * Reference: tests/test_v4_reference.py::ref_v4_tq_k_append()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct V4TqKAppendParams {
    const __nv_bfloat16* __restrict__ k_nope;   // [num_tokens, head_dim]
    const __nv_bfloat16* __restrict__ k_rope;    // [num_tokens, qk_rope_head_dim]
    const __nv_bfloat16* __restrict__ v_nope;    // [num_tokens, head_dim]

    uint8_t* __restrict__ kv_cache;              // flat paged TQ cache
    const int* __restrict__ slot_mapping;         // [num_tokens] → global slot index

    const float* __restrict__ Pi;                 // [head_dim, head_dim] orthogonal rotation
    const float* __restrict__ centroids;          // [16] Lloyd-Max centroids
    const float* __restrict__ decision_boundaries; // [15] interior boundaries (sorted)

    int num_tokens;
    int head_dim;           // 512
    int qk_rope_head_dim;   // 64
    int num_centroids;       // 16
};

void run_v4_tq_k_append(const V4TqKAppendParams& params, cudaStream_t stream);

struct V4TqNormalizeParams {
    const __nv_bfloat16* __restrict__ src;   // [num_vecs, dim]
    __nv_bfloat16* __restrict__ dst_unit;    // [num_vecs, dim] unit vectors
    float* __restrict__ dst_norms;           // [num_vecs] L2 norms
    int num_vecs;
    int dim;                                 // 512
};

struct V4TqQuantPackWriteParams {
    const __nv_bfloat16* __restrict__ k_rot;     // [num_tokens, head_dim] rotated K
    const float* __restrict__ k_norms;            // [num_tokens]
    const __nv_bfloat16* __restrict__ v_rot;      // [num_tokens, head_dim] rotated V
    const float* __restrict__ v_norms;            // [num_tokens]
    const __nv_bfloat16* __restrict__ k_rope;     // [num_tokens, qk_rope_head_dim]

    uint8_t* __restrict__ kv_cache;
    const int* __restrict__ slot_mapping;

    const float* __restrict__ centroids;
    const float* __restrict__ decision_boundaries;

    int num_tokens;
    int head_dim;
    int qk_rope_head_dim;
    int num_centroids;
};

void run_v4_tq_normalize(const V4TqNormalizeParams& params, cudaStream_t stream);
void run_v4_tq_quant_pack_write(const V4TqQuantPackWriteParams& params, cudaStream_t stream);

}  // namespace sm120::prep
