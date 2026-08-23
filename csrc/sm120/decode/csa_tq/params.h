#pragma once
/***************************************************************************************************
 * V4 CSA TQ Decode Parameters
 *
 * TQ sparse decode for V4 compressed entries. Separate K and V TQ entries.
 * Output is FP32 in rotated space — caller applies v_rotate_back(Pi).
 *
 * V4 TQ entry layout (644 bytes):
 *   K: [256B packed_nope | 2B FP16 norm | 128B BF16 rope] = 386B
 *   V: [256B packed_nope | 2B FP16 norm]                   = 258B
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::decode::csa_tq {

struct CsaTqDecodeParams {
    int b, s_q, h_q;
    int head_dim;           // 512
    int qk_rope_head_dim;   // 64
    float sm_scale;

    // Q: pre-rotated FP32 NOPE + BF16 ROPE
    const float* __restrict__ q_rot;              // [b, s_q, h_q, head_dim] FP32
    const __nv_bfloat16* __restrict__ q_rope;     // [b, s_q, h_q, qk_rope_head_dim] BF16

    // Compressed KV: flat V4 TQ cache (644 bytes/entry)
    const uint8_t* __restrict__ kv_cache;

    // Sparse indices from Lightning Indexer
    const int* __restrict__ indices;              // [b, s_q, topk] int32 (-1 = invalid)
    int topk;
    int stride_indices_b, stride_indices_s_q;

    // Codebook
    const float* __restrict__ centroids;          // [16] FP32

    // Output (FP32 in ROTATED space — needs v_rotate_back)
    float* __restrict__ out;                      // [b, s_q, h_q, head_dim] FP32
    float* __restrict__ lse;                      // [b, s_q, h_q]

    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_lse_b, stride_lse_s_q;

    // Split-KV (when num_sm_parts > 1)
    int num_sm_parts;
    float* __restrict__ o_accum;                  // [num_sm_parts * b, s_q, h_q * head_dim]
    float* __restrict__ lse_accum;                // [num_sm_parts * b, s_q, h_q]
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int* __restrict__ num_splits_ptr;
    // Tile scheduler metadata — one entry per partition
    void* __restrict__ tile_scheduler_metadata_ptr;

    cudaStream_t stream;
};

void run_csa_tq_decode(const CsaTqDecodeParams& params);

}  // namespace sm120::decode::csa_tq
