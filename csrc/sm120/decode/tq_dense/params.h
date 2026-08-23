#pragma once
/***************************************************************************************************
 * TQ Dense Decode Parameters
 *
 * Dense MLA decode from TQ-compressed KV cache. Fundamentally different from
 * SnapMLA FP8 decode:
 *   - NOPE scores: scalar codebook lookup (no MMA)
 *   - ROPE scores: BF16 dot product
 *   - PV: accumulated in rotated space (no per-token inverse rotation)
 *   - Output: FP32 in rotated space (needs v_rotate_back after mla_combine)
 *
 * TQ cache per token: [d_c/2 uint8 packed] [2B fp16 norm] [d_rope*2 BF16 rope]
 *
 * Supports split-KV: when num_sm_parts > 1, grid.z = num_sm_parts and each CTA
 * processes a subset of KV blocks via tile scheduler metadata.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::decode::tq_dense {

// Scheduling metadata for split-KV (matches shared DecodingSchedMeta)
struct __align__(32) TqDecodingSchedMeta {
    int begin_req_idx;
    int begin_block_idx;
    int end_req_idx;
    int end_block_idx;
    int begin_split_idx;
    int _pad[3];
};

struct TqDenseDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_c, d_rope;
    float sm_scale;

    // Q: pre-rotated FP32 NOPE + BF16 ROPE (no FP8 quantization, no Q scales)
    const float* __restrict__ q_rot;                // [b, s_q, h_q, d_c] FP32
    const __nv_bfloat16* __restrict__ q_rope;       // [b, s_q, h_q, d_rope] BF16

    // KV: paged TQ cache — [packed_nope | fp16_norm | bf16_rope]
    const uint8_t* __restrict__ kv_cache;
    int64_t cache_stride_block;                     // bytes between page blocks
    int cache_stride_row;                           // bytes between rows

    // Page table
    const int* __restrict__ block_table;
    int block_table_batch_stride;
    int page_block_size;                            // tokens per page (64)

    // Per-batch sequence lengths
    const int* __restrict__ seqlens_k;              // [b]

    // Codebook
    const float* __restrict__ centroids;            // [16] FP32 codebook

    // Output (FP32 in ROTATED space — needs v_rotate_back after mla_combine)
    float* __restrict__ out;                        // [b, s_q, h_q, d_c] FP32
    float* __restrict__ lse;                        // [b, s_q, h_q]

    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_lse_b, stride_lse_s_q;

    // Split-KV infrastructure (when num_sm_parts > 1)
    float* __restrict__ o_accum;                    // [total_splits, num_q_seqs, d_c] FP32
    float* __restrict__ lse_accum;                  // [total_splits, num_q_seqs] FP32
    int stride_o_accum_split;                       // = num_q_seqs * d_c
    int stride_o_accum_s_q;                         // = h_q * d_c
    int stride_o_accum_h_q;                         // = d_c
    int stride_lse_accum_split;                     // = num_q_seqs
    int stride_lse_accum_s_q;                       // = h_q
    TqDecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;  // [num_sm_parts]
    int* __restrict__ num_splits_ptr;               // [b+1] cumulative splits
    int num_sm_parts;

    cudaStream_t stream;
};

void run_tq_dense_decode(const TqDenseDecodeParams& params);

}  // namespace sm120::decode::tq_dense
