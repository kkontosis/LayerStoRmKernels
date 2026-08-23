#pragma once
/***************************************************************************************************
 * TQ Sparse Decode Parameters
 *
 * Sparse MLA decode from TQ-compressed KV cache with topk index array.
 * Same algorithm as tq_dense_decode but reads KV tokens via indices instead
 * of sequential page iteration.
 *
 * Differences from TqDenseDecodeParams:
 *   - indices: [b, s_q, topk] int32 — which tokens to attend to (padded with -1)
 *   - topk: max number of tokens per query
 *   - No seqlens_k — validity determined by indices >= 0
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::decode::tq_sparse {

struct TqSparseDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_c, d_rope;
    float sm_scale;

    // Q: pre-rotated FP32 NOPE + BF16 ROPE
    const float* __restrict__ q_rot;                // [b, s_q, h_q, d_c] FP32
    const __nv_bfloat16* __restrict__ q_rope;       // [b, s_q, h_q, d_rope] BF16

    // KV: paged TQ cache — [packed_nope | fp16_norm | bf16_rope]
    const uint8_t* __restrict__ kv_cache;
    int64_t cache_stride_block;                     // bytes between page blocks
    int cache_stride_row;                           // bytes between rows
    int page_block_size;                            // tokens per page (64)

    // Sparse indices
    const int* __restrict__ indices;                // [b, s_q, topk] int32 (-1 = invalid)
    int topk;                                       // max tokens per query
    int stride_indices_b;                           // stride for batch dim
    int stride_indices_s_q;                         // stride for s_q dim

    // Elements between consecutive head rows of q_rope (0 → d_rope,
    // contiguous). Interleaved [nope|rope] engine layout passes d_c + d_rope
    // (TD-TQ-SPARSE-DECODE-UNWIRED).
    int q_rope_row_stride = 0;

    // Codebook
    const float* __restrict__ centroids;            // [16] FP32 codebook

    // Output (FP32 in ROTATED space — needs v_rotate_back after mla_combine)
    float* __restrict__ out;                        // [b, s_q, h_q, d_c] FP32
    float* __restrict__ lse;                        // [b, s_q, h_q]

    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_lse_b, stride_lse_s_q;

    cudaStream_t stream;
};

void run_tq_sparse_decode(const TqSparseDecodeParams& params);

}  // namespace sm120::decode::tq_sparse
