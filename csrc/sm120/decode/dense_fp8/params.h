#pragma once

//==============================================================================
// SM120 Dense FP8 Decode - Parameters (SnapMLA Path)
//
// Dense attention: attends to ALL tokens up to seqlen (no topk indices).
// Uses paged KV cache with block_table for page resolution.
//==============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include "cutlass/bfloat16.h"

#include "../../components/sparse_config.h"

// Reuse DecodingSchedMeta from sparse decode (same split-KV scheduling)
#include "../sparse_fp8/params.h"

namespace sm120::decode::dense_fp8 {

using sm120::sparse::ModelType;
using sm120::decode::sparse_fp8::DecodingSchedMeta;

struct DenseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v, d_nope;
    float sm_scale;

    // Q: separate FP8 NOPE + BF16 ROPE
    cutlass::bfloat16_t* __restrict__ q_nope;   // [b, s_q, h_q, d_nope] FP8 (cast in kernel)
    cutlass::bfloat16_t* __restrict__ q_rope;   // [b, s_q, h_q, d_rope] BF16
    float* __restrict__ q_scales;               // [b, s_q, h_q]

    // KV: paged cache — [d_c FP8 | float32 scale | d_rope BF16] per token
    cutlass::bfloat16_t* __restrict__ kv_cache;  // Base pointer (cast to fp8 in kernel)
    int stride_kv_block;                         // Stride between page blocks (bytes/FP8 elems)
    int stride_kv_row;                           // Stride between rows within page (bytes/FP8 elems)

    // Page table
    int* __restrict__ block_table;               // [b, max_num_blocks_per_seq]
    int block_table_batch_stride;                // = max_num_blocks_per_seq
    int page_block_size;                         // Tokens per page (e.g., 64)

    // Per-batch sequence lengths
    int* __restrict__ seqlens_k;                 // [b]

    // Output
    cutlass::bfloat16_t* __restrict__ out;       // [b, s_q, h_q, d_v]
    float* __restrict__ lse;                     // [b, s_q, h_q]

    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_lse_b, stride_lse_s_q;

    cudaStream_t stream;

    // Split-KV accumulation
    float* __restrict__ lse_accum;
    float* __restrict__ o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;

    // Optional gated deterministic (bit-reproducible) softmax-denominator
    // reduction. false (default) = legacy cross-warp atomicAdd path.
    bool deterministic_reduce = false;
};

template<ModelType MODEL_TYPE, int NUM_HEADS, bool DETERMINISTIC>
void run_flash_splitkv_mla_dense_fp8_kernel(const DenseAttnDecodeParams &params);

}  // namespace sm120::decode::dense_fp8
