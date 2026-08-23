#pragma once
// Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
// MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.

//==============================================================================
// SM120 Sparse FP8 Decode - Parameters
//
// Re-exports the SparseAttnDecodeParams from the shared FlashMLA params.h.
// This header is MSVC-safe (no CUTLASS includes beyond bfloat16.h).
//
// The SparseAttnDecodeParams struct is defined in github.com/deepseek-ai/FlashMLA: csrc/params.h
// and contains all fields needed for sparse attention decode, including:
// - Main KV cache pointers and strides
// - Extra KV cache (for MODEL1 dynamic topk)
// - TopK indices and lengths
// - Split-KV accumulation buffers
// - Tile scheduler metadata
//==============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include "cutlass/bfloat16.h"

#include "../../components/sparse_config.h"

namespace sm120::decode::sparse_fp8 {

// Use shared ModelType from sparse_config.h
using sm120::sparse::ModelType;

// Scheduling metadata for split-KV (matches FlashMLA TileSchedulerMetaDataSize=8 layout)
// Layout: [begin_idx, begin_block_idx, end_idx, end_block_idx, begin_n_split_idx, _, _, _]
struct __align__(32) DecodingSchedMeta {
    int begin_req_idx;        // [0] First request index (inclusive)
    int begin_block_idx;      // [1] First block index within begin_req (inclusive)
    int end_req_idx;          // [2] Last request index (inclusive)
    int end_block_idx;        // [3] Last block index within end_req (exclusive)
    int begin_split_idx;      // [4] Split index for begin_req
    int _pad[3];              // [5-7] Unused (matches TileSchedulerMetaDataSize=8)
};

// Sparse attention decode parameters (mirrors FlashMLA SparseAttnDecodeParams)
struct SparseAttnDecodeParams {
    int b, s_q;
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale, sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    ModelType model_type;

    cutlass::bfloat16_t* __restrict__ q;     // [b, s_q, h_q, d_nope] FP8 Q_NOPE (cast to fp8 in kernel)
    cutlass::bfloat16_t* __restrict__ q_rope; // [b, s_q, h_q, d_rope] BF16 Q_ROPE (pre-scaled)
    float* __restrict__ q_scales;            // [b, s_q, h_q] per-head Q quantization scales
    cutlass::bfloat16_t* __restrict__ kv;    // Paged FP8 cache [d_c FP8 | scale | d_rope BF16]
    int* __restrict__ indices;               // [b, s_q, topk]
    int* __restrict__ topk_length;           // [b], may be nullptr
    float* __restrict__ attn_sink;           // [h_q], may be nullptr

    float* __restrict__ lse;                 // [b, s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out;   // [b, s_q, h_q, d_v]

    int extra_num_blocks, extra_page_block_size, extra_topk;
    cutlass::bfloat16_t* __restrict__ extra_kv;           // [extra_num_blocks, extra_page_block_size, d_qk] FP8
    int* __restrict__ extra_indices;                       // [b, s_q, extra_topk]
    int* __restrict__ extra_topk_length;                   // [b], may be nullptr

    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    cudaStream_t stream;

    // SplitKV-related parameters
    float* __restrict__ lse_accum;
    float* __restrict__ o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;
};

// Forward declarations
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}  // namespace sm120::decode::sparse_fp8
