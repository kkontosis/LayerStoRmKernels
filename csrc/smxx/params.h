#pragma once

// Split-KV combine + metadata parameter structs.
// Based on FlashMLA (https://github.com/deepseek-ai/FlashMLA)
// Original: github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120 (SM120 fork of github.com/deepseek-ai/FlashMLA): csrc/params.h
// License: MIT, Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md
//
// These structs are used by mla_combine and get_mla_metadata — both are
// arch-generic kernels that operate on split-KV partial results (float32
// lse_accum / o_accum). They never touch the KV cache.

#include <cuda_runtime.h>
#include <cstdint>
#include "cutlass/bfloat16.h"

// Tile scheduler metadata layout:
// [begin_idx, begin_block_idx, end_idx, end_block_idx, begin_n_split_idx, _, _, _]
static constexpr int TileSchedulerMetaDataSize = 8;

// Parameters for mla_combine (merge split-KV partial results)
struct MlaCombineParams {
    using index_t = int64_t;

    int b;              // batch size
    int h_q, h_k;       // number of Q/K heads
    int q_seq_per_hk;   // q sequences per KV head = h_q / h_k * s_q
    int d_v;            // value dimension (512 for MLA)

    // Output
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;

    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // Split-KV scheduling
    int *__restrict__ num_splits_ptr;    // [b+1] cumulative split counts
    int num_sm_parts;

    // Split-KV accumulators
    void *__restrict__ softmax_lseaccum_ptr;  // [total_splits, num_q_seqs] float32
    void *__restrict__ oaccum_ptr;            // [total_splits, num_q_seqs, d_v] float32
};

// Parameters for get_mla_metadata (compute split-KV schedule)
struct GetMlaMetadataParams {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
    int topk;   // -1 = use seqlens_k, else use this value for all batches
};
