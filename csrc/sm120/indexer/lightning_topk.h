#pragma once
/***************************************************************************************************
 * Lightning Indexer Top-K Selection: radix histogram filter
 *
 * Selects the top-k highest-scoring compressed blocks from Lightning Indexer
 * scores, with causality enforcement (future blocks excluded).
 *
 * Algorithm: 4-pass radix histogram selection (matching SGLang/vLLM/TRT-LLM):
 *   1. Convert FP32 scores to sortable uint32 keys
 *   2. 4 passes of 8-bit radix histogram to find exact threshold key
 *   3. Gather all entries above threshold + remaining from threshold bin
 *   4. Bitonic sort output indices ascending (deterministic access pattern)
 *
 * Grid:  (1, 1, 1)  — single CTA per query (called per-query from Python)
 * Block: 256 threads (8 warps)
 *
 * Reference: tests/test_v4_reference.py::ref_lightning_topk()
 * Ported from: SGLang sgl-kernel/csrc/elementwise/topk.cu,
 *              vLLM csrc/topk.cu, TRT-LLM kernels/indexerTopK.cu
 * (SGLang: Apache-2.0, Copyright 2023-2024 SGLang Team; vLLM: Apache-2.0,
 *  Copyright contributors to the vLLM project; TensorRT-LLM: Apache-2.0,
 *  Copyright (c) 2011-2025 NVIDIA CORPORATION & AFFILIATES —
 *  see THIRD_PARTY_NOTICES.md)
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cstdint>

namespace sm120::indexer {

struct LightningTopkParams {
    // Input: importance scores from scoring kernel (FP32)
    const float* __restrict__ scores;            // [num_blocks]

    // Input: endpoint position of each compressed block (for causality)
    const int* __restrict__ block_endpoints;      // [num_blocks]

    // Output: selected block indices, sorted ascending, padded with -1
    int* __restrict__ output_indices;             // [topk]

    // Output: scores of selected blocks, padded with -inf
    float* __restrict__ output_scores;            // [topk]

    // Output: actual number of valid entries written
    int* __restrict__ effective_k_out;            // [1]

    // Dimensions
    int num_blocks;          // total compressed blocks to select from
    int topk;                // 1024 (Pro) / 512 (Flash), max 2048
    int query_position;      // causality cutoff: only blocks with endpoint <= this
};

void run_lightning_topk(const LightningTopkParams& params, cudaStream_t stream);

// TD-SPARSE-PREFILL-SCORE-BATCH: multi-row top-k — ONE launch, one CTA per
// row (grid.x = num_rows), each CTA running the exact single-query CTA body
// (lightning_topk_cta) on row r's scores slice with row r's OWN bound
// (row_num_blocks[r]) and causality cutoff (row_query_position[r]) —
// bit-identical to num_rows separate run_lightning_topk launches. A row
// with num_blocks == 0 gets the valid empty result (indices padded -1,
// effective_k 0), as in the single-query kernel.
struct LightningTopkBatchedParams {
    const float* __restrict__ scores;              // [num_rows, scores_stride]
    const int* __restrict__ block_endpoints;       // shared endpoint iota
    const int* __restrict__ row_num_blocks;        // [num_rows] per-row bound
    const int* __restrict__ row_query_position;    // [num_rows] per-row cutoff

    int* __restrict__ output_indices;              // [num_rows, topk]
    float* __restrict__ output_scores;             // [num_rows, topk]
    int* __restrict__ effective_k_out;             // [num_rows]

    int64_t scores_stride;   // floats between consecutive rows' scores
    int num_rows;
    int topk;                // max 2048
};

void run_lightning_topk_batched(const LightningTopkBatchedParams& params,
                                cudaStream_t stream);

}  // namespace sm120::indexer
