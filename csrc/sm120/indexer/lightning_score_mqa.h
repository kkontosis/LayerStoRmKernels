#pragma once
/***************************************************************************************************
 * Lightning Indexer Scoring — MQA variant (single shared K head, broadcast)
 *
 * Identical math to lightning_score.h, but the indexer K cache stores ONE key
 * vector per block ([num_blocks, INDEX_HEAD_DIM]) that is shared across all
 * INDEX_N_HEADS query heads (multi-query indexer, as in DeepSeek-V3.2/GLM-5.2:
 * wk projects hidden → a single [index_head_dim] key). This avoids the
 * INDEX_N_HEADS× memory blow-up of replicating the shared key into the per-head
 * layout that lightning_score.h expects — important at long (1M) context.
 *
 * Score (same as the per-head kernel, with K broadcast over heads):
 *   dots[n,h] = q_proj[h,:] · K[n,:]          (same K[n] for every head h)
 *   scores[n] = Σ_h ReLU(dots[n,h]) · score_proj[h]
 *
 * The per-head kernel (lightning_score.h) is retained for models that store K
 * per-head (e.g. the V3.2/V4 reference path + tests).
 *
 * Grid: (ceil(num_blocks / BLOCKS_PER_CTA), 1, 1)
 * Block: 256 threads (8 warps)
 *
 * Reference: tests/test_v4_reference.py::ref_lightning_score_mqa()
 * Derived from lightning_score.h — same upstream lineage (SGLang
 * nsa_indexer.py, vLLM sparse_attn_indexer.py; both Apache-2.0 —
 *  see THIRD_PARTY_NOTICES.md)
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace sm120::indexer {

struct LightningScoreMqaParams {
    // Query projected into indexer space (BF16, loaded once to smem)
    const __nv_bfloat16* __restrict__ q_proj;           // [INDEX_N_HEADS, INDEX_HEAD_DIM]

    // Indexer K cache (FP8 E4M3) — ONE shared key per block (no head dim)
    const __nv_fp8_e4m3* __restrict__ indexer_k_cache;  // [num_blocks, INDEX_HEAD_DIM]

    // Per-block K scales (FP32): dequant = fp8_val * k_scale[block]
    const float* __restrict__ k_scales;                 // [num_blocks]

    // Score aggregation weights (FP32) — per query head
    const float* __restrict__ score_proj;               // [INDEX_N_HEADS]

    // Output: importance score per block (FP32)
    float* __restrict__ scores_out;                     // [num_blocks]

    // Dimensions
    int num_blocks;          // number of blocks to score
    int index_n_heads;       // e.g. 32 (GLM-5.2)
    int index_head_dim;      // e.g. 128
};

void run_lightning_score_mqa(const LightningScoreMqaParams& params, cudaStream_t stream);

// TD-SPARSE-PREFILL-SCORE-BATCH: multi-row MQA scoring — ONE launch covers
// num_rows independent queries, each row with its OWN causal block bound.
// Row r scores blocks [0, row_num_blocks[r]) of ITS K view into
// scores_out + r*scores_stride. Per (row, block) the scoring body is the
// exact single-query body (mqa_score_one_block) — bit-identical to
// num_rows separate run_lightning_score_mqa launches.
//
// K storage — exactly one of:
//   * contiguous: k_cache/k_scales shared by all rows, global block index
//     addressing (the executor-arena single-sequence layout);
//   * paged (k_page_table != nullptr): per-row page-pointer table
//     [num_rows, page_table_stride] (device array); each page holds
//     [page_tokens × index_head_dim] FP8 rows followed by [page_tokens]
//     f32 scales. Entries beyond a row's bound may be null (never read).
//
// Grid: (ceil(max_num_blocks / BLOCKS_PER_CTA), num_rows); rows with a
// smaller bound early-out.
struct LightningScoreMqaBatchedParams {
    const __nv_bfloat16* __restrict__ q_all;        // [num_rows, n_heads*head_dim]
    const float* __restrict__ score_proj_all;       // [num_rows, n_heads]
    const int* __restrict__ row_num_blocks;         // [num_rows] per-row bound

    // Contiguous K (used when k_page_table == nullptr)
    const __nv_fp8_e4m3* __restrict__ k_cache;      // [*, index_head_dim]
    const float* __restrict__ k_scales;             // [*]

    // Paged K (used when non-null)
    const void* const* __restrict__ k_page_table;   // [num_rows, page_table_stride]
    int page_table_stride;                          // pages per row slot
    int page_tokens;                                // K rows per page

    float* __restrict__ scores_out;                 // [num_rows, scores_stride]
    int64_t scores_stride;                          // floats between row starts

    int num_rows;
    int max_num_blocks;      // max over row_num_blocks (grid sizing)
    int index_n_heads;
    int index_head_dim;
};

void run_lightning_score_mqa_batched(const LightningScoreMqaBatchedParams& params,
                                     cudaStream_t stream);

}  // namespace sm120::indexer
