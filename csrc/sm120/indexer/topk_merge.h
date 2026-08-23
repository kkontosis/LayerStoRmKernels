#pragma once
/***************************************************************************************************
 * Lightning Indexer Top-K Cross-Rank Merge (DCP local indexer mode)
 *
 * EXACT merge of per-rank shard-local top-k candidate lists into the global
 * top-k. Under dcp_indexer_mode=local each DCP rank stores only its own
 * position shard's indexer-K (round-robin by indexer page: owner(pos) =
 * (pos / page_tokens) % dcp_size) and runs lightning_topk over its shard
 * only, emitting LOCAL slot indices + scores. The per-position score is a
 * sum over indexer heads with replicated weights, so a shard-local score
 * equals the global score for that position — any global-top-k position is
 * therefore contained in its owner's shard-top-k candidate list.
 *
 * This merge:
 *   1. fills a full-length [num_blocks] score scratch with -inf,
 *   2. scatters every rank's candidates back to their GLOBAL positions
 *      (global = (local / page_tokens * dcp_size + rank) * page_tokens
 *               + local % page_tokens; ownership is disjoint, so no races),
 *   3. re-runs the standard lightning_topk radix selection over the scratch.
 *
 * Because each shard's candidate list contains ALL of its causal positions
 * whenever the shard holds <= topk of them, the reconstructed scratch is
 * bit-identical to the replicated-mode score array for contexts up to
 * dcp_size * topk positions — the merged selection is then bit-identical to
 * replicated-mode selection. Beyond that, omitted positions score strictly
 * below their shard's selection threshold <= the global threshold, so the
 * merged top-k still equals the full top-k up to exact FP32 score ties at
 * the threshold (the same tie nondeterminism lightning_topk itself has).
 *
 * Candidate layout (NCCL allgather output, rank-major):
 *   segment n of dcp_size, each seg_words 32-bit words:
 *     [batch * topk int32 local indices][batch * topk f32 scores]
 *   Row `token` of each segment is merged. Padding entries have index -1
 *   (skipped); -inf-scored entries are skipped as belt-and-braces.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cstdint>

#include "lightning_topk.h"

namespace sm120::indexer {

struct LightningTopkMergeParams {
    // Gathered per-rank candidate lists (device, rank-major; layout above).
    const void* __restrict__ gathered;
    int seg_words;           // 32-bit words per rank segment (= 2*batch*topk)
    int batch;               // rows per segment
    int token;               // row to merge

    // Full-length merge scratch (device, [num_blocks] f32; overwritten).
    float* __restrict__ scores_scratch;

    // Standard lightning_topk inputs/outputs (see lightning_topk.h).
    const int* __restrict__ block_endpoints;   // [num_blocks]
    int* __restrict__ output_indices;          // [topk], GLOBAL, sorted asc, pad -1
    float* __restrict__ output_scores;         // [topk]
    int* __restrict__ effective_k_out;         // [1]

    int num_blocks;          // global causal length for this row
    int topk;
    int query_position;      // causality cutoff (global position)

    // Local→global mapping (round-robin by indexer page).
    int dcp_size;
    int page_tokens;         // indexer_k_page_size_tokens
};

void run_lightning_topk_merge(const LightningTopkMergeParams& params,
                              cudaStream_t stream);

}  // namespace sm120::indexer
