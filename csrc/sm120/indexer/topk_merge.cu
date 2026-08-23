#include "topk_merge.h"

#include <cmath>

namespace sm120::indexer {

static constexpr int MERGE_THREADS = 256;

// Phase 1: fill the full-length merge scratch with -inf. Positions never
// scattered (below their shard's selection threshold, or beyond every
// candidate list) must lose against every real candidate.
__global__ void __launch_bounds__(MERGE_THREADS)
topk_merge_fill_kernel(float* __restrict__ scores, int num_blocks) {
    for (int i = blockIdx.x * MERGE_THREADS + threadIdx.x; i < num_blocks;
         i += gridDim.x * MERGE_THREADS) {
        scores[i] = -INFINITY;
    }
}

// Phase 2: scatter every rank's candidates back to their GLOBAL positions.
// Ownership is disjoint (round-robin by indexer page), so at most one rank
// writes any position — no atomics needed. Thread t handles candidate
// k = t % topk of rank n = t / topk.
__global__ void __launch_bounds__(MERGE_THREADS)
topk_merge_scatter_kernel(const LightningTopkMergeParams params) {
    const int total = params.dcp_size * params.topk;
    for (int t = blockIdx.x * MERGE_THREADS + threadIdx.x; t < total;
         t += gridDim.x * MERGE_THREADS) {
        const int n = t / params.topk;
        const int k = t % params.topk;
        const int* seg = static_cast<const int*>(params.gathered)
                         + static_cast<int64_t>(n) * params.seg_words;
        const int local = __ldg(seg + params.token * params.topk + k);
        if (local < 0) continue;  // -1 padding
        const float* seg_scores = reinterpret_cast<const float*>(
            seg + params.batch * params.topk);
        const float score = __ldg(seg_scores + params.token * params.topk + k);
        if (score == -INFINITY) continue;  // belt-and-braces (never selected)
        const int PT = params.page_tokens;
        const int global =
            (local / PT * params.dcp_size + n) * PT + local % PT;
        if (global >= 0 && global < params.num_blocks)
            params.scores_scratch[global] = score;
    }
}

void run_lightning_topk_merge(const LightningTopkMergeParams& params,
                              cudaStream_t stream) {
    if (params.topk == 0 || params.dcp_size < 1 || params.page_tokens <= 0)
        return;

    if (params.num_blocks > 0) {
        const int fill_grid =
            (params.num_blocks + MERGE_THREADS - 1) / MERGE_THREADS;
        topk_merge_fill_kernel<<<fill_grid, MERGE_THREADS, 0, stream>>>(
            params.scores_scratch, params.num_blocks);

        const int total = params.dcp_size * params.topk;
        const int scat_grid = (total + MERGE_THREADS - 1) / MERGE_THREADS;
        topk_merge_scatter_kernel<<<scat_grid, MERGE_THREADS, 0, stream>>>(
            params);
    }

    // Phase 3: the standard radix top-k over the reconstructed scratch —
    // identical selection semantics to replicated mode by construction.
    LightningTopkParams tp{};
    tp.scores = params.scores_scratch;
    tp.block_endpoints = params.block_endpoints;
    tp.output_indices = params.output_indices;
    tp.output_scores = params.output_scores;
    tp.effective_k_out = params.effective_k_out;
    tp.num_blocks = params.num_blocks;
    tp.topk = params.topk;
    tp.query_position = params.query_position;
    run_lightning_topk(tp, stream);
}

}  // namespace sm120::indexer
