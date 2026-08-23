#include "lightning_topk.h"
#include <climits>
#include <cfloat>

namespace sm120::indexer {

static constexpr int TOPK_THREADS = 256;
static constexpr int NUM_BINS = 256;
static constexpr int MAX_TOPK = 2048;

// Convert float to unsigned integer that preserves sort order.
// Positive floats already sort correctly as unsigned ints (IEEE 754 property).
// Negative floats sort backwards, so we flip all bits.
// Matching SGLang/vLLM/TRT-LLM convert_to_uint32().
__device__ __forceinline__ uint32_t float_to_sortable(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Single-CTA top-k selection body, shared by the single-query kernel and the
// batched kernel (TD-SPARSE-PREFILL-SCORE-BATCH): the batched kernel runs
// this exact body once per row CTA — bit-identical to per-query launches by
// construction. (The __shared__ buffers are per-CTA as before.)
__device__ __forceinline__ void lightning_topk_cta(
    const float* __restrict__ scores,
    const int* __restrict__ block_endpoints,
    int* __restrict__ output_indices,
    float* __restrict__ output_scores,
    int* __restrict__ effective_k_out,
    int N, int topk, int query_position) {
    const int tid = threadIdx.x;
    const int K = min(topk, MAX_TOPK);

    __shared__ uint32_t s_hist[NUM_BINS];
    __shared__ int s_out_idx[MAX_TOPK];
    __shared__ float s_out_score[MAX_TOPK];
    __shared__ int s_cnt;
    __shared__ int s_thresh_cnt;
    __shared__ int s_remaining;
    __shared__ uint32_t s_thresh_prefix;

    // --- Count causal-valid elements ---
    if (tid == 0) s_cnt = 0;
    __syncthreads();

    int local_valid = 0;
    for (int i = tid; i < N; i += TOPK_THREADS) {
        if (__ldg(block_endpoints + i) <= query_position)
            local_valid++;
    }
    atomicAdd(&s_cnt, local_valid);
    __syncthreads();

    int num_valid = s_cnt;
    int eff_k = min(num_valid, K);

    if (eff_k == 0) {
        for (int i = tid; i < K; i += TOPK_THREADS) {
            output_indices[i] = -1;
            output_scores[i] = -INFINITY;
        }
        if (tid == 0) *effective_k_out = 0;
        return;
    }

    // --- 4-pass radix histogram selection ---
    // Each pass examines 8 bits of the sortable key, from MSB to LSB.
    // After all passes, thresh_prefix holds the exact 32-bit threshold key.
    uint32_t thresh_prefix = 0;
    int remaining = eff_k;

    for (int pass = 0; pass < 4 && remaining > 0; pass++) {
        const int shift = (3 - pass) * 8;  // 24, 16, 8, 0

        // Clear histogram
        if (tid < NUM_BINS) s_hist[tid] = 0;
        __syncthreads();

        // Build histogram of the current 8-bit slice
        for (int i = tid; i < N; i += TOPK_THREADS) {
            if (__ldg(block_endpoints + i) > query_position) continue;

            uint32_t key = float_to_sortable(__ldg(scores + i));

            // Filter: only elements matching the prefix from higher passes
            if (pass > 0) {
                int check_shift = (4 - pass) * 8;  // 24, 16, 8
                if ((key >> check_shift) != (thresh_prefix >> check_shift)) continue;
            }

            uint8_t bin = (key >> shift) & 0xFF;
            atomicAdd(&s_hist[bin], 1u);
        }
        __syncthreads();

        // Scan histogram from high to low to find threshold bin
        if (tid == 0) {
            int acc = 0;
            uint8_t thresh_bin = 0;
            for (int b = 255; b >= 0; b--) {
                acc += s_hist[b];
                if (acc >= remaining) {
                    thresh_bin = (uint8_t)b;
                    int above = acc - s_hist[b];
                    s_remaining = remaining - above;
                    break;
                }
            }
            s_thresh_prefix = thresh_prefix | ((uint32_t)thresh_bin << shift);
        }
        __syncthreads();

        thresh_prefix = s_thresh_prefix;
        remaining = s_remaining;
    }
    __syncthreads();

    // --- Gather phase: collect elements above threshold ---
    if (tid == 0) {
        s_cnt = 0;
        s_thresh_cnt = 0;
    }
    __syncthreads();

    const uint32_t threshold_key = thresh_prefix;
    const int final_remaining = remaining;

    for (int i = tid; i < N; i += TOPK_THREADS) {
        if (__ldg(block_endpoints + i) > query_position) continue;

        uint32_t key = float_to_sortable(__ldg(scores + i));

        if (key > threshold_key) {
            int pos = atomicAdd(&s_cnt, 1);
            if (pos < K) {
                s_out_idx[pos] = i;
                s_out_score[pos] = __ldg(scores + i);
            }
        } else if (key == threshold_key) {
            int old = atomicAdd(&s_thresh_cnt, 1);
            if (old < final_remaining) {
                int pos = atomicAdd(&s_cnt, 1);
                if (pos < K) {
                    s_out_idx[pos] = i;
                    s_out_score[pos] = __ldg(scores + i);
                }
            }
        }
    }
    __syncthreads();

    int actual_k = min(s_cnt, K);

    // --- Bitonic sort by index (ascending) ---
    // Round up to next power of 2 for bitonic sort
    int sort_n = 1;
    while (sort_n < actual_k) sort_n <<= 1;

    // Pad with INT_MAX so padding elements sort to end
    for (int i = tid + actual_k; i < sort_n; i += TOPK_THREADS) {
        s_out_idx[i] = INT_MAX;
        s_out_score[i] = -INFINITY;
    }
    __syncthreads();

    for (int k_step = 2; k_step <= sort_n; k_step <<= 1) {
        for (int j = k_step >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = tid; i < sort_n; i += TOPK_THREADS) {
                int ixj = i ^ j;
                if (ixj > i && ixj < sort_n) {
                    bool ascending = ((i & k_step) == 0);
                    bool do_swap = ascending
                        ? (s_out_idx[i] > s_out_idx[ixj])
                        : (s_out_idx[i] < s_out_idx[ixj]);
                    if (do_swap) {
                        int tmp_i = s_out_idx[i];
                        s_out_idx[i] = s_out_idx[ixj];
                        s_out_idx[ixj] = tmp_i;
                        float tmp_s = s_out_score[i];
                        s_out_score[i] = s_out_score[ixj];
                        s_out_score[ixj] = tmp_s;
                    }
                }
            }
        }
    }
    __syncthreads();

    // --- Write to global memory ---
    for (int i = tid; i < K; i += TOPK_THREADS) {
        if (i < actual_k) {
            output_indices[i] = s_out_idx[i];
            output_scores[i] = s_out_score[i];
        } else {
            output_indices[i] = -1;
            output_scores[i] = -INFINITY;
        }
    }
    if (tid == 0) {
        *effective_k_out = actual_k;
    }
}

__global__ void __launch_bounds__(TOPK_THREADS)
lightning_topk_kernel(const LightningTopkParams params) {
    lightning_topk_cta(params.scores, params.block_endpoints,
                       params.output_indices, params.output_scores,
                       params.effective_k_out,
                       params.num_blocks, params.topk, params.query_position);
}

// TD-SPARSE-PREFILL-SCORE-BATCH: one CTA per row, each running the exact
// single-query body on its row's scores slice / bound / cutoff / outputs.
__global__ void __launch_bounds__(TOPK_THREADS)
lightning_topk_batched_kernel(const LightningTopkBatchedParams params) {
    const int row = blockIdx.x;
    lightning_topk_cta(
        params.scores + (int64_t)row * params.scores_stride,
        params.block_endpoints,
        params.output_indices + (size_t)row * params.topk,
        params.output_scores + (size_t)row * params.topk,
        params.effective_k_out + row,
        __ldg(params.row_num_blocks + row),
        params.topk,
        __ldg(params.row_query_position + row));
}

void run_lightning_topk(const LightningTopkParams& params, cudaStream_t stream) {
    if (params.topk == 0) return;
    // num_blocks == 0 still launches: the kernel then writes the empty result
    // (effective_k = 0, indices padded -1) — required by the DCP local-indexer
    // mode where a rank may own zero positions of a short sequence and its
    // candidate list must be valid (-1-padded) before the cross-rank merge.
    lightning_topk_kernel<<<1, TOPK_THREADS, 0, stream>>>(params);
}

void run_lightning_topk_batched(const LightningTopkBatchedParams& params,
                                cudaStream_t stream) {
    if (params.topk == 0 || params.num_rows <= 0) return;
    // Rows with num_blocks == 0 still get the valid empty result (see above).
    lightning_topk_batched_kernel<<<params.num_rows, TOPK_THREADS, 0, stream>>>(
        params);
}

}  // namespace sm120::indexer
