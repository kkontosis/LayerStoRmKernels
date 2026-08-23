#include "lightning_score_mqa.h"

namespace sm120::indexer {

static constexpr int MQA_BLOCKS_PER_CTA = 4;

// Per-block scoring body shared by the single-query and the batched kernel
// (TD-SPARSE-PREFILL-SCORE-BATCH). Identical thread mapping (256 threads,
// 4 threads/head covering head_dim/4 dims each) and identical arithmetic /
// reduction order as the original inline body — the batched kernel is
// bit-identical to per-query launches BY CONSTRUCTION (same device code per
// (row, block), only the grid packing differs; a block's score never depends
// on which CTA slot computed it).
__device__ __forceinline__ void mqa_score_one_block(
    const __nv_bfloat16* s_q, const float* s_score_proj, float* s_warp_sums,
    const __nv_fp8_e4m3* k_row, const float* k_scale_ptr, float* score_out,
    int index_n_heads, int index_head_dim) {
    const int tid = threadIdx.x;

    // Thread mapping: 256 threads / heads → 4 threads per head, each covering
    // index_head_dim/4 dims of the dot product.
    const int head_idx = tid / 4;
    const int lane = tid % 4;
    const int warp_id = tid / 32;
    const int dim_start = lane * (index_head_dim / 4);
    const int dims_per_thread = index_head_dim / 4;

    float partial_dot = 0.0f;
    if (head_idx < index_n_heads) {
        // Single shared K per block — NO head offset (this is the MQA change).
        const __nv_fp8_e4m3* k_ptr = k_row + dim_start;
        const __nv_bfloat16* q_ptr = s_q + head_idx * index_head_dim + dim_start;

        // Dot product over this thread's dim slice: FP8 K × BF16 Q → FP32.
        #pragma unroll 4
        for (int d = 0; d < dims_per_thread; d += 4) {
            uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(k_ptr + d));
            const __nv_fp8_e4m3* vals = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);

            float k0 = float(vals[0]);
            float k1 = float(vals[1]);
            float k2 = float(vals[2]);
            float k3 = float(vals[3]);

            float q0 = __bfloat162float(q_ptr[d]);
            float q1 = __bfloat162float(q_ptr[d + 1]);
            float q2 = __bfloat162float(q_ptr[d + 2]);
            float q3 = __bfloat162float(q_ptr[d + 3]);

            partial_dot += k0 * q0 + k1 * q1 + k2 * q2 + k3 * q3;
        }

        // Apply per-block K scale.
        float k_scale = __ldg(k_scale_ptr);
        partial_dot *= k_scale;
    }

    // Reduce within the 4 threads per head.
    partial_dot += __shfl_xor_sync(0xffffffff, partial_dot, 1);
    partial_dot += __shfl_xor_sync(0xffffffff, partial_dot, 2);

    // Lane 0 of each head: ReLU + score_proj weighting.
    float weighted = 0.0f;
    if (lane == 0 && head_idx < index_n_heads) {
        weighted = fmaxf(partial_dot, 0.0f) * s_score_proj[head_idx];
    }

    // Warp reduce across heads in this warp.
    float warp_sum = weighted;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
    }

    if (tid % 32 == 0) {
        s_warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            total += s_warp_sums[w];
        }
        *score_out = total;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(256)
lightning_score_mqa_kernel(const LightningScoreMqaParams params) {
    const int block_base = blockIdx.x * MQA_BLOCKS_PER_CTA;
    const int tid = threadIdx.x;

    // Static shared memory: score_proj (up to 64 heads) + warp reduction
    __shared__ float s_score_proj[64];
    __shared__ float s_warp_sums[8];

    if (tid < params.index_n_heads) {
        s_score_proj[tid] = params.score_proj[tid];
    }

    // Dynamic shared memory: Q in BF16 [INDEX_N_HEADS * INDEX_HEAD_DIM],
    // loaded once and reused across all blocks in this CTA.
    extern __shared__ __nv_bfloat16 s_q_proj_mqa[];
    const int q_total = params.index_n_heads * params.index_head_dim;
    for (int i = tid; i < q_total; i += 256) {
        s_q_proj_mqa[i] = params.q_proj[i];
    }
    __syncthreads();

    for (int bi = 0; bi < MQA_BLOCKS_PER_CTA; bi++) {
        const int block_idx = block_base + bi;
        if (block_idx >= params.num_blocks) return;
        mqa_score_one_block(
            s_q_proj_mqa, s_score_proj, s_warp_sums,
            params.indexer_k_cache + (int64_t)block_idx * params.index_head_dim,
            params.k_scales + block_idx,
            params.scores_out + block_idx,
            params.index_n_heads, params.index_head_dim);
    }
}

// TD-SPARSE-PREFILL-SCORE-BATCH: multi-row variant. Grid y = row; each row
// carries its OWN block bound (row_num_blocks[row]) and its own query row,
// score-proj row and scores-out row. K is either one contiguous cache shared
// by all rows (global block index addressing, identical to the single-query
// kernel) or a per-row page table ([page_tokens × head_dim FP8 rows |
// page_tokens f32 scales] per page — the exact layout the retired per-page
// host loop launched over). Per (row, block) the scoring body is
// mqa_score_one_block above — bit-identical to per-query launches.
__global__ void __launch_bounds__(256)
lightning_score_mqa_batched_kernel(const LightningScoreMqaBatchedParams params) {
    const int row = blockIdx.y;
    const int num_blocks = __ldg(params.row_num_blocks + row);
    const int block_base = blockIdx.x * MQA_BLOCKS_PER_CTA;
    if (block_base >= num_blocks) return;
    const int tid = threadIdx.x;

    __shared__ float s_score_proj[64];
    __shared__ float s_warp_sums[8];

    if (tid < params.index_n_heads) {
        s_score_proj[tid] =
            params.score_proj_all[(size_t)row * params.index_n_heads + tid];
    }

    extern __shared__ __nv_bfloat16 s_q_proj_mqa[];
    const int q_total = params.index_n_heads * params.index_head_dim;
    const __nv_bfloat16* q_row = params.q_all + (size_t)row * q_total;
    for (int i = tid; i < q_total; i += 256) {
        s_q_proj_mqa[i] = q_row[i];
    }
    __syncthreads();

    float* scores_row = params.scores_out + (int64_t)row * params.scores_stride;
    for (int bi = 0; bi < MQA_BLOCKS_PER_CTA; bi++) {
        const int block_idx = block_base + bi;
        if (block_idx >= num_blocks) return;
        const __nv_fp8_e4m3* k_row;
        const float* k_scale_ptr;
        if (params.k_page_table) {
            const int pg = block_idx / params.page_tokens;
            const int sl = block_idx - pg * params.page_tokens;
            const __nv_fp8_e4m3* base = static_cast<const __nv_fp8_e4m3*>(
                params.k_page_table[(size_t)row * params.page_table_stride + pg]);
            k_row = base + (int64_t)sl * params.index_head_dim;
            k_scale_ptr = reinterpret_cast<const float*>(
                base + (size_t)params.page_tokens * params.index_head_dim) + sl;
        } else {
            k_row = params.k_cache + (int64_t)block_idx * params.index_head_dim;
            k_scale_ptr = params.k_scales + block_idx;
        }
        mqa_score_one_block(s_q_proj_mqa, s_score_proj, s_warp_sums,
                            k_row, k_scale_ptr, scores_row + block_idx,
                            params.index_n_heads, params.index_head_dim);
    }
}

void run_lightning_score_mqa(const LightningScoreMqaParams& params, cudaStream_t stream) {
    if (params.num_blocks == 0) return;

    int grid = (params.num_blocks + MQA_BLOCKS_PER_CTA - 1) / MQA_BLOCKS_PER_CTA;
    int smem = params.index_n_heads * params.index_head_dim * sizeof(__nv_bfloat16);

    lightning_score_mqa_kernel<<<grid, 256, smem, stream>>>(params);
}

void run_lightning_score_mqa_batched(const LightningScoreMqaBatchedParams& params,
                                     cudaStream_t stream) {
    if (params.num_rows <= 0 || params.max_num_blocks <= 0) return;

    dim3 grid((params.max_num_blocks + MQA_BLOCKS_PER_CTA - 1) / MQA_BLOCKS_PER_CTA,
              params.num_rows);
    int smem = params.index_n_heads * params.index_head_dim * sizeof(__nv_bfloat16);

    lightning_score_mqa_batched_kernel<<<grid, 256, smem, stream>>>(params);
}

}  // namespace sm120::indexer
