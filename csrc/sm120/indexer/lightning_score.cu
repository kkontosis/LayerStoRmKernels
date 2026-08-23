#include "lightning_score.h"

namespace sm120::indexer {

static constexpr int BLOCKS_PER_CTA = 4;
static constexpr float FP8_E4M3_MAX = 448.0f;

// __ldg for FP8: load as char via read-only cache and reinterpret
__device__ __forceinline__ __nv_fp8_e4m3 ldg_fp8(const __nv_fp8_e4m3* ptr) {
    char raw = __ldg(reinterpret_cast<const char*>(ptr));
    __nv_fp8_e4m3 result;
    memcpy(&result, &raw, 1);
    return result;
}

__global__ void __launch_bounds__(256)
lightning_score_kernel(const LightningScoreParams params) {
    const int block_base = blockIdx.x * BLOCKS_PER_CTA;
    const int tid = threadIdx.x;

    // Static shared memory: score_proj (64 floats = 256B) + warp reduction (32B)
    __shared__ float s_score_proj[64];
    __shared__ float s_warp_sums[8];

    if (tid < params.index_n_heads) {
        s_score_proj[tid] = params.score_proj[tid];
    }

    // Dynamic shared memory: Q in BF16 [INDEX_N_HEADS * INDEX_HEAD_DIM]
    // 64 * 128 * 2B = 16 KB — loaded once, reused across all blocks in this CTA
    extern __shared__ __nv_bfloat16 s_q_proj[];
    const int q_total = params.index_n_heads * params.index_head_dim;
    for (int i = tid; i < q_total; i += 256) {
        s_q_proj[i] = params.q_proj[i];
    }
    __syncthreads();

    // Thread mapping: 256 threads / 64 heads = 4 threads per head
    // Each thread handles 128/4 = 32 dimensions of the dot product
    const int head_idx = tid / 4;
    const int lane = tid % 4;
    const int warp_id = tid / 32;
    const int dim_start = lane * 32;

    // Pre-load Q slice for this thread's head+dims into registers
    float q_reg[8];  // 32 values stored as 8 float4-worth (4 values per iteration below)
    if (head_idx < params.index_n_heads) {
        const __nv_bfloat16* q_ptr = s_q_proj + head_idx * params.index_head_dim + dim_start;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            // Load pairs of BF16, convert to float
            __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(q_ptr + d * 4);
            q_reg[d] = __low2float(pair);
            // We'll interleave the high values in the dot product loop
        }
    }

    for (int bi = 0; bi < BLOCKS_PER_CTA; bi++) {
        const int block_idx = block_base + bi;
        if (block_idx >= params.num_blocks) return;

        float partial_dot = 0.0f;
        if (head_idx < params.index_n_heads) {
            // FP8 K cache pointer for this block+head+dim_start
            const __nv_fp8_e4m3* k_ptr = params.indexer_k_cache +
                (int64_t)block_idx * params.index_n_heads * params.index_head_dim +
                head_idx * params.index_head_dim + dim_start;
            const __nv_bfloat16* q_ptr = s_q_proj + head_idx * params.index_head_dim + dim_start;

            // Dot product: 32 FP8 elements × 32 BF16 elements → FP32 accumulation
            // Load FP8 4 at a time (32 bits = 4 FP8 values)
            #pragma unroll
            for (int d = 0; d < 32; d += 4) {
                // Vectorized FP8 load: 4 bytes at once
                uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(k_ptr + d));
                const __nv_fp8_e4m3* vals = reinterpret_cast<const __nv_fp8_e4m3*>(&packed);

                float k0 = float(vals[0]);
                float k1 = float(vals[1]);
                float k2 = float(vals[2]);
                float k3 = float(vals[3]);

                // Q from shared memory (BF16 → float)
                float q0 = __bfloat162float(q_ptr[d]);
                float q1 = __bfloat162float(q_ptr[d + 1]);
                float q2 = __bfloat162float(q_ptr[d + 2]);
                float q3 = __bfloat162float(q_ptr[d + 3]);

                partial_dot += k0 * q0 + k1 * q1 + k2 * q2 + k3 * q3;
            }

            // Apply per-block K scale
            float k_scale = __ldg(params.k_scales + block_idx);
            partial_dot *= k_scale;
        }

        // Reduce within 4 threads per head
        partial_dot += __shfl_xor_sync(0xffffffff, partial_dot, 1);
        partial_dot += __shfl_xor_sync(0xffffffff, partial_dot, 2);

        // Lane 0: ReLU + score_proj weighting
        float weighted = 0.0f;
        if (lane == 0 && head_idx < params.index_n_heads) {
            weighted = fmaxf(partial_dot, 0.0f) * s_score_proj[head_idx];
        }

        // Warp reduce: sum weighted scores across 8 heads per warp
        float warp_sum = weighted;
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, offset);
        }

        // Cross-warp reduce
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
            params.scores_out[block_idx] = total;
        }
        __syncthreads();
    }
}

void run_lightning_score(const LightningScoreParams& params, cudaStream_t stream) {
    if (params.num_blocks == 0) return;

    int grid = (params.num_blocks + BLOCKS_PER_CTA - 1) / BLOCKS_PER_CTA;
    int smem = params.index_n_heads * params.index_head_dim * sizeof(__nv_bfloat16);

    lightning_score_kernel<<<grid, 256, smem, stream>>>(params);
}

}  // namespace sm120::indexer
