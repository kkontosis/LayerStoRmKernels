#pragma once
/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * TQ Sparse Decode Kernel — Index-Based Gather + Codebook Lookup + Rotated-Space PV
 *
 * Same algorithm as tq_dense_decode but gathers KV tokens via indices array
 * instead of iterating pages sequentially.
 *
 * Key differences from dense:
 *   - Block loop iterates over topk indices in groups of 64
 *   - Each token loaded via gather: indices[block*64 + k] → page resolution → cache read
 *   - Validity: indices[i] >= 0 (padded with -1 for invalid)
 *   - No block_table — direct page resolution from global token index
 **************************************************************************************************/

#include "params.h"
#include <cuda_fp16.h>

namespace sm120::decode::tq_sparse {

static constexpr int NUM_THREADS = 256;
static constexpr int KV_BLOCK_SIZE = 64;
static constexpr int NUM_CENTROIDS = 16;
static constexpr int NUM_WARPS = 8;

// Shared memory layout (same as dense)
struct TqSparseDecodeSmem {
    uint8_t packed_nope[KV_BLOCK_SIZE * 256];    // 16KB (d_c=512)
    __half norms[KV_BLOCK_SIZE];                 // 128B
    __nv_bfloat16 k_rope[KV_BLOCK_SIZE * 64];   // 8KB (d_rope=64)
    float centroids[NUM_CENTROIDS];              // 64B
    float warp_nope[NUM_WARPS][KV_BLOCK_SIZE];   // 2KB (cross-warp NOPE partials)
    float scores[KV_BLOCK_SIZE];                 // 256B
    float P_weights[KV_BLOCK_SIZE];              // 256B
    float reduce_scratch[NUM_WARPS];             // 32B
    int token_valid[KV_BLOCK_SIZE];              // 256B — validity flags
};

__global__ void __launch_bounds__(256)
tq_sparse_decode_kernel(const TqSparseDecodeParams params) {
    const int head_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int lane = tid % 32;

    if (head_idx >= params.h_q) return;

    const int d_c = params.d_c;
    const int d_rope = params.d_rope;
    const int packed_bytes = d_c / 2;
    const int page_block_size = params.page_block_size;

    extern __shared__ char smem_raw[];
    TqSparseDecodeSmem& smem = *reinterpret_cast<TqSparseDecodeSmem*>(smem_raw);

    // Load centroids
    if (tid < NUM_CENTROIDS)
        smem.centroids[tid] = params.centroids[tid];

    // Pointer to this query's indices: [topk]
    const int* my_indices = params.indices +
        batch_idx * params.stride_indices_b +
        s_q_idx * params.stride_indices_s_q;

    // Load q_rot to registers (2 values per thread, FP32)
    const float* q_rot_ptr = params.q_rot +
        (int64_t)batch_idx * params.s_q * params.h_q * d_c +
        (int64_t)s_q_idx * params.h_q * d_c +
        (int64_t)head_idx * d_c;
    float qr0 = 0.0f, qr1 = 0.0f;
    if (tid * 2 < d_c) {
        qr0 = __ldg(q_rot_ptr + tid * 2);
        qr1 = __ldg(q_rot_ptr + tid * 2 + 1);
    }

    // Load q_rope to registers — per LANE, not per tid: the ROPE score loop
    // reduces a full-d_rope dot product inside EACH warp (lane l covers dims
    // 2l, 2l+1), so every warp needs the full q_rope vector. Loading with tid
    // left warps 1..7 with zeros, dropping the rope score for tokens 8..63 of
    // every KV block.
    const int rope_stride = params.q_rope_row_stride > 0
        ? params.q_rope_row_stride : d_rope;
    const __nv_bfloat16* q_rope_ptr = params.q_rope +
        ((int64_t)batch_idx * params.s_q * params.h_q +
         (int64_t)s_q_idx * params.h_q +
         (int64_t)head_idx) * rope_stride;
    float qrope0 = 0.0f, qrope1 = 0.0f;
    if (lane * 2 < d_rope) {
        qrope0 = __bfloat162float(__ldg(q_rope_ptr + lane * 2));
        qrope1 = __bfloat162float(__ldg(q_rope_ptr + lane * 2 + 1));
    }

    // PV accumulator (rotated space) + online softmax state
    float acc0 = 0.0f, acc1 = 0.0f;
    float M_global = -1e30f, L_global = 0.0f;

    __syncthreads();

    // =====================================================================
    // Main loop over topk indices in blocks of KV_BLOCK_SIZE
    // =====================================================================
    int total_blocks = (params.topk + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    for (int blk = 0; blk < total_blocks; ++blk) {
        int blk_start = blk * KV_BLOCK_SIZE;

        // =================================================================
        // Load KV data to smem via index-based gather
        // =================================================================

        // First: determine which tokens are valid and resolve their cache addresses
        // Each of the first 64 threads checks one token's validity
        if (tid < KV_BLOCK_SIZE) {
            int idx_pos = blk_start + tid;
            int token_idx = (idx_pos < params.topk) ? __ldg(my_indices + idx_pos) : -1;
            smem.token_valid[tid] = token_idx;  // store token index (>= 0 means valid)
        }
        __syncthreads();

        // Count valid tokens in this block
        int num_valid = 0;
        for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
            if (smem.token_valid[i] >= 0) num_valid = i + 1;
            else break;  // Indices are contiguous — once we hit -1, rest are invalid
        }
        // Actually, sparse indices may not have -1 only at the end.
        // Count all valid ones properly.
        num_valid = 0;
        for (int i = 0; i < KV_BLOCK_SIZE; ++i) {
            if (smem.token_valid[i] >= 0) num_valid++;
        }

        if (num_valid == 0) continue;

        // Load packed_nope via gather
        int total_packed = KV_BLOCK_SIZE * packed_bytes;
        for (int i = tid; i < total_packed; i += NUM_THREADS) {
            int tok = i / packed_bytes;
            int bi = i % packed_bytes;
            int token_idx = smem.token_valid[tok];
            if (token_idx >= 0) {
                int page_id = token_idx / page_block_size;
                int row_in_page = token_idx % page_block_size;
                const uint8_t* row_ptr = params.kv_cache +
                    (int64_t)page_id * params.cache_stride_block +
                    (int64_t)row_in_page * params.cache_stride_row;
                smem.packed_nope[i] = __ldg(row_ptr + bi);
            } else {
                smem.packed_nope[i] = 0;
            }
        }

        // Load norms via gather
        if (tid < KV_BLOCK_SIZE) {
            int token_idx = smem.token_valid[tid];
            if (token_idx >= 0) {
                int page_id = token_idx / page_block_size;
                int row_in_page = token_idx % page_block_size;
                const uint8_t* row_ptr = params.kv_cache +
                    (int64_t)page_id * params.cache_stride_block +
                    (int64_t)row_in_page * params.cache_stride_row;
                smem.norms[tid] = *reinterpret_cast<const __half*>(row_ptr + packed_bytes);
            } else {
                smem.norms[tid] = __float2half(0.0f);
            }
        }

        // Load k_rope via gather
        int total_rope = KV_BLOCK_SIZE * d_rope;
        for (int i = tid; i < total_rope; i += NUM_THREADS) {
            int tok = i / d_rope;
            int di = i % d_rope;
            int token_idx = smem.token_valid[tok];
            if (token_idx >= 0) {
                int page_id = token_idx / page_block_size;
                int row_in_page = token_idx % page_block_size;
                const __nv_bfloat16* rope_ptr = reinterpret_cast<const __nv_bfloat16*>(
                    params.kv_cache +
                    (int64_t)page_id * params.cache_stride_block +
                    (int64_t)row_in_page * params.cache_stride_row +
                    packed_bytes + 2);
                smem.k_rope[i] = __ldg(rope_ptr + di);
            } else {
                smem.k_rope[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // =================================================================
        // NOPE scores: all 64 tokens, warp reduce + cross-warp via smem
        // (identical to dense kernel from here on)
        // =================================================================
        #pragma unroll 1
        for (int sub = 0; sub < KV_BLOCK_SIZE / 8; ++sub) {
            float p[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int tok = sub * 8 + k;
                uint8_t byte = smem.packed_nope[tok * packed_bytes + tid];
                int i0 = byte & 0x0F, i1 = (byte >> 4) & 0x0F;
                p[k] = qr0 * smem.centroids[i0] + qr1 * smem.centroids[i1];
            }
            // Warp reduce
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                #pragma unroll
                for (int off = 16; off > 0; off /= 2)
                    p[k] += __shfl_xor_sync(0xffffffff, p[k], off);
            }
            // Lane 0 of each warp stores to smem
            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                    smem.warp_nope[warp_idx][sub * 8 + k] = p[k];
            }
        }
        __syncthreads();

        // Sum warp partials: threads 0..63 each handle one token
        if (tid < KV_BLOCK_SIZE) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w)
                s += smem.warp_nope[w][tid];
            // Multiply by norm; invalid tokens have norm=0 so score=0
            smem.scores[tid] = s * __half2float(smem.norms[tid]);
        }
        __syncthreads();

        // =================================================================
        // ROPE scores: 8 warps, each handles 8 tokens
        // =================================================================
        {
            int tok_start = warp_idx * 8;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int tok = tok_start + k;
                float pr = 0.0f;
                if (lane * 2 < d_rope) {
                    pr = qrope0 * __bfloat162float(smem.k_rope[tok * d_rope + lane * 2]) +
                         qrope1 * __bfloat162float(smem.k_rope[tok * d_rope + lane * 2 + 1]);
                }
                #pragma unroll
                for (int off = 16; off > 0; off /= 2)
                    pr += __shfl_xor_sync(0xffffffff, pr, off);
                // Only add rope score if token is valid
                if (lane == 0 && smem.token_valid[tok] >= 0)
                    smem.scores[tok] += pr;
            }
        }
        __syncthreads();

        // =================================================================
        // Online softmax
        // =================================================================
        if (tid < KV_BLOCK_SIZE) {
            smem.scores[tid] = (smem.token_valid[tid] >= 0) ?
                smem.scores[tid] * params.sm_scale : -1e30f;
        }
        __syncthreads();

        // Block max reduction
        float local_max = (tid < KV_BLOCK_SIZE && smem.token_valid[tid] >= 0) ?
            smem.scores[tid] : -1e30f;
        #pragma unroll
        for (int off = 16; off > 0; off /= 2)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        if (lane == 0) smem.reduce_scratch[warp_idx] = local_max;
        __syncthreads();

        float M_block = smem.reduce_scratch[0];
        if (tid == 0) {
            for (int w = 1; w < NUM_WARPS; ++w)
                M_block = fmaxf(M_block, smem.reduce_scratch[w]);
            smem.reduce_scratch[0] = M_block;
        }
        __syncthreads();
        M_block = smem.reduce_scratch[0];

        // Rescale old accumulators
        float M_new = fmaxf(M_global, M_block);
        float rescale = (M_global > -1e29f) ? expf(M_global - M_new) : 0.0f;
        acc0 *= rescale;
        acc1 *= rescale;
        L_global *= rescale;

        // Compute exp(score - M_new) and P_weights
        if (tid < KV_BLOCK_SIZE) {
            float p = (smem.token_valid[tid] >= 0) ?
                expf(smem.scores[tid] - M_new) : 0.0f;
            smem.scores[tid] = p;                                     // for L sum
            smem.P_weights[tid] = p * __half2float(smem.norms[tid]);  // for PV
        }
        __syncthreads();

        // Sum L
        float l_part = (tid < KV_BLOCK_SIZE && smem.token_valid[tid] >= 0) ?
            smem.scores[tid] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off /= 2)
            l_part += __shfl_xor_sync(0xffffffff, l_part, off);
        if (lane == 0) smem.reduce_scratch[warp_idx] = l_part;
        __syncthreads();
        if (tid == 0) {
            float s = 0;
            for (int w = 0; w < NUM_WARPS; ++w) s += smem.reduce_scratch[w];
            smem.reduce_scratch[0] = s;
        }
        __syncthreads();
        L_global += smem.reduce_scratch[0];
        M_global = M_new;

        // =================================================================
        // PV accumulation in rotated space
        // =================================================================
        #pragma unroll 1
        for (int k = 0; k < KV_BLOCK_SIZE; ++k) {
            if (smem.token_valid[k] < 0) continue;
            float w = smem.P_weights[k];
            if (w == 0.0f) continue;
            uint8_t byte = smem.packed_nope[k * packed_bytes + tid];
            int i0 = byte & 0x0F, i1 = (byte >> 4) & 0x0F;
            acc0 += w * smem.centroids[i0];
            acc1 += w * smem.centroids[i1];
        }
        __syncthreads();
    }  // end block loop

    // =====================================================================
    // Epilogue: normalize + write
    // =====================================================================
    float inv_L = (L_global > 0.0f) ? (1.0f / L_global) : 0.0f;
    acc0 *= inv_L;
    acc1 *= inv_L;

    float* o_ptr = params.out +
        (int64_t)batch_idx * params.stride_o_b +
        (int64_t)s_q_idx * params.stride_o_s_q +
        (int64_t)head_idx * params.stride_o_h_q;
    if (tid * 2 < d_c) {
        o_ptr[tid * 2]     = acc0;
        o_ptr[tid * 2 + 1] = acc1;
    }

    if (tid == 0) {
        float lse = (L_global > 0.0f) ? (logf(L_global) + M_global) : -1e30f;
        params.lse[batch_idx * params.stride_lse_b +
                   s_q_idx * params.stride_lse_s_q +
                   head_idx] = lse;
    }
}

void run_tq_sparse_decode(const TqSparseDecodeParams& params) {
    size_t smem_size = sizeof(TqSparseDecodeSmem);
    tq_sparse_decode_kernel<<<dim3(params.h_q, params.s_q, params.b),
                              NUM_THREADS, smem_size, params.stream>>>(params);
}

}  // namespace sm120::decode::tq_sparse
