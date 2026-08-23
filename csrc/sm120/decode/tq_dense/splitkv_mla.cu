#pragma once
/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * TQ Dense Decode Kernel — Scalar Codebook Lookup + Rotated-Space PV
 *
 * v2: Split-KV support via tile scheduler metadata.
 *
 * Grid: (h_q, s_q, num_sm_parts) — each CTA handles one head, iterating
 * over its assigned (batch, block_range) pairs from the tile scheduler.
 *
 * When num_sm_parts == 1, falls back to the simple single-CTA-per-head path
 * (grid.z iterates batches directly, no tile scheduler).
 *
 * Thread assignment (256 threads, d_c=512):
 *   Thread tid handles coordinates [tid*2, tid*2+1] for both score + PV
 *
 * Per KV block (64 tokens):
 *   1. Load packed_nope + norms + k_rope to smem
 *   2. NOPE scores: scalar codebook lookup + warp/cross-warp reduce
 *   3. ROPE scores: warp-parallel dot products
 *   4. Online softmax
 *   5. PV accumulation in rotated space
 **************************************************************************************************/

#include "params.h"
#include <cuda_fp16.h>

namespace sm120::decode::tq_dense {

static constexpr int NUM_THREADS = 256;
static constexpr int KV_BLOCK_SIZE = 64;
static constexpr int NUM_CENTROIDS = 16;
static constexpr int NUM_WARPS = 8;
static constexpr float LOG2E_CONST = 1.4426950408889634f;

// Shared memory layout
struct TqDecodeSmem {
    uint8_t packed_nope[KV_BLOCK_SIZE * 256];    // 16KB (d_c=512)
    __half norms[KV_BLOCK_SIZE];                 // 128B
    __nv_bfloat16 k_rope[KV_BLOCK_SIZE * 64];   // 8KB (d_rope=64)
    float centroids[NUM_CENTROIDS];              // 64B
    float warp_nope[NUM_WARPS][KV_BLOCK_SIZE];   // 2KB (cross-warp NOPE partials)
    float scores[KV_BLOCK_SIZE];                 // 256B
    float P_weights[KV_BLOCK_SIZE];              // 256B
    float reduce_scratch[NUM_WARPS];             // 32B
};
// Total: ~27KB


// Process one (batch, block_range) assignment. Updates acc0/acc1/M_global/L_global.
__device__ __forceinline__ void process_kv_blocks(
    const TqDenseDecodeParams& params,
    TqDecodeSmem& smem,
    int batch_idx, int head_idx, int s_q_idx,
    int start_block, int end_block,
    float qr0, float qr1, float qrope0, float qrope1,
    float& acc0, float& acc1, float& M_global, float& L_global
) {
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int lane = tid % 32;
    const int d_c = params.d_c;
    const int d_rope = params.d_rope;
    const int packed_bytes = d_c / 2;

    for (int blk = start_block; blk < end_block; ++blk) {
        int page_id = __ldg(params.block_table +
                            batch_idx * params.block_table_batch_stride + blk);
        const uint8_t* page_base = params.kv_cache +
                                   (int64_t)page_id * params.cache_stride_block;
        int seqlen = __ldg(params.seqlens_k + batch_idx);
        int blk_start = blk * params.page_block_size;
        int num_valid = min(params.page_block_size, seqlen - blk_start);

        // =================================================================
        // Load KV data to smem (restructured: explicit per-token iteration)
        // =================================================================
        // packed_nope: 256 threads × 256 packed_bytes = 1:1 mapping per token
        for (int tok = 0; tok < KV_BLOCK_SIZE; ++tok) {
            smem.packed_nope[tok * packed_bytes + tid] = (tok < num_valid) ?
                __ldg(page_base + (int64_t)tok * params.cache_stride_row + tid) : 0;
        }
        if (tid < KV_BLOCK_SIZE) {
            smem.norms[tid] = (tid < num_valid) ?
                *reinterpret_cast<const __half*>(
                    page_base + (int64_t)tid * params.cache_stride_row + packed_bytes)
                : __float2half(0.0f);
        }
        int total_rope = KV_BLOCK_SIZE * d_rope;
        for (int i = tid; i < total_rope; i += NUM_THREADS) {
            int tok = i / d_rope;
            int di = i % d_rope;
            smem.k_rope[i] = (tok < num_valid) ?
                __ldg(reinterpret_cast<const __nv_bfloat16*>(
                    page_base + (int64_t)tok * params.cache_stride_row + packed_bytes + 2) + di)
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // =================================================================
        // NOPE scores: 8 tokens at a time, warp reduce + cross-warp via smem
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
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                #pragma unroll
                for (int off = 16; off > 0; off /= 2)
                    p[k] += __shfl_xor_sync(0xffffffff, p[k], off);
            }
            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                    smem.warp_nope[warp_idx][sub * 8 + k] = p[k];
            }
        }
        __syncthreads();

        // Sum warp partials
        if (tid < KV_BLOCK_SIZE) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w)
                s += smem.warp_nope[w][tid];
            smem.scores[tid] = s * __half2float(smem.norms[tid]);
        }
        __syncthreads();

        // =================================================================
        // ROPE scores
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
                if (lane == 0 && tok < num_valid)
                    smem.scores[tok] += pr;
            }
        }
        __syncthreads();

        // =================================================================
        // Online softmax
        // =================================================================
        if (tid < KV_BLOCK_SIZE) {
            smem.scores[tid] = (tid < num_valid) ?
                smem.scores[tid] * params.sm_scale : -1e30f;
        }
        __syncthreads();

        float local_max = (tid < num_valid) ? smem.scores[tid] : -1e30f;
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

        float M_new = fmaxf(M_global, M_block);
        float rescale = (M_global > -1e29f) ? expf(M_global - M_new) : 0.0f;
        acc0 *= rescale;
        acc1 *= rescale;
        L_global *= rescale;

        if (tid < KV_BLOCK_SIZE) {
            float p = (tid < num_valid) ? expf(smem.scores[tid] - M_new) : 0.0f;
            smem.scores[tid] = p;
            smem.P_weights[tid] = p * __half2float(smem.norms[tid]);
        }
        __syncthreads();

        float l_part = (tid < num_valid) ? smem.scores[tid] : 0.0f;
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
        // PV accumulation in rotated space (batched, 8 tokens at a time)
        // =================================================================
        #pragma unroll 1
        for (int sub = 0; sub < KV_BLOCK_SIZE / 8; ++sub) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int tok = sub * 8 + k;
                float w = smem.P_weights[tok];
                if (w != 0.0f) {
                    uint8_t byte = smem.packed_nope[tok * packed_bytes + tid];
                    int i0 = byte & 0x0F, i1 = (byte >> 4) & 0x0F;
                    acc0 += w * smem.centroids[i0];
                    acc1 += w * smem.centroids[i1];
                }
            }
        }
        __syncthreads();
    }  // end block loop
}


__global__ void __launch_bounds__(256)
tq_dense_decode_kernel(const TqDenseDecodeParams params) {
    const int head_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (head_idx >= params.h_q) return;

    const int d_c = params.d_c;
    const int d_rope = params.d_rope;

    extern __shared__ char smem_raw[];
    TqDecodeSmem& smem = *reinterpret_cast<TqDecodeSmem*>(smem_raw);

    // Load centroids to smem
    if (tid < NUM_CENTROIDS)
        smem.centroids[tid] = params.centroids[tid];

    // Determine if we're using split-KV or simple mode
    const bool use_splitkv = (params.num_sm_parts > 1);
    const int partition_idx = blockIdx.z;

    if (!use_splitkv) {
        // =====================================================================
        // Simple mode: grid.z = batch, one CTA per (head, s_q, batch)
        // =====================================================================
        const int batch_idx = partition_idx;
        if (batch_idx >= params.b) return;

        // Load q_rot to registers
        const float* q_rot_ptr = params.q_rot +
            (int64_t)batch_idx * params.s_q * params.h_q * d_c +
            (int64_t)s_q_idx * params.h_q * d_c +
            (int64_t)head_idx * d_c;
        float qr0 = 0.0f, qr1 = 0.0f;
        if (tid * 2 < d_c) {
            qr0 = __ldg(q_rot_ptr + tid * 2);
            qr1 = __ldg(q_rot_ptr + tid * 2 + 1);
        }

        // Load q_rope to registers — per LANE, not per tid: the ROPE score
        // loop in process_kv_blocks reduces a full d_rope dot product inside
        // EACH warp (lane l covers dims 2l, 2l+1), so every warp needs the
        // full q_rope vector. Loading with tid left warps 1..7 with zeros,
        // silently dropping the rope score for tokens 8..63 of every block.
        const __nv_bfloat16* q_rope_ptr = params.q_rope +
            (int64_t)batch_idx * params.s_q * params.h_q * d_rope +
            (int64_t)s_q_idx * params.h_q * d_rope +
            (int64_t)head_idx * d_rope;
        const int lane_q = tid % 32;
        float qrope0 = 0.0f, qrope1 = 0.0f;
        if (lane_q * 2 < d_rope) {
            qrope0 = __bfloat162float(__ldg(q_rope_ptr + lane_q * 2));
            qrope1 = __bfloat162float(__ldg(q_rope_ptr + lane_q * 2 + 1));
        }

        float acc0 = 0.0f, acc1 = 0.0f;
        float M_global = -1e30f, L_global = 0.0f;

        __syncthreads();

        int seqlen = __ldg(params.seqlens_k + batch_idx);
        int total_blocks = (seqlen + params.page_block_size - 1) / params.page_block_size;

        process_kv_blocks(params, smem, batch_idx, head_idx, s_q_idx,
                          0, total_blocks,
                          qr0, qr1, qrope0, qrope1,
                          acc0, acc1, M_global, L_global);

        // Epilogue: normalize + write directly
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
            float lse_val = (L_global > 0.0f) ? (logf(L_global) + M_global) : -1e30f;
            params.lse[batch_idx * params.stride_lse_b +
                       s_q_idx * params.stride_lse_s_q +
                       head_idx] = lse_val;
        }

    } else {
        // =====================================================================
        // Split-KV mode: grid.z = num_sm_parts, tile scheduler assigns work
        // =====================================================================
        const TqDecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
        if (sched_meta.begin_req_idx >= params.b) return;

        for (int batch_idx = sched_meta.begin_req_idx;
             batch_idx <= sched_meta.end_req_idx; ++batch_idx) {

            int seqlen = __ldg(params.seqlens_k + batch_idx);
            int total_blocks = (seqlen + params.page_block_size - 1) / params.page_block_size;

            int start_block = (batch_idx == sched_meta.begin_req_idx) ?
                sched_meta.begin_block_idx : 0;
            int end_block = (batch_idx == sched_meta.end_req_idx) ?
                sched_meta.end_block_idx : total_blocks;
            bool is_no_split = (start_block == 0 && end_block == total_blocks);

            // Load Q for this batch
            const float* q_rot_ptr = params.q_rot +
                (int64_t)batch_idx * params.s_q * params.h_q * d_c +
                (int64_t)s_q_idx * params.h_q * d_c +
                (int64_t)head_idx * d_c;
            float qr0 = 0.0f, qr1 = 0.0f;
            if (tid * 2 < d_c) {
                qr0 = __ldg(q_rot_ptr + tid * 2);
                qr1 = __ldg(q_rot_ptr + tid * 2 + 1);
            }
            // q_rope loaded per LANE (see simple-mode note): every warp
            // computes full-d_rope dot products, so all warps need q_rope.
            const __nv_bfloat16* q_rope_ptr = params.q_rope +
                (int64_t)batch_idx * params.s_q * params.h_q * d_rope +
                (int64_t)s_q_idx * params.h_q * d_rope +
                (int64_t)head_idx * d_rope;
            const int lane_q = tid % 32;
            float qrope0 = 0.0f, qrope1 = 0.0f;
            if (lane_q * 2 < d_rope) {
                qrope0 = __bfloat162float(__ldg(q_rope_ptr + lane_q * 2));
                qrope1 = __bfloat162float(__ldg(q_rope_ptr + lane_q * 2 + 1));
            }

            float acc0 = 0.0f, acc1 = 0.0f;
            float M_global = -1e30f, L_global = 0.0f;

            __syncthreads();

            process_kv_blocks(params, smem, batch_idx, head_idx, s_q_idx,
                              start_block, end_block,
                              qr0, qr1, qrope0, qrope1,
                              acc0, acc1, M_global, L_global);

            // Epilogue
            float inv_L = (L_global > 0.0f) ? (1.0f / L_global) : 0.0f;
            acc0 *= inv_L;
            acc1 *= inv_L;

            if (is_no_split) {
                // Direct write (no other partition covers this batch)
                float* o_ptr = params.out +
                    (int64_t)batch_idx * params.stride_o_b +
                    (int64_t)s_q_idx * params.stride_o_s_q +
                    (int64_t)head_idx * params.stride_o_h_q;
                if (tid * 2 < d_c) {
                    o_ptr[tid * 2]     = acc0;
                    o_ptr[tid * 2 + 1] = acc1;
                }
                if (tid == 0) {
                    float lse_val = (L_global > 0.0f) ? (logf(L_global) + M_global) : -1e30f;
                    params.lse[batch_idx * params.stride_lse_b +
                               s_q_idx * params.stride_lse_s_q +
                               head_idx] = lse_val;
                }
            } else {
                // Write to split-KV accumulators for mla_combine
                int nsi = (batch_idx == sched_meta.begin_req_idx) ?
                    sched_meta.begin_split_idx : 0;
                int si = __ldg(params.num_splits_ptr + batch_idx) + nsi;

                int q_seq = s_q_idx * params.h_q + head_idx;
                float* oa_ptr = params.o_accum +
                    (int64_t)si * params.stride_o_accum_split +
                    (int64_t)q_seq * d_c;
                if (tid * 2 < d_c) {
                    oa_ptr[tid * 2]     = acc0;
                    oa_ptr[tid * 2 + 1] = acc1;
                }
                if (tid == 0) {
                    // LSE in log2 domain for mla_combine compatibility
                    float lse_ln = (L_global > 0.0f) ? (logf(L_global) + M_global) : -1e30f;
                    float lse_log2 = lse_ln * LOG2E_CONST;
                    params.lse_accum[si * params.stride_lse_accum_split + q_seq] = lse_log2;
                }
            }
        }  // end batch loop
    }
}

void run_tq_dense_decode(const TqDenseDecodeParams& params) {
    size_t smem_size = sizeof(TqDecodeSmem);
    if (params.num_sm_parts <= 1) {
        // Simple mode: grid.z = batch
        tq_dense_decode_kernel<<<dim3(params.h_q, params.s_q, params.b),
                                 NUM_THREADS, smem_size, params.stream>>>(params);
    } else {
        // Split-KV mode: grid.z = num_sm_parts
        tq_dense_decode_kernel<<<dim3(params.h_q, params.s_q, params.num_sm_parts),
                                 NUM_THREADS, smem_size, params.stream>>>(params);
    }
}

}  // namespace sm120::decode::tq_dense
