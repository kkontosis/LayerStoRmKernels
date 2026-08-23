#pragma once
/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * V4 CSA TQ Decode — Sparse Index-Based TQ Scoring + Separate K/V
 *
 * Adapted from tq_sparse/splitkv_mla.cu (V3.2). Key change: V4 has separate K
 * and V TQ entries in 644-byte cache slots. K packed data is used for scoring,
 * V packed data for PV accumulation (reusing the same smem buffer in two phases).
 *
 * Output is FP32 in rotated space. Caller applies:
 *   1. tq_v_rotate_back (Pi) to get original-space output
 *   2. LSE combine with SWA output from CSA FP8 decode (topk=0)
 *
 * Grid: (h_q, s_q, batch)
 * Block: 256 threads (8 warps)
 * KV Block Size: 64 tokens per iteration
 *
 * Reference: tests/test_v4_reference.py::ref_v4_tq_decode_csa()
 **************************************************************************************************/

#include "splitkv_csa_tq.h"
#include "../tq_dense/params.h"   // TqDecodingSchedMeta
#include <cuda_fp16.h>

namespace sm120::decode::csa_tq {

using sm120::decode::tq_dense::TqDecodingSchedMeta;

static constexpr int NUM_THREADS = 256;
static constexpr int KV_BLOCK_SIZE = 64;
static constexpr int NUM_CENTROIDS = 16;
static constexpr int NUM_WARPS = 8;
static constexpr float LOG2E_CONST = 1.4426950408889634f;

// V4 TQ entry byte offsets (644 total)
static constexpr int K_PACKED_OFF = 0;
static constexpr int K_NORM_OFF   = 256;
static constexpr int K_ROPE_OFF   = 258;
static constexpr int V_PACKED_OFF = 386;
static constexpr int V_NORM_OFF   = 642;
static constexpr int ENTRY_BYTES  = 644;

struct CsaTqDecodeSmem {
    uint8_t packed_nope[KV_BLOCK_SIZE * 256];    // 16KB — reused for K then V
    __half k_norms[KV_BLOCK_SIZE];               // 128B
    __nv_bfloat16 k_rope[KV_BLOCK_SIZE * 64];   // 8KB
    __half v_norms[KV_BLOCK_SIZE];               // 128B
    float centroids[NUM_CENTROIDS];              // 64B
    float warp_nope[NUM_WARPS][KV_BLOCK_SIZE];   // 2KB
    float scores[KV_BLOCK_SIZE];                 // 256B
    float P_weights[KV_BLOCK_SIZE];              // 256B
    float reduce_scratch[NUM_WARPS];             // 32B
    int token_valid[KV_BLOCK_SIZE];              // 256B
};

__global__ void __launch_bounds__(256)
csa_tq_decode_kernel(const CsaTqDecodeParams params) {
    const int head_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_idx = tid / 32;
    const int lane = tid % 32;

    if (head_idx >= params.h_q) return;

    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;
    const int packed_bytes = hd / 2;

    // Split-KV: determine block range from tile scheduler
    const TqDecodingSchedMeta* sched = reinterpret_cast<const TqDecodingSchedMeta*>(
        params.tile_scheduler_metadata_ptr);
    TqDecodingSchedMeta meta;
    int batch_idx;
    int total_blocks_all;
    int start_block, end_block;
    bool is_no_split;

    if (params.num_sm_parts <= 1) {
        batch_idx = 0;
        total_blocks_all = (params.topk + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        start_block = 0;
        end_block = total_blocks_all;
        is_no_split = true;
    } else {
        meta = sched[partition_idx];
        if (meta.begin_req_idx >= params.b) return;
        batch_idx = meta.begin_req_idx;
        total_blocks_all = (params.topk + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        start_block = meta.begin_block_idx;
        end_block = meta.end_block_idx;
        is_no_split = (start_block == 0 && end_block == total_blocks_all);
    }

    extern __shared__ char smem_raw[];
    CsaTqDecodeSmem& smem = *reinterpret_cast<CsaTqDecodeSmem*>(smem_raw);

    if (tid < NUM_CENTROIDS)
        smem.centroids[tid] = params.centroids[tid];

    const int* my_indices = params.indices +
        batch_idx * params.stride_indices_b +
        s_q_idx * params.stride_indices_s_q;

    // Load q_rot (FP32, 2 values per thread)
    const float* q_rot_ptr = params.q_rot +
        (int64_t)batch_idx * params.s_q * params.h_q * hd +
        (int64_t)s_q_idx * params.h_q * hd +
        (int64_t)head_idx * hd;
    float qr0 = 0.0f, qr1 = 0.0f;
    if (tid * 2 < hd) {
        qr0 = __ldg(q_rot_ptr + tid * 2);
        qr1 = __ldg(q_rot_ptr + tid * 2 + 1);
    }

    // Load q_rope (BF16) — per LANE, not per tid: the ROPE score loop below
    // reduces a full-rd dot product inside EACH warp (lane l covers dims 2l,
    // 2l+1), so every warp needs the full q_rope vector. Loading with tid
    // left warps 1..7 with zeros, dropping the rope score for tokens 8..63
    // of every KV block.
    const __nv_bfloat16* q_rope_ptr = params.q_rope +
        (int64_t)batch_idx * params.s_q * params.h_q * rd +
        (int64_t)s_q_idx * params.h_q * rd +
        (int64_t)head_idx * rd;
    float qrope0 = 0.0f, qrope1 = 0.0f;
    if (lane * 2 < rd) {
        qrope0 = __bfloat162float(__ldg(q_rope_ptr + lane * 2));
        qrope1 = __bfloat162float(__ldg(q_rope_ptr + lane * 2 + 1));
    }

    float acc0 = 0.0f, acc1 = 0.0f;
    float M_global = -1e30f, L_global = 0.0f;

    __syncthreads();

    for (int blk = start_block; blk < end_block; ++blk) {
        int blk_start = blk * KV_BLOCK_SIZE;

        // === Load validity ===
        if (tid < KV_BLOCK_SIZE) {
            int idx_pos = blk_start + tid;
            int token_idx = (idx_pos < params.topk) ? __ldg(my_indices + idx_pos) : -1;
            smem.token_valid[tid] = token_idx;
        }
        __syncthreads();

        int num_valid = 0;
        for (int i = 0; i < KV_BLOCK_SIZE; ++i)
            if (smem.token_valid[i] >= 0) num_valid++;
        if (num_valid == 0) continue;

        // === Phase 1: Load K data for scoring ===

        // K packed_nope (from offset 0) — vectorized 4B loads
        int total_packed = KV_BLOCK_SIZE * packed_bytes;
        {
            constexpr int VEC_BYTES = 4;
            int total_vec = total_packed / VEC_BYTES;
            int bytes_per_tok = packed_bytes / VEC_BYTES;  // 64
            for (int vi = tid; vi < total_vec; vi += NUM_THREADS) {
                int tok = vi / bytes_per_tok;
                int wi = vi % bytes_per_tok;
                int token_idx = smem.token_valid[tok];
                uint32_t val = 0;
                if (token_idx >= 0) {
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(
                        params.kv_cache + (int64_t)token_idx * ENTRY_BYTES + K_PACKED_OFF);
                    val = __ldg(src + wi);
                }
                reinterpret_cast<uint32_t*>(smem.packed_nope)[vi] = val;
            }
        }

        // K norms
        if (tid < KV_BLOCK_SIZE) {
            int token_idx = smem.token_valid[tid];
            if (token_idx >= 0) {
                const uint8_t* entry = params.kv_cache + (int64_t)token_idx * ENTRY_BYTES;
                smem.k_norms[tid] = *reinterpret_cast<const __half*>(entry + K_NORM_OFF);
            } else {
                smem.k_norms[tid] = __float2half(0.0f);
            }
        }

        // K rope (K_ROPE_OFF=258 is 2-byte aligned, use BF16 loads directly)
        int total_rope = KV_BLOCK_SIZE * rd;
        for (int i = tid; i < total_rope; i += NUM_THREADS) {
            int tok = i / rd;
            int di = i % rd;
            int token_idx = smem.token_valid[tok];
            if (token_idx >= 0) {
                const __nv_bfloat16* rope_ptr = reinterpret_cast<const __nv_bfloat16*>(
                    params.kv_cache + (int64_t)token_idx * ENTRY_BYTES + K_ROPE_OFF);
                smem.k_rope[i] = __ldg(rope_ptr + di);
            } else {
                smem.k_rope[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // === NOPE scores ===
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

        // Sum warp partials + multiply by K norm
        if (tid < KV_BLOCK_SIZE) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w)
                s += smem.warp_nope[w][tid];
            smem.scores[tid] = s * __half2float(smem.k_norms[tid]);
        }
        __syncthreads();

        // === ROPE scores ===
        {
            int tok_start = warp_idx * 8;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                int tok = tok_start + k;
                float pr = 0.0f;
                if (lane * 2 < rd) {
                    pr = qrope0 * __bfloat162float(smem.k_rope[tok * rd + lane * 2]) +
                         qrope1 * __bfloat162float(smem.k_rope[tok * rd + lane * 2 + 1]);
                }
                #pragma unroll
                for (int off = 16; off > 0; off /= 2)
                    pr += __shfl_xor_sync(0xffffffff, pr, off);
                if (lane == 0 && smem.token_valid[tok] >= 0)
                    smem.scores[tok] += pr;
            }
        }
        __syncthreads();

        // === Online softmax ===
        if (tid < KV_BLOCK_SIZE) {
            smem.scores[tid] = (smem.token_valid[tid] >= 0) ?
                smem.scores[tid] * params.sm_scale : -1e30f;
        }
        __syncthreads();

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

        float M_new = fmaxf(M_global, M_block);
        float rescale = (M_global > -1e29f) ? expf(M_global - M_new) : 0.0f;
        acc0 *= rescale;
        acc1 *= rescale;
        L_global *= rescale;

        // P_weights = exp(score - M_new) — raw probability, no V norm yet
        if (tid < KV_BLOCK_SIZE) {
            float p = (smem.token_valid[tid] >= 0) ?
                expf(smem.scores[tid] - M_new) : 0.0f;
            smem.scores[tid] = p;          // for L sum
            smem.P_weights[tid] = p;       // raw P, V norm applied during PV
        }
        __syncthreads();

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

        // === Phase 2: Load V data for PV (reuse packed_nope buffer) ===

        // V packed_nope (from offset 386) — vectorized 2B loads (V_PACKED_OFF is 2-byte aligned)
        {
            constexpr int VEC_BYTES = 2;
            int total_vec = total_packed / VEC_BYTES;
            int elems_per_tok = packed_bytes / VEC_BYTES;
            for (int vi = tid; vi < total_vec; vi += NUM_THREADS) {
                int tok = vi / elems_per_tok;
                int wi = vi % elems_per_tok;
                int token_idx = smem.token_valid[tok];
                if (token_idx >= 0) {
                    const uint16_t* src = reinterpret_cast<const uint16_t*>(
                        params.kv_cache + (int64_t)token_idx * ENTRY_BYTES + V_PACKED_OFF);
                    reinterpret_cast<uint16_t*>(smem.packed_nope)[vi] = __ldg(src + wi);
                }
            }
        }

        // V norms
        if (tid < KV_BLOCK_SIZE) {
            int token_idx = smem.token_valid[tid];
            if (token_idx >= 0) {
                const uint8_t* entry = params.kv_cache + (int64_t)token_idx * ENTRY_BYTES;
                smem.v_norms[tid] = *reinterpret_cast<const __half*>(entry + V_NORM_OFF);
            } else {
                smem.v_norms[tid] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // === PV accumulation in rotated space (using V data) ===
        {
            float c[NUM_CENTROIDS];
            #pragma unroll
            for (int ci = 0; ci < NUM_CENTROIDS; ++ci)
                c[ci] = smem.centroids[ci];

            #pragma unroll 8
            for (int k = 0; k < KV_BLOCK_SIZE; ++k) {
                float w = smem.P_weights[k] * __half2float(smem.v_norms[k]);
                uint8_t byte = smem.packed_nope[k * packed_bytes + tid];
                int i0 = byte & 0x0F, i1 = (byte >> 4) & 0x0F;
                acc0 += w * c[i0];
                acc1 += w * c[i1];
            }
        }
        __syncthreads();
    }

    // === Epilogue ===
    // Normalize in BOTH modes: mla_combine expects NORMALIZED per-split
    // outputs (it forms a convex combination with weights 2^(lse_accum-global)),
    // exactly like tq_dense/splitkv_mla.cu.
    float inv_L = (L_global > 0.0f) ? (1.0f / L_global) : 0.0f;
    acc0 *= inv_L;
    acc1 *= inv_L;

    if (is_no_split) {
        float* o_ptr = params.out +
            (int64_t)batch_idx * params.stride_o_b +
            (int64_t)s_q_idx * params.stride_o_s_q +
            (int64_t)head_idx * params.stride_o_h_q;
        if (tid * 2 < hd) {
            o_ptr[tid * 2]     = acc0;
            o_ptr[tid * 2 + 1] = acc1;
        }

        if (tid == 0) {
            float lse = (L_global > 0.0f) ? (logf(L_global) + M_global) : -1e30f;
            params.lse[batch_idx * params.stride_lse_b +
                       s_q_idx * params.stride_lse_s_q +
                       head_idx] = lse;
        }
    } else {
        // Split-KV: write NORMALIZED partial + log2-unit LSE to accumulators
        int nsi = (batch_idx == meta.begin_req_idx) ? meta.begin_split_idx : 0;
        int si = __ldg(params.num_splits_ptr + batch_idx) + nsi;
        float* oa = params.o_accum + si * params.stride_o_accum_split +
                    s_q_idx * params.stride_o_accum_s_q +
                    head_idx * params.stride_o_accum_h_q;
        if (tid * 2 < hd) {
            oa[tid * 2]     = acc0;
            oa[tid * 2 + 1] = acc1;
        }
        if (tid == 0) {
            // LSE in log2 domain for mla_combine compatibility (weights are
            // exp2f(lse_accum - global); a natural-unit value here silently
            // mis-weights the multi-split combine).
            float lse_ln = (L_global > 0.0f) ? (logf(L_global) + M_global) : -1e30f;
            params.lse_accum[si * params.stride_lse_accum_split +
                             s_q_idx * params.stride_lse_accum_s_q +
                             head_idx] = lse_ln * LOG2E_CONST;
        }
    }
}

void run_csa_tq_decode(const CsaTqDecodeParams& params) {
    size_t smem_size = sizeof(CsaTqDecodeSmem);
    int grid_z = (params.num_sm_parts > 1) ? params.num_sm_parts : params.b;
    csa_tq_decode_kernel<<<dim3(params.h_q, params.s_q, grid_z),
                           NUM_THREADS, smem_size, params.stream>>>(params);
}

}  // namespace sm120::decode::csa_tq
