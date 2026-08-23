/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * SM120 Absorbed Dense Prefill (head64) — Pipelined Producer/Consumer
 *
 * Dense counterpart to the sparse prefill kernel: attends to ALL s_kv tokens
 * sequentially (no index indirection). Same BF16 MMA pipeline, same
 * producer/consumer warp split, same shared memory plan.
 *
 * Key simplifications vs sparse:
 *   - K/V loaded sequentially (no indices/gather)
 *   - Valid mask from s_kv bound (not index validity)
 *   - No topk/topk_length parameters
 **************************************************************************************************/

#pragma once

#include "phase1.h"

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "traits.h"
#include "config.h"
#include "../../../../components/helpers.h"

using namespace cute;
using namespace sm120::sparse;

namespace sm120::prefill::dense::head64 {

// Load one K-tile from bf16 KV — sequential read (no index indirection)
template<int D_QK, typename Traits>
__device__ void load_k_tile_bf16(
    const cutlass::bfloat16_t* __restrict__ kv_ptr,
    const bool* __restrict__ sValid,
    cutlass::bfloat16_t* __restrict__ sK_ptr,
    int k_block, int k_offset, int kv_stride,
    int producer_thread_idx  // 0-127
) {
    using T = Traits;
    using InputT = typename T::InputT;
    auto sK = make_tensor(make_smem_ptr(sK_ptr), typename T::SmemLayoutK{});

    constexpr int k_tile_elems = B_TOPK * K_TILE_DIM;
    constexpr int per_thread = (k_tile_elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = producer_thread_idx + i * 128;
        if (idx < k_tile_elems) {
            int token_row = idx / K_TILE_DIM;
            int k_col = idx % K_TILE_DIM;
            int global_k = k_offset + k_col;
            InputT val = InputT(0);
            if (sValid[token_row] && global_k < D_QK) {
                int token_index = k_block * B_TOPK + token_row;
                val = kv_ptr[(int64_t)token_index * kv_stride + global_k];
            }
            sK(token_row, k_col) = val;
        }
    }
}

template<int D_QK, bool DETERMINISTIC>
__global__ void __launch_bounds__(NUM_THREADS, 1)
dense_attn_fwd_sm120_kernel(const DenseAttnFwdParams params) {
    using T = Traits<D_QK>;
    using InputT = typename T::InputT;
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

    const int s_q_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const bool is_producer = (warp_idx >= 4);
    const bool is_consumer = (warp_idx < 4);
    // 4 consumer warps — matches is_consumer + the hardcoded 128-thread split.
    constexpr int NUM_CONSUMER_WARPS = 4;
    // DETERMINISTIC-only per-warp denominator partials (see splitkv_mla.cu).
    // Size 1 / never referenced in the default path → zero footprint there.
    __shared__ float sL_partial[DETERMINISTIC ? B_H * NUM_CONSUMER_WARPS : 1];

    // Per-row causal KV bound: this CTA (query row s_q_idx) attends staged
    // rows [0, row_s_kv). nullptr → legacy flat s_kv (non-causal). Clamped to
    // s_kv so a bad bound can never read past the staging. Both the block
    // count and the valid mask derive from row_s_kv, so a bounded row runs
    // exactly the same k_block sequence as an s_kv==row_s_kv batch-of-1 call
    // (bit-equal outputs; see params.h CAUSAL-CHUNK CONTRACT).
    const int row_s_kv = params.s_kv_per_row
        ? min(max(params.s_kv_per_row[s_q_idx], 0), params.s_kv)
        : params.s_kv;
    const int num_k_blocks = max(ceil_div(row_s_kv, (int)B_TOPK), 1);

    extern __shared__ char smem_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    float* sScores = smem.smem_pv_tile.data();
    float* sM = smem.smem_M.data();
    float* sL = smem.smem_L.data();
    float* sScale = smem.smem_scale.data();
    bool* sValid = smem.is_kv_valid;
    InputT* sK_bufs[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

    TiledMma64x64 tiled_mma;

    constexpr int O_ELEMS_PER_THREAD = (B_H * D_V + NUM_THREADS - 1) / NUM_THREADS;
    float rO[O_ELEMS_PER_THREAD];

    const InputT* q_ptr = (const InputT*)params.q + s_q_idx * params.stride_q_s_q;
    const InputT* kv_ptr = (const InputT*)params.kv;
    int kv_stride = params.stride_kv_s_kv;

    if (thread_idx < B_H) { sM[thread_idx] = MAX_INIT_VAL; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
    #pragma unroll
    for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) rO[i] = 0.0f;
    __syncthreads();

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        // Valid mask (per-row causal bound; == s_kv when s_kv_per_row is null)
        if (is_producer) {
            for (int i = (thread_idx - 128); i < B_TOPK; i += 128) {
                if (i >= 0) {
                    sValid[i] = (k_block * B_TOPK + i) < row_s_kv;
                }
            }
        }
        __syncthreads();

        // Pipelined QK GEMM
        auto rS = partition_fragment_C(tiled_mma, Shape<_64, _64>{});
        clear(rS);

        constexpr int q_tile_elems = B_H * K_TILE_DIM;
        constexpr int q_per_thread = (q_tile_elems + NUM_THREADS - 1) / NUM_THREADS;

        // Bootstrap: producers load K[tile 0], all load Q[tile 0]
        if (is_producer) {
            load_k_tile_bf16<D_QK, T>(kv_ptr, sValid, sK_bufs[0],
                                       k_block, 0, kv_stride, thread_idx - 128);
        }
        {
            auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
            #pragma unroll
            for (int i = 0; i < q_per_thread; ++i) {
                int idx = thread_idx + i * NUM_THREADS;
                if (idx < q_tile_elems) {
                    int row = idx / K_TILE_DIM, col = idx % K_TILE_DIM;
                    sQ(row, col) = (row < params.h_q && col < D_QK) ?
                        q_ptr[row * params.stride_q_h_q + col] : InputT(0);
                }
            }
        }
        __syncthreads();

        for (int k_tile = 0; k_tile < T::NUM_K_TILES_TOTAL; ++k_tile) {
            // Phase A: All 256 threads cooperatively load Q[tile k_tile] (tile 0 in bootstrap)
            if (k_tile > 0) {
                auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
                #pragma unroll
                for (int i = 0; i < q_per_thread; ++i) {
                    int idx = thread_idx + i * NUM_THREADS;
                    if (idx < q_tile_elems) {
                        int row = idx / K_TILE_DIM, col = idx % K_TILE_DIM;
                        int gk = k_tile * K_TILE_DIM + col;
                        sQ(row, col) = (row < params.h_q && gk < D_QK) ?
                            q_ptr[row * params.stride_q_h_q + gk] : InputT(0);
                    }
                }
                __syncthreads();
            }

            // Phase B (overlapped): producers load K[i+1], consumers MMA on K[i]
            if (is_producer && k_tile + 1 < T::NUM_K_TILES_TOTAL) {
                load_k_tile_bf16<D_QK, T>(kv_ptr, sValid,
                    sK_bufs[(k_tile + 1) % 2], k_block,
                    (k_tile + 1) * K_TILE_DIM, kv_stride, thread_idx - 128);
            }
            if (is_consumer) {
                auto sQ_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
                auto sK_t = make_tensor(make_smem_ptr(sK_bufs[k_tile % 2]), typename T::SmemLayoutK{});
                cute_gemm_ss<false>(tiled_mma, sQ_t, sK_t, rS, thread_idx);
            }
            __syncthreads();
        }

        // Register-resident softmax
        if (is_consumer) {
            CUTE_UNROLL
            for (int i = 0; i < size(rS); ++i) rS(i) *= params.sm_scale;
        }
        if (thread_idx < B_H) { sScale[thread_idx] = sM[thread_idx]; sM[thread_idx] = NEGATIVE_INFINITY; }
        __syncthreads();
        if (is_consumer) fragment_masked_row_max(rS, sValid, sM, thread_idx, B_H);
        __syncthreads();
        if (thread_idx < B_H) {
            float old_max = sScale[thread_idx], new_max = sM[thread_idx];
            float rescale = (old_max == NEGATIVE_INFINITY) ? 1.0f : exp2f((old_max - new_max) * LOG2E);
            sL[thread_idx] *= rescale; sScale[thread_idx] = rescale;
            if constexpr (DETERMINISTIC) {
                #pragma unroll
                for (int w = 0; w < NUM_CONSUMER_WARPS; ++w)
                    sL_partial[thread_idx * NUM_CONSUMER_WARPS + w] = 0.0f;
            }
        }
        __syncthreads();
        if (is_consumer)
            fragment_masked_exp_sum<DETERMINISTIC>(
                rS, sValid, sM, sL, thread_idx, B_H, sL_partial, NUM_CONSUMER_WARPS);
        __syncthreads();
        // DETERMINISTIC: fixed-index cross-warp combine into sL (running denom).
        // No extra sync — next sL reader is gated by the P-store sync below.
        if constexpr (DETERMINISTIC) {
            if (thread_idx < B_H) {
                float acc = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_CONSUMER_WARPS; ++w)
                    acc += sL_partial[thread_idx * NUM_CONSUMER_WARPS + w];
                sL[thread_idx] += acc;
            }
        }

        #pragma unroll
        for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
            int row = (thread_idx + i * NUM_THREADS) / D_V;
            if (row < B_H) rO[i] *= sScale[row];
        }

        // P store: fragment -> swizzled smem
        if (is_consumer) fragment_to_smem_bf16(rS, smem.smem_P.data(), typename T::SmemLayoutP{}, thread_idx, B_H, B_TOPK);
        __syncthreads();

        // V load — sequential, all threads
        constexpr int v_elems = B_TOPK * D_V;
        constexpr int v_per_thread = (v_elems + NUM_THREADS - 1) / NUM_THREADS;
        #pragma unroll
        for (int i = 0; i < v_per_thread; ++i) {
            int idx = thread_idx + i * NUM_THREADS;
            if (idx < v_elems) {
                int tr = idx / D_V, vc = idx % D_V;
                InputT val = InputT(0);
                if (sValid[tr]) {
                    int token_index = k_block * B_TOPK + tr;
                    val = kv_ptr[(int64_t)token_index * kv_stride + vc];
                }
                int vti = vc / 64, vci = vc % 64;
                auto sVt = make_tensor(
                    make_smem_ptr(smem.smem_V.data() + vti * T::V_tile_elems),
                    typename T::SmemLayoutVTile{});
                sVt(vci, tr) = val;
            }
        }
        __syncthreads();

        // PV GEMM
        constexpr int V_TILE_DIM = 64;
        #pragma unroll 2
        for (int vt = 0; vt < T::NUM_V_TILES; ++vt) {
            if (is_consumer) {
                auto sP = make_tensor(make_smem_ptr(smem.smem_P.data()), typename T::SmemLayoutP{});
                auto sVt = make_tensor(make_smem_ptr(smem.smem_V.data() + vt * 64 * B_TOPK), typename T::SmemLayoutVTile{});
                auto rPV = partition_fragment_C(tiled_mma, Shape<_64, _64>{});
                cute_gemm_ss<true>(tiled_mma, sP, sVt, rPV, thread_idx);
                fragment_to_smem_f32(rPV, sScores, thread_idx, 64, 64);
            }
            __syncthreads();
            int tc = vt * V_TILE_DIM;
            #pragma unroll
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int gi = thread_idx + i * NUM_THREADS, r = gi / D_V, gc = gi % D_V;
                if (gc >= tc && gc < tc + V_TILE_DIM && r < B_H) rO[i] += sScores[r * V_TILE_DIM + gc - tc];
            }
            __syncthreads();
        }
    }

    // Epilogue
    int nvh = min(params.h_q, (int)B_H);
    InputT* o_ptr = (InputT*)params.out + s_q_idx * params.h_q * params.d_v;
    #pragma unroll
    for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
        int idx = thread_idx + i * NUM_THREADS, r = idx / D_V, c = idx % D_V;
        if (r < nvh) {
            float inv = sL[r] == 0.0f ? 0.0f : (1.0f / sL[r]);
            if (params.attn_sink) {
                // sM is a NATURAL-unit max: the sink logit enters the
                // denominator as e^(sink - m) == exp2f((sink - m)*LOG2E).
                float sl = __ldg(params.attn_sink + r);
                inv = sL[r] == 0.0f ? 0.0f : __fdividef(1.0f, sL[r] + exp2f((sl - sM[r]) * LOG2E));
            }
            o_ptr[r * params.d_v + c] = InputT(rO[i] * inv);
        }
    }
    if (thread_idx < nvh) {
        float cL = sL[thread_idx], cM = sM[thread_idx];
        // sM/rS are in NATURAL log units in this kernel (rS *= sm_scale, exp
        // via exp2f(x*LOG2E) == e^x), so LSE = m + log(l) directly. The old
        // fmaf(cM, ln2, ...) assumed log2-unit maxima (FlashMLA's
        // sm_scale_div_log2 convention) and under-reported LSE by
        // (1-ln2)*m — corrupting any cross-rank LSE consumer.
        float lse = (cL == 0.0f) ? INFINITY : (cM + logf(cL));
        int gi = s_q_idx * params.h_q + thread_idx;
        if (params.max_logits) params.max_logits[gi] = cM;
        params.lse[gi] = (lse == -INFINITY) ? INFINITY : lse;
    }
}

template<int D_QK, bool DETERMINISTIC>
void run_dense_fwd_phase1_kernel(const DenseAttnFwdParams& params) {
    using T = Traits<D_QK>;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &dense_attn_fwd_sm120_kernel<D_QK, DETERMINISTIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<params.s_q, NUM_THREADS, smem_size, params.stream>>>(params);
}

}  // namespace sm120::prefill::dense::head64
