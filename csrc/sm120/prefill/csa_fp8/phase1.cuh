/***************************************************************************************************
 * SM120 V4 CSA FP8 Prefill — Non-Absorbed Dense Attention with Causal Masking
 *
 * Adapted from dense/fwd/head64/phase1.cuh for DeepSeek V4:
 *   - Separate K and V pointers (V4's K_NOPE ≠ V_NOPE, unlike absorbed MLA)
 *   - Per-query causal masking via causal_seqlens array
 *   - Head-group indexing (blockIdx.y) for h_q > 64
 *   - Single KV head broadcast to all Q heads (no h_kv dimension in K/V)
 *
 * Grid: (s_q, ceil(h_q / B_H), 1)
 * Block: 256 threads (warps 0-3 = consumer MMA, warps 4-7 = producer K/V load)
 *
 * Source attribution: adapted from FlashMLA dense prefill (MIT License,
 * Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md)
 **************************************************************************************************/

#pragma once

#include "phase1.h"

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "../sparse/fwd/head64/traits.h"
#include "../sparse/fwd/head64/config.h"
#include "../../components/helpers.h"

using namespace cute;
using namespace sm120::sparse;

namespace sm120::prefill::csa_fp8 {

using namespace sm120::prefill::sparse::head64;
template<int D_QK>
using Traits = sm120::prefill::sparse::head64::Traits<D_QK>;

// Load one K-tile from separate K tensor (not absorbed KV)
template<int D_QK, typename T>
__device__ void load_k_tile_separate(
    const cutlass::bfloat16_t* __restrict__ k_ptr,
    const bool* __restrict__ sValid,
    cutlass::bfloat16_t* __restrict__ sK_ptr,
    int k_block, int k_offset, int k_stride,
    int producer_thread_idx
) {
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
                val = k_ptr[(int64_t)token_index * k_stride + global_k];
            }
            sK(token_row, k_col) = val;
        }
    }
}

template<int D_QK>
__global__ void __launch_bounds__(NUM_THREADS, 1)
csa_fp8_prefill_kernel(const CsaFp8PrefillParams params) {
    using T = Traits<D_QK>;
    using InputT = typename T::InputT;
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

    const int s_q_idx = blockIdx.x;
    const int hg = blockIdx.y;
    const int h_base = hg * B_H;
    const int h_count = min((int)B_H, params.h_q - h_base);
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const bool is_producer = (warp_idx >= 4);
    const bool is_consumer = (warp_idx < 4);

    // Per-query causal masking
    int s_kv_this = params.s_kv;
    if (params.causal_seqlens) {
        s_kv_this = min(s_kv_this, __ldg(params.causal_seqlens + s_q_idx));
    }
    const int num_k_blocks = max(ceil_div(s_kv_this, (int)B_TOPK), 1);

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

    // Q pointer offset by query position and head group
    const InputT* q_ptr = (const InputT*)params.q
        + s_q_idx * params.stride_q_s_q
        + h_base * params.stride_q_h_q;

    // K/V: single KV head (no head offset)
    const InputT* k_ptr = (const InputT*)params.k;
    const InputT* v_ptr = (const InputT*)params.v;

    if (thread_idx < B_H) { sM[thread_idx] = NEGATIVE_INFINITY; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
    #pragma unroll
    for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) rO[i] = 0.0f;
    __syncthreads();

    if (s_kv_this <= 0) goto epilogue;

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        // Valid mask: causal bound
        if (is_producer) {
            for (int i = (thread_idx - 128); i < B_TOPK; i += 128) {
                if (i >= 0) {
                    sValid[i] = (k_block * B_TOPK + i) < s_kv_this;
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
            load_k_tile_separate<D_QK, T>(k_ptr, sValid, sK_bufs[0],
                                           k_block, 0, params.stride_k_s_kv,
                                           thread_idx - 128);
        }
        {
            auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
            #pragma unroll
            for (int i = 0; i < q_per_thread; ++i) {
                int idx = thread_idx + i * NUM_THREADS;
                if (idx < q_tile_elems) {
                    int row = idx / K_TILE_DIM, col = idx % K_TILE_DIM;
                    sQ(row, col) = (row < h_count && col < D_QK) ?
                        q_ptr[row * params.stride_q_h_q + col] : InputT(0);
                }
            }
        }
        __syncthreads();

        for (int k_tile = 0; k_tile < T::NUM_K_TILES_TOTAL; ++k_tile) {
            if (k_tile > 0) {
                auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
                #pragma unroll
                for (int i = 0; i < q_per_thread; ++i) {
                    int idx = thread_idx + i * NUM_THREADS;
                    if (idx < q_tile_elems) {
                        int row = idx / K_TILE_DIM, col = idx % K_TILE_DIM;
                        int gk = k_tile * K_TILE_DIM + col;
                        sQ(row, col) = (row < h_count && gk < D_QK) ?
                            q_ptr[row * params.stride_q_h_q + gk] : InputT(0);
                    }
                }
                __syncthreads();
            }

            if (is_producer && k_tile + 1 < T::NUM_K_TILES_TOTAL) {
                load_k_tile_separate<D_QK, T>(k_ptr, sValid,
                    sK_bufs[(k_tile + 1) % 2], k_block,
                    (k_tile + 1) * K_TILE_DIM, params.stride_k_s_kv,
                    thread_idx - 128);
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
        }
        __syncthreads();
        if (is_consumer) fragment_masked_exp_sum(rS, sValid, sM, sL, thread_idx, B_H);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
            int row = (thread_idx + i * NUM_THREADS) / D_V;
            if (row < B_H) rO[i] *= sScale[row];
        }

        // P store
        if (is_consumer) fragment_to_smem_bf16(rS, smem.smem_P.data(), typename T::SmemLayoutP{}, thread_idx, B_H, B_TOPK);
        __syncthreads();

        // V load — from SEPARATE V tensor (not absorbed)
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
                    val = v_ptr[(int64_t)token_index * params.stride_v_s_kv + vc];
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

epilogue:
    // Write output for this head group
    int nvh = min(h_count, (int)B_H);
    InputT* o_ptr = (InputT*)params.out + s_q_idx * params.h_q * params.d_v + h_base * params.d_v;
    #pragma unroll
    for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
        int idx = thread_idx + i * NUM_THREADS, r = idx / D_V, c = idx % D_V;
        if (r < nvh) {
            float inv = sL[r] == 0.0f ? 0.0f : (1.0f / sL[r]);
            o_ptr[r * params.d_v + c] = InputT(rO[i] * inv);
        }
    }
    if (thread_idx < nvh) {
        float cL = sL[thread_idx], cM = sM[thread_idx];
        // sM/rS are NATURAL log units here (rS *= sm_scale, exp via
        // exp2f(x*LOG2E) == e^x): LSE = m + log(l); no ln2 conversion.
        // The old fmaf(cM, ln2, ...) assumed FlashMLA's log2-unit maxima
        // (see prefill/dense fwd head64 — same fix, submodule 23a945c).
        float lse_val = (cL == 0.0f) ? INFINITY : (cM + logf(cL));
        int gi = s_q_idx * params.h_q + h_base + thread_idx;
        params.lse[gi] = (lse_val == -INFINITY) ? INFINITY : lse_val;
    }
}

template<int D_QK>
void run_csa_fp8_prefill_kernel(const CsaFp8PrefillParams& params) {
    using T = Traits<D_QK>;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &csa_fp8_prefill_kernel<D_QK>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    int num_head_groups = (params.h_q + B_H - 1) / B_H;
    dim3 grid(params.s_q, num_head_groups, 1);
    kernel<<<grid, NUM_THREADS, smem_size, params.stream>>>(params);
}

}  // namespace sm120::prefill::csa_fp8
