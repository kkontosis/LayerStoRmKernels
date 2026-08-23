/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * SM120 Sparse Prefill/Decode Small TopK (head64) — Pipelined Producer/Consumer
 *
 * Dual-mode:
 *   Prefill: BF16 absorbed path (standard MMA, BF16 KV)
 *   Decode:  SnapMLA FP8-native path (FP8 MMA, scale-fused softmax, FP8 P quant)
 *
 * Decode SnapMLA pipeline (mirrors csrc/sm120/decode/sparse_fp8/splitkv_mla.cu):
 *   1. QK GEMM: FP8 Q @ FP8 K^T → f32 scores (SM89_16x8x32 MMA, K=32)
 *   2. Post-QK dequant: scores *= K_scale[col]
 *   3. Online softmax with V-scale fusion: P'[i,j] = exp(S[i,j]) * V_scale[j]
 *   4. Block-wise dynamic FP8 quantization of P' → P_fp8
 *   5. PV GEMM: FP8 P @ FP8 V → f32 output, post-multiply by P_scale
 *
 * IMPORTANT: Decode path requires SnapMLA per-token KV quantization format.
 **************************************************************************************************/

#pragma once

#include "phase1.h"

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cuda_fp8.h>

#include "traits.h"
#include "config.h"
#include "../../../../components/helpers.h"
#include "../../../../components/dequant.h"
#include "../../../../decode/sparse_fp8/params.h"

using namespace cute;
using namespace sm120::sparse;

namespace sm120::prefill::sparse::small_topk::head64 {

// SnapMlaDecodeParams / SmallTopkArgT come from phase1.h (params
// unification: decode mode takes the SnapMLA decode param struct).

// __ldg helper for FP8
__device__ __forceinline__ fp8 ldg_fp8(const fp8* ptr) {
    char raw = __ldg(reinterpret_cast<const char*>(ptr));
    fp8 result;
    memcpy(&result, &raw, 1);
    return result;
}

//==============================================================================
// Prefill mode K gather (bf16, absorbed) — unchanged
//==============================================================================
template<int D_QK, typename Traits>
__device__ void gather_k_tile_prefill(
    const cutlass::bfloat16_t* kv_ptr, const int* gIndices, const bool* sValid,
    cutlass::bfloat16_t* sK_ptr, int k_block, int k_offset, int kv_stride, int ptid
) {
    using InputT = typename Traits::InputT;
    auto sK = make_tensor(make_smem_ptr(sK_ptr), typename Traits::SmemLayoutK{});
    constexpr int elems = B_TOPK * K_TILE_DIM, per = (elems + 128 - 1) / 128;
    #pragma unroll
    for (int i = 0; i < per; ++i) {
        int idx = ptid + i * 128;
        if (idx < elems) {
            int tr = idx / K_TILE_DIM, kc = idx % K_TILE_DIM, gk = k_offset + kc;
            InputT val = InputT(0);
            if (sValid[tr] && gk < D_QK)
                val = InputT(__ldg(reinterpret_cast<const __nv_bfloat16*>(kv_ptr) + (int64_t)__ldg(gIndices + k_block * B_TOPK + tr) * kv_stride + gk));
            sK(tr, kc) = val;
        }
    }
}

//==============================================================================
// Decode mode FP8 K gather — SnapMLA raw byte load + per-token scale
// All K dims treated uniformly as FP8 (NOPE + pre-scaled ROPE concatenated).
//==============================================================================
template<int D_QK>
__device__ void gather_k_tile_decode_fp8(
    const fp8* __restrict__ kv_ptr,
    const int* __restrict__ cur_indices,
    const bool* __restrict__ sValid,
    cutlass::float_e4m3_t* __restrict__ sK_ptr,
    float* __restrict__ sK_scales,
    int k_offset,
    int page_block_size,
    int kv_block_stride,
    int kv_row_stride,
    int producer_thread_idx,
    bool load_scales
) {
    using TF = TraitsFp8<D_QK>;
    auto sK = make_tensor(make_smem_ptr(sK_ptr), typename TF::SmemLayoutK{});

    constexpr int D_NOPE_LOCAL = D_QK - D_ROPE;
    constexpr int k_tile_elems = B_TOPK * K_TILE_DIM;
    constexpr int per_thread = (k_tile_elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = producer_thread_idx + i * 128;
        if (idx < k_tile_elems) {
            int token_row = idx / K_TILE_DIM;
            int k_col = idx % K_TILE_DIM;
            int global_k = k_offset + k_col;

            cutlass::float_e4m3_t val = cutlass::float_e4m3_t(0);

            if (sValid[token_row] && global_k < D_NOPE_LOCAL) {
                int token_index = __ldg(cur_indices + token_row);
                int block_index = (int)((unsigned)token_index / (unsigned)page_block_size);
                int rel_idx = (unsigned)token_index % (unsigned)page_block_size;

                const fp8* gK_base = kv_ptr + (int64_t)block_index * kv_block_stride +
                                     (int64_t)rel_idx * kv_row_stride;
                val = cutlass::float_e4m3_t(ldg_fp8(gK_base + global_k));

                // Load per-token float32 scale on first K-tile only
                if (load_scales && k_col == 0) {
                    float scale_val = __ldg((float*)(gK_base + D_NOPE_LOCAL));
                    sK_scales[token_row] = scale_val;
                }
            }

            sK(token_row, k_col) = val;
        }
    }
}

//==============================================================================
// Prefill Mode Kernel (BF16, unchanged)
//==============================================================================
template<int D_QK>
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_attn_fwd_small_topk_prefill_sm120_kernel(const SparseAttnFwdParams params) {
    using T = Traits<D_QK>;
    using InputT = typename T::InputT;
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

    const int s_q_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const bool is_producer = (warp_idx >= 4);
    const bool is_consumer = (warp_idx < 4);
    const int ptid = thread_idx - 128;  // producer thread id

    const int topk_length = params.topk_length ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_k_blocks = max(ceil_div(topk_length, (int)B_TOPK), 1);

    extern __shared__ char smem_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);
    float* sScores = smem.smem_pv_tile.data();
    float* sM = smem.smem_M.data(); float* sL = smem.smem_L.data(); float* sScale = smem.smem_scale.data();
    bool* sValid = smem.is_kv_valid;
    InputT* sK_bufs[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

    TiledMma64x64 tiled_mma;
    constexpr int O_ELEMS = (B_H * D_V + NUM_THREADS - 1) / NUM_THREADS;
    float rO[O_ELEMS];

    const InputT* q_ptr = (const InputT*)params.q + s_q_idx * params.stride_q_s_q;
    const InputT* kv_ptr = (const InputT*)params.kv;
    int kv_stride = params.stride_kv_s_kv;
    int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;

    if (thread_idx < B_H) { sM[thread_idx] = MAX_INIT_VAL; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
    #pragma unroll
    for (int i = 0; i < O_ELEMS; ++i) rO[i] = 0.0f;
    __syncthreads();

    constexpr int q_tile_elems = B_H * K_TILE_DIM;
    constexpr int q_per_thread = (q_tile_elems + NUM_THREADS - 1) / NUM_THREADS;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        if (is_producer) {
            for (int i = ptid; i < B_TOPK; i += 128) {
                int gi = kb * B_TOPK + i;
                sValid[i] = (gi < topk_length) && (__ldg(gIndices + gi) >= 0) && (__ldg(gIndices + gi) < params.s_kv);
            }
        }
        __syncthreads();

        auto rS = partition_fragment_C(tiled_mma, Shape<_64, _64>{}); clear(rS);

        // Bootstrap
        if (is_producer) gather_k_tile_prefill<D_QK, T>(kv_ptr, gIndices, sValid, sK_bufs[0], kb, 0, kv_stride, ptid);
        { auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
          #pragma unroll
          for (int i = 0; i < q_per_thread; ++i) { int idx = thread_idx + i * NUM_THREADS; if (idx < q_tile_elems) { int r = idx / K_TILE_DIM, c = idx % K_TILE_DIM; sQ(r, c) = (r < params.h_q && c < D_QK) ? InputT(__ldg(reinterpret_cast<const __nv_bfloat16*>(q_ptr) + r * params.stride_q_h_q + c)) : InputT(0); } } }
        __syncthreads();

        for (int kt = 0; kt < T::NUM_K_TILES_TOTAL; ++kt) {
            if (kt > 0) {
                auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{});
                #pragma unroll
                for (int i = 0; i < q_per_thread; ++i) { int idx = thread_idx + i * NUM_THREADS; if (idx < q_tile_elems) { int r = idx / K_TILE_DIM, c = idx % K_TILE_DIM; int gk = kt * K_TILE_DIM + c; sQ(r, c) = (r < params.h_q && gk < D_QK) ? InputT(__ldg(reinterpret_cast<const __nv_bfloat16*>(q_ptr) + r * params.stride_q_h_q + gk)) : InputT(0); } }
                __syncthreads();
            }
            if (is_producer && kt + 1 < T::NUM_K_TILES_TOTAL)
                gather_k_tile_prefill<D_QK, T>(kv_ptr, gIndices, sValid, sK_bufs[(kt+1)%2], kb, (kt+1)*K_TILE_DIM, kv_stride, ptid);
            if (is_consumer) {
                cute_gemm_ss<false>(tiled_mma, make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_tile.data()), typename T::SmemLayoutQ{}),
                    make_tensor(make_smem_ptr(sK_bufs[kt%2]), typename T::SmemLayoutK{}), rS, thread_idx);
            }
            __syncthreads();
        }

        // Softmax (register-resident)
        if (is_consumer) {
            CUTE_UNROLL
            for (int i = 0; i < size(rS); ++i) rS(i) *= params.sm_scale;
        }
        if (thread_idx < B_H) { sScale[thread_idx] = sM[thread_idx]; sM[thread_idx] = NEGATIVE_INFINITY; }
        __syncthreads();
        if (is_consumer) fragment_masked_row_max(rS, sValid, sM, thread_idx, B_H);
        __syncthreads();
        if (thread_idx < B_H) { float om = sScale[thread_idx], nm = sM[thread_idx]; float rs = (om == NEGATIVE_INFINITY) ? 1.0f : exp2f((om - nm) * LOG2E); sL[thread_idx] *= rs; sScale[thread_idx] = rs; }
        __syncthreads();
        if (is_consumer) fragment_masked_exp_sum(rS, sValid, sM, sL, thread_idx, B_H);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < O_ELEMS; ++i) { int r = (thread_idx + i * NUM_THREADS) / D_V; if (r < B_H) rO[i] *= sScale[r]; }
        if (is_consumer) fragment_to_smem_bf16(rS, smem.smem_P.data(), typename T::SmemLayoutP{}, thread_idx, B_H, B_TOPK);
        __syncthreads();

        // V gather (transposed, all threads).
        // Store through the swizzled SmemLayoutVTile matching the PV GEMM
        // read path — the old raw [tile][col][row] indexing bypassed the
        // swizzle and permuted V within the tile (same write/read-mismatch
        // class fixed in fwd/decode kernels, submodule 06de0b7).
        constexpr int v_elems = B_TOPK * D_V, v_per = (v_elems + NUM_THREADS - 1) / NUM_THREADS;
        #pragma unroll
        for (int i = 0; i < v_per; ++i) { int idx = thread_idx + i * NUM_THREADS; if (idx < v_elems) { int tr = idx / D_V, vc = idx % D_V; InputT val = InputT(0); if (sValid[tr]) val = InputT(__ldg(reinterpret_cast<const __nv_bfloat16*>(kv_ptr) + (int64_t)__ldg(gIndices + kb * B_TOPK + tr) * kv_stride + vc)); auto sVt = make_tensor(make_smem_ptr(smem.smem_V.data() + (vc/64) * T::V_tile_elems), typename T::SmemLayoutVTile{}); sVt(vc % 64, tr) = val; } }
        __syncthreads();

        // PV GEMM
        #pragma unroll 2
        for (int vt = 0; vt < T::NUM_V_TILES; ++vt) {
            if (is_consumer) { auto rPV = partition_fragment_C(tiled_mma, Shape<_64, _64>{}); cute_gemm_ss<true>(tiled_mma, make_tensor(make_smem_ptr(smem.smem_P.data()), typename T::SmemLayoutP{}), make_tensor(make_smem_ptr(smem.smem_V.data() + vt*64*B_TOPK), typename T::SmemLayoutVTile{}), rPV, thread_idx); fragment_to_smem_f32(rPV, sScores, thread_idx, 64, 64); }
            __syncthreads();
            int tc = vt * 64;
            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) { int gi = thread_idx + i * NUM_THREADS, r = gi / D_V, gc = gi % D_V; if (gc >= tc && gc < tc + 64 && r < B_H) rO[i] += sScores[r * 64 + gc - tc]; }
            __syncthreads();
        }
    }

    // Epilogue
    // sM/sL are NATURAL log units in this kernel (rS *= sm_scale, exp via
    // exp2f(x*LOG2E) == e^x), so LSE = m + log(l) directly and the sink logit
    // enters the denominator as e^(sink - m) == exp2f((sink - m)*LOG2E). The
    // old fmaf(cM, ln2, ...) / cM*ln2 / exp2f(sink*LOG2E - m) forms assumed
    // FlashMLA's log2-unit maxima and corrupted any cross-rank LSE consumer
    // (see prefill/dense fwd head64 — same fix, submodule 23a945c).
    int nvh = min(params.h_q, (int)B_H);
    InputT* o_ptr = (InputT*)params.out + s_q_idx * params.h_q * params.d_v;
    #pragma unroll
    for (int i = 0; i < O_ELEMS; ++i) { int idx = thread_idx + i * NUM_THREADS, r = idx / D_V, c = idx % D_V; if (r < nvh) { float inv = sL[r] == 0.0f ? 0.0f : (1.0f / sL[r]); if (params.attn_sink) { float sl = __ldg(params.attn_sink + r); inv = sL[r] == 0.0f ? 0.0f : __fdividef(1.0f, sL[r] + exp2f((sl - sM[r]) * LOG2E)); } o_ptr[r * params.d_v + c] = InputT(rO[i] * inv); } }
    if (thread_idx < nvh) { float cL = sL[thread_idx], cM = sM[thread_idx]; float lse = (cL == 0.0f) ? INFINITY : (cM + logf(cL)); int gi = s_q_idx * params.h_q + thread_idx; if (params.max_logits) params.max_logits[gi] = cM; params.lse[gi] = (lse == -INFINITY) ? INFINITY : lse; }
}

//==============================================================================
// Decode-with-SplitKV Mode Kernel — SnapMLA FP8-Native
//
// FP8 throughout: Q, K, V as FP8 e4m3. FP8 MMA via SM89_16x8x32.
// Scale-fused softmax pipeline with FP8 P quantization.
//==============================================================================
template<int D_QK>
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_attn_fwd_small_topk_decode_sm120_kernel(const SnapMlaDecodeParams params) {
    using T = TraitsFp8<D_QK>;
    using InputT = typename T::InputT;  // float_e4m3_t
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

    const int head_block_idx = blockIdx.x, s_q_idx = blockIdx.y, partition_idx = blockIdx.z;
    const int thread_idx = threadIdx.x, warp_idx = thread_idx / 32;
    const bool is_producer = (warp_idx >= 4), is_consumer = (warp_idx < 4);
    const int ptid = thread_idx - 128;

    extern __shared__ char smem_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);
    float* sPVtile = smem.smem_pv_tile.data();
    float* sM = smem.smem_M.data(); float* sL = smem.smem_L.data(); float* sScale = smem.smem_scale.data();
    float* sKscales = smem.smem_K_scales.data();
    float* sQscales = smem.smem_Q_scales.data();
    float* sPscales = smem.smem_P_scales.data();
    bool* sValid = smem.is_kv_valid;
    InputT* sK_bufs[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

    TiledMmaFp8_64x64 tiled_mma_fp8;   // For NOPE FP8 tiles + PV GEMM
    TiledMma64x64 tiled_mma_bf16;       // For ROPE BF16 tile

    constexpr int O_ELEMS = (B_H * D_V + NUM_THREADS - 1) / NUM_THREADS;
    float rO[O_ELEMS];

    sm120::decode::sparse_fp8::DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    constexpr int q_tile_elems = B_H * K_TILE_DIM;
    constexpr int q_per_thread = (q_tile_elems + NUM_THREADS - 1) / NUM_THREADS;

    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        int topk_val = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
        int total_kv_blocks = max(ceil_div(topk_val, (int)B_TOPK), 1);
        int sbi = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int ebi = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_kv_blocks;
        bool is_no_split = sbi == 0 && ebi == total_kv_blocks;
        int start_head_idx = head_block_idx * B_H;

        if (thread_idx < B_H) { sM[thread_idx] = MAX_INIT_VAL_SM; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
        #pragma unroll
        for (int i = 0; i < O_ELEMS; ++i) rO[i] = 0.0f;
        __syncthreads();

        // Q: separate FP8 NOPE and BF16 ROPE arrays
        const InputT* q_nope_ptr = (const InputT*)params.q +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_NOPE +
            s_q_idx * params.h_q * T::HEAD_DIM_NOPE + start_head_idx * T::HEAD_DIM_NOPE;
        const cutlass::bfloat16_t* q_rope_ptr = (const cutlass::bfloat16_t*)params.q_rope +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_ROPE +
            s_q_idx * params.h_q * T::HEAD_DIM_ROPE + start_head_idx * T::HEAD_DIM_ROPE;
        int* gIndices = params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
        fp8* kv_ptr = (fp8*)params.kv;

        // Load per-head Q scales to smem
        if (thread_idx < B_H && start_head_idx + thread_idx < params.h_q) {
            int q_scale_idx = batch_idx * params.s_q * params.h_q + s_q_idx * params.h_q + start_head_idx + thread_idx;
            sQscales[thread_idx] = __ldg(params.q_scales + q_scale_idx);
        }
        __syncthreads();

        for (int bi = sbi; bi < ebi; ++bi) {
            int* cur_indices = gIndices + bi * B_TOPK;
            if (is_producer) { for (int i = ptid; i < B_TOPK; i += 128) { int gi = bi * B_TOPK + i; sValid[i] = (gi < topk_val) && (__ldg(cur_indices + i) >= 0); } }
            __syncthreads();

            //==============================================================
            // DUAL-MMA QK GEMM: FP8 NOPE tiles + BF16 ROPE tile
            //==============================================================
            auto rS = partition_fragment_C(tiled_mma_fp8, Shape<_64, _64>{}); clear(rS);

            // Bootstrap: producers load K_NOPE[tile 0] + scales, all load Q_NOPE[tile 0]
            if (is_producer) gather_k_tile_decode_fp8<D_QK>(kv_ptr, cur_indices, sValid, sK_bufs[0], sKscales, 0, params.page_block_size, params.stride_kv_block, params.stride_kv_row, ptid, true);
            { auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
              #pragma unroll
              for (int i = 0; i < q_per_thread; ++i) { int idx = thread_idx + i * NUM_THREADS; if (idx < q_tile_elems) { int r = idx / K_TILE_DIM, c = idx % K_TILE_DIM; sQ(r, c) = (r + start_head_idx < params.h_q && c < T::HEAD_DIM_NOPE) ? InputT(ldg_fp8(reinterpret_cast<const fp8*>(q_nope_ptr) + r * T::HEAD_DIM_NOPE + c)) : InputT(0); } } }
            __syncthreads();

            // Pipelined FP8 NOPE loop
            for (int kt = 0; kt < T::NUM_NOPE_TILES; ++kt) {
                if (kt > 0) {
                    auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                    #pragma unroll
                    for (int i = 0; i < q_per_thread; ++i) { int idx = thread_idx + i * NUM_THREADS; if (idx < q_tile_elems) { int r = idx / K_TILE_DIM, c = idx % K_TILE_DIM; int gk = kt * K_TILE_DIM + c; sQ(r, c) = (r + start_head_idx < params.h_q && gk < T::HEAD_DIM_NOPE) ? InputT(ldg_fp8(reinterpret_cast<const fp8*>(q_nope_ptr) + r * T::HEAD_DIM_NOPE + gk)) : InputT(0); } }
                    __syncthreads();
                }
                if (is_producer && kt + 1 < T::NUM_NOPE_TILES)
                    gather_k_tile_decode_fp8<D_QK>(kv_ptr, cur_indices, sValid, sK_bufs[(kt+1)%2], sKscales, (kt+1)*K_TILE_DIM, params.page_block_size, params.stride_kv_block, params.stride_kv_row, ptid, false);
                if (is_consumer) {
                    cute_gemm_ss<false>(tiled_mma_fp8, make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{}), make_tensor(make_smem_ptr(sK_bufs[kt%2]), typename T::SmemLayoutK{}), rS, thread_idx);
                }
                __syncthreads();
            }

            // BF16 ROPE tile
            {
                constexpr int qr_elems = B_H * D_ROPE;
                constexpr int qr_per = (qr_elems + NUM_THREADS - 1) / NUM_THREADS;
                auto sQR = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_rope.data()), typename T::SmemLayoutQRope{});
                #pragma unroll
                for (int i = 0; i < qr_per; ++i) { int idx = thread_idx + i * NUM_THREADS; if (idx < qr_elems) { int r = idx / D_ROPE, c = idx % D_ROPE; sQR(r, c) = (r + start_head_idx < params.h_q) ? cutlass::bfloat16_t(__ldg(reinterpret_cast<const __nv_bfloat16*>(q_rope_ptr) + r * T::HEAD_DIM_ROPE + c)) : cutlass::bfloat16_t(0); } }
                // Producers gather K_ROPE (BF16 from cache)
                if (is_producer) {
                    constexpr int kr_elems = B_TOPK * D_ROPE;
                    constexpr int kr_per = (kr_elems + 128 - 1) / 128;
                    auto sKR = make_tensor(make_smem_ptr(smem.qk_phase.smem_K_rope.data()), typename T::SmemLayoutKRope{});
                    #pragma unroll
                    for (int i = 0; i < kr_per; ++i) {
                        int idx = ptid + i * 128;
                        if (idx < kr_elems) {
                            int tr = idx / D_ROPE, rc = idx % D_ROPE;
                            cutlass::bfloat16_t val = cutlass::bfloat16_t(0);
                            if (sValid[tr]) {
                                int ti = __ldg(cur_indices + tr);
                                int bii = (int)((unsigned)ti / (unsigned)params.page_block_size);
                                int ri = (unsigned)ti % (unsigned)params.page_block_size;
                                const fp8* base = kv_ptr + (int64_t)bii * params.stride_kv_block + (int64_t)ri * params.stride_kv_row;
                                const __nv_bfloat16* rope_base = reinterpret_cast<const __nv_bfloat16*>(base + T::HEAD_DIM_NOPE + 4);
                                val = cutlass::bfloat16_t(__ldg(rope_base + rc));
                            }
                            sKR(tr, rc) = val;
                        }
                    }
                }
                __syncthreads();
                if (is_consumer) {
                    cute_gemm_ss<false>(tiled_mma_bf16, make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_rope.data()), typename T::SmemLayoutQRope{}), make_tensor(make_smem_ptr(smem.qk_phase.smem_K_rope.data()), typename T::SmemLayoutKRope{}), rS, thread_idx);
                }
                __syncthreads();
            }

            //==============================================================
            // Post-QK dequant: scores *= Q_scale[row] * K_scale[col]
            //==============================================================
            if (is_consumer) fragment_qk_dequant(rS, sQscales, sKscales, thread_idx);

            //==============================================================
            // Standard softmax (no V-scale fusion — matches reference)
            //==============================================================
            if (is_consumer) {
                CUTE_UNROLL
                for (int i = 0; i < size(rS); ++i) rS(i) *= params.sm_scale;
            }
            if (thread_idx < B_H) { sScale[thread_idx] = sM[thread_idx]; sM[thread_idx] = MAX_INIT_VAL_MASK; }
            __syncthreads();

            if (is_consumer) fragment_masked_row_max(rS, sValid, sM, thread_idx, B_H);
            __syncthreads();

            if (thread_idx < B_H) { float om = sScale[thread_idx], nm = sM[thread_idx]; float rs = (om <= MAX_INIT_VAL_SM) ? 1.0f : exp2f((om - nm) * LOG2E); sL[thread_idx] *= rs; sScale[thread_idx] = rs; }
            __syncthreads();

            if (is_consumer) fragment_masked_exp_sum(rS, sValid, sM, sL, thread_idx, B_H);
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) { int r = (thread_idx + i * NUM_THREADS) / D_V; if (r < B_H) rO[i] *= sScale[r]; }

            //==============================================================
            // Per-row FP8 P quantization
            //==============================================================
            if (thread_idx < B_H) sPscales[thread_idx] = 0.0f;
            __syncthreads();

            if (is_consumer) fragment_fp8_compute_row_scales(rS, sPscales, thread_idx);
            __syncthreads();

            if (thread_idx < B_H) sPscales[thread_idx] = fmaxf(sPscales[thread_idx], P_SCALE_EPS) / FP8_E4M3_MAX;
            __syncthreads();

            if (is_consumer) fragment_fp8_row_quantize_store(rS, smem.smem_P.data(), typename T::SmemLayoutP{}, sPscales, thread_idx);
            __syncthreads();

            //==============================================================
            // V gather — raw FP8 bytes, transposed, all threads
            //==============================================================
            constexpr int v_elems = B_TOPK * D_V, v_per = (v_elems + NUM_THREADS - 1) / NUM_THREADS;
            #pragma unroll
            for (int i = 0; i < v_per; ++i) {
                int idx = thread_idx + i * NUM_THREADS;
                if (idx < v_elems) {
                    int tr = idx / D_V, vc = idx % D_V;
                    InputT val = InputT(0);
                    if (sValid[tr]) {
                        int ti = __ldg(cur_indices + tr);
                        int bii = (int)((unsigned)ti / (unsigned)params.page_block_size);
                        int ri = (unsigned)ti % (unsigned)params.page_block_size;
                        const fp8* base = kv_ptr + (int64_t)bii * params.stride_kv_block + (int64_t)ri * params.stride_kv_row;
                        val = InputT(base[vc]);
                    }
                    // Swizzled store matching the PV GEMM read (see prefill
                    // epilogue note; FP8 tile layout here).
                    { auto sVt = make_tensor(make_smem_ptr(smem.smem_V.data() + (vc/64) * T::V_tile_elems), typename T::SmemLayoutVTile{}); sVt(vc % 64, tr) = val; }
                }
            }
            __syncthreads();

            //==============================================================
            // FP8 PV GEMM + per-row P_scale dequant
            //==============================================================
            #pragma unroll 2
            for (int vt = 0; vt < T::NUM_V_TILES; ++vt) {
                if (is_consumer) {
                    auto rPV = partition_fragment_C(tiled_mma_fp8, Shape<_64, _64>{});
                    cute_gemm_ss<true>(tiled_mma_fp8, make_tensor(make_smem_ptr(smem.smem_P.data()), typename T::SmemLayoutP{}), make_tensor(make_smem_ptr(smem.smem_V.data() + vt*64*B_TOPK), typename T::SmemLayoutVTile{}), rPV, thread_idx);

                    fragment_pv_row_dequant(rPV, sPscales, thread_idx);

                    fragment_to_smem_f32(rPV, sPVtile, thread_idx, 64, 64);
                }
                __syncthreads();
                int tc = vt * 64;
                #pragma unroll
                for (int i = 0; i < O_ELEMS; ++i) { int gi = thread_idx + i * NUM_THREADS, r = gi / D_V, gc = gi % D_V; if (gc >= tc && gc < tc + 64 && r < B_H) rO[i] += sPVtile[r * 64 + gc - tc]; }
                __syncthreads();
            }
        }

        // Epilogue — output is always BF16
        // LSE units (TD-LSE-UNITS-DECODE): sM/sL are NATURAL log units here
        // (rS *= sm_scale, exp via exp2f(x*LOG2E) == e^x) and sL is the
        // FULL-PRECISION exp row-sum (fragment_masked_exp_sum runs BEFORE the
        // FP8 P quantization), so:
        //   - external lse (no-split) = m + log(l). The old
        //     logf(l*Ps) + m/LOG2E carried a spurious ×P_scale (a leftover of
        //     FlashMLA's quantized-sum accumulator) and a ×ln2 on a natural m.
        //   - lse_accum (split, consumed by mla_combine in LOG2 units)
        //     = log2(l) + m*LOG2E. The old log2f(l*Ps) + m mixed a natural m
        //     into a log2 expression (and the same spurious ×P_scale),
        //     mis-weighting the split combine.
        //   - the sink logit enters the denominator as e^(sink - m).
        int nvh = min(params.h_q - start_head_idx, (int)B_H);
        if (is_no_split) {
            cutlass::bfloat16_t* o_ptr = (cutlass::bfloat16_t*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q + start_head_idx * params.stride_o_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) { int idx = thread_idx + i * NUM_THREADS, r = idx / D_V, c = idx % D_V; if (r < nvh) { float inv = sL[r] == 0.0f ? 0.0f : (1.0f / sL[r]); if (params.attn_sink) { float sl = __ldg(params.attn_sink + start_head_idx + r); inv = sL[r] == 0.0f ? 0.0f : __fdividef(1.0f, sL[r] + exp2f((sl - sM[r]) * LOG2E)); } o_ptr[r * params.stride_o_h_q + c] = cutlass::bfloat16_t(rO[i] * inv); } }
            if (thread_idx < nvh) { float* g = params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + start_head_idx; float cL = sL[thread_idx]; g[thread_idx] = (cL == 0.0f) ? INFINITY : (sM[thread_idx] + logf(cL)); }
        } else {
            int nsi = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            int si = __ldg(params.num_splits_ptr + batch_idx) + nsi;
            float* oa = params.o_accum + si * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q + start_head_idx * params.stride_o_accum_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) { int idx = thread_idx + i * NUM_THREADS, r = idx / D_V, c = idx % D_V; if (r < nvh) oa[r * params.stride_o_accum_h_q + c] = rO[i] * (sL[r] == 0.0f ? 0.0f : (1.0f / sL[r])); }
            if (thread_idx < nvh) { float* g = params.lse_accum + si * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q + start_head_idx; float cL = sL[thread_idx]; g[thread_idx] = (cL == 0.0f) ? -INFINITY : (log2f(cL) + sM[thread_idx] * LOG2E); }
        }
        __syncthreads();
    }
}

//==============================================================================
// Launch functions
//==============================================================================
template<int D_QK>
void run_fwd_for_small_topk_phase1_kernel_prefill(const SparseAttnFwdParams& params) {
    using T = Traits<D_QK>;
    constexpr size_t ss = sizeof(typename T::SharedMemoryPlan);
    auto k = &sparse_attn_fwd_small_topk_prefill_sm120_kernel<D_QK>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, ss);
    k<<<params.s_q, NUM_THREADS, ss, params.stream>>>(params);
}

template<int D_QK>
void run_fwd_for_small_topk_phase1_kernel_decode(const SnapMlaDecodeParams& params) {
    using T = TraitsFp8<D_QK>;
    constexpr size_t ss = sizeof(typename T::SharedMemoryPlan);
    auto k = &sparse_attn_fwd_small_topk_decode_sm120_kernel<D_QK>;
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, ss);
    k<<<dim3(1, params.s_q, params.num_sm_parts), NUM_THREADS, ss, params.stream>>>(params);
}

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_fwd_for_small_topk_phase1_kernel(const SmallTopkArgT<FWD_MODE>& params) {
    if constexpr (FWD_MODE == SparseAttnFwdMode::Prefill) run_fwd_for_small_topk_phase1_kernel_prefill<D_QK>(params);
    else run_fwd_for_small_topk_phase1_kernel_decode<D_QK>(params);
}

}  // namespace sm120::prefill::sparse::small_topk::head64
