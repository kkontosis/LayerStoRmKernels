/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * SM120 Sparse FP8 Decode Kernel — SnapMLA Dual-MMA Path
 *
 * Mixed-precision QK GEMM matching SGLang-FluentLLM flashmla-fp8 reference:
 *   NOPE tiles: FP8 Q_NOPE @ FP8 K_NOPE^T (SM89_16x8x32 MMA, K=32)
 *   ROPE tile:  BF16 Q_ROPE @ BF16 K_ROPE^T (SM80_16x8x16 MMA, K=16)
 *   Both accumulate into the same f32 fragment (SM80_16x8_Row CLayout).
 *
 * SnapMLA pipeline:
 *   1. QK GEMM: dual MMA (FP8 NOPE + BF16 ROPE) → f32 scores
 *   2. Post-QK dequant: scores *= Q_scale[row] * K_scale[col]
 *   3. Standard online softmax (no V-scale fusion)
 *   4. Per-row dynamic FP8 quantization of P
 *   5. PV GEMM: FP8 P @ FP8 V → f32 output
 *   6. Post-PV dequant: output *= P_scale[row]
 *
 * KV cache row layout: [d_c FP8 | float32 scale | d_rope BF16]
 * Q layout: separate q_nope (FP8) and q_rope (BF16) arrays
 *
 * Producer/consumer pipeline:
 *   Warps 0-3: MMA consumer (FP8 + BF16 QK, FP8 PV, softmax)
 *   Warps 4-7: gather producer (FP8 NOPE + BF16 ROPE from cache)
 **************************************************************************************************/

#include <cutlass/cutlass.h>
#include <cutlass/arch/memory_sm80.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cuda_fp8.h>

#include "traits.h"
#include "params.h"
#include "../../components/dequant.h"
#include "../../components/helpers.h"

using namespace cute;
using namespace sm120::sparse;

namespace sm120::decode::sparse_fp8 {

// __ldg helper for FP8: __ldg doesn't natively support __nv_fp8_e4m3,
// so load as char via read-only cache and reinterpret.
__device__ __forceinline__ fp8 ldg_fp8(const fp8* ptr) {
    char raw = __ldg(reinterpret_cast<const char*>(ptr));
    fp8 result;
    memcpy(&result, &raw, 1);
    return result;
}

//==============================================================================
// FP8 K NOPE tile gather: raw byte load from paged cache
// Loads per-token scale on first tile. NOPE dims only (d_c range).
//==============================================================================
template<ModelType MODEL_TYPE, typename Traits>
__device__ void gather_k_nope_fp8(
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
    using T = Traits;
    auto sK = make_tensor(make_smem_ptr(sK_ptr), typename T::SmemLayoutK{});

    constexpr int k_tile_elems = T::TOPK_BLOCK_SIZE * T::K_TILE_DIM;
    constexpr int per_thread = (k_tile_elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = producer_thread_idx + i * 128;
        if (idx < k_tile_elems) {
            int token_row = idx / T::K_TILE_DIM;
            int k_col = idx % T::K_TILE_DIM;
            int global_k = k_offset + k_col;

            cutlass::float_e4m3_t val = cutlass::float_e4m3_t(0.0f);

            if (sValid[token_row] && global_k < T::HEAD_DIM_NOPE) {
                int token_index = __ldg(cur_indices + token_row);
                int block_index = (int)((unsigned)token_index / (unsigned)page_block_size);
                int rel_idx = (unsigned)token_index % (unsigned)page_block_size;

                const fp8* gK_base = kv_ptr + (int64_t)block_index * kv_block_stride +
                                     (int64_t)rel_idx * kv_row_stride;
                val = cutlass::float_e4m3_t(ldg_fp8(gK_base + global_k));

                // Load per-token float32 scale on first K-tile only
                if (load_scales && k_col == 0) {
                    float scale_val = __ldg(reinterpret_cast<const float*>(gK_base + T::HEAD_DIM_NOPE));
                    sK_scales[token_row] = scale_val;
                }
            }

            sK(token_row, k_col) = val;
        }
    }
}

//==============================================================================
// BF16 K ROPE tile gather: loads pre-scaled BF16 ROPE from paged cache
// Cache row: [d_c FP8 | 4B float32 scale | d_rope BF16]
// ROPE starts at byte offset d_c + 4, stored as BF16.
//==============================================================================
template<typename Traits>
__device__ void gather_k_rope_bf16(
    const fp8* __restrict__ kv_ptr,
    const int* __restrict__ cur_indices,
    const bool* __restrict__ sValid,
    cutlass::bfloat16_t* __restrict__ sKR_ptr,
    int page_block_size,
    int kv_block_stride,
    int kv_row_stride,
    int producer_thread_idx
) {
    using T = Traits;
    auto sKR = make_tensor(make_smem_ptr(sKR_ptr), typename T::SmemLayoutKRope{});

    constexpr int rope_elems = T::TOPK_BLOCK_SIZE * T::HEAD_DIM_ROPE;
    constexpr int per_thread = (rope_elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = producer_thread_idx + i * 128;
        if (idx < rope_elems) {
            int token_row = idx / T::HEAD_DIM_ROPE;
            int rope_col = idx % T::HEAD_DIM_ROPE;

            cutlass::bfloat16_t val = cutlass::bfloat16_t(0.0f);

            if (sValid[token_row]) {
                int token_index = __ldg(cur_indices + token_row);
                int block_index = (int)((unsigned)token_index / (unsigned)page_block_size);
                int rel_idx = (unsigned)token_index % (unsigned)page_block_size;

                const fp8* gK_base = kv_ptr + (int64_t)block_index * kv_block_stride +
                                     (int64_t)rel_idx * kv_row_stride;
                // ROPE starts after FP8 NOPE + float32 scale
                const __nv_bfloat16* rope_base = reinterpret_cast<const __nv_bfloat16*>(
                    gK_base + T::HEAD_DIM_NOPE + 4);
                val = cutlass::bfloat16_t(__ldg(rope_base + rope_col));
            }

            sKR(token_row, rope_col) = val;
        }
    }
}

//==============================================================================
// Main Kernel — SnapMLA FP8-Native Pipelined Decode
//==============================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
__global__ void __launch_bounds__(256, 1)
flash_fwd_splitkv_mla_fp8_sparse_sm120_kernel(const SparseAttnDecodeParams params) {
    using T = Traits<MODEL_TYPE>;
    using InputT = typename T::InputT;  // float_e4m3_t
    using SharedMemoryPlan = typename T::SharedMemoryPlan;
    using MC = ModelConfig<MODEL_TYPE>;

    static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;

    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / 32;
    const bool is_producer = (warp_idx >= 4);
    const bool is_consumer = (warp_idx < 4);

    extern __shared__ char smem_buf[];
    SharedMemoryPlan& smem = *reinterpret_cast<SharedMemoryPlan*>(smem_buf);

    float* sPVtile = smem.smem_pv_tile.data();
    float* sM = smem.smem_M.data();
    float* sL = smem.smem_L.data();
    float* sScale = smem.smem_scale.data();
    float* sKscales = smem.smem_K_scales.data();
    float* sQscales = smem.smem_Q_scales.data();
    float* sPscales = smem.smem_P_scales.data();
    bool* sValid = smem.is_kv_valid;
    InputT* sK_bufs[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

    TiledMmaFp8_64x64 tiled_mma_fp8;   // For NOPE FP8 tiles + PV GEMM
    TiledMma64x64 tiled_mma_bf16;       // For ROPE BF16 tile

    constexpr int O_ELEMS_PER_THREAD = (T::BLOCK_SIZE_M * T::HEAD_DIM_V + T::NUM_THREADS - 1) / T::NUM_THREADS;
    float rO[O_ELEMS_PER_THREAD];

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        // --- Topk range (unchanged) ---
        int topk_val, extra_topk_val = 0, num_orig_kv_blocks;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            topk_val = params.topk;
            num_orig_kv_blocks = ceil_div(topk_val, (int)T::TOPK_BLOCK_SIZE);
        } else {
            int tl = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            topk_val = tl;
            num_orig_kv_blocks = max(ceil_div(tl, (int)T::TOPK_BLOCK_SIZE), 1);
            extra_topk_val = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
        }
        int total_topk_padded = (MODEL_TYPE == ModelType::V32) ? topk_val :
            num_orig_kv_blocks * T::TOPK_BLOCK_SIZE + ceil_div(extra_topk_val, (int)T::TOPK_BLOCK_SIZE) * T::TOPK_BLOCK_SIZE;
        int total_kv_blocks = total_topk_padded / T::TOPK_BLOCK_SIZE;

        int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_kv_blocks;
        bool is_no_split = start_block_idx == 0 && end_block_idx == total_kv_blocks;

        if (thread_idx < T::BLOCK_SIZE_M) { sM[thread_idx] = MAX_INIT_VAL_SM; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
        #pragma unroll
        for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) rO[i] = 0.0f;
        __syncthreads();

        int start_head_idx = head_block_idx * T::BLOCK_SIZE_M;
        // Q_NOPE is FP8, Q_ROPE is BF16 (separate arrays from fused_q_quant)
        const InputT* q_nope_ptr = (const InputT*)params.q +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_NOPE +
            s_q_idx * params.h_q * T::HEAD_DIM_NOPE + start_head_idx * T::HEAD_DIM_NOPE;
        const cutlass::bfloat16_t* q_rope_ptr = (const cutlass::bfloat16_t*)params.q_rope +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_ROPE +
            s_q_idx * params.h_q * T::HEAD_DIM_ROPE + start_head_idx * T::HEAD_DIM_ROPE;
        int* gIndices = params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;

        // Load per-head Q scales to smem — q_scales layout: [b, s_q, h_q]
        if (thread_idx < T::BLOCK_SIZE_M && start_head_idx + thread_idx < params.h_q) {
            int q_scale_idx = batch_idx * params.s_q * params.h_q +
                              s_q_idx * params.h_q + start_head_idx + thread_idx;
            sQscales[thread_idx] = __ldg(params.q_scales + q_scale_idx);
        }
        __syncthreads();
        int* gExtraIndices = params.extra_indices ?
            (params.extra_indices + batch_idx * params.stride_extra_indices_b + s_q_idx * params.stride_extra_indices_s_q) : nullptr;

        //======================================================================
        // Main loop over topk blocks
        //======================================================================
        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            bool is_extra_block = false;
            int* cur_indices;
            fp8* kv_ptr;
            int page_block_size_cur, kv_block_stride, kv_row_stride;

            if constexpr (MODEL_TYPE == ModelType::V32) {
                cur_indices = gIndices + block_idx * T::TOPK_BLOCK_SIZE;
                kv_ptr = (fp8*)params.kv; page_block_size_cur = params.page_block_size;
                kv_block_stride = params.stride_kv_block; kv_row_stride = params.stride_kv_row;
            } else {
                is_extra_block = (block_idx >= num_orig_kv_blocks);
                if (!is_extra_block) {
                    cur_indices = gIndices + block_idx * T::TOPK_BLOCK_SIZE;
                    kv_ptr = (fp8*)params.kv; page_block_size_cur = params.page_block_size;
                    kv_block_stride = params.stride_kv_block; kv_row_stride = params.stride_kv_row;
                } else {
                    int ebi = block_idx - num_orig_kv_blocks;
                    cur_indices = gExtraIndices + ebi * T::TOPK_BLOCK_SIZE;
                    kv_ptr = (fp8*)params.extra_kv; page_block_size_cur = params.extra_page_block_size;
                    kv_block_stride = params.stride_extra_kv_block; kv_row_stride = params.stride_extra_kv_row;
                }
            }

            // Valid mask (producer warps)
            if (is_producer) {
                int ptid = thread_idx - 128;
                for (int i = ptid; i < T::TOPK_BLOCK_SIZE; i += 128) {
                    int token_idx = __ldg(cur_indices + i);
                    sValid[i] = (token_idx >= 0);
                }
            }
            __syncthreads();

            //==============================================================
            // DUAL-MMA QK GEMM: FP8 NOPE tiles + BF16 ROPE tile
            //==============================================================
            auto rS = partition_fragment_C(tiled_mma_fp8, Shape<_64, _64>{});
            clear(rS);

            constexpr int q_tile_elems = T::BLOCK_SIZE_M * T::K_TILE_DIM;
            constexpr int q_per_thread = (q_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

            // Bootstrap: producers load K_NOPE[tile 0] + scales, all load Q_NOPE[tile 0]
            if (is_producer) {
                gather_k_nope_fp8<MODEL_TYPE, T>(
                    kv_ptr, cur_indices, sValid, sK_bufs[0], sKscales,
                    0, page_block_size_cur, kv_block_stride, kv_row_stride,
                    thread_idx - 128, true);
            }
            {
                auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                #pragma unroll
                for (int i = 0; i < q_per_thread; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < q_tile_elems) {
                        int row = idx / T::K_TILE_DIM, col = idx % T::K_TILE_DIM;
                        sQ(row, col) = (row + start_head_idx < params.h_q && col < T::HEAD_DIM_NOPE) ?
                            InputT(ldg_fp8(reinterpret_cast<const fp8*>(q_nope_ptr) + row * T::HEAD_DIM_NOPE + col)) : InputT(0);
                    }
                }
            }
            __syncthreads();

            // Pipelined FP8 NOPE loop
            for (int k_tile = 0; k_tile < T::NUM_NOPE_TILES; ++k_tile) {
                if (k_tile > 0) {
                    auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                    #pragma unroll
                    for (int i = 0; i < q_per_thread; ++i) {
                        int idx = thread_idx + i * T::NUM_THREADS;
                        if (idx < q_tile_elems) {
                            int row = idx / T::K_TILE_DIM, col = idx % T::K_TILE_DIM;
                            int gk = k_tile * T::K_TILE_DIM + col;
                            sQ(row, col) = (row + start_head_idx < params.h_q && gk < T::HEAD_DIM_NOPE) ?
                                InputT(ldg_fp8(reinterpret_cast<const fp8*>(q_nope_ptr) + row * T::HEAD_DIM_NOPE + gk)) : InputT(0);
                        }
                    }
                    __syncthreads();
                }

                if (is_producer && k_tile + 1 < T::NUM_NOPE_TILES) {
                    gather_k_nope_fp8<MODEL_TYPE, T>(
                        kv_ptr, cur_indices, sValid,
                        sK_bufs[(k_tile + 1) % 2], sKscales,
                        (k_tile + 1) * T::K_TILE_DIM,
                        page_block_size_cur, kv_block_stride, kv_row_stride,
                        thread_idx - 128, false);
                }
                if (is_consumer) {
                    auto sQ_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                    auto sK_t = make_tensor(make_smem_ptr(sK_bufs[k_tile % 2]), typename T::SmemLayoutK{});
                    cute_gemm_ss<false>(tiled_mma_fp8, sQ_t, sK_t, rS, thread_idx);
                }
                __syncthreads();
            }

            // BF16 ROPE tile: load Q_ROPE + gather K_ROPE, then BF16 MMA
            {
                // All threads load Q_ROPE to BF16 smem
                constexpr int qr_elems = T::BLOCK_SIZE_M * T::HEAD_DIM_ROPE;
                constexpr int qr_per = (qr_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;
                auto sQR = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_rope.data()), typename T::SmemLayoutQRope{});
                #pragma unroll
                for (int i = 0; i < qr_per; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < qr_elems) {
                        int row = idx / T::HEAD_DIM_ROPE, col = idx % T::HEAD_DIM_ROPE;
                        sQR(row, col) = (row + start_head_idx < params.h_q) ?
                            cutlass::bfloat16_t(__ldg(reinterpret_cast<const __nv_bfloat16*>(q_rope_ptr) + row * T::HEAD_DIM_ROPE + col)) : cutlass::bfloat16_t(0.0f);
                    }
                }
                // Producers gather K_ROPE (BF16 from cache)
                if (is_producer) {
                    gather_k_rope_bf16<T>(
                        kv_ptr, cur_indices, sValid,
                        smem.qk_phase.smem_K_rope.data(),
                        page_block_size_cur, kv_block_stride, kv_row_stride,
                        thread_idx - 128);
                }
                __syncthreads();

                // BF16 MMA: accumulates into same rS fragment (compatible CLayout)
                if (is_consumer) {
                    auto sQR_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_rope.data()), typename T::SmemLayoutQRope{});
                    auto sKR_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_K_rope.data()), typename T::SmemLayoutKRope{});
                    cute_gemm_ss<false>(tiled_mma_bf16, sQR_t, sKR_t, rS, thread_idx);
                }
                __syncthreads();
            }

            //==============================================================
            // Post-QK dequant: scores *= Q_scale[row] * K_scale[col]
            //==============================================================
            if (is_consumer) {
                fragment_qk_dequant(rS, sQscales, sKscales, thread_idx);
            }

            //==============================================================
            // Standard softmax (no V-scale fusion — matches reference)
            //==============================================================
            if (is_consumer) {
                CUTE_UNROLL
                for (int i = 0; i < size(rS); ++i) rS(i) *= params.sm_scale;
            }

            if (thread_idx < T::BLOCK_SIZE_M) {
                sScale[thread_idx] = sM[thread_idx];
                sM[thread_idx] = MAX_INIT_VAL_MASK;
            }
            __syncthreads();

            if (is_consumer) fragment_masked_row_max(rS, sValid, sM, thread_idx, T::BLOCK_SIZE_M);
            __syncthreads();

            if (thread_idx < T::BLOCK_SIZE_M) {
                float old_max = sScale[thread_idx], new_max = sM[thread_idx];
                float rescale = (old_max <= MAX_INIT_VAL_SM) ? 1.0f : exp2f((old_max - new_max) * LOG2E);
                sL[thread_idx] *= rescale;
                sScale[thread_idx] = rescale;
            }
            __syncthreads();

            // Standard exp + row-sum (no V-scale fusion)
            if (is_consumer) fragment_masked_exp_sum(rS, sValid, sM, sL, thread_idx, T::BLOCK_SIZE_M);
            __syncthreads();

            // Rescale previous output
            #pragma unroll
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int row = (thread_idx + i * T::NUM_THREADS) / T::HEAD_DIM_V;
                if (row < T::BLOCK_SIZE_M) rO[i] *= sScale[row];
            }

            //==============================================================
            // Per-row FP8 P quantization (matching reference row-by-row)
            //==============================================================

            // Phase 1: init per-row amax to 0
            if (thread_idx < T::BLOCK_SIZE_M) sPscales[thread_idx] = 0.0f;
            __syncthreads();

            // Phase 2: compute per-row amax (consumers only)
            if (is_consumer) fragment_fp8_compute_row_scales(rS, sPscales, thread_idx);
            __syncthreads();

            // Phase 3: convert per-row amax → per-row scale
            if (thread_idx < T::BLOCK_SIZE_M)
                sPscales[thread_idx] = fmaxf(sPscales[thread_idx], P_SCALE_EPS) / FP8_E4M3_MAX;
            __syncthreads();

            // Phase 4: quantize P to FP8 with per-row scales
            if (is_consumer)
                fragment_fp8_row_quantize_store(rS, smem.smem_P.data(), typename T::SmemLayoutP{}, sPscales, thread_idx);
            __syncthreads();

            //==============================================================
            // FP8 PV GEMM + per-row P_scale + per-tile V_scale dequant
            // V loaded per-tile: gather FP8, dequant by K_scale, requant
            //==============================================================
            constexpr int V_TILE_DIM = 64;
            constexpr int v_tile_elems = T::TOPK_BLOCK_SIZE * V_TILE_DIM;
            constexpr int v_tile_per_thread = (v_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

            // Precompute per-element base pointers (invariant across V-tiles).
            // Avoids redundant __ldg + integer division per V-tile iteration.
            const fp8* local_gK_base[v_tile_per_thread];
            float local_kscale[v_tile_per_thread];
            #pragma unroll
            for (int i = 0; i < v_tile_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                local_gK_base[i] = nullptr;
                local_kscale[i] = 0.0f;
                if (idx < v_tile_elems) {
                    int token_row = idx / V_TILE_DIM;
                    if (sValid[token_row]) {
                        int token_index = __ldg(cur_indices + token_row);
                        int bi = (int)((unsigned)token_index / (unsigned)page_block_size_cur);
                        int ri = (unsigned)token_index % (unsigned)page_block_size_cur;
                        local_gK_base[i] = kv_ptr + (int64_t)bi * kv_block_stride + (int64_t)ri * kv_row_stride;
                        local_kscale[i] = sKscales[token_row];
                    }
                }
            }

            for (int v_tile = 0; v_tile < T::NUM_V_TILES; ++v_tile) {
                // Load V-tile: gather raw FP8, dequant by K_scale, requant
                float local_vvals[v_tile_per_thread];
                float local_vamax = 0.0f;
                int v_dim_offset = v_tile * V_TILE_DIM;

                #pragma unroll
                for (int i = 0; i < v_tile_per_thread; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    float v_deq = 0.0f;
                    if (idx < v_tile_elems && local_gK_base[i] != nullptr) {
                        int v_col = v_dim_offset + idx % V_TILE_DIM;
                        if (v_col < T::HEAD_DIM_V) {
                            v_deq = float(InputT(ldg_fp8(local_gK_base[i] + v_col))) * local_kscale[i];
                        }
                    }
                    local_vvals[i] = v_deq;
                    local_vamax = fmaxf(local_vamax, fabsf(v_deq));
                }

                // Warp-reduce amax
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2)
                    local_vamax = fmaxf(local_vamax, __shfl_xor_sync(0xffffffff, local_vamax, offset));

                // Cross-warp amax reduce via smem (reuse sPVtile[0..7] as scratch)
                if (thread_idx % 32 == 0)
                    sPVtile[thread_idx / 32] = local_vamax;
                __syncthreads();
                if (thread_idx == 0) {
                    float amax = sPVtile[0];
                    #pragma unroll
                    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, sPVtile[w]);
                    sPVtile[0] = amax;
                }
                __syncthreads();

                float v_scale = fmaxf(sPVtile[0], 1e-12f) / FP8_E4M3_MAX;
                float v_inv_scale = 1.0f / v_scale;

                // Requant and store to smem (swizzled layout)
                auto sVt_store = make_tensor(
                    make_smem_ptr(smem.smem_V.data() + v_tile * T::V_tile_elems),
                    typename T::SmemLayoutVTile{});
                #pragma unroll
                for (int i = 0; i < v_tile_per_thread; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < v_tile_elems) {
                        int token_row = idx / V_TILE_DIM;
                        int vci = idx % V_TILE_DIM;
                        float val = local_vvals[i] * v_inv_scale;
                        val = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, val));
                        sVt_store(vci, token_row) = InputT(val);
                    }
                }
                __syncthreads();

                if (is_consumer) {
                    auto sP = make_tensor(make_smem_ptr(smem.smem_P.data()), typename T::SmemLayoutP{});
                    auto sVt = make_tensor(
                        make_smem_ptr(smem.smem_V.data() + v_tile * T::V_tile_elems),
                        typename T::SmemLayoutVTile{});

                    auto rPV = partition_fragment_C(tiled_mma_fp8, Shape<_64, _64>{});
                    cute_gemm_ss<true>(tiled_mma_fp8, sP, sVt, rPV, thread_idx);

                    // Dequant: P_scale[row] * V_tile_scale
                    fragment_pv_row_dequant(rPV, sPscales, thread_idx);
                    CUTE_UNROLL
                    for (int i = 0; i < size(rPV); ++i) rPV(i) *= v_scale;

                    fragment_to_smem_f32(rPV, sPVtile, thread_idx, 64, 64);
                }
                __syncthreads();

                const int tc = v_tile * V_TILE_DIM;
                #pragma unroll
                for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                    int gi = thread_idx + i * T::NUM_THREADS;
                    int row = gi / T::HEAD_DIM_V, gcol = gi % T::HEAD_DIM_V;
                    if (gcol >= tc && gcol < tc + V_TILE_DIM && row < T::BLOCK_SIZE_M)
                        rO[i] += sPVtile[row * V_TILE_DIM + gcol - tc];
                }
                __syncthreads();
            }
        }  // end topk block loop

        //======================================================================
        // Epilogue
        //======================================================================
        int num_valid = min(params.h_q - start_head_idx, (int)T::BLOCK_SIZE_M);

        if (is_no_split) {
            cutlass::bfloat16_t* o_ptr = (cutlass::bfloat16_t*)params.out +
                            batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                            start_head_idx * params.stride_o_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V, col = idx % T::HEAD_DIM_V;
                if (row < num_valid) {
                    float inv = sL[row] == 0.0f ? 0.0f : (1.0f / sL[row]);
                    if (params.attn_sink) {
                        // sM is a NATURAL-unit max: the sink logit enters the
                        // denominator as e^(sink - m) == exp2f((sink - m)*LOG2E).
                        float sl = __ldg(params.attn_sink + start_head_idx + row);
                        inv = sL[row] == 0.0f ? 0.0f : __fdividef(1.0f, sL[row] + exp2f((sl - sM[row]) * LOG2E));
                    }
                    o_ptr[row * params.stride_o_h_q + col] = cutlass::bfloat16_t(rO[i] * inv);
                }
            }
            if (thread_idx < num_valid) {
                float* g = params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + start_head_idx;
                // TD-LSE-UNITS-DECODE: sM/sL are NATURAL log units in this port
                // (rS *= sm_scale, exp via exp2f(x*LOG2E) == e^x) and sL is the
                // FULL-PRECISION exp row-sum (fragment_masked_exp_sum runs BEFORE
                // the FP8 P quantization), so LSE = m + log(l). The old
                // logf(L*P_scale) + M/LOG2E carried a spurious ×P_scale (leftover
                // of FlashMLA's quantized-sum accumulator, where L was Σ(exp/Ps))
                // and applied ×ln2 to a natural-unit m (FlashMLA's log2-unit M
                // convention) — corrupting any cross-rank LSE consumer (DCP).
                float cur_L = sL[thread_idx];
                g[thread_idx] = (cur_L == 0.0f) ? INFINITY : (sM[thread_idx] + logf(cur_L));
            }
        } else {
            int nsi = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            int si = __ldg(params.num_splits_ptr + batch_idx) + nsi;
            float* oa = params.o_accum + si * params.stride_o_accum_split +
                        s_q_idx * params.stride_o_accum_s_q + start_head_idx * params.stride_o_accum_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V, col = idx % T::HEAD_DIM_V;
                if (row < num_valid) oa[row * params.stride_o_accum_h_q + col] = rO[i] * (sL[row] == 0.0f ? 0.0f : (1.0f / sL[row]));
            }
            if (thread_idx < num_valid) {
                float* g = params.lse_accum + si * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q + start_head_idx;
                // lse_accum is consumed by mla_combine in LOG2 units
                // (exp2f(lse_accum - global)): true log2-LSE = log2(l) + m*LOG2E
                // with the NATURAL-unit m of this port. The old
                // log2f(L*P_scale) + M mixed a natural m into a log2 expression
                // and carried a spurious ×P_scale — mis-weighting the split
                // combine (TD-LSE-UNITS-DECODE).
                float cur_L = sL[thread_idx];
                g[thread_idx] = (cur_L == 0.0f) ? -INFINITY : (log2f(cur_L) + sM[thread_idx] * LOG2E);
            }
        }
        __syncthreads();
    }
}

//==============================================================================
// Launch
//==============================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    using T = Traits<MODEL_TYPE>;
    static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_sm120_kernel<MODEL_TYPE, NUM_HEADS>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts), T::NUM_THREADS, smem_size, params.stream>>>(params);
}

}  // namespace sm120::decode::sparse_fp8
