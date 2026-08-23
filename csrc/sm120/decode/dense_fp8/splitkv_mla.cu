/***************************************************************************************************
 * Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
 * MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.
 *
 * SM120 Dense FP8 Decode Kernel — SnapMLA Dual-MMA Path
 *
 * Dense attention: attends to ALL tokens up to seqlen (no topk/indices).
 * Uses paged KV cache with block_table for page resolution.
 *
 * Same dual-MMA pipeline as sparse decode:
 *   NOPE tiles: FP8 Q_NOPE @ FP8 K_NOPE^T (SM89_16x8x32)
 *   ROPE tile:  BF16 Q_ROPE @ BF16 K_ROPE^T (SM80_16x8x16)
 *
 * Key simplification vs sparse: K/V loaded from contiguous pages instead of
 * scattered gather via indices. No valid mask from indices — mask from seqlen.
 *
 * KV cache row: [d_c FP8 | float32 scale | d_rope BF16]
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

namespace sm120::decode::dense_fp8 {

// __ldg helper for FP8: __ldg doesn't natively support __nv_fp8_e4m3,
// so load as char via read-only cache and reinterpret.
__device__ __forceinline__ fp8 ldg_fp8(const fp8* ptr) {
    char raw = __ldg(reinterpret_cast<const char*>(ptr));
    fp8 result;
    memcpy(&result, &raw, 1);
    return result;
}

//==============================================================================
// Dense K NOPE tile load: contiguous page read
//
// For QK GEMM: B operand is K. CuTe partition_B maps the smem
// tensor as (N=token, K=dim). So we store K as sK(token, dim).
//==============================================================================
template<typename Traits>
__device__ void load_k_nope_fp8(
    const fp8* __restrict__ page_base,
    int stride_row,
    float* __restrict__ sK_scales,
    cutlass::float_e4m3_t* __restrict__ sK_ptr,
    int k_offset,
    int num_valid_tokens,
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

            if (token_row < num_valid_tokens && global_k < T::HEAD_DIM_NOPE) {
                const fp8* row_ptr = page_base + (int64_t)token_row * stride_row;
                val = cutlass::float_e4m3_t(ldg_fp8(row_ptr + global_k));

                if (load_scales && k_col == 0) {
                    sK_scales[token_row] = __ldg(reinterpret_cast<const float*>(row_ptr + T::HEAD_DIM_NOPE));
                }
            }

            sK(token_row, k_col) = val;
        }
    }
}

//==============================================================================
// Dense K ROPE tile load: contiguous BF16 read from page
//==============================================================================
template<typename Traits>
__device__ void load_k_rope_bf16(
    const fp8* __restrict__ page_base,
    int stride_row,
    cutlass::bfloat16_t* __restrict__ sKR_ptr,
    int num_valid_tokens,
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

            if (token_row < num_valid_tokens) {
                const fp8* row_ptr = page_base + (int64_t)token_row * stride_row;
                const __nv_bfloat16* rope_ptr = reinterpret_cast<const __nv_bfloat16*>(
                    row_ptr + T::HEAD_DIM_NOPE + 4);
                val = cutlass::bfloat16_t(__ldg(rope_ptr + rope_col));
            }

            sKR(token_row, rope_col) = val;
        }
    }
}

//==============================================================================
// Dense V-tile load with dequant + requant
//
// Loads one 64-dim V-tile from the page, dequantizes by per-token K_scale,
// then requantizes with a single per-tile V_scale. This ensures the PV GEMM
// operates on properly scaled FP8 values. Without this, the output is ~1400x
// too large (raw FP8 V missing K_scale dequant).
//
// Returns the per-tile V_scale (all threads compute the same value).
// Requires 2 internal __syncthreads() for the amax reduce.
//==============================================================================
template<typename Traits>
__device__ float load_v_tile_dequant_requant(
    const fp8* __restrict__ page_base,
    int stride_row,
    cutlass::float_e4m3_t* __restrict__ sV_tile_ptr,
    const float* __restrict__ sKscales,
    float* __restrict__ s_amax_scratch,  // shared mem scratch (1 float, e.g. sPVtile[0])
    int num_valid_tokens,
    int v_dim_offset,
    int thread_idx,
    int num_threads
) {
    using T = Traits;
    constexpr int tile_tokens = T::TOPK_BLOCK_SIZE;  // 64
    constexpr int tile_dims = 64;
    constexpr int tile_elems = tile_tokens * tile_dims;
    constexpr int per_thread = (tile_elems + 255) / 256;  // 16

    // Pass 1: Read V from global, dequant by K_scale, track amax
    float local_vals[per_thread];
    float local_amax = 0.0f;

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = thread_idx + i * num_threads;
        float v_deq = 0.0f;
        if (idx < tile_elems) {
            int token_row = idx / tile_dims;
            int v_col = v_dim_offset + idx % tile_dims;
            if (token_row < num_valid_tokens && v_col < T::HEAD_DIM_V) {
                const fp8* row_ptr = page_base + (int64_t)token_row * stride_row;
                v_deq = float(cutlass::float_e4m3_t(ldg_fp8(row_ptr + v_col))) * sKscales[token_row];
            }
        }
        local_vals[i] = v_deq;
        local_amax = fmaxf(local_amax, fabsf(v_deq));
    }

    // Warp-reduce amax
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));

    // Cross-warp amax reduce via smem (reuse s_amax_scratch[0..7] as scratch)
    if (thread_idx % 32 == 0)
        s_amax_scratch[thread_idx / 32] = local_amax;
    __syncthreads();
    if (thread_idx == 0) {
        float amax = s_amax_scratch[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) amax = fmaxf(amax, s_amax_scratch[w]);
        s_amax_scratch[0] = amax;
    }
    __syncthreads();

    float v_amax = s_amax_scratch[0];
    float v_scale = fmaxf(v_amax, 1e-12f) / FP8_E4M3_MAX;
    float v_inv_scale = 1.0f / v_scale;

    // Pass 2: Requant and store to smem (swizzled layout for PV GEMM)
    auto sVt = make_tensor(make_smem_ptr(sV_tile_ptr), typename T::SmemLayoutVTile{});

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = thread_idx + i * num_threads;
        if (idx < tile_elems) {
            int token_row = idx / tile_dims;
            int vci = idx % tile_dims;
            float val = local_vals[i] * v_inv_scale;
            val = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, val));
            sVt(vci, token_row) = cutlass::float_e4m3_t(val);
        }
    }

    return v_scale;
}

//==============================================================================
// Main Kernel — Dense SnapMLA Dual-MMA Decode
//==============================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS, bool DETERMINISTIC>
__global__ void __launch_bounds__(256, 1)
flash_fwd_splitkv_mla_dense_fp8_sm120_kernel(const DenseAttnDecodeParams params) {
    using T = Traits<MODEL_TYPE>;
    using InputT = typename T::InputT;
    using RopeT = typename T::RopeT;
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

    static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
    // 4 consumer warps (warp_idx 0..3) — matches is_consumer below + the
    // hardcoded 128-thread producer/consumer split.
    static constexpr int NUM_CONSUMER_WARPS = 4;

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
    InputT* sK_bufs[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

    // DETERMINISTIC-only: per-warp private softmax-denominator partials. Sized 1
    // (and never referenced) in the default path → zero footprint there. ~1 KB
    // static smem when enabled (BLOCK_SIZE_M=64 × 4 warps); fits alongside the
    // ~54 KB dynamic SharedMemoryPlan under the 99 KB SM120 cap.
    __shared__ float sL_partial[DETERMINISTIC ? T::BLOCK_SIZE_M * NUM_CONSUMER_WARPS : 1];

    TiledMmaFp8_64x64 tiled_mma_fp8;
    TiledMma64x64 tiled_mma_bf16;

    constexpr int O_ELEMS = (T::BLOCK_SIZE_M * T::HEAD_DIM_V + T::NUM_THREADS - 1) / T::NUM_THREADS;
    float rO[O_ELEMS];

    sparse_fp8::DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    constexpr int q_tile_elems = T::BLOCK_SIZE_M * T::K_TILE_DIM;
    constexpr int q_per_thread = (q_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        int seqlen = __ldg(params.seqlens_k + batch_idx);
        int total_kv_blocks = ceil_div(seqlen, params.page_block_size);

        int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_kv_blocks;
        bool is_no_split = start_block_idx == 0 && end_block_idx == total_kv_blocks;

        int start_head_idx = head_block_idx * T::BLOCK_SIZE_M;

        if (thread_idx < T::BLOCK_SIZE_M) { sM[thread_idx] = MAX_INIT_VAL_SM; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
        #pragma unroll
        for (int i = 0; i < O_ELEMS; ++i) rO[i] = 0.0f;
        __syncthreads();

        // Q NOPE (FP8) and ROPE (BF16) from separate arrays
        const InputT* q_nope_ptr = (const InputT*)params.q_nope +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_NOPE +
            s_q_idx * params.h_q * T::HEAD_DIM_NOPE + start_head_idx * T::HEAD_DIM_NOPE;
        const RopeT* q_rope_ptr = (const RopeT*)params.q_rope +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_ROPE +
            s_q_idx * params.h_q * T::HEAD_DIM_ROPE + start_head_idx * T::HEAD_DIM_ROPE;

        // Load per-head Q scales
        if (thread_idx < T::BLOCK_SIZE_M && start_head_idx + thread_idx < params.h_q) {
            int q_scale_idx = batch_idx * params.s_q * params.h_q + s_q_idx * params.h_q + start_head_idx + thread_idx;
            sQscales[thread_idx] = __ldg(params.q_scales + q_scale_idx);
        }
        __syncthreads();

        //==================================================================
        // Main loop over KV pages
        //==================================================================
        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            // Resolve page
            int page_id = __ldg(params.block_table + batch_idx * params.block_table_batch_stride + block_idx);
            const fp8* page_base = (const fp8*)params.kv_cache + (int64_t)page_id * params.stride_kv_block;
            int block_start_token = block_idx * params.page_block_size;
            int num_valid = min(params.page_block_size, seqlen - block_start_token);

            // Valid mask (seqlen-based, not index-based)
            if (is_producer) {
                int ptid = thread_idx - 128;
                for (int i = ptid; i < T::TOPK_BLOCK_SIZE; i += 128)
                    smem.is_kv_valid[i] = (i < num_valid);
            }
            __syncthreads();

            //==============================================================
            // DUAL-MMA QK GEMM: FP8 NOPE tiles + BF16 ROPE tile
            //==============================================================
            auto rS = partition_fragment_C(tiled_mma_fp8, Shape<_64, _64>{});
            clear(rS);

            // Bootstrap: producers load K_NOPE[tile 0] + scales
            if (is_producer) {
                load_k_nope_fp8<T>(page_base, params.stride_kv_row, sKscales,
                    sK_bufs[0], 0, num_valid, thread_idx - 128, true);
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
                    load_k_nope_fp8<T>(page_base, params.stride_kv_row, sKscales,
                        sK_bufs[(k_tile + 1) % 2], (k_tile + 1) * T::K_TILE_DIM,
                        num_valid, thread_idx - 128, false);
                }
                if (is_consumer) {
                    auto sQ_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                    auto sK_t = make_tensor(make_smem_ptr(sK_bufs[k_tile % 2]), typename T::SmemLayoutK{});
                    cute_gemm_ss<false>(tiled_mma_fp8, sQ_t, sK_t, rS, thread_idx);
                }
                __syncthreads();

            }

            // BF16 ROPE tile
            {
                constexpr int qr_elems = T::BLOCK_SIZE_M * T::HEAD_DIM_ROPE;
                constexpr int qr_per = (qr_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;
                auto sQR = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_rope.data()), typename T::SmemLayoutQRope{});
                #pragma unroll
                for (int i = 0; i < qr_per; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < qr_elems) {
                        int row = idx / T::HEAD_DIM_ROPE, col = idx % T::HEAD_DIM_ROPE;
                        sQR(row, col) = (row + start_head_idx < params.h_q) ?
                            RopeT(__ldg(reinterpret_cast<const __nv_bfloat16*>(q_rope_ptr) + row * T::HEAD_DIM_ROPE + col)) : RopeT(0);
                    }
                }
                if (is_producer) {
                    load_k_rope_bf16<T>(page_base, params.stride_kv_row,
                        smem.qk_phase.smem_K_rope.data(), num_valid, thread_idx - 128);
                }
                __syncthreads();

                if (is_consumer) {
                    auto sQR_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_rope.data()), typename T::SmemLayoutQRope{});
                    auto sKR_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_K_rope.data()), typename T::SmemLayoutKRope{});
                    cute_gemm_ss<false>(tiled_mma_bf16, sQR_t, sKR_t, rS, thread_idx);
                }
                __syncthreads();
            }

            //==============================================================
            // Post-QK dequant + softmax + P quant + PV GEMM
            // (identical pipeline to sparse decode)
            //==============================================================
            if (is_consumer) fragment_qk_dequant(rS, sQscales, sKscales, thread_idx);

            if (is_consumer) {
                CUTE_UNROLL
                for (int i = 0; i < size(rS); ++i) rS(i) *= params.sm_scale;
            }
            if (thread_idx < T::BLOCK_SIZE_M) { sScale[thread_idx] = sM[thread_idx]; sM[thread_idx] = MAX_INIT_VAL_MASK; }
            __syncthreads();

            if (is_consumer) fragment_masked_row_max(rS, smem.is_kv_valid, sM, thread_idx, T::BLOCK_SIZE_M);
            __syncthreads();

            if (thread_idx < T::BLOCK_SIZE_M) {
                float old_max = sScale[thread_idx], new_max = sM[thread_idx];
                float rescale = (old_max <= MAX_INIT_VAL_SM) ? 1.0f : exp2f((old_max - new_max) * LOG2E);
                sL[thread_idx] *= rescale;
                sScale[thread_idx] = rescale;
                // Zero this row's per-warp partial slots (covered by the
                // __syncthreads below before the consumer warps write them).
                if constexpr (DETERMINISTIC) {
                    #pragma unroll
                    for (int w = 0; w < NUM_CONSUMER_WARPS; ++w)
                        sL_partial[thread_idx * NUM_CONSUMER_WARPS + w] = 0.0f;
                }
            }
            __syncthreads();

            if (is_consumer)
                fragment_masked_exp_sum<DETERMINISTIC>(
                    rS, smem.is_kv_valid, sM, sL, thread_idx, T::BLOCK_SIZE_M,
                    sL_partial, NUM_CONSUMER_WARPS);
            __syncthreads();

            // DETERMINISTIC: fixed-index cross-warp combine into sL (the only
            // added cost vs the atomic path: an N-add per KV tile; no extra
            // sync — the next sL reader is gated by the §428/§429 sync). sL is
            // the running denominator → accumulate (+=) this tile's contribution.
            if constexpr (DETERMINISTIC) {
                if (thread_idx < T::BLOCK_SIZE_M) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < NUM_CONSUMER_WARPS; ++w)
                        acc += sL_partial[thread_idx * NUM_CONSUMER_WARPS + w];
                    sL[thread_idx] += acc;
                }
            }

            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) {
                int row = (thread_idx + i * T::NUM_THREADS) / T::HEAD_DIM_V;
                if (row < T::BLOCK_SIZE_M) rO[i] *= sScale[row];
            }

            // Per-row P quantization
            if (thread_idx < T::BLOCK_SIZE_M) sPscales[thread_idx] = 0.0f;
            __syncthreads();
            if (is_consumer) fragment_fp8_compute_row_scales(rS, sPscales, thread_idx);
            __syncthreads();
            if (thread_idx < T::BLOCK_SIZE_M)
                sPscales[thread_idx] = fmaxf(sPscales[thread_idx], P_SCALE_EPS) / FP8_E4M3_MAX;
            __syncthreads();
            if (is_consumer)
                fragment_fp8_row_quantize_store(rS, smem.smem_P.data(), typename T::SmemLayoutP{}, sPscales, thread_idx);
            __syncthreads();

            // FP8 PV GEMM + per-row P_scale + per-tile V_scale dequant
            // V is loaded per-tile with dequant (K_scale) + requant (V_tile_scale)
            constexpr int V_TILE_DIM = 64;
            for (int v_tile = 0; v_tile < T::NUM_V_TILES; ++v_tile) {
                // Load V-tile: read raw FP8, dequant by K_scale, requant with tile V_scale
                float v_tile_scale = load_v_tile_dequant_requant<T>(
                    page_base, params.stride_kv_row,
                    smem.smem_V.data() + v_tile * T::V_tile_elems,
                    sKscales, sPVtile,  // sPVtile[0..7] as amax scratch
                    num_valid, v_tile * V_TILE_DIM,
                    thread_idx, T::NUM_THREADS);
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
                    for (int i = 0; i < size(rPV); ++i) rPV(i) *= v_tile_scale;

                    fragment_to_smem_f32(rPV, sPVtile, thread_idx, 64, 64);
                }
                __syncthreads();

                const int tc = v_tile * V_TILE_DIM;
                #pragma unroll
                for (int i = 0; i < O_ELEMS; ++i) {
                    int gi = thread_idx + i * T::NUM_THREADS;
                    int row = gi / T::HEAD_DIM_V, gcol = gi % T::HEAD_DIM_V;
                    if (gcol >= tc && gcol < tc + V_TILE_DIM && row < T::BLOCK_SIZE_M)
                        rO[i] += sPVtile[row * V_TILE_DIM + gcol - tc];
                }
                __syncthreads();
            }
        }  // end page loop

        //==================================================================
        // Epilogue
        //==================================================================
        int num_valid_heads = min(params.h_q - start_head_idx, (int)T::BLOCK_SIZE_M);

        if (is_no_split) {
            cutlass::bfloat16_t* o_ptr = (cutlass::bfloat16_t*)params.out +
                batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                start_head_idx * params.stride_o_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V, col = idx % T::HEAD_DIM_V;
                if (row < num_valid_heads) {
                    float inv = sL[row] == 0.0f ? 0.0f : (1.0f / sL[row]);
                    o_ptr[row * params.stride_o_h_q + col] = cutlass::bfloat16_t(rO[i] * inv);
                }
            }
            if (thread_idx < num_valid_heads) {
                float* g = params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + start_head_idx;
                // TD-LSE-UNITS-DECODE: sM/sL are NATURAL log units (rS *=
                // sm_scale, exp2f(x*LOG2E) == e^x) and sL is the FULL-PRECISION
                // exp row-sum (exp_sum runs BEFORE FP8 P quantization), so
                // LSE = m + log(l). The old logf(L*P_scale) + M/LOG2E carried a
                // spurious ×P_scale and a ×ln2 on a natural m (FlashMLA log2
                // convention) — see decode/sparse_fp8/splitkv_mla.cu.
                float cur_L = sL[thread_idx];
                g[thread_idx] = (cur_L == 0.0f) ? INFINITY : (sM[thread_idx] + logf(cur_L));
            }
        } else {
            int nsi = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            int si = __ldg(params.num_splits_ptr + batch_idx) + nsi;
            float* oa = params.o_accum + si * params.stride_o_accum_split +
                        s_q_idx * params.stride_o_accum_s_q + start_head_idx * params.stride_o_accum_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V, col = idx % T::HEAD_DIM_V;
                if (row < num_valid_heads)
                    oa[row * params.stride_o_accum_h_q + col] = rO[i] * (sL[row] == 0.0f ? 0.0f : (1.0f / sL[row]));
            }
            if (thread_idx < num_valid_heads) {
                float* g = params.lse_accum + si * params.stride_lse_accum_split + s_q_idx * params.stride_lse_accum_s_q + start_head_idx;
                // lse_accum: LOG2 units for mla_combine = log2(l) + m*LOG2E
                // (natural-unit m). Old form mixed a natural m into a log2
                // expression + spurious ×P_scale — mis-weighted the split
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
template<ModelType MODEL_TYPE, int NUM_HEADS, bool DETERMINISTIC>
void run_flash_splitkv_mla_dense_fp8_kernel(const DenseAttnDecodeParams &params) {
    using T = Traits<MODEL_TYPE>;
    static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    auto kernel = &flash_fwd_splitkv_mla_dense_fp8_sm120_kernel<MODEL_TYPE, NUM_HEADS, DETERMINISTIC>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    kernel<<<dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts), T::NUM_THREADS, smem_size, params.stream>>>(params);
}

}  // namespace sm120::decode::dense_fp8
