/***************************************************************************************************
 * SM120 CSA FP8 Decode Kernel — V4 Compressed Sparse Attention
 *
 * Adapted from sparse_fp8/splitkv_mla.cu (SnapMLA) for DeepSeek V4 CSA layers.
 * Key differences from SnapMLA sparse decode:
 *   - V4 cache: 1160 B/entry with separate K and V at distinct offsets
 *   - SWA combine: sparse compressed + sliding window in a single kernel
 *   - Single KV head broadcast to all Q heads (num_key_value_heads=1)
 *   - Q arrives as BF16, quantized to FP8 in-kernel with per-head dynamic scales
 *
 * Source attribution: adapted from FlashMLA sparse decode (MIT License,
 * Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md)
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

namespace sm120::decode::csa_fp8 {

using fp8 = __nv_fp8_e4m3;

__device__ __forceinline__ fp8 ldg_fp8(const fp8* ptr) {
    char raw = __ldg(reinterpret_cast<const char*>(ptr));
    fp8 result;
    memcpy(&result, &raw, 1);
    return result;
}

//==============================================================================
// V4 K NOPE gather (sparse indices)
//==============================================================================
template<typename T>
__device__ void v4_gather_k_nope(
    const char* __restrict__ cache_ptr,
    const int* __restrict__ cur_indices,
    const bool* __restrict__ sValid,
    cutlass::float_e4m3_t* __restrict__ sK_ptr,
    float* __restrict__ sK_scales,
    int k_offset, int page_block_size,
    int64_t block_stride, int row_stride,
    int ptid, bool load_scales
) {
    auto sK = make_tensor(make_smem_ptr(sK_ptr), typename T::SmemLayoutK{});
    constexpr int tile_elems = T::TOPK_BLOCK_SIZE * T::K_TILE_DIM;
    constexpr int per_thr = (tile_elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thr; ++i) {
        int idx = ptid + i * 128;
        if (idx < tile_elems) {
            int row = idx / T::K_TILE_DIM, col = idx % T::K_TILE_DIM;
            int gk = k_offset + col;
            cutlass::float_e4m3_t val(0.0f);
            if (sValid[row] && gk < T::HEAD_DIM_NOPE) {
                int ti = __ldg(cur_indices + row);
                int bi = (unsigned)ti / (unsigned)page_block_size;
                int ri = (unsigned)ti % (unsigned)page_block_size;
                const char* entry = cache_ptr + (int64_t)bi * block_stride + (int64_t)ri * row_stride;
                val = cutlass::float_e4m3_t(ldg_fp8((const fp8*)(entry + V4CacheLayout::K_NOPE_OFFSET) + gk));
                if (load_scales && col == 0)
                    sK_scales[row] = __ldg((const float*)(entry + V4CacheLayout::K_SCALE_OFFSET));
            }
            sK(row, col) = val;
        }
    }
}

//==============================================================================
// V4 K ROPE gather (sparse indices)
//==============================================================================
template<typename T>
__device__ void v4_gather_k_rope(
    const char* __restrict__ cache_ptr,
    const int* __restrict__ cur_indices,
    const bool* __restrict__ sValid,
    cutlass::bfloat16_t* __restrict__ sKR_ptr,
    int page_block_size, int64_t block_stride, int row_stride, int ptid
) {
    auto sKR = make_tensor(make_smem_ptr(sKR_ptr), typename T::SmemLayoutKRope{});
    constexpr int elems = T::TOPK_BLOCK_SIZE * T::HEAD_DIM_ROPE;
    constexpr int per_thr = (elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thr; ++i) {
        int idx = ptid + i * 128;
        if (idx < elems) {
            int row = idx / T::HEAD_DIM_ROPE, col = idx % T::HEAD_DIM_ROPE;
            cutlass::bfloat16_t val(0.0f);
            if (sValid[row]) {
                int ti = __ldg(cur_indices + row);
                int bi = (unsigned)ti / (unsigned)page_block_size;
                int ri = (unsigned)ti % (unsigned)page_block_size;
                const char* entry = cache_ptr + (int64_t)bi * block_stride + (int64_t)ri * row_stride;
                val = cutlass::bfloat16_t(__ldg((const __nv_bfloat16*)(entry + V4CacheLayout::K_ROPE_OFFSET) + col));
            }
            sKR(row, col) = val;
        }
    }
}

//==============================================================================
// V4 K NOPE load (SWA dense, block_table)
//==============================================================================
template<typename T>
__device__ void v4_swa_load_k_nope(
    const char* __restrict__ cache_ptr,
    const int* __restrict__ block_table, int bt_stride, int batch_idx,
    int page_block_size, int64_t block_stride, int row_stride,
    int token_start, int swa_len,
    cutlass::float_e4m3_t* __restrict__ sK_ptr,
    float* __restrict__ sK_scales, bool* __restrict__ sValid,
    int k_offset, int ptid, bool load_scales
) {
    auto sK = make_tensor(make_smem_ptr(sK_ptr), typename T::SmemLayoutK{});
    constexpr int tile_elems = T::TOPK_BLOCK_SIZE * T::K_TILE_DIM;
    constexpr int per_thr = (tile_elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thr; ++i) {
        int idx = ptid + i * 128;
        if (idx < tile_elems) {
            int row = idx / T::K_TILE_DIM, col = idx % T::K_TILE_DIM;
            int gk = k_offset + col;
            int tpos = token_start + row;
            bool valid = (tpos < swa_len && gk < T::HEAD_DIM_NOPE);
            if (load_scales && col == 0) sValid[row] = (tpos < swa_len);
            cutlass::float_e4m3_t val(0.0f);
            if (valid) {
                int pid = __ldg(block_table + batch_idx * bt_stride + tpos / page_block_size);
                int ri = tpos % page_block_size;
                const char* entry = cache_ptr + (int64_t)pid * block_stride + (int64_t)ri * row_stride;
                val = cutlass::float_e4m3_t(ldg_fp8((const fp8*)(entry + V4CacheLayout::K_NOPE_OFFSET) + gk));
                if (load_scales && col == 0)
                    sK_scales[row] = __ldg((const float*)(entry + V4CacheLayout::K_SCALE_OFFSET));
            }
            sK(row, col) = val;
        }
    }
}

//==============================================================================
// V4 K ROPE load (SWA dense, block_table)
//==============================================================================
template<typename T>
__device__ void v4_swa_load_k_rope(
    const char* __restrict__ cache_ptr,
    const int* __restrict__ block_table, int bt_stride, int batch_idx,
    int page_block_size, int64_t block_stride, int row_stride,
    int token_start, const bool* __restrict__ sValid,
    cutlass::bfloat16_t* __restrict__ sKR_ptr, int ptid
) {
    auto sKR = make_tensor(make_smem_ptr(sKR_ptr), typename T::SmemLayoutKRope{});
    constexpr int elems = T::TOPK_BLOCK_SIZE * T::HEAD_DIM_ROPE;
    constexpr int per_thr = (elems + 128 - 1) / 128;

    #pragma unroll
    for (int i = 0; i < per_thr; ++i) {
        int idx = ptid + i * 128;
        if (idx < elems) {
            int row = idx / T::HEAD_DIM_ROPE, col = idx % T::HEAD_DIM_ROPE;
            cutlass::bfloat16_t val(0.0f);
            if (sValid[row]) {
                int tpos = token_start + row;
                int pid = __ldg(block_table + batch_idx * bt_stride + tpos / page_block_size);
                int ri = tpos % page_block_size;
                const char* entry = cache_ptr + (int64_t)pid * block_stride + (int64_t)ri * row_stride;
                val = cutlass::bfloat16_t(__ldg((const __nv_bfloat16*)(entry + V4CacheLayout::K_ROPE_OFFSET) + col));
            }
            sKR(row, col) = val;
        }
    }
}

//==============================================================================
// Main Kernel — V4 CSA FP8 Decode
//==============================================================================
template<int NUM_HEADS, bool DETERMINISTIC>
__global__ void __launch_bounds__(256, 1)
csa_fp8_decode_sm120_kernel(const CsaFp8DecodeParams params) {
    using T = Traits;
    using InputT = typename T::InputT;
    using SharedMemoryPlan = typename T::SharedMemoryPlan;

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
    float* sVscales = smem.smem_V_scales.data();
    float* sQscales = smem.smem_Q_scales.data();
    float* sPscales = smem.smem_P_scales.data();
    bool* sValid = smem.is_kv_valid;
    InputT* sK_bufs[2] = {smem.qk_phase.smem_K_tile0.data(), smem.qk_phase.smem_K_tile1.data()};

    // DET-REDUCE (mirrors dense_fp8/splitkv_mla.cu): per-warp private
    // softmax-denominator partials. Sized 1 (never referenced) in the
    // default path; ~1 KB static smem when enabled.
    constexpr int NUM_CONSUMER_WARPS = 4;
    __shared__ float sL_partial[DETERMINISTIC ? T::BLOCK_SIZE_M * NUM_CONSUMER_WARPS : 1];

    TiledMmaFp8_64x64 tiled_mma_fp8;
    TiledMma64x64 tiled_mma_bf16;

    constexpr int O_ELEMS_PER_THREAD = (T::BLOCK_SIZE_M * T::HEAD_DIM_V + T::NUM_THREADS - 1) / T::NUM_THREADS;
    float rO[O_ELEMS_PER_THREAD];

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;

    constexpr int q_tile_elems = T::BLOCK_SIZE_M * T::K_TILE_DIM;
    constexpr int q_per_thread = (q_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        // --- Block range for this partition ---
        int topk_blocks = (params.topk + T::TOPK_BLOCK_SIZE - 1) / T::TOPK_BLOCK_SIZE;
        int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : topk_blocks;
        bool is_no_split = (start_block_idx == 0 && end_block_idx == topk_blocks);

        // SWA: only first partition (start_block_idx==0) handles SWA
        bool handle_swa = (start_block_idx == 0);
        int swa_len = handle_swa ? __ldg(params.swa_seqlens + batch_idx) : 0;
        int swa_blocks = (swa_len + T::TOPK_BLOCK_SIZE - 1) / T::TOPK_BLOCK_SIZE;

        int num_compressed_iters = end_block_idx - start_block_idx;
        int total_iters = num_compressed_iters + swa_blocks;

        // --- Init softmax state ---
        if (thread_idx < T::BLOCK_SIZE_M) { sM[thread_idx] = MAX_INIT_VAL_SM; sL[thread_idx] = 0.0f; sScale[thread_idx] = 1.0f; }
        #pragma unroll
        for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) rO[i] = 0.0f;
        __syncthreads();

        int start_head_idx = head_block_idx * T::BLOCK_SIZE_M;

        // --- Q FP8 quantization: compute per-head scales ---
        // Read BF16 Q_NOPE, compute per-head amax, store scale = amax / FP8_MAX
        {
            if (thread_idx < T::BLOCK_SIZE_M) sQscales[thread_idx] = 0.0f;
            __syncthreads();

            if (params.q_scales) {
                // Pre-computed scales provided — layout: [b, s_q, h_q]
                if (thread_idx < T::BLOCK_SIZE_M && start_head_idx + thread_idx < params.h_q) {
                    int idx = batch_idx * params.s_q * params.h_q +
                              s_q_idx * params.h_q + start_head_idx + thread_idx;
                    sQscales[thread_idx] = __ldg(params.q_scales + idx);
                }
            } else {
                // In-kernel dynamic per-head quantization
                // 256 threads, 64 heads × 512 dims = 32768 elems → 128 per thread
                const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q_nope) +
                    batch_idx * params.stride_q_b + s_q_idx * params.stride_q_s_q +
                    start_head_idx * params.stride_q_h_q;

                // Each group of 4 threads handles one head (512/4 = 128 dims per thread)
                int head_local = thread_idx / 4;
                int thr_in_head = thread_idx % 4;
                constexpr int DIMS_PER_THR = T::HEAD_DIM_NOPE / 4;  // 128
                float my_max = 0.0f;
                if (start_head_idx + head_local < params.h_q) {
                    const __nv_bfloat16* h_ptr = q_ptr + head_local * T::HEAD_DIM_NOPE;
                    #pragma unroll 4
                    for (int d = 0; d < DIMS_PER_THR; ++d) {
                        float v = fabsf((float)__ldg(h_ptr + thr_in_head * DIMS_PER_THR + d));
                        my_max = fmaxf(my_max, v);
                    }
                }
                // 4-way shuffle reduce within thread group (always within same warp)
                my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, 1));
                my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, 2));
                if (thr_in_head == 0 && head_local < T::BLOCK_SIZE_M)
                    sQscales[head_local] = fmaxf(my_max, 1e-12f) / FP8_E4M3_MAX;
            }
            __syncthreads();
        }

        // Q pointers (BF16) — stride_q_* is in elements of head_dim
        const __nv_bfloat16* q_nope_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q_nope) +
            batch_idx * params.stride_q_b + s_q_idx * params.stride_q_s_q +
            start_head_idx * params.stride_q_h_q;
        // Q_ROPE uses same batch/seq strides but with ROPE head dim
        const __nv_bfloat16* q_rope_ptr = reinterpret_cast<const __nv_bfloat16*>(params.q_rope) +
            batch_idx * params.s_q * params.h_q * T::HEAD_DIM_ROPE +
            s_q_idx * params.h_q * T::HEAD_DIM_ROPE + start_head_idx * T::HEAD_DIM_ROPE;

        // Sparse indices pointer
        const int* gIndices = params.sparse_indices +
            batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;

        //======================================================================
        // Main loop: compressed blocks + SWA blocks
        //======================================================================
        for (int iter = 0; iter < total_iters; ++iter) {
            bool is_swa = (iter >= num_compressed_iters);

            // --- Determine gather params ---
            const int* cur_indices = nullptr;   // sparse mode
            int swa_token_start = 0;            // swa mode
            const char* cache_ptr;
            int pbs;
            int64_t bstride;
            int rstride;

            if (!is_swa) {
                int block_idx = start_block_idx + iter;
                cur_indices = gIndices + block_idx * T::TOPK_BLOCK_SIZE;
                cache_ptr = params.compressed_kv;
                pbs = params.compressed_page_block_size;
                bstride = params.stride_compressed_block;
                rstride = params.stride_compressed_row;
            } else {
                swa_token_start = (iter - num_compressed_iters) * T::TOPK_BLOCK_SIZE;
                cache_ptr = params.swa_kv;
                pbs = params.swa_page_block_size;
                bstride = params.stride_swa_block;
                rstride = params.stride_swa_row;
            }

            // --- Valid mask (producer warps) ---
            if (is_producer) {
                int ptid = thread_idx - 128;
                if (!is_swa) {
                    for (int i = ptid; i < T::TOPK_BLOCK_SIZE; i += 128)
                        sValid[i] = (__ldg(cur_indices + i) >= 0);
                } else {
                    for (int i = ptid; i < T::TOPK_BLOCK_SIZE; i += 128)
                        sValid[i] = (swa_token_start + i < swa_len);
                }
            }
            __syncthreads();

            //==============================================================
            // DUAL-MMA QK GEMM: FP8 NOPE tiles + BF16 ROPE tile
            //==============================================================
            auto rS = partition_fragment_C(tiled_mma_fp8, Shape<_64, _64>{});
            clear(rS);

            // Bootstrap: producers load K_NOPE[tile 0] + scales, all load Q_NOPE[tile 0]
            if (is_producer) {
                int ptid = thread_idx - 128;
                if (!is_swa) {
                    v4_gather_k_nope<T>(cache_ptr, cur_indices, sValid,
                        sK_bufs[0], sKscales, 0, pbs, bstride, rstride, ptid, true);
                } else {
                    v4_swa_load_k_nope<T>(cache_ptr,
                        params.swa_block_table, params.swa_block_table_stride, batch_idx,
                        pbs, bstride, rstride, swa_token_start, swa_len,
                        sK_bufs[0], sKscales, sValid, 0, ptid, true);
                }
            }
            // All threads: load Q_NOPE[tile 0] as BF16 → quantize to FP8
            {
                auto sQ = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                #pragma unroll
                for (int i = 0; i < q_per_thread; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < q_tile_elems) {
                        int row = idx / T::K_TILE_DIM, col = idx % T::K_TILE_DIM;
                        InputT val(0.0f);
                        if (row + start_head_idx < params.h_q && col < T::HEAD_DIM_NOPE) {
                            float qf = (float)__ldg(q_nope_ptr + row * T::HEAD_DIM_NOPE + col);
                            float inv_s = 1.0f / sQscales[row];
                            qf *= inv_s;
                            qf = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, qf));
                            val = InputT(qf);
                        }
                        sQ(row, col) = val;
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
                            InputT val(0.0f);
                            if (row + start_head_idx < params.h_q && gk < T::HEAD_DIM_NOPE) {
                                float qf = (float)__ldg(q_nope_ptr + row * T::HEAD_DIM_NOPE + gk);
                                float inv_s = 1.0f / sQscales[row];
                                qf *= inv_s;
                                qf = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, qf));
                                val = InputT(qf);
                            }
                            sQ(row, col) = val;
                        }
                    }
                    __syncthreads();
                }

                // Producers prefetch next K_NOPE tile
                if (is_producer && k_tile + 1 < T::NUM_NOPE_TILES) {
                    int ptid = thread_idx - 128;
                    if (!is_swa) {
                        v4_gather_k_nope<T>(cache_ptr, cur_indices, sValid,
                            sK_bufs[(k_tile + 1) % 2], sKscales,
                            (k_tile + 1) * T::K_TILE_DIM, pbs, bstride, rstride, ptid, false);
                    } else {
                        v4_swa_load_k_nope<T>(cache_ptr,
                            params.swa_block_table, params.swa_block_table_stride, batch_idx,
                            pbs, bstride, rstride, swa_token_start, swa_len,
                            sK_bufs[(k_tile + 1) % 2], sKscales, sValid,
                            (k_tile + 1) * T::K_TILE_DIM, ptid, false);
                    }
                }
                // Consumers: FP8 MMA on current tile
                if (is_consumer) {
                    auto sQ_t = make_tensor(make_smem_ptr(smem.qk_phase.smem_Q_nope.data()), typename T::SmemLayoutQ{});
                    auto sK_t = make_tensor(make_smem_ptr(sK_bufs[k_tile % 2]), typename T::SmemLayoutK{});
                    cute_gemm_ss<false>(tiled_mma_fp8, sQ_t, sK_t, rS, thread_idx);
                }
                __syncthreads();
            }

            //==============================================================
            // Post-QK dequant of the FP8 NOPE accumulation ONLY:
            // scores *= Q_scale[row] * K_scale[col]. MUST run BEFORE the
            // BF16 ROPE MMA below — the rope tiles are raw BF16 (never
            // quantized), so multiplying a combined nope+rope accumulator
            // by the FP8 scales erased the rope term (qs·ks ~ 1e-5;
            // DeepSeek-V4 rope magnitudes dwarf the nope amax — found by
            // the ticket-H V4 golden boot, engine score == 448-dot only).
            // Register-only + reads sKscales (stable since tile 0): no
            // extra sync needed.
            //==============================================================
            if (is_consumer)
                fragment_qk_dequant(rS, sQscales, sKscales, thread_idx);

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
                            cutlass::bfloat16_t(__ldg(q_rope_ptr + row * T::HEAD_DIM_ROPE + col)) :
                            cutlass::bfloat16_t(0.0f);
                    }
                }
                if (is_producer) {
                    int ptid = thread_idx - 128;
                    if (!is_swa) {
                        v4_gather_k_rope<T>(cache_ptr, cur_indices, sValid,
                            smem.qk_phase.smem_K_rope.data(), pbs, bstride, rstride, ptid);
                    } else {
                        v4_swa_load_k_rope<T>(cache_ptr,
                            params.swa_block_table, params.swa_block_table_stride, batch_idx,
                            pbs, bstride, rstride, swa_token_start, sValid,
                            smem.qk_phase.smem_K_rope.data(), ptid);
                    }
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
            // Standard softmax (QK dequant already applied to the NOPE
            // accumulation BEFORE the rope MMA — see above)
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
                float rescale;
                if (new_max <= MAX_INIT_VAL_SM) {
                    // Fully-masked tile (every index in this 64-tile is -1):
                    // new_max stayed at the init sentinel. The tile must be
                    // an online-softmax NO-OP — keep the running max and
                    // rescale by 1 (the old code computed exp2(old − (-1e33))
                    // = inf, exploding sL/rO for any topk padded by whole
                    // tiles; sub-tile -1 padding was unaffected).
                    sM[thread_idx] = old_max;
                    rescale = 1.0f;
                } else {
                    rescale = (old_max <= MAX_INIT_VAL_SM)
                        ? 1.0f : exp2f((old_max - new_max) * LOG2E);
                }
                sL[thread_idx] *= rescale;
                sScale[thread_idx] = rescale;
                // DET-REDUCE: zero this row's per-warp partial slots
                // (covered by the __syncthreads below before the consumer
                // warps write them). Fully-masked tiles leave them 0.
                if constexpr (DETERMINISTIC) {
                    #pragma unroll
                    for (int w = 0; w < NUM_CONSUMER_WARPS; ++w)
                        sL_partial[thread_idx * NUM_CONSUMER_WARPS + w] = 0.0f;
                }
            }
            __syncthreads();

            if (is_consumer)
                fragment_masked_exp_sum<DETERMINISTIC>(
                    rS, sValid, sM, sL, thread_idx, T::BLOCK_SIZE_M,
                    sL_partial, NUM_CONSUMER_WARPS);
            __syncthreads();

            // DET-REDUCE: fixed-index cross-warp combine into the running
            // denominator (arrival-order atomicAdd made sL — and thus lse
            // and the 1/sL output normalization — carry run-to-run f32-lsb
            // jitter; bf16 rounding hid it until a near-boundary value
            // flipped, the V4 decode run-to-run drift). No extra sync: the
            // next cross-thread sL read is gated by the P-quant sync below.
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
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int row = (thread_idx + i * T::NUM_THREADS) / T::HEAD_DIM_V;
                if (row < T::BLOCK_SIZE_M) rO[i] *= sScale[row];
            }

            //==============================================================
            // Per-row FP8 P quantization
            //==============================================================
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

            //==============================================================
            // PV GEMM — V4: V at offset 644, V_scale at offset 1156
            //==============================================================
            constexpr int V_TILE_DIM = 64;
            constexpr int v_tile_elems = T::TOPK_BLOCK_SIZE * V_TILE_DIM;
            constexpr int v_tile_per_thread = (v_tile_elems + T::NUM_THREADS - 1) / T::NUM_THREADS;

            // Precompute per-element V entry base + V scale
            const char* local_entry[v_tile_per_thread];
            float local_vscale[v_tile_per_thread];
            #pragma unroll
            for (int i = 0; i < v_tile_per_thread; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                local_entry[i] = nullptr;
                local_vscale[i] = 0.0f;
                if (idx < v_tile_elems) {
                    int row = idx / V_TILE_DIM;
                    if (sValid[row]) {
                        if (!is_swa) {
                            int ti = __ldg(cur_indices + row);
                            int bi = (unsigned)ti / (unsigned)pbs;
                            int ri = (unsigned)ti % (unsigned)pbs;
                            local_entry[i] = cache_ptr + (int64_t)bi * bstride + (int64_t)ri * rstride;
                        } else {
                            int tpos = swa_token_start + row;
                            int pid = __ldg(params.swa_block_table + batch_idx * params.swa_block_table_stride + tpos / pbs);
                            int ri = tpos % pbs;
                            local_entry[i] = cache_ptr + (int64_t)pid * bstride + (int64_t)ri * rstride;
                        }
                        local_vscale[i] = __ldg((const float*)(local_entry[i] + V4CacheLayout::V_SCALE_OFFSET));
                    }
                }
            }

            for (int v_tile = 0; v_tile < T::NUM_V_TILES; ++v_tile) {
                int v_dim_offset = v_tile * V_TILE_DIM;
                float local_vvals[v_tile_per_thread];
                float local_vamax = 0.0f;

                #pragma unroll
                for (int i = 0; i < v_tile_per_thread; ++i) {
                    float v_deq = 0.0f;
                    if (local_entry[i] != nullptr) {
                        int idx = thread_idx + i * T::NUM_THREADS;
                        int v_col = v_dim_offset + idx % V_TILE_DIM;
                        if (v_col < T::HEAD_DIM_V) {
                            v_deq = float(InputT(ldg_fp8((const fp8*)(local_entry[i] + V4CacheLayout::V_NOPE_OFFSET) + v_col)))
                                    * local_vscale[i];
                        }
                    }
                    local_vvals[i] = v_deq;
                    local_vamax = fmaxf(local_vamax, fabsf(v_deq));
                }

                // Warp-reduce amax
                #pragma unroll
                for (int off = 16; off > 0; off /= 2)
                    local_vamax = fmaxf(local_vamax, __shfl_xor_sync(0xffffffff, local_vamax, off));

                if (thread_idx % 32 == 0) sPVtile[thread_idx / 32] = local_vamax;
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

                auto sVt_store = make_tensor(
                    make_smem_ptr(smem.smem_V.data() + v_tile * T::V_tile_elems),
                    typename T::SmemLayoutVTile{});
                #pragma unroll
                for (int i = 0; i < v_tile_per_thread; ++i) {
                    int idx = thread_idx + i * T::NUM_THREADS;
                    if (idx < v_tile_elems) {
                        int vrow = idx / V_TILE_DIM, vci = idx % V_TILE_DIM;
                        float val = local_vvals[i] * v_inv_scale;
                        val = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, val));
                        sVt_store(vci, vrow) = InputT(val);
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
        }  // end main loop

        //======================================================================
        // Epilogue
        //======================================================================
        int num_valid = min(params.h_q - start_head_idx, (int)T::BLOCK_SIZE_M);

        if (is_no_split) {
            cutlass::bfloat16_t* o_ptr = params.out +
                batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                start_head_idx * params.stride_o_h_q;
            #pragma unroll
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V, col = idx % T::HEAD_DIM_V;
                if (row < num_valid) {
                    float inv = sL[row] == 0.0f ? 0.0f : (1.0f / sL[row]);
                    o_ptr[row * params.stride_o_h_q + col] = cutlass::bfloat16_t(rO[i] * inv);
                }
            }
            if (thread_idx < num_valid) {
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
            for (int i = 0; i < O_ELEMS_PER_THREAD; ++i) {
                int idx = thread_idx + i * T::NUM_THREADS;
                int row = idx / T::HEAD_DIM_V, col = idx % T::HEAD_DIM_V;
                if (row < num_valid)
                    oa[row * params.stride_o_accum_h_q + col] = rO[i] * (sL[row] == 0.0f ? 0.0f : (1.0f / sL[row]));
            }
            if (thread_idx < num_valid) {
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
template<int NUM_HEADS>
void run_csa_fp8_decode_kernel(const CsaFp8DecodeParams &params) {
    using T = Traits;
    static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    dim3 grid(NUM_M_BLOCKS, params.s_q, params.num_sm_parts);
    if (params.deterministic_reduce) {
        auto kernel = &csa_fp8_decode_sm120_kernel<NUM_HEADS, true>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, T::NUM_THREADS, smem_size, params.stream>>>(params);
    } else {
        auto kernel = &csa_fp8_decode_sm120_kernel<NUM_HEADS, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        kernel<<<grid, T::NUM_THREADS, smem_size, params.stream>>>(params);
    }
}

}  // namespace sm120::decode::csa_fp8
