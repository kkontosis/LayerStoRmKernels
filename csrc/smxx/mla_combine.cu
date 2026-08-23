// Split-KV combine kernel — merges partial attention results from split-KV decode.
//
// Based on FlashMLA (https://github.com/deepseek-ai/FlashMLA)
// Original: github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120 (SM120 fork of github.com/deepseek-ai/FlashMLA): csrc/smxx/mla_combine.cu
// License: MIT, Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md
//
// Modifications:
//   - Param struct renamed: DecodingParams → MlaCombineParams
//   - MAX_SPLITS extended to 192 (was 64) for RTX 5090 (170 SMs) batch=1 coverage
//   - FP16 path disabled by default (we only use BF16)
//
// This kernel is arch-generic — no SM-specific instructions.
// It reads float32 lse_accum / o_accum and writes BF16 merged output.
// It never touches the KV cache.

#include "mla_combine.h"

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <math_constants.h>   // CUDART_INF_F
#include <cmath>              // isnan, isinf

#include "params.h"
#include "utils.h"

using namespace cute;

template<typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
mla_combine_kernel(__grid_constant__ const MlaCombineParams params) {
    // grid_shape: [batch_size, num_q_heads*s_q / BLOCK_SIZE_M]
    // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result
    static_assert(NUM_THREADS/32 == BLOCK_SIZE_M); // The number of warps == block_size_m
    const int batch_idx = blockIdx.x;
    const int m_block_idx = blockIdx.y;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;

    const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
    const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
    const int my_num_splits = end_split_idx - start_split_idx;
    FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);
    if (my_num_splits == 1) {
        return;
    }

    const int num_q_seqs = params.q_seq_per_hk * params.h_k;
    const int num_cur_valid_q_seqs = min(BLOCK_SIZE_M, num_q_seqs - m_block_idx*BLOCK_SIZE_M);
    Tensor gLseAccum = make_tensor(
        make_gmem_ptr((float*)params.softmax_lseaccum_ptr + start_split_idx*num_q_seqs + m_block_idx*BLOCK_SIZE_M),
        Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
        make_stride(num_q_seqs, _1{})
    );
    Tensor gLse = make_tensor(
        make_gmem_ptr((float*)params.softmax_lse_ptr + batch_idx*num_q_seqs + m_block_idx*BLOCK_SIZE_M),
        Shape<Int<BLOCK_SIZE_M>>{},
        Stride<_1>{}
    );

    extern __shared__ float smem_buf[];
    Tensor sLseScale = make_tensor(
        make_smem_ptr(smem_buf),
        Shape<Int<BLOCK_SIZE_M>, Int<MAX_SPLITS>>{},
        Stride<Int<MAX_SPLITS+1>, _1>{} // +1 to avoid bank conflict
    );

    // Wait for the previous kernel (the MLA kernel) to finish
    cudaGridDependencySynchronize();

    // Read gLseAccum into sLseScale
    {
        #pragma unroll 4
        for (int elem_idx = threadIdx.x; elem_idx < my_num_splits*BLOCK_SIZE_M; elem_idx += NUM_THREADS) {
            int split_idx = elem_idx / BLOCK_SIZE_M;
            int seq_idx = elem_idx % BLOCK_SIZE_M;
            sLseScale(seq_idx, split_idx) = seq_idx < num_cur_valid_q_seqs ? gLseAccum(split_idx, seq_idx) : -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (warp_idx >= num_cur_valid_q_seqs)
        return;

    // Warp #i gathers LseAccum for seq #i
    {
        constexpr int NUM_LSE_PER_THREAD = cute::ceil_div(MAX_SPLITS, 32);
        float local_lse[NUM_LSE_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            local_lse[i] = split_idx < my_num_splits ? sLseScale(warp_idx, split_idx) : -CUDART_INF_F;
        }

        float max_lse = -CUDART_INF_F;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            max_lse = max(max_lse, local_lse[i]);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = (isinf(max_lse) && max_lse < 0.f) ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || isnan(sum_lse)) ? CUDART_INF_F : log2f(sum_lse) + max_lse;
        if (lane_idx == 0)
            gLse(warp_idx) = global_lse / (float)M_LOG2E;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            if (split_idx < my_num_splits) sLseScale(warp_idx, split_idx) = exp2f(local_lse[i] - global_lse);
        }
    }

    __syncwarp();

    // Warp #i accumulates activation for seq #i
    {
        const int64_t row_offset_oaccum = (int64_t)(start_split_idx*num_q_seqs+m_block_idx*BLOCK_SIZE_M+warp_idx) * HEAD_DIM_V;
        Tensor gOaccum = make_tensor(
            make_gmem_ptr(reinterpret_cast<float *>(params.oaccum_ptr) + row_offset_oaccum),
            Shape<Int<MAX_SPLITS>, Int<HEAD_DIM_V>>{},
            make_stride(num_q_seqs*HEAD_DIM_V, _1{})
        );

        static_assert(HEAD_DIM_V % 32 == 0);
        constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / 32;
        float result[ELEMS_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            result[i] = 0.0f;

        #pragma unroll 2
        for (int split = 0; split < my_num_splits; ++split) {
            float lse_scale = sLseScale(warp_idx, split);
            if (lse_scale != 0.f) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                    result[i] += lse_scale * gOaccum(split, lane_idx + i*32);
                }
            }
        }

        cudaTriggerProgrammaticLaunchCompletion();

        const int q_seq_idx = m_block_idx*BLOCK_SIZE_M + warp_idx;
        const int k_head_idx = q_seq_idx / params.q_seq_per_hk;
        auto o_ptr = reinterpret_cast<ElementT *>(params.o_ptr) + batch_idx*params.o_batch_stride + k_head_idx*params.o_head_stride + (q_seq_idx%params.q_seq_per_hk)*params.o_row_stride;
        Tensor gO = make_tensor(
            make_gmem_ptr(o_ptr),
            Shape<Int<HEAD_DIM_V>>{},
            Stride<_1>{}
        );

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            gO(lane_idx+i*32) = (ElementT)result[i];
    }
}


// MAX_SPLITS=192 covers RTX 5090 (170 SMs) at batch=1.
// Original FlashMLA: 32/64. SGLang-FluentLLM: up to 160.
// Each tier is a separate template instantiation — only the matching one runs.
// Smem cost: 8 × (MAX_SPLITS+1) × 4 bytes = 6.2KB at 192. Trivial.
#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)       \
    [&] {                                                  \
        if (NUM_SPLITS <= 32) {                            \
            constexpr static int NAME = 32;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 64) {                     \
            constexpr static int NAME = 64;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 128) {                    \
            constexpr static int NAME = 128;               \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 192) {                    \
            constexpr static int NAME = 192;               \
            return __VA_ARGS__();                          \
        } else {                                           \
            FLASH_ASSERT(false);                           \
        }                                                  \
    }()


template<typename ElementT>
void run_mla_combine_kernel(MlaCombineParams &params, cudaStream_t stream) {
    static constexpr int HEAD_DIM_V = 512;
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
        constexpr int BLOCK_SIZE_M = 8;
        constexpr int NUM_THREADS = BLOCK_SIZE_M*32;
        constexpr size_t smem_size = BLOCK_SIZE_M*(NUM_SPLITS+1)*sizeof(float);
        auto combine_kernel = &mla_combine_kernel<ElementT, HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS, NUM_THREADS>;
        CHECK_CUDA(cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t combine_kernel_config = {
            dim3(params.b, cute::ceil_div(params.h_k*params.q_seq_per_hk, BLOCK_SIZE_M), 1),
            dim3(NUM_THREADS, 1, 1),
            smem_size,
            stream,
            attribute,
            1
        };
        cudaLaunchKernelEx(&combine_kernel_config, combine_kernel, params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_mla_combine_kernel<cutlass::bfloat16_t>(MlaCombineParams &params, cudaStream_t stream);
template void run_mla_combine_kernel<float>(MlaCombineParams &params, cudaStream_t stream);
