#pragma once
// Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
// MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.

//==============================================================================
// SM120 Sparse MLA Attention - CuTe MMA Helpers + Pipeline Support
//
// Architecture:
//   Warps 0-3 (consumer): CuTe SM80 MMA (QK, PV) + softmax
//   Warps 4-7 (producer): FP8 gather + dequant
//
// Pipeline: Double-buffered K tiles with role-split between __syncthreads():
//   Between sync N and sync N+1:
//     Producers gather K[tile i+1] → buf[(i+1)%2]
//     Consumers MMA on K[tile i] from buf[i%2]
//   Both groups do useful work — gather latency hidden behind MMA compute.
//
// Softmax: Smem-based. CuTe fragment materialized to smem via partition_C
//   retile mapping (handles multi-atom TiledMMA correctly). Softmax runs
//   on the smem scores, then P is written to swizzled smem for PV GEMM.
//==============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "sparse_config.h"
#include "dequant.h"

namespace sm120::sparse {

using namespace cute;

//==============================================================================
// CuTe MMA Type Definitions — BF16 (for absorbed prefill, bf16 KV)
//==============================================================================
using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;

using TiledMma64x64 = TiledMMA<
    MmaAtom,
    Layout<Shape<_2, _2, _1>>,
    Tile<_64, _64, _16>>;

//==============================================================================
// CuTe MMA Type Definitions — FP8 (SnapMLA decode path)
//
// Uses CUTLASS-provided SM89_16x8x32_F32E4M3E4M3F32_TN from mma_sm89.hpp
// m16n8k32, e4m3 inputs, f32 accumulator
// K=32 means 2× fewer inner loop iterations vs BF16 K=16
// Works on SM89+ (Ada Lovelace, Blackwell)
//==============================================================================

#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_traits_sm89.hpp>

using MmaAtomFp8 = MMA_Atom<cute::SM89_16x8x32_F32E4M3E4M3F32_TN>;

// TiledMMA for 64x64 output tile with FP8 inputs, 4 warps (2x2 layout)
// K=32 per MMA step → inner loop iterates K_TILE_DIM/32 times
using TiledMmaFp8_64x64 = TiledMMA<
    MmaAtomFp8,
    Layout<Shape<_2, _2, _1>>,
    Tile<_64, _64, _32>>;

//==============================================================================
// Swizzled Shared Memory Layouts
//==============================================================================

// BF16 smem layout (for absorbed prefill path)
static constexpr int SmemAlignmentBf16 = 128 / sizeof_bits_v<cutlass::bfloat16_t>;  // 8

using SmemSwizzle = Swizzle<3, 3, 3>;

using SmemLayoutAtom = decltype(
    composition(SmemSwizzle{},
                Layout<Shape<_8, Int<SmemAlignmentBf16>>,
                       Stride<Int<SmemAlignmentBf16>, _1>>{}));

using SmemLayoutTile64x64 = decltype(
    tile_to_shape(SmemLayoutAtom{}, Shape<_64, _64>{}));

// FP8 smem layout (for SnapMLA decode path)
// FP8 elements are 1 byte → alignment = 128/8 = 16 elements per 128-bit line
// Swizzle<3,4,3> matches TRT-LLM's Ada FP8 GEMM pattern
static constexpr int SmemAlignmentFp8 = 128 / sizeof_bits_v<cutlass::float_e4m3_t>;  // 16

using SmemSwizzleFp8 = Swizzle<3, 4, 3>;

using SmemLayoutAtomFp8 = decltype(
    composition(SmemSwizzleFp8{},
                Layout<Shape<_16, Int<SmemAlignmentFp8>>,
                       Stride<Int<SmemAlignmentFp8>, _1>>{}));

using SmemLayoutTile64x64Fp8 = decltype(
    tile_to_shape(SmemLayoutAtomFp8{}, Shape<_64, _64>{}));

// FP8 64x64 tile for Q (used by QK GEMM A operand)
using SmemLayoutTile64x64Fp8Q = SmemLayoutTile64x64Fp8;

//==============================================================================
// CuTe GEMM: smem A × smem B → register C
//
// Uses retile_A/B to create smem views compatible with MMA, then
// copies to register fragments and calls gemm per K-step.
//==============================================================================
template<bool zero_init, typename TiledMma, typename TensorA, typename TensorB, typename TensorC>
__forceinline__ __device__ void cute_gemm_ss(
    TiledMma& tiled_mma,
    TensorA const& sA, TensorB const& sB, TensorC& rC,
    int thread_idx
) {
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    // Smem views partitioned for this thread's MMA slices
    auto tCsA = thr_mma.partition_A(sA);
    auto tCsB = thr_mma.partition_B(sB);

    // Register fragments matching the MMA layout
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);

    if constexpr (zero_init) { clear(rC); }

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
        // Load from smem to registers
        cute::copy(tCsA(_, _, k), tCrA(_, _, k));
        cute::copy(tCsB(_, _, k), tCrB(_, _, k));
        // Execute MMA with register operands
        cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), rC);
    }
}

//==============================================================================
// Fragment → smem materialization using CuTe partition_C mapping
//
// Correctly handles multi-atom TiledMMA fragment layout by using the
// MMA's own thread-to-output mapping. Stores f32 fragment to linear
// smem [M, N] for softmax processing.
//==============================================================================
template<typename TiledMma, typename TensorFrag>
__forceinline__ __device__ void fragment_to_scores_smem(
    TiledMma& tiled_mma,
    TensorFrag const& rC,
    float* __restrict__ sScores,
    int thread_idx, int M, int N
) {
    auto sC = make_tensor(make_smem_ptr(sScores),
                          make_layout(make_shape(M, N), make_stride(N, 1)));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCsC = thr_mma.partition_C(sC);

    CUTE_UNROLL
    for (int i = 0; i < size(rC); ++i) {
        tCsC(i) = rC(i);
    }
}

//==============================================================================
// Smem scores → bf16 P in swizzled smem (for PV GEMM B operand)
//==============================================================================
__device__ __forceinline__
void scores_to_P_smem(
    const float* __restrict__ sScores,
    cutlass::bfloat16_t* __restrict__ sP_ptr,
    int thread_idx, int num_threads
) {
    constexpr int elems = BLOCK_M * TOPK_BLOCK_SIZE;
    constexpr int per_thread = (elems + 256 - 1) / 256;
    auto sP = make_tensor(make_smem_ptr(sP_ptr), SmemLayoutTile64x64{});

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = thread_idx + i * num_threads;
        if (idx < elems) {
            sP(idx / TOPK_BLOCK_SIZE, idx % TOPK_BLOCK_SIZE) =
                cutlass::bfloat16_t(sScores[idx]);
        }
    }
}

//==============================================================================
// PV fragment → linear smem (f32) for all-thread scatter
//==============================================================================
template<typename TiledMma, typename TensorFrag>
__forceinline__ __device__ void pv_fragment_to_smem(
    TiledMma& tiled_mma,
    TensorFrag const& rC,
    float* __restrict__ sPVtile,
    int thread_idx, int M, int N
) {
    // Same as fragment_to_scores_smem — uses partition_C for correct mapping
    auto sC = make_tensor(make_smem_ptr(sPVtile),
                          make_layout(make_shape(M, N), make_stride(N, 1)));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCsC = thr_mma.partition_C(sC);

    CUTE_UNROLL
    for (int i = 0; i < size(rC); ++i) {
        tCsC(i) = rC(i);
    }
}

//==============================================================================
// Masked online softmax (smem-based)
//==============================================================================
__device__ __forceinline__
void masked_online_softmax(
    float* __restrict__ sScores,
    const bool* __restrict__ sValid,
    float* __restrict__ sM,
    float* __restrict__ sL,
    float* __restrict__ sScale,
    float scale_softmax,
    int thread_idx, int num_threads
) {
    constexpr int score_elems = BLOCK_M * TOPK_BLOCK_SIZE;
    constexpr int per_thread = (score_elems + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < per_thread; ++i) {
        int idx = thread_idx + i * num_threads;
        if (idx < score_elems) sScores[idx] *= scale_softmax;
    }
    __syncthreads();

    if (thread_idx < BLOCK_M) {
        float row_max = NEGATIVE_INFINITY;
        #pragma unroll
        for (int j = 0; j < TOPK_BLOCK_SIZE; ++j) {
            if (sValid[j])
                row_max = fmaxf(row_max, sScores[thread_idx * TOPK_BLOCK_SIZE + j]);
        }
        float old_max = sM[thread_idx];
        float new_max = fmaxf(old_max, row_max);
        float rescale = (old_max == NEGATIVE_INFINITY) ? 1.0f :
                        exp2f((old_max - new_max) * LOG2E);
        sL[thread_idx] *= rescale;
        sM[thread_idx] = new_max;
        sScale[thread_idx] = rescale;
    }
    __syncthreads();

    if (thread_idx < BLOCK_M) {
        float row_max = sM[thread_idx];
        float row_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < TOPK_BLOCK_SIZE; ++j) {
            int idx = thread_idx * TOPK_BLOCK_SIZE + j;
            if (sValid[j]) {
                float exp_val = exp2f((sScores[idx] - row_max) * LOG2E);
                sScores[idx] = exp_val;
                row_sum += exp_val;
            } else {
                sScores[idx] = 0.0f;
            }
        }
        sL[thread_idx] += row_sum;
    }
    __syncthreads();
}

//==============================================================================
// Valid mask generation
//==============================================================================
__device__ __forceinline__
void generate_valid_mask(
    const int* __restrict__ indices, int block_idx, int topk_length,
    bool* __restrict__ sValid, int thread_idx, int num_threads
) {
    const int base = block_idx * TOPK_BLOCK_SIZE;
    for (int i = thread_idx; i < TOPK_BLOCK_SIZE; i += num_threads) {
        int global_idx = base + i;
        sValid[i] = (global_idx < topk_length) && (__ldg(indices + global_idx) >= 0);
    }
}

//==============================================================================
// Utilities
//==============================================================================
__host__ __device__ __forceinline__
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

//==============================================================================
// Atomic float max via CAS (correct for all float values including negatives)
// Needed because integer atomicMax on reinterpreted float bits fails for negatives.
//==============================================================================
__device__ __forceinline__ void atomic_max_f32(float* addr, float val) {
    int* addr_int = reinterpret_cast<int*>(addr);
    int old = *addr_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

//==============================================================================
// Register-resident softmax helpers
//
// Operate on a CuTe C-fragment (f32, [64, 64] output) and recover per-element
// (row, col) via make_identity_tensor + partition_C.
// Call from compute warps (thread_idx in [0, 127]) only.
// __syncthreads() is the call site's responsibility.
//
// Uses shuffle-reduce instead of atomics:
// 1. Each thread accumulates per-row partial max/sum in 4 register slots
//    (each thread owns elements from exactly 4 distinct rows in the 2×2 warp layout)
// 2. shfl_xor(1) + shfl_xor(2) reduces across the 4 lanes sharing each row within a warp
// 3. Lane leader (lane_id % 4 == 0) writes one value per row to sM/sL
//    → Only 2 writes per row (one from each warp in a pair), vs ~64 atomics before
// 4. These 2 writes still use atomics but contention is negligible (2 vs 64)
//==============================================================================

// Per-row shuffle-reduced max from masked f32 fragment.
template<typename TensorFrag>
__forceinline__ __device__ void fragment_masked_row_max(
    TensorFrag const& rS,
    const bool* __restrict__ sValid,
    float* __restrict__ sM,
    int thread_idx, int /* block_m */
) {
    TiledMmaFp8_64x64 tiled_mma;
    auto ref = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCoords = thr_mma.partition_C(ref);

    const int lane_id = thread_idx % 32;

    // Each thread owns elements from 4 distinct rows. Accumulate per-row max.
    float local_max[4] = {NEGATIVE_INFINITY, NEGATIVE_INFINITY, NEGATIVE_INFINITY, NEGATIVE_INFINITY};
    int local_rows[4] = {-1, -1, -1, -1};
    int num_rows = 0;

    CUTE_UNROLL
    for (int i = 0; i < size(rS); ++i) {
        auto c = tCoords(i);
        int row = (int)get<0>(c);
        int col = (int)get<1>(c);
        if (!sValid[col]) continue;

        // Find slot for this row (at most 4 distinct rows per thread)
        int slot = -1;
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s) {
            if (local_rows[s] == row) { slot = s; break; }
        }
        if (slot == -1) { slot = num_rows++; local_rows[slot] = row; }
        local_max[slot] = fmaxf(local_max[slot], rS(i));
    }

    // Shuffle-reduce across the 4 lanes sharing each row (lane_id ^ {1,2})
    // Then lane leader writes to sM (at most 2 atomics per row across warp pairs)
    // NOTE: All threads must participate in __shfl_xor_sync(0xffffffff, ...) —
    // skipping via `continue` when local_rows[s]<0 causes deadlock when some
    // threads in a warp have no valid columns (e.g. s_kv=1).
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
        float val = local_max[s];
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
        if (local_rows[s] >= 0 && (lane_id % 4) == 0) {
            atomic_max_f32(&sM[local_rows[s]], val);
        }
    }
}

// Per-element exp2f in-place with shuffle-reduced row-sum accumulation.
//
// DETERMINISTIC=false (default): each warp's lane-leader atomic-adds its
//   shuffle-reduced per-row partial into sL[row]. Cross-warp accumulation
//   order is warp-scheduling-dependent (fp32 non-associativity) → run-to-run
//   denominator jitter. This is the legacy / byte-identical path.
// DETERMINISTIC=true: each consumer warp writes its partial to a PRIVATE slot
//   sL_partial[row * num_consumer_warps + warp_idx] (NO contention). The caller
//   then __syncthreads() and sums sL_partial[row][0..num_consumer_warps-1] into
//   sL[row] in FIXED index order — bit-reproducible (vLLM block_sum style).
//   sL_partial must be zero-initialized by the caller before this call (warps
//   that do not own a given row leave their slot untouched). The fixed-order
//   combine into sL is the CALLER's responsibility (see splitkv_mla.cu /
//   prefill phase1.cuh). The within-warp __shfl reduction is unchanged.
template<bool DETERMINISTIC = false, typename TensorFrag>
__forceinline__ __device__ void fragment_masked_exp_sum(
    TensorFrag& rS,
    const bool* __restrict__ sValid,
    const float* __restrict__ sM,
    float* __restrict__ sL,
    int thread_idx, int /* block_m */,
    float* __restrict__ sL_partial = nullptr,   // [block_m * num_consumer_warps], DETERMINISTIC only
    int num_consumer_warps = 0                   // partial-slot stride, DETERMINISTIC only
) {
    TiledMmaFp8_64x64 tiled_mma;
    auto ref = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCoords = thr_mma.partition_C(ref);

    const int lane_id = thread_idx % 32;
    const int warp_idx = thread_idx / 32;

    // Pass 1: compute exp values in-place, accumulate per-row partial sums
    float local_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int local_rows[4] = {-1, -1, -1, -1};
    int num_rows = 0;

    CUTE_UNROLL
    for (int i = 0; i < size(rS); ++i) {
        auto c = tCoords(i);
        int row = (int)get<0>(c);
        int col = (int)get<1>(c);

        if (sValid[col]) {
            float exp_val = exp2f((rS(i) - sM[row]) * LOG2E);
            rS(i) = exp_val;

            int slot = -1;
            CUTE_UNROLL
            for (int s = 0; s < 4; ++s) {
                if (local_rows[s] == row) { slot = s; break; }
            }
            if (slot == -1) { slot = num_rows++; local_rows[slot] = row; }
            local_sum[slot] += exp_val;
        } else {
            rS(i) = 0.0f;
        }
    }

    // Pass 2: shuffle-reduce sums, lane leader writes to sL
    // NOTE: All threads must participate in __shfl_xor_sync — see row_max note.
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
        float val = local_sum[s];
        val += __shfl_xor_sync(0xffffffff, val, 1);
        val += __shfl_xor_sync(0xffffffff, val, 2);
        if (local_rows[s] >= 0 && (lane_id % 4) == 0) {
            if constexpr (DETERMINISTIC) {
                // Private slot per (row, warp): no contention, fixed-order
                // combine into sL done by the caller after __syncthreads().
                sL_partial[local_rows[s] * num_consumer_warps + warp_idx] = val;
            } else {
                atomicAdd(&sL[local_rows[s]], val);
            }
        }
    }
}

//==============================================================================
// Fragment → swizzled bf16 smem (P store after online softmax)
// Converts f32 C-fragment to bf16 using partition_C on the swizzled P layout.
//==============================================================================
template<typename SmemLayout, typename TensorFrag>
__forceinline__ __device__ void fragment_to_smem_bf16(
    TensorFrag const& rC,
    cutlass::bfloat16_t* __restrict__ sP_ptr,
    SmemLayout const& layout,
    int thread_idx, int /* M */, int /* N */
) {
    TiledMma64x64 tiled_mma;
    auto sP = make_tensor(make_smem_ptr(sP_ptr), layout);
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCsP = thr_mma.partition_C(sP);

    CUTE_UNROLL
    for (int i = 0; i < size(rC); ++i) {
        tCsP(i) = cutlass::bfloat16_t(rC(i));
    }
}

//==============================================================================
// Fragment → linear f32 smem (wrapper for pv_fragment_to_smem without tiled_mma arg)
// Uses TiledMmaFp8_64x64 for PV GEMM fragments (must match the MMA that produced rC).
//==============================================================================
template<typename TensorFrag>
__forceinline__ __device__ void fragment_to_smem_f32(
    TensorFrag const& rC,
    float* __restrict__ sOut,
    int thread_idx, int M, int N
) {
    TiledMmaFp8_64x64 tiled_mma;
    pv_fragment_to_smem(tiled_mma, rC, sOut, thread_idx, M, N);
}

//==============================================================================
// SnapMLA: Post-QK dequant + standard softmax + per-row FP8 P quantization
//
// Corrected pipeline matching SGLang-FluentLLM reference (flashmla-fp8):
//   1. QK GEMM: FP8 Q_NOPE @ FP8 K_NOPE^T + BF16 Q_ROPE @ BF16 K_ROPE^T → f32
//   2. Post-QK dequant: scores[i,j] *= Q_scale[i] * K_scale[j]
//   3. Standard online softmax (no V-scale fusion — V has no per-token scale in kernel)
//   4. Per-row dynamic FP8 quantization of P → P_fp8 with per-row P_scale[]
//   5. FP8 PV GEMM: P_fp8 @ V_fp8 → f32 output
//   6. Post-PV: output[row] /= (L[row] * P_scale[row])
//   7. LSE: log(L * P_scale) + M / log2e
//
// Tiered numerical constants (matching reference):
//   MAX_INIT_VAL_SM = -1e30  (softmax M initialization)
//   MAX_INIT_VAL    = -1e33  (masking, must satisfy MAX_INIT_VAL * sm_scale < MAX_INIT_VAL_SM)
//   P_SCALE_EPS     = 1e-26  (P amax floor to prevent division by zero)
//==============================================================================

static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL_MASK = -1e33f;
static constexpr float P_SCALE_EPS = 1e-26f;

// Post-QK dequant: applies Q_scale[row] * K_scale[col] to QK scores fragment.
// Call from consumer warps only, after QK GEMM accumulation is complete.
template<typename TensorFrag>
__forceinline__ __device__ void fragment_qk_dequant(
    TensorFrag& rS,
    const float* __restrict__ sQscales,  // [BLOCK_SIZE_M] per-head Q scales
    const float* __restrict__ sKscales,  // [TOPK_BLOCK_SIZE] per-token K scales
    int thread_idx
) {
    TiledMmaFp8_64x64 tiled_mma;
    auto ref = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCoords = thr_mma.partition_C(ref);

    CUTE_UNROLL
    for (int i = 0; i < size(rS); ++i) {
        int row = (int)get<0>(tCoords(i));
        int col = (int)get<1>(tCoords(i));
        rS(i) *= sQscales[row] * sKscales[col];
    }
}

//==============================================================================
// Per-row FP8 P quantization (matching reference row-by-row dynamic scaling)
//
// TWO-PHASE design (avoids __syncthreads inside consumer-only code):
//   Phase 1 (fragment_fp8_compute_row_scales): per-row amax via shuffle-reduce
//     + atomic to sP_scales[row]. Caller inits sP_scales[0..M-1]=0 then syncs.
//   Between phases: caller syncs, then threads 0..M-1 convert amax to scale.
//   Phase 2 (fragment_fp8_row_quantize_store): read per-row scale, quantize,
//     store to FP8 swizzled smem.
//
// Caller protocol:
//   1. threads 0..M-1: sP_scales[i] = 0;  __syncthreads();
//   2. consumers: fragment_fp8_compute_row_scales(rS, sP_scales, tid);  __syncthreads();
//   3. threads 0..M-1: sP_scales[i] = max(sP_scales[i], P_SCALE_EPS) / FP8_MAX;  __syncthreads();
//   4. consumers: fragment_fp8_row_quantize_store(rS, sP, layout, sP_scales, tid);  __syncthreads();
//==============================================================================

// Phase 1: Per-row amax from fragment via shuffle-reduce + atomic.
template<typename TensorFrag>
__forceinline__ __device__ void fragment_fp8_compute_row_scales(
    TensorFrag const& rC,
    float* __restrict__ sP_scales,       // [BLOCK_SIZE_M] output: per-row amax
    int thread_idx
) {
    TiledMmaFp8_64x64 tiled_mma;
    auto ref = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCoords = thr_mma.partition_C(ref);

    const int lane_id = thread_idx % 32;

    float local_amax[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int local_rows[4] = {-1, -1, -1, -1};
    int num_rows = 0;

    CUTE_UNROLL
    for (int i = 0; i < size(rC); ++i) {
        int row = (int)get<0>(tCoords(i));
        float absval = fabsf(rC(i));

        int slot = -1;
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s) {
            if (local_rows[s] == row) { slot = s; break; }
        }
        if (slot == -1) { slot = num_rows++; local_rows[slot] = row; }
        local_amax[slot] = fmaxf(local_amax[slot], absval);
    }

    // Shuffle-reduce across 4 lanes sharing each row, then atomic to smem
    // NOTE: All threads must participate in __shfl_xor_sync — see row_max note.
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
        float val = local_amax[s];
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
        if (local_rows[s] >= 0 && (lane_id % 4) == 0) {
            atomic_max_f32(&sP_scales[local_rows[s]], val);
        }
    }
}

// Phase 2: Per-row quantize fragment → FP8 swizzled smem using per-row scales.
// sP_scales[row] must contain the SCALE (not amax) at this point.
template<typename SmemLayout, typename TensorFrag>
__forceinline__ __device__ void fragment_fp8_row_quantize_store(
    TensorFrag const& rC,
    cutlass::float_e4m3_t* __restrict__ sP_ptr,
    SmemLayout const& layout,
    const float* __restrict__ sP_scales,  // [BLOCK_SIZE_M] per-row scales
    int thread_idx
) {
    TiledMmaFp8_64x64 tiled_mma;
    auto ref = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCoords = thr_mma.partition_C(ref);

    auto sP = make_tensor(make_smem_ptr(sP_ptr), layout);
    auto tCsP = thr_mma.partition_C(sP);

    CUTE_UNROLL
    for (int i = 0; i < size(rC); ++i) {
        int row = (int)get<0>(tCoords(i));
        float scale = sP_scales[row];
        float inv_scale = 1.0f / scale;
        float val = rC(i) * inv_scale;
        val = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, val));
        tCsP(i) = cutlass::float_e4m3_t(val);
    }
}

// Post-PV dequant: applies per-row P_scale to PV GEMM output fragment.
// Call from consumer warps only, after PV GEMM.
template<typename TensorFrag>
__forceinline__ __device__ void fragment_pv_row_dequant(
    TensorFrag& rPV,
    const float* __restrict__ sP_scales,  // [BLOCK_SIZE_M] per-row P scales
    int thread_idx
) {
    TiledMmaFp8_64x64 tiled_mma;
    auto ref = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    auto tCoords = thr_mma.partition_C(ref);

    CUTE_UNROLL
    for (int i = 0; i < size(rPV); ++i) {
        int row = (int)get<0>(tCoords(i));
        rPV(i) *= sP_scales[row];
    }
}

}  // namespace sm120::sparse
