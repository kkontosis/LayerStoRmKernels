#pragma once

//==============================================================================
// SM120 Sparse FP8 Decode - Traits (SnapMLA Path, Dual MMA)
//
// Mixed-precision: NOPE in FP8 (SM89_16x8x32 MMA), ROPE in BF16 (SM80_16x8x16 MMA).
// Both accumulate into the same f32 fragment (same SM80_16x8_Row CLayout).
//
// Smem budget with dual MMA:
//   QK Phase: Q_nope[64,64] FP8=4KB + Q_rope[64,64] BF16=8KB
//           + K_nope[64,64] FP8 ×2=8KB + K_rope[64,64] BF16=8KB → 28KB
//   PV Phase: V[64,512] FP8=32KB (overlaps QK via union)
//   P[64,64] FP8=4KB
//   PV_tile[64,64] f32=16KB
//   Scales + stats: ~2KB
//   Total: max(28, 32) + 4 + 16 + 2 = ~54KB << 99KB
//==============================================================================

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "config.h"
#include "../../components/helpers.h"

namespace sm120::decode::sparse_fp8 {

using namespace cute;
using sm120::sparse::SmemLayoutAtomFp8;
using sm120::sparse::SmemLayoutTile64x64Fp8;
using sm120::sparse::SmemLayoutAtom;
using sm120::sparse::SmemLayoutTile64x64;

template<ModelType MODEL_TYPE>
struct Traits {
    using Cfg = Config<MODEL_TYPE>;
    using InputT = cutlass::float_e4m3_t;    // FP8 e4m3 for K_NOPE, V, Q_NOPE
    using RopeT = cutlass::bfloat16_t;       // BF16 for K_ROPE, Q_ROPE
    using ComputeT = cutlass::bfloat16_t;

    static constexpr int BLOCK_SIZE_M = Cfg::BLOCK_SIZE_M;
    static constexpr int TOPK_BLOCK_SIZE = Cfg::TOPK_BLOCK_SIZE;
    static constexpr int HEAD_DIM_K = Cfg::HEAD_DIM_K;
    static constexpr int HEAD_DIM_V = Cfg::HEAD_DIM_V;
    static constexpr int HEAD_DIM_NOPE = Cfg::HEAD_DIM_NOPE;
    static constexpr int HEAD_DIM_ROPE = Cfg::HEAD_DIM_ROPE;
    static constexpr int NUM_SCALES = Cfg::NUM_SCALES;
    static constexpr int NUM_THREADS = Cfg::NUM_THREADS;
    static constexpr int NUM_WARPS = Cfg::NUM_WARPS;
    static constexpr int NUM_COMPUTE_WARPS = Cfg::NUM_COMPUTE_WARPS;
    static constexpr int NUM_PRODUCE_WARPS = Cfg::NUM_PRODUCE_WARPS;
    static constexpr int K_TILE_DIM = Cfg::K_TILE_DIM;
    static constexpr int NUM_K_TILES = Cfg::NUM_K_TILES;
    static constexpr int NUM_K_BUFS = Cfg::NUM_K_BUFS;
    static constexpr int NUM_V_TILES = HEAD_DIM_V / 64;  // 8

    // Number of FP8 NOPE K-tiles (excludes ROPE tile)
    static constexpr int NUM_NOPE_TILES = (HEAD_DIM_NOPE + K_TILE_DIM - 1) / K_TILE_DIM;

    // FP8 swizzled smem layouts (NOPE, V, P)
    using SmemLayoutQ = SmemLayoutTile64x64Fp8;
    using SmemLayoutK = SmemLayoutTile64x64Fp8;
    using SmemLayoutP = SmemLayoutTile64x64Fp8;
    using SmemLayoutVTile = SmemLayoutTile64x64Fp8;

    // BF16 swizzled smem layouts (ROPE)
    using SmemLayoutQRope = SmemLayoutTile64x64;
    using SmemLayoutKRope = SmemLayoutTile64x64;

    // Element counts
    static constexpr size_t Q_nope_elems = cosize_v<SmemLayoutQ>;
    static constexpr size_t Q_rope_elems = cosize_v<SmemLayoutQRope>;
    static constexpr size_t K_tile_elems = cosize_v<SmemLayoutK>;
    static constexpr size_t K_rope_elems = cosize_v<SmemLayoutKRope>;
    static constexpr size_t P_elems = cosize_v<SmemLayoutP>;
    static constexpr size_t V_tile_elems = cosize_v<SmemLayoutVTile>;
    static constexpr size_t V_total_elems = NUM_V_TILES * V_tile_elems;
    static constexpr size_t PV_tile_elems = BLOCK_SIZE_M * 64;

    //==========================================================================
    // Shared Memory Plan — Dual MMA SnapMLA (~54KB << 99KB)
    //==========================================================================
    struct SharedMemoryPlan {
        union {
            struct {
                cute::array_aligned<InputT, Q_nope_elems> smem_Q_nope;      // 4KB FP8
                cute::array_aligned<RopeT, Q_rope_elems> smem_Q_rope;       // 8KB BF16
                cute::array_aligned<InputT, K_tile_elems> smem_K_tile0;     // 4KB FP8
                cute::array_aligned<InputT, K_tile_elems> smem_K_tile1;     // 4KB FP8
                cute::array_aligned<RopeT, K_rope_elems> smem_K_rope;       // 8KB BF16
            } qk_phase;                                                      // 28KB total

            cute::array_aligned<InputT, V_total_elems> smem_V;              // 32KB FP8
        };

        cute::array_aligned<InputT, P_elems> smem_P;                        // 4KB FP8
        cute::array_aligned<float, PV_tile_elems> smem_pv_tile;             // 16KB

        // Softmax statistics
        cute::array_aligned<float, BLOCK_SIZE_M> smem_M;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_L;
        cute::array_aligned<float, BLOCK_SIZE_M> smem_scale;

        // Per-token K quantization scales
        cute::array_aligned<float, TOPK_BLOCK_SIZE> smem_K_scales;          // 256B
        // Per-head Q quantization scales
        cute::array_aligned<float, BLOCK_SIZE_M> smem_Q_scales;             // 256B
        // Per-row P quantization scales
        cute::array_aligned<float, BLOCK_SIZE_M> smem_P_scales;             // 256B

        // Valid mask
        bool is_kv_valid[TOPK_BLOCK_SIZE];
    };

    static constexpr size_t SharedMemSize = sizeof(SharedMemoryPlan);
    static_assert(SharedMemSize <= 101376, "Shared memory exceeds SM120 limit (99KB)");
};

}  // namespace sm120::decode::sparse_fp8
