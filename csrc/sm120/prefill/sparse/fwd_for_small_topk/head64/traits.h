#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "config.h"
#include "../../../../components/helpers.h"

namespace sm120::prefill::sparse::small_topk::head64 {

using namespace cute;
using sm120::sparse::SmemLayoutTile64x64;
using sm120::sparse::SmemLayoutTile64x64Fp8;

//==============================================================================
// BF16 Traits — used by prefill mode (absorbed, bf16 KV)
//==============================================================================
template<int D_QK>
struct Traits {
    using InputT = cutlass::bfloat16_t;
    static constexpr int HEAD_DIM_K = D_QK;
    static constexpr int HEAD_DIM_V = D_V;
    static constexpr int BLOCK_SIZE_M = B_H;
    static constexpr int TOPK_BLOCK_SIZE = B_TOPK;
    static constexpr int K_TILE = K_TILE_DIM;
    static constexpr int NUM_K_TILES_TOTAL = (D_QK + K_TILE_DIM - 1) / K_TILE_DIM;
    static constexpr int NUM_V_TILES = D_V / 64;

    using SmemLayoutQ = SmemLayoutTile64x64;
    using SmemLayoutK = SmemLayoutTile64x64;
    using SmemLayoutP = SmemLayoutTile64x64;
    using SmemLayoutVTile = SmemLayoutTile64x64;

    static constexpr size_t Q_tile_elems = cosize_v<SmemLayoutQ>;
    static constexpr size_t K_tile_elems = cosize_v<SmemLayoutK>;
    static constexpr size_t P_elems = cosize_v<SmemLayoutP>;
    static constexpr size_t V_tile_elems = cosize_v<SmemLayoutVTile>;
    static constexpr size_t V_total_elems = NUM_V_TILES * V_tile_elems;
    static constexpr size_t PV_tile_elems = B_H * 64;

    struct SharedMemoryPlan {
        union {
            struct {
                cute::array_aligned<InputT, Q_tile_elems> smem_Q_tile;
                cute::array_aligned<InputT, K_tile_elems> smem_K_tile0;
                cute::array_aligned<InputT, K_tile_elems> smem_K_tile1;
            } qk_phase;
            cute::array_aligned<InputT, V_total_elems> smem_V;
        };
        cute::array_aligned<InputT, P_elems> smem_P;
        cute::array_aligned<float, PV_tile_elems> smem_pv_tile;
        cute::array_aligned<float, B_H> smem_M;
        cute::array_aligned<float, B_H> smem_L;
        cute::array_aligned<float, B_H> smem_scale;
        bool is_kv_valid[B_TOPK];
    };

    static constexpr size_t SharedMemSize = sizeof(SharedMemoryPlan);
    static_assert(SharedMemSize <= 101376, "Shared memory exceeds SM120 limit (99KB)");
};

//==============================================================================
// FP8 Traits — used by decode mode (SnapMLA FP8-native path)
//
// FP8 smem (1 byte/elem) cuts buffer sizes in half vs BF16:
//   QK Phase: Q[64,64] FP8=4KB + K[64,64] FP8 ×2=8KB → 12KB
//   PV Phase: V[64,512] FP8=32KB (overlaps QK via union)
//   P[64,64] FP8=4KB + P_scale[1]=4B
//   PV_tile[64,64] f32=16KB
//   Scales + stats: ~1.5KB
//   Total: max(12, 32) + 4 + 16 + 1.5 ≈ 53KB << 99KB
//==============================================================================
template<int D_QK>
struct TraitsFp8 {
    using InputT = cutlass::float_e4m3_t;
    using RopeT = cutlass::bfloat16_t;
    static constexpr int HEAD_DIM_K = D_QK;
    static constexpr int HEAD_DIM_V = D_V;
    static constexpr int HEAD_DIM_NOPE = D_QK - D_ROPE;
    static constexpr int HEAD_DIM_ROPE = D_ROPE;
    static constexpr int BLOCK_SIZE_M = B_H;
    static constexpr int TOPK_BLOCK_SIZE = B_TOPK;
    static constexpr int K_TILE = K_TILE_DIM;
    static constexpr int NUM_K_TILES_TOTAL = (D_QK + K_TILE_DIM - 1) / K_TILE_DIM;
    static constexpr int NUM_NOPE_TILES = (HEAD_DIM_NOPE + K_TILE_DIM - 1) / K_TILE_DIM;
    static constexpr int NUM_V_TILES = D_V / 64;

    // FP8 smem layouts (NOPE, V, P)
    using SmemLayoutQ = SmemLayoutTile64x64Fp8;
    using SmemLayoutK = SmemLayoutTile64x64Fp8;
    using SmemLayoutP = SmemLayoutTile64x64Fp8;
    using SmemLayoutVTile = SmemLayoutTile64x64Fp8;

    // BF16 smem layouts (ROPE)
    using SmemLayoutQRope = SmemLayoutTile64x64;
    using SmemLayoutKRope = SmemLayoutTile64x64;

    static constexpr size_t Q_nope_elems = cosize_v<SmemLayoutQ>;
    static constexpr size_t Q_rope_elems = cosize_v<SmemLayoutQRope>;
    static constexpr size_t K_tile_elems = cosize_v<SmemLayoutK>;
    static constexpr size_t K_rope_elems = cosize_v<SmemLayoutKRope>;
    static constexpr size_t P_elems = cosize_v<SmemLayoutP>;
    static constexpr size_t V_tile_elems = cosize_v<SmemLayoutVTile>;
    static constexpr size_t V_total_elems = NUM_V_TILES * V_tile_elems;
    static constexpr size_t PV_tile_elems = B_H * 64;

    struct SharedMemoryPlan {
        union {
            struct {
                cute::array_aligned<InputT, Q_nope_elems> smem_Q_nope;
                cute::array_aligned<RopeT, Q_rope_elems> smem_Q_rope;
                cute::array_aligned<InputT, K_tile_elems> smem_K_tile0;
                cute::array_aligned<InputT, K_tile_elems> smem_K_tile1;
                cute::array_aligned<RopeT, K_rope_elems> smem_K_rope;
            } qk_phase;
            cute::array_aligned<InputT, V_total_elems> smem_V;
        };
        cute::array_aligned<InputT, P_elems> smem_P;
        cute::array_aligned<float, PV_tile_elems> smem_pv_tile;
        cute::array_aligned<float, B_H> smem_M;
        cute::array_aligned<float, B_H> smem_L;
        cute::array_aligned<float, B_H> smem_scale;
        cute::array_aligned<float, B_TOPK> smem_K_scales;
        cute::array_aligned<float, B_H> smem_Q_scales;
        cute::array_aligned<float, B_H> smem_P_scales;
        bool is_kv_valid[B_TOPK];
    };

    static constexpr size_t SharedMemSize = sizeof(SharedMemoryPlan);
    static_assert(SharedMemSize <= 101376, "Shared memory exceeds SM120 limit (99KB)");
};

}  // namespace sm120::prefill::sparse::small_topk::head64
