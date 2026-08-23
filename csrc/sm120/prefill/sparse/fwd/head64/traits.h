#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "config.h"
#include "../../../../components/helpers.h"

namespace sm120::prefill::sparse::head64 {

using namespace cute;
using sm120::sparse::SmemLayoutTile64x64;

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

}  // namespace sm120::prefill::sparse::head64
