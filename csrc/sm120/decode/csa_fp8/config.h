#pragma once

//==============================================================================
// SM120 CSA FP8 Decode — Configuration
//
// V4 CSA attention: head_dim=512 (NOPE), qk_rope_head_dim=64, d_v=512.
// Same tiling as SnapMLA sparse FP8 (64×64 tiles, 256 threads, dual MMA).
//==============================================================================

#include "../../components/sparse_config.h"

namespace sm120::decode::csa_fp8 {

using namespace sm120::sparse;

struct Config {
    // V4 head dimensions — identical to V3.2 V32 numerically
    static constexpr int HEAD_DIM_NOPE = 512;
    static constexpr int HEAD_DIM_ROPE = 64;
    static constexpr int HEAD_DIM_K = HEAD_DIM_NOPE + HEAD_DIM_ROPE;  // 576
    static constexpr int HEAD_DIM_V = 512;

    // Tile sizes
    static constexpr int BLOCK_SIZE_M = 64;       // Query heads per CTA
    static constexpr int TOPK_BLOCK_SIZE = 64;    // KV tokens per tile

    // Thread configuration
    static constexpr int NUM_THREADS = 256;       // 8 warps
    static constexpr int NUM_WARPS = 8;
    static constexpr int NUM_COMPUTE_WARPS = 4;   // Warps 0-3: CuTe MMA compute
    static constexpr int NUM_PRODUCE_WARPS = 4;   // Warps 4-7: gather + dequant

    // K-dimension tiling
    static constexpr int K_TILE_DIM = 64;
    static constexpr int NUM_NOPE_TILES = HEAD_DIM_NOPE / K_TILE_DIM;  // 8
    static constexpr int NUM_K_BUFS = 2;          // Double-buffered K tiles

    // V4 has separate K and V scales (unlike MLA where K_scale covers both)
    static constexpr int NUM_K_SCALES = 1;        // Single float32 scale per entry
    static constexpr int NUM_V_SCALES = 1;

    // SWA window
    static constexpr int SWA_MAX_TOKENS = 128;
    static constexpr int SWA_MAX_BLOCKS = (SWA_MAX_TOKENS + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;  // 2
};

}  // namespace sm120::decode::csa_fp8
