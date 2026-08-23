#pragma once

//==============================================================================
// SM120 Sparse FP8 Decode - Configuration
//
// Configuration for the sparse FP8 MLA decode kernel on SM120.
// Based on SM120 dense decode config but adapted for sparse attention
// with FP8 KV cache and topk-based token selection.
//==============================================================================

#include "../../components/sparse_config.h"

namespace sm120::decode::sparse_fp8 {

using sm120::sparse::ModelType;
using sm120::sparse::ModelConfig;

template<ModelType MODEL_TYPE>
struct Config {
    using MC = ModelConfig<MODEL_TYPE>;

    // Tile sizes
    static constexpr int BLOCK_SIZE_M = 64;       // Query tile rows (= h_q per CTA)
    static constexpr int TOPK_BLOCK_SIZE = 64;    // Tokens per topk block
    static constexpr int HEAD_DIM_K = MC::HEAD_DIM_K;
    static constexpr int HEAD_DIM_V = 512;
    static constexpr int HEAD_DIM_NOPE = MC::HEAD_DIM_NOPE;
    static constexpr int HEAD_DIM_ROPE = 64;
    static constexpr int NUM_SCALES = MC::NUM_SCALES;

    // Thread configuration
    static constexpr int NUM_THREADS = 256;       // 8 warps
    static constexpr int NUM_WARPS = 8;
    static constexpr int NUM_COMPUTE_WARPS = 4;   // Warps 0-3: CuTe MMA compute
    static constexpr int NUM_PRODUCE_WARPS = 4;   // Warps 4-7: gather + dequant

    // K-dimension tiling to fit 99KB smem
    static constexpr int K_TILE_DIM = 64;
    static constexpr int NUM_K_TILES = (HEAD_DIM_K + K_TILE_DIM - 1) / K_TILE_DIM;
    // V32: 576/64 = 9 tiles, MODEL1: 512/64 = 8 tiles

    // Number of K buffers for double-buffering
    static constexpr int NUM_K_BUFS = 2;
};

}  // namespace sm120::decode::sparse_fp8
