#pragma once

//==============================================================================
// SM120 Sparse Prefill - Configuration (head64)
//
// Configuration for the absorbed sparse prefill kernel on SM120.
// Based on SM100 sparse prefill config but adapted for SM120 constraints:
// - 99KB shared memory (vs SM100's 227KB)
// - CuTe SM80_16x8x16 MMA atoms (no UMMA/TMEM)
// - No cluster multicast
//
// "Absorbed" means the KV cache stores the compressed latent directly,
// so K = [NOPE (bf16), ROPE (bf16)] and V = NOPE portion.
//==============================================================================

#include "../../../../components/sparse_config.h"

namespace sm120::prefill::sparse::head64 {

using sm120::sparse::ModelType;

// Dimensions
static constexpr int D_Q = 576;
static constexpr int D_K = 576;
static constexpr int D_V = 512;
static constexpr int D_NOPE = 512;    // = D_V for absorbed path
static constexpr int D_ROPE = 64;     // = D_Q - D_V

// Block sizes
static constexpr int B_H = 64;        // Query heads per CTA
static constexpr int B_TOPK = 64;     // Tokens per topk block

// Thread configuration: 256 threads (8 warps)
// Warps 0-3: CuTe MMA compute (QK + PV)
// Warps 4-5: K/V loading
// Warps 6-7: softmax/masking + RoPE loading
static constexpr int NUM_THREADS = 256;
static constexpr int NUM_WARPS = 8;
static constexpr int NUM_COMPUTE_WARPS = 4;

// K-dimension tiling
static constexpr int K_TILE_DIM = 64;
static constexpr int NUM_K_TILES_NOPE = D_NOPE / K_TILE_DIM;  // 8
static constexpr int NUM_K_TILES_ROPE = D_ROPE / K_TILE_DIM;   // 1
static constexpr int NUM_K_TILES = (D_K + K_TILE_DIM - 1) / K_TILE_DIM;  // 9

// Number of K buffers (absorbed path: KV is same tensor)
static constexpr int NUM_BUFS = 2;

// Numeric constants
static constexpr float MAX_INIT_VAL = -1e30f;
static constexpr float LOG2E = 1.4426950408889634f;

}  // namespace sm120::prefill::sparse::head64
