#pragma once

//==============================================================================
// SM120 Sparse Prefill - Small TopK Configuration (head64)
//
// Specialization of the sparse prefill kernel for small topk values.
// Key differences from the standard prefill (Kernel 2):
// - Fewer K buffers (NUM_BUFS=2 vs standard's 2, but tighter loops)
// - Dual mode: supports both prefill AND decode-with-splitKV
// - Optimized for small topk (tighter loops, less buffer overhead)
//
// Based on SM100 fwd_for_small_topk/head128 but adapted for:
// - head64 (not head128)
// - SM120 99KB smem (no TMEM/UMMA)
// - CuTe SM80_16x8x16 MMA instead of UMMA
//==============================================================================

#include "../../../../components/sparse_config.h"

namespace sm120::prefill::sparse::small_topk::head64 {

using sm120::sparse::ModelType;

// Dimensions
static constexpr int D_V = 512;
static constexpr int D_NOPE = 448;     // For MODEL1
static constexpr int D_ROPE = 64;

// Block sizes
static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;

// Thread configuration
static constexpr int NUM_THREADS = 256;
static constexpr int NUM_WARPS = 8;
static constexpr int NUM_COMPUTE_WARPS = 4;

// K-dimension tiling
static constexpr int K_TILE_DIM = 64;

// Small topk: fewer buffers for tighter memory
static constexpr int NUM_BUFS = 2;

// Numeric constants
static constexpr float MAX_INIT_VAL = -1e30f;
static constexpr float LOG2E = 1.4426950408889634f;
static constexpr float NEGATIVE_INFINITY = -1e30f;

}  // namespace sm120::prefill::sparse::small_topk::head64
