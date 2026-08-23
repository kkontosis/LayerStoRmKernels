#pragma once
// Portions derived from FlashMLA (https://github.com/deepseek-ai/FlashMLA;
// MIT License, Copyright (c) 2025 DeepSeek) — see THIRD_PARTY_NOTICES.md.

//==============================================================================
// SM120 Sparse MLA Attention - Common Configuration
//
// Shared constants and types for all sparse attention kernels on SM120.
// These match the SM90/SM100 sparse attention conventions from FlashMLA.
//==============================================================================

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

namespace sm120::sparse {

// Model types - matches FlashMLA params.h
enum class ModelType {
    V32,
    MODEL1
};

//==============================================================================
// Dimension constants for head64 MLA
//==============================================================================
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_ROPE = 64;

// Per-model K dimensions
template<ModelType M>
struct ModelConfig;

template<>
struct ModelConfig<ModelType::V32> {
    static constexpr int HEAD_DIM_K = 576;
    static constexpr int HEAD_DIM_NOPE = 512;  // 576 - 64
    static constexpr int QUANT_TILE_SIZE = 128;
    static constexpr int NUM_SCALES = 4;        // float32 scales
    // V3.2 SnapMLA: [512 FP8 nope | 4B float32 scale | 64 BF16 rope] = 644 bytes/token
    static constexpr int SNAPMLA_BYTES_PER_TOKEN = HEAD_DIM_NOPE + 4 + HEAD_DIM_ROPE * 2;
    // V3.2 FlashMLA: 512 FP8 nope + 4 float32 scales + 64 bf16 rope = 656 bytes/token
    static constexpr int FLASHMLA_BYTES_PER_TOKEN = 656;
};

template<>
struct ModelConfig<ModelType::MODEL1> {
    static constexpr int HEAD_DIM_K = 512;
    static constexpr int HEAD_DIM_NOPE = 448;  // 512 - 64
    static constexpr int QUANT_TILE_SIZE = 64;
    static constexpr int NUM_SCALES = 8;        // fp8_e8m0 scales (7 + 1 padding)
    // MODEL1 SnapMLA: [448 FP8 nope | 4B float32 scale | 64 BF16 rope] = 580 bytes/token
    static constexpr int SNAPMLA_BYTES_PER_TOKEN = HEAD_DIM_NOPE + 4 + HEAD_DIM_ROPE * 2;
    // MODEL1 FlashMLA: 448 FP8 nope + 64 bf16 rope + 8 fp8_e8m0 scales = 584 bytes/token
    static constexpr int FLASHMLA_BYTES_PER_TOKEN = 448 + 64 * 2 + 8;
};

//==============================================================================
// Tiling constants (must precede CacheAlignmentCheck which uses K_TILE_DIM)
//==============================================================================
static constexpr int K_TILE_DIM = 64;             // K-dimension tile for outer tiling

//==============================================================================
// Compile-time alignment checks for interleaved SnapMLA cache format
//
// Cache row: [d_c FP8 | float32 scale (4B) | d_rope BF16]
// Ensures any new ModelConfig respects alignment requirements:
//   - float32 scale at offset d_c must be 4-byte aligned
//   - BF16 ROPE at offset d_c+4 must be 2-byte aligned
//   - NOPE must tile evenly into K_TILE_DIM for the FP8 MMA loop
//   - ROPE must equal K_TILE_DIM for a single BF16 MMA tile
//==============================================================================
template<ModelType M>
struct CacheAlignmentCheck {
    using MC = ModelConfig<M>;

    // float32 scale at byte offset HEAD_DIM_NOPE must be 4-byte aligned
    static_assert(MC::HEAD_DIM_NOPE % 4 == 0,
        "HEAD_DIM_NOPE must be 4-byte aligned for float32 scale in interleaved cache");

    // BF16 ROPE at byte offset HEAD_DIM_NOPE+4 must be 2-byte aligned
    static_assert((MC::HEAD_DIM_NOPE + 4) % 2 == 0,
        "HEAD_DIM_NOPE+4 must be 2-byte aligned for BF16 ROPE in interleaved cache");

    // Row stride must preserve float32 alignment across rows
    static_assert(MC::SNAPMLA_BYTES_PER_TOKEN % 4 == 0,
        "SnapMLA row size must be 4-byte aligned to preserve float32 scale alignment across rows");

    // NOPE dims must tile evenly into K_TILE_DIM for FP8 MMA loop
    static_assert(MC::HEAD_DIM_NOPE % K_TILE_DIM == 0,
        "HEAD_DIM_NOPE must be a multiple of K_TILE_DIM for even FP8 tiling");

    // ROPE dims must equal one K_TILE_DIM for single BF16 MMA tile
    static_assert(HEAD_DIM_ROPE == K_TILE_DIM,
        "HEAD_DIM_ROPE must equal K_TILE_DIM for single BF16 ROPE tile");

    // HEAD_DIM_K = HEAD_DIM_NOPE + HEAD_DIM_ROPE
    static_assert(MC::HEAD_DIM_K == MC::HEAD_DIM_NOPE + HEAD_DIM_ROPE,
        "HEAD_DIM_K must equal HEAD_DIM_NOPE + HEAD_DIM_ROPE");
};

// Force instantiation for both models
template struct CacheAlignmentCheck<ModelType::V32>;
template struct CacheAlignmentCheck<ModelType::MODEL1>;

//==============================================================================
// Sparse attention block sizes
//==============================================================================
static constexpr int TOPK_BLOCK_SIZE = 64;     // Tokens per topk block
static constexpr int BLOCK_M = 64;             // Query tile size (head dim)

//==============================================================================
// SM120-specific constraints
//==============================================================================
static constexpr int SM120_SMEM_BYTES = 101376;  // 99KB = 101,376 bytes
static constexpr int NUM_THREADS = 256;           // 8 warps
static constexpr int NUM_WARPS = 8;

//==============================================================================
// Tiling constants (continued)
//==============================================================================

//==============================================================================
// Numeric constants
//==============================================================================
static constexpr float NEGATIVE_INFINITY = -1e30f;
static constexpr float LOG2E = 1.4426950408889634f;

// MMA atom configurations (CuTe)
// BF16: SM80_16x8x16_F32BF16BF16F32_TN — used for absorbed prefill (bf16 KV)
static constexpr int MMA_BF16_M = 16;
static constexpr int MMA_BF16_N = 8;
static constexpr int MMA_BF16_K = 16;

// FP8: SM89_16x8x32_F32F8F8F32_TN — used for FP8 decode (SnapMLA path)
// K=32 means 2× fewer inner loop iterations vs BF16 K=16
static constexpr int MMA_FP8_M = 16;
static constexpr int MMA_FP8_N = 8;
static constexpr int MMA_FP8_K = 32;

// FP8 max value for dynamic quantization (e4m3: max = 448.0)
static constexpr float FP8_E4M3_MAX = 448.0f;

}  // namespace sm120::sparse
