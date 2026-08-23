#pragma once

//==============================================================================
// SM120 CSA FP8 Decode — Parameters
//
// DeepSeek V4 Compressed Sparse Attention decode with FP8 KV cache.
// Adapted from sparse_fp8/params.h (SnapMLA) for V4's separate K+V cache
// layout, SWA combine, and single KV head.
//
// V4 cache entry (1160 bytes):
//   [K_NOPE 512B FP8 | K_scale 4B | K_ROPE 128B BF16 | V_NOPE 512B FP8 | V_scale 4B]
//
// Source attribution: adapted from FlashMLA sparse decode (MIT License,
// Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md)
//==============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include "cutlass/bfloat16.h"

#include "../sparse_fp8/params.h"   // DecodingSchedMeta

namespace sm120::decode::csa_fp8 {

using sm120::decode::sparse_fp8::DecodingSchedMeta;

// V4 FP8 cache entry byte offsets
struct V4CacheLayout {
    static constexpr int K_NOPE_OFFSET  = 0;
    static constexpr int K_SCALE_OFFSET = 512;
    static constexpr int K_ROPE_OFFSET  = 516;
    static constexpr int V_NOPE_OFFSET  = 644;
    static constexpr int V_SCALE_OFFSET = 1156;
    static constexpr int ENTRY_BYTES    = 1160;
};

struct CsaFp8DecodeParams {
    int b, s_q;
    int h_q;                            // 128 (Pro) / 64 (Flash)
    float sm_scale;
    float sm_scale_log2;                // sm_scale * log2(e)

    // Q: BF16 — separate NOPE and ROPE arrays
    cutlass::bfloat16_t* __restrict__ q_nope;   // [b, s_q, h_q, 512] BF16
    cutlass::bfloat16_t* __restrict__ q_rope;   // [b, s_q, h_q, 64] BF16
    float* __restrict__ q_scales;               // [b, s_q, h_q] per-head Q FP8 scales, or nullptr

    // Compressed KV cache (FP8, V4 format, 1160 B/entry)
    const char* __restrict__ compressed_kv;     // raw byte pointer
    int compressed_page_block_size;             // entries per page (e.g. 64)
    int64_t stride_compressed_block;            // bytes per page (page_block_size * 1160)
    int stride_compressed_row;                  // bytes per entry (1160)

    // Sparse indices from Lightning Indexer.
    // IMPORTANT: topk MUST be a multiple of TOPK_BLOCK_SIZE (64). The kernel reads
    // indices in full 64-token tiles. If topk is not aligned, the caller must pad
    // sparse_indices to ceil(topk/64)*64 with -1 sentinels and set strides accordingly.
    // (FlashMLA and vLLM enforce the same constraint.)
    const int* __restrict__ sparse_indices;     // [b, s_q, padded_topk] — token-level indices
    int topk;                                   // 1024 (Pro) / 512 (Flash) — must be multiple of 64
    int stride_indices_b, stride_indices_s_q;

    // SWA KV cache (FP8, same V4 format, uncompressed recent tokens)
    const char* __restrict__ swa_kv;            // raw byte pointer
    int swa_page_block_size;
    int64_t stride_swa_block;
    int stride_swa_row;                         // 1160
    const int* __restrict__ swa_block_table;    // [b, max_swa_blocks]
    int swa_block_table_stride;

    // Per-batch sequence info
    const int* __restrict__ swa_seqlens;        // [b] number of SWA tokens (up to 128)

    // Output
    cutlass::bfloat16_t* __restrict__ out;      // [b, s_q, h_q, 512]
    float* __restrict__ lse;                    // [b, s_q, h_q]

    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_lse_b, stride_lse_s_q;

    cudaStream_t stream;

    // Split-KV accumulation (same pattern as SnapMLA)
    float* __restrict__ lse_accum;
    float* __restrict__ o_accum;
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr;
    int* __restrict__ num_splits_ptr;
    int num_sm_parts;

    // DET-REDUCE: bit-reproducible softmax-denominator reduction (fixed-order
    // cross-warp combine instead of arrival-order atomicAdd on sL). Same
    // contract as dense_fp8/sparse_fp8 deterministic_reduce.
    bool deterministic_reduce = false;
};

// Launch wrapper — templated on NUM_HEADS for compile-time grid calculation
template<int NUM_HEADS>
void run_csa_fp8_decode_kernel(const CsaFp8DecodeParams &params);

}  // namespace sm120::decode::csa_fp8
