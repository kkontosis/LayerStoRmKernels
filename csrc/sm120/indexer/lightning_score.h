#pragma once
/***************************************************************************************************
 * Lightning Indexer Scoring: FP8 multi-head dot-product scoring
 *
 * Computes importance scores for each compressed block by:
 *   1. Multi-head dot product: q_proj[h,d] @ indexer_k_cache[n,h,d]^T → dots[n,h]
 *      (index_n_heads=64 heads, index_head_dim=128)
 *   2. ReLU activation: dots = max(dots, 0)
 *   3. Score projection: scores[n] = sum_h(dots[n,h] * score_proj[h])
 *
 * Following SGLang/vLLM/TRT-LLM approach: indexer K cache is stored in FP8 E4M3
 * with per-block scales, halving memory bandwidth vs BF16. Q stays BF16 (loaded
 * once to shared memory). Score computation uses FP32 accumulation.
 *
 * Grid: (ceil(num_blocks / BLOCKS_PER_CTA), 1, 1)
 * Block: 256 threads (8 warps)
 *
 * Reference: tests/test_v4_reference.py::ref_lightning_score()
 * Ported from: SGLang nsa_indexer.py (fp8_mqa_logits), vLLM sparse_attn_indexer.py
 * (SGLang: Apache-2.0, Copyright 2023-2024 SGLang Team; vLLM: Apache-2.0,
 *  Copyright contributors to the vLLM project — see THIRD_PARTY_NOTICES.md)
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace sm120::indexer {

struct LightningScoreParams {
    // Query projected into indexer space (BF16, loaded once to smem)
    const __nv_bfloat16* __restrict__ q_proj;          // [INDEX_N_HEADS, INDEX_HEAD_DIM]

    // Indexer K cache (FP8 E4M3) — per compressed block
    const __nv_fp8_e4m3* __restrict__ indexer_k_cache;  // [num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM]

    // Per-block K scales (FP32): dequant = fp8_val * k_scale[block]
    const float* __restrict__  k_scales;                // [num_blocks]

    // Score aggregation weights (FP32)
    const float* __restrict__ score_proj;               // [INDEX_N_HEADS]

    // Output: importance score per block (FP32)
    float* __restrict__ scores_out;                     // [num_blocks]

    // Dimensions
    int num_blocks;          // number of compressed blocks to score
    int index_n_heads;       // 64
    int index_head_dim;      // 128
};

void run_lightning_score(const LightningScoreParams& params, cudaStream_t stream);

}  // namespace sm120::indexer
