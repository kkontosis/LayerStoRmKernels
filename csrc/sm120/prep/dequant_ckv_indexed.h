#pragma once
/***************************************************************************************************
 * Dequantize CKV Fused Indexed: Gather + dequant from paged FP8 cache → contiguous BF16
 *
 * Reads selected tokens from the SnapMLA paged KV cache (interleaved FP8 format),
 * dequantizes NOPE and un-scales ROPE by the per-token scale, and writes contiguous
 * BF16 output for consumption by the absorbed BF16 prefill kernel.
 *
 * Used for chunked prefill: past tokens live in the FP8 cache and must be
 * materialized as BF16 before the prefill attention kernel can read them.
 *
 * SnapMLA cache row layout (per token):
 *   [d_c FP8 bytes] [1 float32 scale (4 bytes)] [d_rope BF16 bytes]
 *   ROPE is stored pre-scaled (k_rope / scale) by fused_k_append.
 *
 * Output layout (per fetched token):
 *   [d_c BF16] [d_rope BF16] = d_qk contiguous BF16 values
 *   (NOPE dequantized from FP8: fp8 * scale, ROPE un-scaled: bf16 * scale)
 *
 * Grid:  (num_fetch, 1, 1) — one CTA per fetched token
 * Block: 128 threads (4 warps)
 *
 * Vectorized uint (4-byte) loads for FP8 data, uint2 (8-byte) stores for BF16 output.
 * Per-token scale broadcast via __shfl_sync within each warp.
 *
 * Compare to SGLang-FluentLLM's dequantize_ckv_fused_indexed which reads from 3 separate tensors.
 * Our version reads from a single interleaved paged cache — better locality for random
 * gather patterns (all token data colocated in same cache lines).
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct DequantCKVIndexedParams {
    // Input: paged FP8 KV cache (SnapMLA interleaved format)
    const __nv_fp8_e4m3* __restrict__ kv_cache;
    int64_t cache_stride_block;     // Stride between page blocks (bytes, as FP8 element count)
    int cache_stride_row;           // Stride between rows within a page (bytes, as FP8 element count)
    int page_size;                  // Tokens per page block

    // Token selection
    const int* __restrict__ indices;  // [num_fetch] → flat slot indices in cache
    int num_fetch;

    // Output: contiguous BF16
    __nv_bfloat16* __restrict__ k_out;  // [num_fetch, d_c + d_rope]

    int d_c;        // Compressed latent dim (e.g., 512 for V3.2, 448 for MODEL1)
    int d_rope;     // RoPE dim (e.g., 64)
};

void run_dequant_ckv_fused_indexed(const DequantCKVIndexedParams& params, cudaStream_t stream);

}  // namespace sm120::prep
