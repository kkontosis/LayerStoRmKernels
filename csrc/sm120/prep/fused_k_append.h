#pragma once
/***************************************************************************************************
 * Fused-K-Append: Per-token KV quantization + RoPE pre-scaling + paged cache write
 *
 * Single kernel that:
 * 1. Computes per-token amax over c_KV (compressed latent) → scale = amax / 448.0
 * 2. Quantizes c_KV to FP8 e4m3: KV_fp8 = c_KV / scale
 * 3. Pre-scales K_ROPE by inverse content scale: K_R' = K_R / scale
 *    (SnapMLA Key Step 1 — unifies domains for uniform FP8 QK GEMM)
 * 4. Writes pre-scaled K_ROPE as BF16 (preserves ROPE precision, matches reference)
 * 5. Writes [d_c FP8 bytes | 1 float32 scale | d_rope BF16 bytes] to paged KV cache
 *
 * Grid: (num_tokens, 1, 1) — one CTA per token being appended
 * Block: 256 threads
 *
 * Runs once per generated token per layer during autoregressive decoding.
 * Integrates PagedAttention-style non-contiguous writes.
 *
 * Optimized: single-pass amax+quantize (reads c_KV data once, not twice),
 * vectorized __nv_bfloat162 loads (halves memory transactions).
 *
 * SnapMLA per-token format per row in KV cache page:
 *   [d_c FP8 bytes] [1 float32 scale (4 bytes)] [d_rope BF16 bytes]
 *   Total: d_c + 4 + d_rope*2 bytes per token
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

#ifndef SM120_PREP_FP8_E4M3_MAX
#define SM120_PREP_FP8_E4M3_MAX
static constexpr float FP8_E4M3_MAX = 448.0f;
#endif

struct FusedKAppendParams {
    // Input: new token's latent + RoPE (bf16). Rows may be STRIDED: consecutive
    // tokens' rows are src_stride_ckv / src_stride_rope ELEMENTS apart
    // (0 = tight, i.e. d_c / d_rope). The engine passes both pointers into ONE
    // interleaved [num_tokens, d_c + d_rope] kv_a projection output (row stride
    // d_c + d_rope for both). Assuming tight rows here was exact only at
    // num_tokens == 1 and read garbage for every later token of a multi-token
    // prefill chunk (TD-PREFILL-CHUNK-ATTN).
    const __nv_bfloat16* __restrict__ c_kv;     // [num_tokens, d_c] compressed latent
    const __nv_bfloat16* __restrict__ k_rope;    // [num_tokens, d_rope] RoPE embeddings
    int src_stride_ckv = 0;   // elements between c_kv token rows (0 → d_c)
    int src_stride_rope = 0;  // elements between k_rope token rows (0 → d_rope)

    // Output: paged KV cache
    __nv_fp8_e4m3* __restrict__ kv_cache;        // Paged KV cache base pointer
    int64_t cache_stride_block;                   // Stride between page blocks (bytes)
    int cache_stride_row;                         // Stride between rows within a page (bytes)

    // Page table for non-contiguous writes
    const int* __restrict__ slot_mapping;         // [num_tokens] → flat slot index in cache

    int num_tokens;
    int d_c;          // Compressed latent dim (e.g., 512)
    int d_rope;       // RoPE dim (e.g., 64)
    int page_size;    // Tokens per page block (e.g., 64)

    // Output layout per token row:
    // [d_c FP8 | 1 float32 scale | d_rope BF16]
    // Total row bytes = d_c + 4 + d_rope * 2
};

void run_fused_k_append(const FusedKAppendParams& params, cudaStream_t stream);

}  // namespace sm120::prep
