#pragma once
/***************************************************************************************************
 * TQ Fused-K-Append: Per-token TurboQuant MSE 4-bit quantization + paged cache write
 *
 * Single kernel that:
 * 1. Computes per-token L2 norm of c_KV (compressed latent)
 * 2. Normalizes to unit sphere: c_unit = c_KV / ||c_KV||
 * 3. Applies random rotation: y = c_unit @ Pi^T (Pi is orthogonal, d_c × d_c)
 * 4. Quantizes each rotated coordinate via binary search on codebook boundaries
 *    (Lloyd-Max optimal for Beta distribution, 4-bit = 16 centroids)
 * 5. Packs 4-bit indices: 2 per byte → d_c/2 bytes
 * 6. Writes [packed_nope | fp16_norm | bf16_rope] to paged KV cache
 *
 * TQ cache layout per token:
 *   [d_c/2 uint8 packed] [2B float16 norm] [d_rope*2 BF16 rope]
 *   V3.2: 256 + 2 + 128 = 386 bytes/token (vs SnapMLA FP8: 644)
 *
 * Grid: (num_tokens, NSPLIT) — the rotation (d_c × d_c mat-vec, the dominant
 *   cost) is split across NSPLIT CTAs/token, and each output coordinate's dot is
 *   computed co-operatively by a warp (lanes stride the d_c contraction). At
 *   decode (num_tokens=1) this replaces the old single-CTA (~95 µs floor) with
 *   NSPLIT×8 warps of work (TD-TQ-K-APPEND-SINGLE-CTA). The packed output is
 *   bit-identical to the single-CTA version up to FP rotation-sum reassociation
 *   (goldens are argmax/tolerance; quantization is searchsorted).
 * Block: 256 threads
 *
 * Pi rows are read from global memory with L2 caching. c_unit is in shared memory.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct TqFusedKAppendParams {
    // Input: new token's latent + RoPE (bf16). Rows may be STRIDED: consecutive
    // tokens' rows are src_stride_ckv / src_stride_rope ELEMENTS apart
    // (0 = tight, i.e. d_c / d_rope) — the engine passes pointers into ONE
    // interleaved [num_tokens, d_c + d_rope] kv_a output (see FusedKAppendParams,
    // TD-PREFILL-CHUNK-ATTN).
    const __nv_bfloat16* __restrict__ c_kv;       // [num_tokens, d_c]
    const __nv_bfloat16* __restrict__ k_rope;      // [num_tokens, d_rope]
    int src_stride_ckv = 0;   // elements between c_kv token rows (0 → d_c)
    int src_stride_rope = 0;  // elements between k_rope token rows (0 → d_rope)

    // Output: paged TQ cache
    uint8_t* __restrict__ kv_cache;                 // Paged TQ cache base pointer
    int64_t cache_stride_block;                     // Stride between page blocks (bytes)
    int cache_stride_row;                           // Stride between rows (bytes)

    // Page table
    const int* __restrict__ slot_mapping;           // [num_tokens] → flat slot index

    // Rotation matrix and codebook (device pointers)
    const float* __restrict__ Pi;                   // [d_c, d_c] orthogonal rotation matrix
    const float* __restrict__ centroids;            // [16] Lloyd-Max centroids
    const float* __restrict__ decision_boundaries;  // [15] interior boundaries (sorted)

    int num_tokens;
    int d_c;           // Compressed latent dim (512 for V3.2, 448 for MODEL1)
    int d_rope;        // RoPE dim (64)
    int page_size;     // Tokens per page block
    int num_centroids; // 2^bits = 16 for 4-bit
};

void run_tq_fused_k_append(const TqFusedKAppendParams& params, cudaStream_t stream);

}  // namespace sm120::prep
