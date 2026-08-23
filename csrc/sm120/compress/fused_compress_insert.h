#pragma once
/***************************************************************************************************
 * Fused Compress + Insert: Compressor weighted sum + RoPE + FP8 quant + cache write
 *
 * Single kernel replacing 2 separate launches (compressor + v4_fp8_k_append).
 * Eliminates 3 intermediate BF16 tensors (K_nope, K_rope, V) in global memory.
 *
 * CSA variant (window=8, stride=4):
 *   1. Softmax gate weights from learned params
 *   2. Weighted sum over 8-token window for K_nope, K_rope_raw, V
 *   3. Apply compressed RoPE (theta=160000) to K_rope accumulator
 *   4. FP8 quantize K_nope + V_nope, write directly to paged cache
 *   5. Copy K_rope as BF16 to cache
 *
 * HCA variant (window=128, stride=128):
 *   Same pipeline, register-accumulation over 128 tokens (multi-pass).
 *
 * Mirrors SnapMLA fused_k_append API pattern — one CTA per compressed entry,
 * writes directly to paged cache via slot_mapping.
 *
 * Grid: (num_compressed, 1, 1)
 * Block: 256 threads
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::compress {

struct FusedCompressInsertParams {
    // Input: raw token vectors (BF16)
    const __nv_bfloat16* __restrict__ input_k_nope;    // [num_tokens, head_dim]
    const __nv_bfloat16* __restrict__ input_k_rope_raw; // [num_tokens, qk_rope_head_dim]
    const __nv_bfloat16* __restrict__ input_v;          // [num_tokens, head_dim]

    // Learned gate parameters (BF16, per-layer)
    const __nv_bfloat16* __restrict__ gate_weights;     // [window]
    const __nv_bfloat16* __restrict__ positional_bias;  // [window] (CSA) or nullptr (HCA)

    // Compressed RoPE cos/sin tables (FP32)
    const float* __restrict__ compress_cos;  // [max_pos, qk_rope_head_dim/2]
    const float* __restrict__ compress_sin;  // [max_pos, qk_rope_head_dim/2]
    int cos_sin_stride;

    // Output: paged FP8 cache (written directly)
    uint8_t* __restrict__ kv_cache;
    const int* __restrict__ slot_mapping;  // [num_compressed] → flat slot index

    // Dimensions
    int num_tokens;
    int num_compressed;
    int head_dim;            // 512
    int qk_rope_head_dim;   // 64
    int window;              // 8 (CSA) or 128 (HCA)
    int stride;              // 4 (CSA) or 128 (HCA)
};

void run_fused_csa_compress_insert(const FusedCompressInsertParams& params, cudaStream_t stream);
void run_fused_hca_compress_insert(const FusedCompressInsertParams& params, cudaStream_t stream);

}  // namespace sm120::compress
