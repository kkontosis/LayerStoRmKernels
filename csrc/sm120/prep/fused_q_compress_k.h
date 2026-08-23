#pragma once
/***************************************************************************************************
 * Fused Q Norm + Compressed K RoPE + K Insert
 *
 * Single-launch kernel replacing 3 tiny per-decode-step operations:
 *   1. Q RMSNorm: per-head normalization on Q [h_q, d_qk]
 *   2. Compressed K RoPE: apply compress_rope_theta RoPE to K rope dims
 *   3. K cache insert: FP8 quantize and write K+V to paged cache
 *
 * The 10-20x speedup (vLLM) comes from eliminating kernel launch overhead
 * (~5 us each) on these tiny single-token operations.
 *
 * Grid: (h_q + 1, 1, 1)
 *   - Blocks 0..h_q-1: Q RMSNorm (one head per block)
 *   - Block h_q: compressed K RoPE + cache insert
 * Block: 256 threads
 *
 * Mirrors SnapMLA fused_q_quant pattern for the Q normalization portion.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct FusedQCompressKParams {
    // Q normalization (in-place)
    __nv_bfloat16* __restrict__ q;     // [h_q, d_qk] BF16 (modified in-place)
    int h_q;
    int d_qk;                          // 576
    float rms_eps;                     // 1e-6

    // Compressed K: RoPE + cache insert
    const __nv_bfloat16* __restrict__ k_nope;     // [1, head_dim] BF16
    const __nv_bfloat16* __restrict__ k_rope_raw; // [1, rope_dim] BF16 (pre-RoPE)
    const __nv_bfloat16* __restrict__ v_nope;     // [1, head_dim] BF16
    const float* __restrict__ compress_cos;       // [max_pos, rope_dim/2]
    const float* __restrict__ compress_sin;       // [max_pos, rope_dim/2]
    int cos_sin_stride;
    int rope_position;                            // compressed RoPE position

    // Cache output
    uint8_t* __restrict__ kv_cache;
    int slot;                                     // target slot in cache
    int head_dim;                                 // 512
    int qk_rope_head_dim;                         // 64
};

void run_fused_q_compress_k(const FusedQCompressKParams& params, cudaStream_t stream);

}  // namespace sm120::prep
