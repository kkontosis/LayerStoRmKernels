#pragma once
/***************************************************************************************************
 * CSA Token Compressor: Softmax-gated pooling with window=8, stride=4
 *
 * Compresses every stride=4 tokens into 1 compressed token using a window of 8
 * input tokens. Per compressed position j:
 *   1. Gather tokens [j*stride, j*stride + window) for K_nope, K_rope_raw, V
 *   2. Compute softmax gate weights: w = softmax(gate_weights + positional_bias)
 *   3. Weighted sum: compressed_k_nope[j] = sum(w * input_k_nope[window])
 *                    compressed_k_rope_raw[j] = sum(w * input_k_rope_raw[window])
 *                    compressed_v[j] = sum(w * input_v[window])
 *   4. Apply compressed RoPE (theta=160000) at position (j*stride + window - 1)
 *      to the rope dimensions of K
 *
 * Grid: (num_compressed, 1, 1) — one CTA per output compressed token
 * Block: 256 threads (8 warps)
 *
 * Shared memory: 8 tokens * 512 dims * 2B = 8 KB for K_nope (same for V) = ~24 KB total
 * Well within SM120's 99 KB limit.
 *
 * Adapted from: no direct template — new kernel for V4 CSA compression.
 * Reference: tests/test_v4_reference.py::ref_csa_compress()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::compress {

struct CsaCompressorParams {
    // Input: raw token vectors (BF16)
    const __nv_bfloat16* __restrict__ input_k_nope;    // [num_tokens, head_dim]
    const __nv_bfloat16* __restrict__ input_k_rope_raw; // [num_tokens, qk_rope_head_dim]
    const __nv_bfloat16* __restrict__ input_v;          // [num_tokens, head_dim]

    // Learned gate parameters (BF16, per-layer)
    const __nv_bfloat16* __restrict__ gate_weights;     // [window=8]
    const __nv_bfloat16* __restrict__ positional_bias;  // [window=8]

    // Compressed RoPE cos/sin tables (FP32)
    const float* __restrict__ compress_cos;  // [max_pos, qk_rope_head_dim/2]
    const float* __restrict__ compress_sin;  // [max_pos, qk_rope_head_dim/2]
    int cos_sin_stride;                      // qk_rope_head_dim / 2

    // Output: compressed vectors (BF16)
    __nv_bfloat16* __restrict__ out_k_nope;      // [num_compressed, head_dim]
    __nv_bfloat16* __restrict__ out_k_rope;       // [num_compressed, qk_rope_head_dim]
    __nv_bfloat16* __restrict__ out_v;            // [num_compressed, head_dim]

    // Dimensions
    int num_tokens;
    int num_compressed;
    int head_dim;            // 512
    int qk_rope_head_dim;   // 64
    int window;              // 8
    int stride;              // 4
};

void run_csa_compressor(const CsaCompressorParams& params, cudaStream_t stream);

}  // namespace sm120::compress
