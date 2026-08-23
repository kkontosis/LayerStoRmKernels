#pragma once
/***************************************************************************************************
 * HCA Token Compressor: Softmax-gated pooling with window=128, stride=128
 *
 * Heavy compression: every 128 tokens → 1 compressed entry. No overlap, no residual.
 * Same mechanism as CSA compressor but with larger window.
 *
 * Per compressed position j:
 *   1. Gather 128 tokens [j*128, (j+1)*128)
 *   2. w = softmax(gate_weights)  (no separate positional_bias for HCA)
 *   3. compressed_k_nope[j] = sum(w * input_k_nope[window])
 *   4. Apply compressed RoPE (theta=160000) at position 128*j + 127
 *
 * Grid: (num_compressed, 1, 1)
 * Block: 256 threads
 *
 * Window data (128 * 512 * 2B = 128 KB) exceeds SM120's 99 KB smem limit.
 * Strategy: accumulate weighted sum in registers, streaming tokens from global
 * memory with L2 caching. Only softmax weights (128 * 4B = 512B) go to smem.
 *
 * Reference: tests/test_v4_reference.py::ref_hca_compress()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::compress {

struct HcaCompressorParams {
    const __nv_bfloat16* __restrict__ input_k_nope;    // [num_tokens, head_dim]
    const __nv_bfloat16* __restrict__ input_k_rope_raw; // [num_tokens, qk_rope_head_dim]
    const __nv_bfloat16* __restrict__ input_v;          // [num_tokens, head_dim]

    const __nv_bfloat16* __restrict__ gate_weights;     // [window=128]

    const float* __restrict__ compress_cos;  // [max_pos, qk_rope_head_dim/2]
    const float* __restrict__ compress_sin;  // [max_pos, qk_rope_head_dim/2]
    int cos_sin_stride;                      // qk_rope_head_dim / 2

    __nv_bfloat16* __restrict__ out_k_nope;      // [num_compressed, head_dim]
    __nv_bfloat16* __restrict__ out_k_rope;       // [num_compressed, qk_rope_head_dim]
    __nv_bfloat16* __restrict__ out_v;            // [num_compressed, head_dim]

    int num_tokens;
    int num_compressed;
    int head_dim;            // 512
    int qk_rope_head_dim;   // 64
    int window;              // 128
    int stride;              // 128
};

void run_hca_compressor(const HcaCompressorParams& params, cudaStream_t stream);

}  // namespace sm120::compress
