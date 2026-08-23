#pragma once
/***************************************************************************************************
 * Fused Inverse RoPE + FP8 Quantization
 *
 * Single kernel replacing 2 launches (inverse_rope + fp8_quant).
 * Operates on the attention output [N, head_dim=512]:
 *   1. Apply inverse RoPE to last qk_rope_head_dim=64 dims
 *   2. FP8-quantize all head_dim dims with per-row scale
 *   3. Write FP8 values + float32 scale
 *
 * Mirrors SnapMLA fused_q_quant pattern (per-row quant + element-wise op).
 *
 * Grid:  (N, 1, 1) — one block per row
 * Block: 256 threads — 2 elements per thread covers head_dim=512
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct FusedInvRopeFp8Params {
    const __nv_bfloat16* __restrict__ x;    // [N, head_dim] BF16 attention output
    const float* __restrict__ cos_table;    // [max_pos, rope_dim/2]
    const float* __restrict__ sin_table;    // [max_pos, rope_dim/2]
    const int* __restrict__ positions;      // [N] position per row

    __nv_fp8_e4m3* __restrict__ out_fp8;    // [N, head_dim] FP8
    float* __restrict__ out_scales;         // [N] per-row scales

    int N;
    int head_dim;          // 512
    int qk_rope_head_dim;  // 64 (last dims get inverse RoPE)
};

void run_fused_inv_rope_fp8(const FusedInvRopeFp8Params& params, cudaStream_t stream);

}  // namespace sm120::prep
