#pragma once
#include "csa_compressor.h"
#include <math_constants.h>

namespace sm120::compress {

__global__ void __launch_bounds__(256)
csa_compressor_kernel(const CsaCompressorParams params) {
    const int comp_idx = blockIdx.x;
    if (comp_idx >= params.num_compressed) return;

    const int tid = threadIdx.x;
    const int head_dim = params.head_dim;
    const int rope_dim = params.qk_rope_head_dim;
    const int window = params.window;
    const int stride = params.stride;

    const int win_start = comp_idx * stride;

    // =========================================================================
    // Step 1: Compute softmax gate weights (all 8 values, replicated per thread)
    // gate_logits[j] = gate_weights[j] + positional_bias[j], then softmax
    // =========================================================================
    __shared__ float s_weights[8];

    if (tid < window) {
        float gw = __bfloat162float(__ldg(params.gate_weights + tid));
        float pb = __bfloat162float(__ldg(params.positional_bias + tid));
        s_weights[tid] = gw + pb;
    }
    __syncthreads();

    // Softmax: find max, subtract, exp, normalize
    if (tid == 0) {
        float max_val = s_weights[0];
        for (int j = 1; j < window; ++j)
            max_val = fmaxf(max_val, s_weights[j]);
        float sum_exp = 0.0f;
        for (int j = 0; j < window; ++j) {
            s_weights[j] = expf(s_weights[j] - max_val);
            sum_exp += s_weights[j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < window; ++j)
            s_weights[j] *= inv_sum;
    }
    __syncthreads();

    // Load softmax weights into registers
    float w[8];
    for (int j = 0; j < window; ++j)
        w[j] = s_weights[j];

    // =========================================================================
    // Step 2: Weighted sum over K_nope and V (both head_dim=512)
    // 256 threads, each handles 2 dimensions → covers 512 dims exactly
    // =========================================================================
    const int d0 = tid * 2;
    const int d1 = tid * 2 + 1;

    if (d0 < head_dim) {
        // K_nope weighted sum
        float acc_k0 = 0.0f, acc_k1 = 0.0f;
        for (int j = 0; j < window; ++j) {
            int token = win_start + j;
            const __nv_bfloat16* row = params.input_k_nope + (int64_t)token * head_dim;
            __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(row) + tid);
            acc_k0 += w[j] * __low2float(pair);
            acc_k1 += w[j] * __high2float(pair);
        }
        __nv_bfloat16* out_k = params.out_k_nope + (int64_t)comp_idx * head_dim;
        out_k[d0] = __float2bfloat16_rn(acc_k0);
        out_k[d1] = __float2bfloat16_rn(acc_k1);

        // V weighted sum
        float acc_v0 = 0.0f, acc_v1 = 0.0f;
        for (int j = 0; j < window; ++j) {
            int token = win_start + j;
            const __nv_bfloat16* row = params.input_v + (int64_t)token * head_dim;
            __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(row) + tid);
            acc_v0 += w[j] * __low2float(pair);
            acc_v1 += w[j] * __high2float(pair);
        }
        __nv_bfloat16* out_v = params.out_v + (int64_t)comp_idx * head_dim;
        out_v[d0] = __float2bfloat16_rn(acc_v0);
        out_v[d1] = __float2bfloat16_rn(acc_v1);
    }

    // =========================================================================
    // Step 3: Weighted sum over K_rope_raw (qk_rope_head_dim=64) + apply RoPE
    // Only first 32 threads needed (32 threads handle 32 interleaved pairs)
    // Reference uses interleaved layout: x1=x[0::2], x2=x[1::2]
    // =========================================================================
    const int half_rope = rope_dim / 2;  // 32
    if (tid < half_rope) {
        // Weighted sum for interleaved pair: dims [2*tid] and [2*tid+1]
        float acc_even = 0.0f, acc_odd = 0.0f;
        for (int j = 0; j < window; ++j) {
            int token = win_start + j;
            const __nv_bfloat16* row = params.input_k_rope_raw + (int64_t)token * rope_dim;
            acc_even += w[j] * __bfloat162float(__ldg(row + 2 * tid));
            acc_odd  += w[j] * __bfloat162float(__ldg(row + 2 * tid + 1));
        }

        // Apply RoPE: out_even = x_even*cos - x_odd*sin, out_odd = x_even*sin + x_odd*cos
        int rope_pos = win_start + window - 1;
        float cos_val = __ldg(params.compress_cos + rope_pos * params.cos_sin_stride + tid);
        float sin_val = __ldg(params.compress_sin + rope_pos * params.cos_sin_stride + tid);

        float out_even = acc_even * cos_val - acc_odd * sin_val;
        float out_odd  = acc_even * sin_val + acc_odd * cos_val;

        __nv_bfloat16* out_rope = params.out_k_rope + (int64_t)comp_idx * rope_dim;
        out_rope[2 * tid]     = __float2bfloat16_rn(out_even);
        out_rope[2 * tid + 1] = __float2bfloat16_rn(out_odd);
    }
}

void run_csa_compressor(const CsaCompressorParams& params, cudaStream_t stream) {
    if (params.num_compressed <= 0) return;
    csa_compressor_kernel<<<params.num_compressed, 256, 0, stream>>>(params);
}

}  // namespace sm120::compress
