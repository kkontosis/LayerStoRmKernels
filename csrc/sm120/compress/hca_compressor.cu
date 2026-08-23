#pragma once
#include "hca_compressor.h"

namespace sm120::compress {

__global__ void __launch_bounds__(256)
hca_compressor_kernel(const HcaCompressorParams params) {
    const int comp_idx = blockIdx.x;
    if (comp_idx >= params.num_compressed) return;

    const int tid = threadIdx.x;
    const int head_dim = params.head_dim;
    const int rope_dim = params.qk_rope_head_dim;
    const int window = params.window;
    const int win_start = comp_idx * params.stride;

    // =========================================================================
    // Step 1: Compute softmax gate weights (128 values)
    // HCA has no separate positional_bias — softmax applied directly to gate_weights
    // =========================================================================
    __shared__ float s_weights[128];

    // Load gate weights: 128 values, 256 threads → first 128 threads load
    if (tid < window) {
        s_weights[tid] = __bfloat162float(__ldg(params.gate_weights + tid));
    }
    __syncthreads();

    // Parallel max reduction
    __shared__ float s_reduce[8];
    float local_max = -1e30f;
    for (int i = tid; i < window; i += 256)
        local_max = fmaxf(local_max, s_weights[i]);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (tid % 32 == 0) s_reduce[tid / 32] = local_max;
    __syncthreads();
    if (tid == 0) {
        float gmax = s_reduce[0];
        for (int w = 1; w < 8; ++w) gmax = fmaxf(gmax, s_reduce[w]);
        s_reduce[0] = gmax;
    }
    __syncthreads();
    float global_max = s_reduce[0];

    // Exp and parallel sum reduction
    if (tid < window) {
        s_weights[tid] = expf(s_weights[tid] - global_max);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < window; i += 256)
        local_sum += s_weights[i];

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid % 32 == 0) s_reduce[tid / 32] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float gsum = s_reduce[0];
        for (int w = 1; w < 8; ++w) gsum += s_reduce[w];
        s_reduce[0] = 1.0f / gsum;
    }
    __syncthreads();
    float inv_sum = s_reduce[0];

    if (tid < window) {
        s_weights[tid] *= inv_sum;
    }
    __syncthreads();

    // =========================================================================
    // Step 2: Weighted sum over K_nope and V (head_dim=512)
    // 256 threads * 2 dims = 512 dims. Accumulate over 128 tokens in registers.
    // =========================================================================
    const int d0 = tid * 2;
    const int d1 = tid * 2 + 1;

    if (d0 < head_dim) {
        float acc_k0 = 0.0f, acc_k1 = 0.0f;
        float acc_v0 = 0.0f, acc_v1 = 0.0f;
        for (int j = 0; j < window; ++j) {
            float wj = s_weights[j];
            int token = win_start + j;

            const __nv_bfloat16* k_row = params.input_k_nope + (int64_t)token * head_dim;
            __nv_bfloat162 k_pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(k_row) + tid);
            acc_k0 += wj * __low2float(k_pair);
            acc_k1 += wj * __high2float(k_pair);

            const __nv_bfloat16* v_row = params.input_v + (int64_t)token * head_dim;
            __nv_bfloat162 v_pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(v_row) + tid);
            acc_v0 += wj * __low2float(v_pair);
            acc_v1 += wj * __high2float(v_pair);
        }

        __nv_bfloat16* out_k = params.out_k_nope + (int64_t)comp_idx * head_dim;
        out_k[d0] = __float2bfloat16_rn(acc_k0);
        out_k[d1] = __float2bfloat16_rn(acc_k1);

        __nv_bfloat16* out_v = params.out_v + (int64_t)comp_idx * head_dim;
        out_v[d0] = __float2bfloat16_rn(acc_v0);
        out_v[d1] = __float2bfloat16_rn(acc_v1);
    }

    // =========================================================================
    // Step 3: Weighted sum over K_rope_raw (64 dims) + apply RoPE
    // Interleaved pairs: thread tid handles dims [2*tid, 2*tid+1]
    // =========================================================================
    const int half_rope = rope_dim / 2;
    if (tid < half_rope) {
        float acc_even = 0.0f, acc_odd = 0.0f;
        for (int j = 0; j < window; ++j) {
            float wj = s_weights[j];
            int token = win_start + j;
            const __nv_bfloat16* row = params.input_k_rope_raw + (int64_t)token * rope_dim;
            acc_even += wj * __bfloat162float(__ldg(row + 2 * tid));
            acc_odd  += wj * __bfloat162float(__ldg(row + 2 * tid + 1));
        }

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

void run_hca_compressor(const HcaCompressorParams& params, cudaStream_t stream) {
    if (params.num_compressed <= 0) return;
    hca_compressor_kernel<<<params.num_compressed, 256, 0, stream>>>(params);
}

}  // namespace sm120::compress
