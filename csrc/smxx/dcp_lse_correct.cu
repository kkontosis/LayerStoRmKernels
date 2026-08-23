// DCP (Decode Context Parallelism) LSE Correction kernel.
//
// Based on vLLM's _correct_attn_cp_out_kernel (Apache-2.0).
// ref/vllm/vllm/v1/attention/ops/common.py
//
// Ported from Triton to CUDA. Arch-generic — no SM-specific instructions.
// Grid: (B, H, 1), Block: 256 threads.
// Each block handles one (batch, head) pair, looping over D dimensions.

#include "dcp_lse_correct.h"
#include <cuda_bf16.h>
#include <cmath>

__global__ void __launch_bounds__(256)
dcp_lse_correct_kernel(const DcpLseCorrectParams params) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    if (batch_idx >= params.B || head_idx >= params.H) return;

    const int tid = threadIdx.x;
    const int N = params.N;

    // --- Step 1: Load all LSE values for this (batch, head) across ranks ---
    // Compute global LSE via numerically stable log-sum-exp
    float lse_max = -INFINITY;
    for (int n = 0; n < N; ++n) {
        float lse_n = params.lses[n * params.stride_lse_N +
                                   batch_idx * params.stride_lse_B +
                                   head_idx * params.stride_lse_H];
        // Replace NaN/inf with -inf (invalid rank)
        if (isnan(lse_n) || isinf(lse_n)) lse_n = -INFINITY;
        lse_max = fmaxf(lse_max, lse_n);
    }
    // If all ranks are -inf, set max to 0 to avoid NaN
    if (isinf(lse_max)) lse_max = 0.0f;

    float lse_sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        float lse_n = params.lses[n * params.stride_lse_N +
                                   batch_idx * params.stride_lse_B +
                                   head_idx * params.stride_lse_H];
        if (isnan(lse_n) || isinf(lse_n)) lse_n = -INFINITY;
        lse_sum += expf(lse_n - lse_max);
    }
    float global_lse = logf(lse_sum) + lse_max;

    // --- Step 2: Store global LSE ---
    if (tid == 0) {
        params.global_lse[batch_idx * params.stride_lse_B +
                          head_idx * params.stride_lse_H] = global_lse;
    }

    // --- Step 3: Correct this rank's output ---
    // factor = exp(local_lse - global_lse)
    float local_lse = params.lses[params.rank * params.stride_lse_N +
                                   batch_idx * params.stride_lse_B +
                                   head_idx * params.stride_lse_H];
    if (isnan(local_lse) || isinf(local_lse)) local_lse = -INFINITY;
    float diff = local_lse - global_lse;
    if (isnan(diff) || isinf(diff)) diff = -INFINITY;
    float factor = expf(diff);

    // Scale output elements: out[b, h, d] *= factor
    __nv_bfloat16* out = static_cast<__nv_bfloat16*>(params.output);
    int base = batch_idx * params.stride_o_B + head_idx * params.stride_o_H;

    for (int d = tid; d < params.D; d += blockDim.x) {
        int idx = base + d * params.stride_o_D;
        float val = __bfloat162float(out[idx]);
        out[idx] = __float2bfloat16_rn(val * factor);
    }
}

void run_dcp_lse_correct_kernel(const DcpLseCorrectParams& params, cudaStream_t stream) {
    dim3 grid(params.B, params.H, 1);
    dcp_lse_correct_kernel<<<grid, 256, 0, stream>>>(params);
}
