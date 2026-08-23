#pragma once

#include <cuda_bf16.h>

// DeepSeek-V4 mHC (Manifold-constrained Hyper-Connections) kernels.
//
// The V4 residual stream is [num_tokens, hc_mult, hidden] (hc_mult = 4).
// Around every attention / FFN module:
//   (post, comb, x) = mhc_pre(R)   — collapse hc streams -> module input
//   y = module(x)
//   R' = mhc_post(y, R, post, comb) — re-expand + doubly-stochastic mix
// and before the LM head:
//   x = mhc_head(R)                 — pre-branch-only weighted collapse
//
// Math reference (bit-matched in structure):
//   ref/vllm/vllm/model_executor/kernels/mhc/torch.py (mhc_pre_torch/mhc_post_torch)
//   ref/llama.cpp/src/models/deepseek4.cpp (build_hc_pre :281 / build_hc_post :331 /
//     build_hc_head :360 / build_hc_sinkhorn :244)
//   ref/sglang/python/sglang/srt/layers/mhc.py (tilelang kernels)
//   (vLLM, SGLang: Apache-2.0; llama.cpp: MIT, Copyright (c) 2023-2026
//   The ggml authors — see THIRD_PARTY_NOTICES.md)
//
//   mixes[hc_mix]   = fn @ rms_norm_unweighted(flatten(R), rms_eps)   (all FP32)
//   pre[hc]         = sigmoid(mixes[0:hc]      * scale[0] + base[0:hc])      + hc_eps
//   post[hc]        = sigmoid(mixes[hc:2hc]    * scale[1] + base[hc:2hc])    * post_mult (2.0)
//   comb[hc,hc]     = sinkhorn(mixes[2hc:] * scale[2] + base[2hc:]) with
//                     row-softmax(dst) -> +eps -> col-normalize(src) ->
//                     (iters-1) x { row-normalize, col-normalize }, eps = hc_eps
//   x[hidden]       = sum_s pre[s] * R[s,:]
//   R'[dst,:]       = post[dst]*y + sum_src comb[src,dst] * R[src,:]
//
// comb memory layout everywhere: row-major [src][dst] (dst fastest), matching
// the vLLM torch reference view (num_tokens, hc, hc) with softmax over dim=-1.
//
// All internal math FP32; residual/x/y are BF16; fn/scale/base/post/comb are FP32.

namespace smxx::mhc {

struct MhcPreParams {
    // Input residual, flattened: [num_tokens, hc * hidden] BF16.
    const __nv_bfloat16* __restrict__ residual;
    int64_t residual_row_stride;   // elements between token rows (>= hc*hidden)

    // Learned mix function: [hc_mix = (2+hc)*hc, hc*hidden] FP32, row-major.
    const float* __restrict__ fn;
    const float* __restrict__ hc_scale;  // [3] FP32
    const float* __restrict__ hc_base;   // [hc_mix] FP32

    float rms_eps;        // RMS over the flat hc*hidden vector (model rms_norm_eps)
    float hc_eps;         // pre-mix epsilon AND sinkhorn epsilon (V4: 1e-6)
    float post_mult;      // hc_post multiplier (V4: 2.0)
    int sinkhorn_iters;   // V4: 20

    // Outputs.
    float* __restrict__ post_out;        // [num_tokens, hc] FP32
    float* __restrict__ comb_out;        // [num_tokens, hc*hc] FP32, [src][dst]
    __nv_bfloat16* __restrict__ x_out;   // [num_tokens, hidden] BF16
    int64_t x_out_row_stride;            // elements between token rows (>= hidden)

    int num_tokens;
    int hc;        // 4
    int hidden;    // 4096 (must be a multiple of 2*blockDim)
};

struct MhcPostParams {
    const __nv_bfloat16* __restrict__ y;         // [num_tokens, hidden] BF16 (module out)
    int64_t y_row_stride;                        // elements between token rows
    const __nv_bfloat16* __restrict__ residual;  // [num_tokens, hc*hidden] BF16
    int64_t residual_row_stride;
    const float* __restrict__ post;              // [num_tokens, hc] FP32
    const float* __restrict__ comb;              // [num_tokens, hc*hc] FP32, [src][dst]

    // Output residual; may alias `residual` (in-place safe: each thread owns
    // all hc slots of its hidden elements).
    __nv_bfloat16* __restrict__ residual_out;
    int64_t residual_out_row_stride;

    int num_tokens;
    int hc;
    int hidden;
};

struct MhcHeadParams {
    const __nv_bfloat16* __restrict__ residual;  // [num_tokens, hc*hidden] BF16
    int64_t residual_row_stride;

    // Head mix function: [hc, hc*hidden] FP32; scalar scale; base [hc].
    const float* __restrict__ fn;
    const float* __restrict__ hc_scale;  // [1] FP32
    const float* __restrict__ hc_base;   // [hc] FP32

    float rms_eps;
    float hc_eps;

    __nv_bfloat16* __restrict__ x_out;   // [num_tokens, hidden] BF16
    int64_t x_out_row_stride;

    int num_tokens;
    int hc;
    int hidden;
};

// hc must be 4 (the only instantiated stream count; V4-Flash and V4-Pro).
void run_mhc_pre(const MhcPreParams& params, cudaStream_t stream);
void run_mhc_post(const MhcPostParams& params, cudaStream_t stream);
void run_mhc_head(const MhcHeadParams& params, cudaStream_t stream);

}  // namespace smxx::mhc
