#pragma once
#include "mhc.h"
#include "utils.h"

#include <cstdio>

// DeepSeek-V4 mHC kernels. Math ported from:
//   ref/vllm/vllm/model_executor/kernels/mhc/torch.py
//     (mhc_pre_torch / mhc_post_torch; Apache-2.0, vLLM project) and
//   ref/llama.cpp/src/models/deepseek4.cpp (build_hc_* ; MIT, llama.cpp authors).
// See mhc.h for the full contract.

namespace smxx::mhc {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// Block-wide sum reduction of `n_vals` independent accumulators per thread.
// vals[] is reduced in place; on return, thread 0's vals[] hold the block sums,
// and s_bcast[0..n_vals) hold the same (all threads may read after syncthreads).
template <int N_VALS>
__device__ void block_reduce_sums(float (&vals)[N_VALS], float* s_scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int v = 0; v < N_VALS; ++v) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            vals[v] += __shfl_down_sync(0xffffffffu, vals[v], off);
        }
        if (lane == 0) s_scratch[v * kWarps + warp] = vals[v];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int v = 0; v < N_VALS; ++v) {
            float x = (lane < kWarps) ? s_scratch[v * kWarps + lane] : 0.0f;
#pragma unroll
            for (int off = kWarps / 2; off > 0; off >>= 1) {
                x += __shfl_down_sync(0xffffffffu, x, off);
            }
            if (lane == 0) s_scratch[v * kWarps] = x;
        }
    }
    __syncthreads();
#pragma unroll
    for (int v = 0; v < N_VALS; ++v) vals[v] = s_scratch[v * kWarps];
    __syncthreads();
}

__device__ __forceinline__ float sigmoidf_stable(float x) {
    // 1/(1+e^-x): numerically fine across the fp32 range.
    return 1.0f / (1.0f + expf(-x));
}

// Sinkhorn on a 4x4 matrix, single thread. comb layout [src][dst], dst fastest.
// Reference order (vLLM torch.py:78-82 / llama.cpp deepseek4.cpp:244-279):
//   stable row-softmax over dst -> +eps -> col-normalize (over src, +eps in denom)
//   -> (iters-1) x { row-normalize (+eps), col-normalize (+eps) }.
template <int HC>
__device__ void sinkhorn_4x4(float (&c)[HC * HC], float eps, int iters) {
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        float m = c[s * HC];
#pragma unroll
        for (int d = 1; d < HC; ++d) m = fmaxf(m, c[s * HC + d]);
        float sum = 0.0f;
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            c[s * HC + d] = expf(c[s * HC + d] - m);
            sum += c[s * HC + d];
        }
        const float inv = 1.0f / sum;
#pragma unroll
        for (int d = 0; d < HC; ++d) c[s * HC + d] = c[s * HC + d] * inv + eps;
    }
    // First column normalization, then (iters-1) x {row, col}.
    auto norm_cols = [&]() {
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < HC; ++s) sum += c[s * HC + d];
            const float inv = 1.0f / (sum + eps);
#pragma unroll
            for (int s = 0; s < HC; ++s) c[s * HC + d] *= inv;
        }
    };
    auto norm_rows = [&]() {
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            float sum = 0.0f;
#pragma unroll
            for (int d = 0; d < HC; ++d) sum += c[s * HC + d];
            const float inv = 1.0f / (sum + eps);
#pragma unroll
            for (int d = 0; d < HC; ++d) c[s * HC + d] *= inv;
        }
    };
    norm_cols();
    for (int it = 1; it < iters; ++it) {
        norm_rows();
        norm_cols();
    }
}

// Shared phase-1: per-token flat-RMS + fn GEMV. Each thread accumulates
// sqrsum + N_MIX dot products over the hc_dim vector, then block-reduces.
// On return s_mixes[0..N_MIX) hold rms-normalized mixes (valid for all threads).
template <int HC, int N_MIX>
__device__ void mhc_mixes(const __nv_bfloat16* __restrict__ residual_row,
                          const float* __restrict__ fn, float rms_eps,
                          int hidden, float* s_scratch, float* s_mixes) {
    const int hc_dim = HC * hidden;
    float acc[N_MIX + 1];
#pragma unroll
    for (int v = 0; v <= N_MIX; ++v) acc[v] = 0.0f;

    for (int i = threadIdx.x; i < hc_dim; i += kThreads) {
        const float x = __bfloat162float(residual_row[i]);
        acc[N_MIX] += x * x;
#pragma unroll
        for (int j = 0; j < N_MIX; ++j) {
            acc[j] += x * __ldg(fn + (int64_t)j * hc_dim + i);
        }
    }
    block_reduce_sums<N_MIX + 1>(acc, s_scratch);

    if (threadIdx.x == 0) {
        const float inv_rms = rsqrtf(acc[N_MIX] / (float)hc_dim + rms_eps);
#pragma unroll
        for (int j = 0; j < N_MIX; ++j) s_mixes[j] = acc[j] * inv_rms;
    }
    __syncthreads();
}

template <int HC>
__global__ void __launch_bounds__(kThreads) mhc_pre_kernel(const MhcPreParams p) {
    constexpr int N_MIX = (2 + HC) * HC;
    const int t = blockIdx.x;
    if (t >= p.num_tokens) return;

    __shared__ float s_scratch[(N_MIX + 1) * kWarps];
    __shared__ float s_mixes[N_MIX];
    __shared__ float s_pre[HC];

    const __nv_bfloat16* row = p.residual + (int64_t)t * p.residual_row_stride;
    mhc_mixes<HC, N_MIX>(row, p.fn, p.rms_eps, p.hidden, s_scratch, s_mixes);

    if (threadIdx.x == 0) {
        const float scale_pre = __ldg(p.hc_scale + 0);
        const float scale_post = __ldg(p.hc_scale + 1);
        const float scale_comb = __ldg(p.hc_scale + 2);

        float comb[HC * HC];
#pragma unroll
        for (int k = 0; k < HC; ++k) {
            s_pre[k] = sigmoidf_stable(s_mixes[k] * scale_pre + __ldg(p.hc_base + k)) +
                       p.hc_eps;
            p.post_out[(int64_t)t * HC + k] =
                sigmoidf_stable(s_mixes[HC + k] * scale_post + __ldg(p.hc_base + HC + k)) *
                p.post_mult;
        }
#pragma unroll
        for (int e = 0; e < HC * HC; ++e) {
            comb[e] = s_mixes[2 * HC + e] * scale_comb + __ldg(p.hc_base + 2 * HC + e);
        }
        sinkhorn_4x4<HC>(comb, p.hc_eps, p.sinkhorn_iters);
#pragma unroll
        for (int e = 0; e < HC * HC; ++e) p.comb_out[(int64_t)t * HC * HC + e] = comb[e];
    }
    __syncthreads();

    // Weighted collapse: x[i] = sum_s pre[s] * R[s*hidden + i].
    float pre[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) pre[s] = s_pre[s];
    __nv_bfloat16* xo = p.x_out + (int64_t)t * p.x_out_row_stride;
    for (int i = threadIdx.x; i < p.hidden; i += kThreads) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            acc += pre[s] * __bfloat162float(row[s * p.hidden + i]);
        }
        xo[i] = __float2bfloat16_rn(acc);
    }
}

template <int HC>
__global__ void __launch_bounds__(kThreads) mhc_post_kernel(const MhcPostParams p) {
    const int t = blockIdx.x;
    if (t >= p.num_tokens) return;

    __shared__ float s_post[HC];
    __shared__ float s_comb[HC * HC];
    if (threadIdx.x < HC) s_post[threadIdx.x] = p.post[(int64_t)t * HC + threadIdx.x];
    if (threadIdx.x < HC * HC) {
        s_comb[threadIdx.x] = p.comb[(int64_t)t * HC * HC + threadIdx.x];
    }
    __syncthreads();

    const __nv_bfloat16* yr = p.y + (int64_t)t * p.y_row_stride;
    const __nv_bfloat16* rr = p.residual + (int64_t)t * p.residual_row_stride;
    __nv_bfloat16* out = p.residual_out + (int64_t)t * p.residual_out_row_stride;

    for (int i = threadIdx.x; i < p.hidden; i += kThreads) {
        const float yv = __bfloat162float(yr[i]);
        float r[HC];
#pragma unroll
        for (int s = 0; s < HC; ++s) r[s] = __bfloat162float(rr[s * p.hidden + i]);
#pragma unroll
        for (int d = 0; d < HC; ++d) {
            float acc = s_post[d] * yv;
#pragma unroll
            for (int s = 0; s < HC; ++s) acc += s_comb[s * HC + d] * r[s];
            out[d * p.hidden + i] = __float2bfloat16_rn(acc);
        }
    }
}

template <int HC>
__global__ void __launch_bounds__(kThreads) mhc_head_kernel(const MhcHeadParams p) {
    constexpr int N_MIX = HC;
    const int t = blockIdx.x;
    if (t >= p.num_tokens) return;

    __shared__ float s_scratch[(N_MIX + 1) * kWarps];
    __shared__ float s_mixes[N_MIX];
    __shared__ float s_pre[HC];

    const __nv_bfloat16* row = p.residual + (int64_t)t * p.residual_row_stride;
    mhc_mixes<HC, N_MIX>(row, p.fn, p.rms_eps, p.hidden, s_scratch, s_mixes);

    if (threadIdx.x == 0) {
        const float scale = __ldg(p.hc_scale);
#pragma unroll
        for (int k = 0; k < HC; ++k) {
            s_pre[k] =
                sigmoidf_stable(s_mixes[k] * scale + __ldg(p.hc_base + k)) + p.hc_eps;
        }
    }
    __syncthreads();

    float pre[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) pre[s] = s_pre[s];
    __nv_bfloat16* xo = p.x_out + (int64_t)t * p.x_out_row_stride;
    for (int i = threadIdx.x; i < p.hidden; i += kThreads) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            acc += pre[s] * __bfloat162float(row[s * p.hidden + i]);
        }
        xo[i] = __float2bfloat16_rn(acc);
    }
}

}  // namespace

void run_mhc_pre(const MhcPreParams& params, cudaStream_t stream) {
    if (params.num_tokens <= 0) return;
    FLASH_ASSERT(params.hc == 4);
    mhc_pre_kernel<4><<<params.num_tokens, kThreads, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void run_mhc_post(const MhcPostParams& params, cudaStream_t stream) {
    if (params.num_tokens <= 0) return;
    FLASH_ASSERT(params.hc == 4);
    mhc_post_kernel<4><<<params.num_tokens, kThreads, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

void run_mhc_head(const MhcHeadParams& params, cudaStream_t stream) {
    if (params.num_tokens <= 0) return;
    FLASH_ASSERT(params.hc == 4);
    mhc_head_kernel<4><<<params.num_tokens, kThreads, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace smxx::mhc
