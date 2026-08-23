#pragma once
#include "tq_v_rotate_back.h"
#include "tq_q_rotate.h"
#include <cstdlib>

namespace sm120::prep {

// out_final[j] = sum_i out_rotated[i] * Pi[i][j]
// This is out_rotated @ Pi (not Pi^T) — the inverse rotation.

__global__ void __launch_bounds__(256)
tq_v_rotate_back_kernel(const TqVRotateBackParams params) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= params.batch_heads) return;

    const int d_c = params.d_c;

    const float* in = params.out_rotated + (int64_t)head_idx * d_c;
    __nv_bfloat16* out = params.out_final + (int64_t)head_idx * d_c;

    // Load out_rotated into shared memory
    extern __shared__ float smem[];
    float* s_in = smem;

    int base = tid * 2;
    if (base < d_c) {
        // Float32 input — load 2 floats per thread
        s_in[base]     = __ldg(in + base);
        s_in[base + 1] = __ldg(in + base + 1);
    }
    __syncthreads();

    // out_final[j] = sum_i s_in[i] * Pi[i][j]
    // Thread computes 2 output coordinates
    if (base < d_c) {
        int out_j0 = base;
        int out_j1 = base + 1;

        float c0 = 0.0f, c1 = 0.0f;
        for (int i = 0; i < d_c; i += 4) {
            float v0 = s_in[i], v1 = s_in[i + 1], v2 = s_in[i + 2], v3 = s_in[i + 3];

            c0 += v0 * __ldg(params.Pi + (int64_t)i * d_c + out_j0);
            c0 += v1 * __ldg(params.Pi + (int64_t)(i + 1) * d_c + out_j0);
            c0 += v2 * __ldg(params.Pi + (int64_t)(i + 2) * d_c + out_j0);
            c0 += v3 * __ldg(params.Pi + (int64_t)(i + 3) * d_c + out_j0);

            if (out_j1 < d_c) {
                c1 += v0 * __ldg(params.Pi + (int64_t)i * d_c + out_j1);
                c1 += v1 * __ldg(params.Pi + (int64_t)(i + 1) * d_c + out_j1);
                c1 += v2 * __ldg(params.Pi + (int64_t)(i + 2) * d_c + out_j1);
                c1 += v3 * __ldg(params.Pi + (int64_t)(i + 3) * d_c + out_j1);
            }
        }

        out[out_j0] = __float2bfloat16_rn(c0);
        if (out_j1 < d_c) {
            out[out_j1] = __float2bfloat16_rn(c1);
        }
    }
}

// ── Tiled variant (see tq_q_rotate.cu) ──────────────────────────────────────
// out_final[h, j] = Σ_i out_rotated[h, i] · Pi[i][j]. The legacy per-head CTA
// walks Pi COLUMNS (strided) per head; this tiles Pi rows through smem once
// per 32-head group. Ascending-i single-accumulator per (h, j) — bit-identical.
namespace {
constexpr int kJTileV = 16, kHTileV = 32, kIChunkV = 128, kRotThreadsV = 256;
inline bool rotate_back_tiled_enabled() {
    // Default LEGACY: the tiled variant MEASURED SLOWER here (30.8 vs 20.5 µs
    // — Pi is L2-resident, so the legacy strided column walk is not
    // DRAM-bound; the tiled version adds smem staging latency). Opt-in via
    // LS_TQ_ROTATE_TILED=1 for re-testing on other shapes/hardware.
    static const bool on = [] {
        const char* v = std::getenv("LS_TQ_ROTATE_TILED");
        return v && v[0] == '1';
    }();
    return on;
}
}

__global__ void __launch_bounds__(kRotThreadsV)
tq_v_rotate_back_tiled_kernel(const TqVRotateBackParams params) {
    const int d_c = params.d_c;
    const int j0 = blockIdx.x * kJTileV;
    const int h0 = blockIdx.y * kHTileV;
    __shared__ float sPi[kIChunkV][kJTileV];
    __shared__ float sIn[kHTileV][kIChunkV];
    const int tj = threadIdx.x >> 4;
    const int thp = threadIdx.x & 15;
    const int h1 = h0 + thp * 2, h2 = h1 + 1;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (int ic = 0; ic < d_c; ic += kIChunkV) {
        for (int t = threadIdx.x; t < kIChunkV * kJTileV; t += kRotThreadsV) {
            const int ii = t / kJTileV, jl = t % kJTileV;   // coalesced over j
            sPi[ii][jl] = __ldg(params.Pi + (int64_t)(ic + ii) * d_c + j0 + jl);
        }
        for (int t = threadIdx.x; t < kHTileV * kIChunkV; t += kRotThreadsV) {
            const int hl = t / kIChunkV, ii = t % kIChunkV;
            const int hg = h0 + hl;
            sIn[hl][ii] = (hg < params.batch_heads)
                ? __ldg(params.out_rotated + (int64_t)hg * d_c + ic + ii)
                : 0.0f;
        }
        __syncthreads();
        #pragma unroll 4
        for (int i = 0; i < kIChunkV; ++i) {
            const float p = sPi[i][tj];
            acc1 += sIn[thp * 2][i] * p;
            acc2 += sIn[thp * 2 + 1][i] * p;
        }
        __syncthreads();
    }
    if (h1 < params.batch_heads)
        params.out_final[(int64_t)h1 * d_c + j0 + tj] = __float2bfloat16_rn(acc1);
    if (h2 < params.batch_heads)
        params.out_final[(int64_t)h2 * d_c + j0 + tj] = __float2bfloat16_rn(acc2);
}

void run_tq_v_rotate_back(const TqVRotateBackParams& params, cudaStream_t stream) {
    if (params.Pi_t && params.d_c <= 512) {
        run_tq_rotate_rows_f32_to_bf16(
            params.out_rotated, params.Pi_t, params.out_final,
            params.batch_heads, params.d_c, stream);
        return;
    }
    if (rotate_back_tiled_enabled()
        && params.d_c % kIChunkV == 0 && params.d_c % kJTileV == 0) {
        dim3 grid(params.d_c / kJTileV,
                  (params.batch_heads + kHTileV - 1) / kHTileV);
        tq_v_rotate_back_tiled_kernel<<<grid, kRotThreadsV, 0, stream>>>(params);
        return;
    }
    int smem_bytes = params.d_c * sizeof(float);
    tq_v_rotate_back_kernel<<<params.batch_heads, 256, smem_bytes, stream>>>(params);
}

}  // namespace sm120::prep
