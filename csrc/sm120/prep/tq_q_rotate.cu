#pragma once
#include "tq_q_rotate.h"
#include <cstdlib>

namespace sm120::prep {

// q_rot[j] = sum_i q_nope[i] * Pi^T[i][j] = sum_i q_nope[i] * Pi[j][i]
// Same structure as the rotation in k_append: dot product of input with rows of Pi.

__global__ void __launch_bounds__(256)
tq_q_rotate_kernel(const TqQRotateParams params) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= params.batch_heads) return;

    const int d_c = params.d_c;

    const int in_stride = params.q_row_stride > 0 ? params.q_row_stride : d_c;
    const __nv_bfloat16* q_in = params.q_nope + (int64_t)head_idx * in_stride;
    float* q_out = params.q_rot + (int64_t)head_idx * d_c;

    // Load q_nope into shared memory as float32
    extern __shared__ float smem[];
    float* s_q = smem;  // [d_c]

    // Each thread loads 2 elements (d_c=512, 256 threads)
    int base = tid * 2;
    if (base < d_c) {
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(q_in) + tid);
        s_q[base]     = __low2float(pair);
        s_q[base + 1] = __high2float(pair);
    }
    __syncthreads();

    // Each thread computes 2 output coordinates
    // q_rot[j] = sum_i s_q[i] * Pi[j][i]  (Pi[j][i] = Pi stored row-major)
    if (base < d_c) {
        int out_j0 = base;
        int out_j1 = base + 1;

        const float* pi_row0 = params.Pi + (int64_t)out_j0 * d_c;
        float y0 = 0.0f;
        for (int i = 0; i < d_c; i += 4) {
            y0 += s_q[i]     * __ldg(pi_row0 + i);
            y0 += s_q[i + 1] * __ldg(pi_row0 + i + 1);
            y0 += s_q[i + 2] * __ldg(pi_row0 + i + 2);
            y0 += s_q[i + 3] * __ldg(pi_row0 + i + 3);
        }

        float y1 = 0.0f;
        if (out_j1 < d_c) {
            const float* pi_row1 = params.Pi + (int64_t)out_j1 * d_c;
            for (int i = 0; i < d_c; i += 4) {
                y1 += s_q[i]     * __ldg(pi_row1 + i);
                y1 += s_q[i + 1] * __ldg(pi_row1 + i + 1);
                y1 += s_q[i + 2] * __ldg(pi_row1 + i + 2);
                y1 += s_q[i + 3] * __ldg(pi_row1 + i + 3);
            }
        }

        q_out[out_j0] = y0;
        if (out_j1 < d_c) {
            q_out[out_j1] = y1;
        }
    }
}

// ── Tiled variant (TD-TQ-SPARSE-DECODE follow-up) ───────────────────────────
// The per-head kernel above re-reads the whole [d_c, d_c] Pi matrix from DRAM
// for EVERY head (batch_heads × 1 MB = 64 MB at d_c=512, h=64 — measured
// 94 µs/layer, Pi-bound). This variant tiles Pi through shared memory ONCE
// per 32-head group: grid (d_c/16 j-tiles, ceil(bh/32) head-groups), each CTA
// owning 16 output coords × 32 heads. Per-(head, j) accumulation stays a
// single running sum over ascending i — bit-identical to the legacy kernel.
namespace {
constexpr int kJTile = 16, kHTile = 32, kIChunk = 128, kRotThreads = 256;
}

__global__ void __launch_bounds__(kRotThreads)
tq_q_rotate_tiled_kernel(const TqQRotateParams params) {
    const int d_c = params.d_c;
    const int j0 = blockIdx.x * kJTile;
    const int h0 = blockIdx.y * kHTile;
    const int in_stride = params.q_row_stride > 0 ? params.q_row_stride : d_c;
    __shared__ float sPi[kJTile][kIChunk];
    __shared__ __nv_bfloat16 sQ[kHTile][kIChunk];
    const int tj = threadIdx.x >> 4;      // j-local 0..15
    const int thp = threadIdx.x & 15;     // head-pair 0..15
    const int h1 = h0 + thp * 2, h2 = h1 + 1;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (int ic = 0; ic < d_c; ic += kIChunk) {
        for (int t = threadIdx.x; t < kJTile * kIChunk; t += kRotThreads) {
            const int jl = t / kIChunk, ii = t % kIChunk;
            sPi[jl][ii] = __ldg(params.Pi + (int64_t)(j0 + jl) * d_c + ic + ii);
        }
        for (int t = threadIdx.x; t < kHTile * kIChunk; t += kRotThreads) {
            const int hl = t / kIChunk, ii = t % kIChunk;
            const int hg = h0 + hl;
            sQ[hl][ii] = (hg < params.batch_heads)
                ? __ldg(params.q_nope + (int64_t)hg * in_stride + ic + ii)
                : __nv_bfloat16(0.0f);
        }
        __syncthreads();
        #pragma unroll 4
        for (int i = 0; i < kIChunk; ++i) {
            const float p = sPi[tj][i];
            acc1 += __bfloat162float(sQ[thp * 2][i]) * p;
            acc2 += __bfloat162float(sQ[thp * 2 + 1][i]) * p;
        }
        __syncthreads();
    }
    if (h1 < params.batch_heads)
        params.q_rot[(int64_t)h1 * d_c + j0 + tj] = acc1;
    if (h2 < params.batch_heads)
        params.q_rot[(int64_t)h2 * d_c + j0 + tj] = acc2;
}

// ── Warp-per-output GEMV (§12l rotate optimization, round 2) ────────────────
// out[h][j] = Σ_k in[h][k] · M[j][k] for M with ROW-major dot rows (M = Pi for
// the forward rotation; M = Pi^T for the inverse — see TqResources::device_Pi_t).
// 8 warps/CTA, each warp owns ONE output j for a 4-head register tile: Pi row
// traffic drops 16× vs per-head CTAs and lanes stride k (coalesced M reads,
// conflict-free smem input). Reduction = fixed shuffle tree per (h, j) —
// deterministic per shape. ~16 serial FMA-iterations per lane.
namespace {
constexpr int kGvWarps = 8, kGvHeads = 4;

template <bool kInBf16, bool kOutBf16>
__global__ void __launch_bounds__(kGvWarps * 32)
tq_rotate_rows_kernel(const void* __restrict__ in, const float* __restrict__ M,
                      void* __restrict__ out, int batch_heads, int d_c,
                      int in_row_stride) {
    const int j = blockIdx.x * kGvWarps + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int h0 = blockIdx.y * kGvHeads;
    __shared__ float sIn[kGvHeads][512];
    for (int t = threadIdx.x; t < kGvHeads * d_c; t += kGvWarps * 32) {
        const int hl = t / d_c, k = t % d_c;
        const int hg = h0 + hl;
        float v = 0.0f;
        if (hg < batch_heads) {
            if (kInBf16)
                v = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(
                    in)[(int64_t)hg * in_row_stride + k]);
            else
                v = reinterpret_cast<const float*>(
                    in)[(int64_t)hg * in_row_stride + k];
        }
        sIn[hl][k] = v;
    }
    __syncthreads();
    if (j >= d_c) return;
    const float* mrow = M + (int64_t)j * d_c;
    float acc[kGvHeads] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lane; k < d_c; k += 32) {
        const float m = __ldg(mrow + k);
        #pragma unroll
        for (int t = 0; t < kGvHeads; ++t) acc[t] += sIn[t][k] * m;
    }
    #pragma unroll
    for (int t = 0; t < kGvHeads; ++t) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            acc[t] += __shfl_xor_sync(0xffffffff, acc[t], o);
    }
    if (lane < kGvHeads) {
        const int hg = h0 + lane;
        if (hg < batch_heads) {
            if (kOutBf16)
                reinterpret_cast<__nv_bfloat16*>(
                    out)[(int64_t)hg * d_c + j] = __float2bfloat16_rn(acc[lane]);
            else
                reinterpret_cast<float*>(out)[(int64_t)hg * d_c + j] = acc[lane];
        }
    }
}
}  // namespace

// Host launchers for both directions (shared with tq_v_rotate_back.cu).
void run_tq_rotate_rows_bf16_to_f32(const __nv_bfloat16* in, const float* M,
                                    float* out, int batch_heads, int d_c,
                                    int in_row_stride, cudaStream_t stream) {
    dim3 grid((d_c + kGvWarps - 1) / kGvWarps,
              (batch_heads + kGvHeads - 1) / kGvHeads);
    tq_rotate_rows_kernel<true, false><<<grid, kGvWarps * 32, 0, stream>>>(
        in, M, out, batch_heads, d_c, in_row_stride);
}
void run_tq_rotate_rows_f32_to_bf16(const float* in, const float* M,
                                    __nv_bfloat16* out, int batch_heads,
                                    int d_c, cudaStream_t stream) {
    dim3 grid((d_c + kGvWarps - 1) / kGvWarps,
              (batch_heads + kGvHeads - 1) / kGvHeads);
    tq_rotate_rows_kernel<false, true><<<grid, kGvWarps * 32, 0, stream>>>(
        in, M, out, batch_heads, d_c, d_c);
}

namespace {
inline bool rotate_legacy_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LS_TQ_ROTATE_LEGACY");
        return v && v[0] == '1';
    }();
    return on;
}
}

void run_tq_q_rotate(const TqQRotateParams& params, cudaStream_t stream) {
    if (!rotate_legacy_enabled() && params.d_c <= 512) {
        run_tq_rotate_rows_bf16_to_f32(
            params.q_nope, params.Pi, params.q_rot, params.batch_heads,
            params.d_c,
            params.q_row_stride > 0 ? params.q_row_stride : params.d_c,
            stream);
        return;
    }
    if (!rotate_legacy_enabled()
        && params.d_c % kIChunk == 0 && params.d_c % kJTile == 0) {
        dim3 grid(params.d_c / kJTile,
                  (params.batch_heads + kHTile - 1) / kHTile);
        tq_q_rotate_tiled_kernel<<<grid, kRotThreads, 0, stream>>>(params);
        return;
    }
    int smem_bytes = params.d_c * sizeof(float);
    tq_q_rotate_kernel<<<params.batch_heads, 256, smem_bytes, stream>>>(params);
}

}  // namespace sm120::prep
