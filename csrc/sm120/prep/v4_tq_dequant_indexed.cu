#pragma once
#include "v4_tq_dequant_indexed.h"

namespace sm120::prep {

// Dequant one TQ sub-entry: unpack → codebook → inverse rotate → scale → write BF16.
// Reads packed_bytes from cache_row, writes d elements to nope_out.
__device__ void v4_tq_dequant_sub(
    const uint8_t* __restrict__ cache_row,
    int packed_bytes, int d,
    float norm,
    const float* __restrict__ Pi,
    float* __restrict__ s_y_hat,
    const float* __restrict__ s_centroids,
    __nv_bfloat16* __restrict__ nope_out,
    int tid
) {
    // Step 1: Unpack 4-bit → codebook lookup → s_y_hat
    if (tid < packed_bytes) {
        uint8_t packed = cache_row[tid];
        int idx0 = packed & 0x0F;
        int idx1 = (packed >> 4) & 0x0F;
        s_y_hat[tid * 2]     = s_centroids[idx0];
        s_y_hat[tid * 2 + 1] = s_centroids[idx1];
    }
    __syncthreads();

    // Step 2: Inverse rotation: c_hat[j] = sum_i y_hat[i] * Pi[i][j], then scale
    if (tid * 2 < d) {
        int out_j0 = tid * 2;
        int out_j1 = tid * 2 + 1;

        float c0 = 0.0f;
        float c1 = 0.0f;
        for (int i = 0; i < d; i += 4) {
            float y0 = s_y_hat[i];
            float y1 = s_y_hat[i + 1];
            float y2 = s_y_hat[i + 2];
            float y3 = s_y_hat[i + 3];

            c0 += y0 * __ldg(Pi + (int64_t)i * d + out_j0);
            c0 += y1 * __ldg(Pi + (int64_t)(i + 1) * d + out_j0);
            c0 += y2 * __ldg(Pi + (int64_t)(i + 2) * d + out_j0);
            c0 += y3 * __ldg(Pi + (int64_t)(i + 3) * d + out_j0);

            if (out_j1 < d) {
                c1 += y0 * __ldg(Pi + (int64_t)i * d + out_j1);
                c1 += y1 * __ldg(Pi + (int64_t)(i + 1) * d + out_j1);
                c1 += y2 * __ldg(Pi + (int64_t)(i + 2) * d + out_j1);
                c1 += y3 * __ldg(Pi + (int64_t)(i + 3) * d + out_j1);
            }
        }

        nope_out[out_j0] = __float2bfloat16_rn(c0 * norm);
        if (out_j1 < d)
            nope_out[out_j1] = __float2bfloat16_rn(c1 * norm);
    }
    __syncthreads();
}


__global__ void __launch_bounds__(256)
v4_tq_dequant_indexed_kernel(const V4TqDequantIndexedParams params) {
    const int fetch_idx = blockIdx.x;
    if (fetch_idx >= params.num_fetch) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;

    // V4 TQ entry layout (644 bytes)
    const int k_packed_bytes = hd / 2;        // 256
    const int k_norm_off = k_packed_bytes;    // 256
    const int k_rope_off = k_norm_off + 2;    // 258
    const int v_off = k_rope_off + rd * 2;    // 386
    const int v_packed_bytes = hd / 2;        // 256
    const int v_norm_off = v_off + v_packed_bytes;  // 642
    const int entry_bytes = v_norm_off + 2;   // 644

    const int slot = __ldg(params.indices + fetch_idx);
    const uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    // Shared memory: y_hat[hd] + centroids[16]
    extern __shared__ float smem[];
    float* s_y_hat = smem;
    float* s_centroids = smem + hd;

    if (tid < 16) {
        s_centroids[tid] = params.centroids[tid];
    }
    __syncthreads();

    // --- Dequant K NOPE ---
    float k_norm = __half2float(*reinterpret_cast<const __half*>(entry + k_norm_off));
    __nv_bfloat16* k_out = params.k_nope_out + (int64_t)fetch_idx * hd;
    v4_tq_dequant_sub(entry, k_packed_bytes, hd, k_norm,
                       params.Pi, s_y_hat, s_centroids, k_out, tid);

    // --- Copy K ROPE (BF16 → BF16) ---
    const __nv_bfloat16* rope_src = reinterpret_cast<const __nv_bfloat16*>(entry + k_rope_off);
    __nv_bfloat16* rope_out = params.k_rope_out + (int64_t)fetch_idx * rd;
    for (int d = tid; d < rd; d += 256)
        rope_out[d] = __ldg(rope_src + d);

    // --- Dequant V NOPE ---
    float v_norm = __half2float(*reinterpret_cast<const __half*>(entry + v_norm_off));
    __nv_bfloat16* v_out = params.v_nope_out + (int64_t)fetch_idx * hd;
    v4_tq_dequant_sub(entry + v_off, v_packed_bytes, hd, v_norm,
                       params.Pi, s_y_hat, s_centroids, v_out, tid);
}

void run_v4_tq_dequant_indexed(const V4TqDequantIndexedParams& params, cudaStream_t stream) {
    if (params.num_fetch == 0) return;
    int smem_bytes = (params.head_dim + 16) * sizeof(float);
    v4_tq_dequant_indexed_kernel<<<params.num_fetch, 256, smem_bytes, stream>>>(params);
}

}  // namespace sm120::prep
