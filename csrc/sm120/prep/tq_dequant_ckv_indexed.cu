#pragma once
#include "tq_dequant_ckv_indexed.h"

namespace sm120::prep {

__global__ void __launch_bounds__(256)
tq_dequant_ckv_indexed_kernel(const TqDequantCKVIndexedParams params) {
    const int fetch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    if (fetch_idx >= params.num_fetch) return;

    const int d_c = params.d_c;
    const int d_rope = params.d_rope;
    const int packed_nope_bytes = d_c / 2;

    // Resolve paged cache source
    int slot = __ldg(params.indices + fetch_idx);
    int page_idx = slot / params.page_size;
    int row_in_page = slot % params.page_size;
    const uint8_t* cache_row = params.kv_cache +
                                (int64_t)page_idx * params.cache_stride_block +
                                (int64_t)row_in_page * params.cache_stride_row;

    // Shared memory: y_hat[d_c] (codebook-looked-up rotated coords) + centroids[16]
    extern __shared__ float smem[];
    float* s_y_hat = smem;                // [d_c]
    float* s_centroids = smem + d_c;      // [16]

    // Load codebook centroids to smem
    if (tid < 16) {
        s_centroids[tid] = params.centroids[tid];
    }
    __syncthreads();

    // =========================================================================
    // Step 1: Unpack 4-bit indices → codebook lookup → s_y_hat
    // Each thread handles 1 packed byte = 2 coordinates
    // =========================================================================
    if (tid < packed_nope_bytes) {
        uint8_t packed = cache_row[tid];
        int idx0 = packed & 0x0F;
        int idx1 = (packed >> 4) & 0x0F;
        s_y_hat[tid * 2]     = s_centroids[idx0];
        s_y_hat[tid * 2 + 1] = s_centroids[idx1];
    }
    __syncthreads();

    // Read norm (FP16)
    __half norm_fp16 = *reinterpret_cast<const __half*>(cache_row + packed_nope_bytes);
    float norm = __half2float(norm_fp16);

    // =========================================================================
    // Step 2: Inverse rotation: c_hat = y_hat @ Pi, then scale by norm
    //
    // c_hat[j] = sum_{i=0}^{d_c-1} y_hat[i] * Pi[i][j]
    //          = column j of Pi dotted with y_hat
    //          = y_hat @ Pi (row j of result)
    //
    // But since Pi is stored row-major as Pi[row][col],
    // c_hat[j] = sum_i y_hat[i] * Pi[i][j]
    // = sum_i s_y_hat[i] * Pi[i * d_c + j]
    //
    // Each thread computes 2 output coordinates (same as k_append).
    // =========================================================================
    __nv_bfloat16* nope_out = params.k_out + (int64_t)fetch_idx * (d_c + d_rope);

    if (tid * 2 < d_c) {
        int out_j0 = tid * 2;
        int out_j1 = tid * 2 + 1;

        float c0 = 0.0f;
        float c1 = 0.0f;
        for (int i = 0; i < d_c; i += 4) {
            float y0 = s_y_hat[i];
            float y1 = s_y_hat[i + 1];
            float y2 = s_y_hat[i + 2];
            float y3 = s_y_hat[i + 3];

            // Pi[i][out_j0] = Pi[i * d_c + out_j0]
            c0 += y0 * __ldg(params.Pi + (int64_t)i * d_c + out_j0);
            c0 += y1 * __ldg(params.Pi + (int64_t)(i + 1) * d_c + out_j0);
            c0 += y2 * __ldg(params.Pi + (int64_t)(i + 2) * d_c + out_j0);
            c0 += y3 * __ldg(params.Pi + (int64_t)(i + 3) * d_c + out_j0);

            if (out_j1 < d_c) {
                c1 += y0 * __ldg(params.Pi + (int64_t)i * d_c + out_j1);
                c1 += y1 * __ldg(params.Pi + (int64_t)(i + 1) * d_c + out_j1);
                c1 += y2 * __ldg(params.Pi + (int64_t)(i + 2) * d_c + out_j1);
                c1 += y3 * __ldg(params.Pi + (int64_t)(i + 3) * d_c + out_j1);
            }
        }

        // Scale by norm and write BF16
        nope_out[out_j0] = __float2bfloat16_rn(c0 * norm);
        if (out_j1 < d_c) {
            nope_out[out_j1] = __float2bfloat16_rn(c1 * norm);
        }
    }

    // =========================================================================
    // Step 3: Copy ROPE (BF16 → BF16, direct copy)
    // =========================================================================
    const __nv_bfloat16* rope_src = reinterpret_cast<const __nv_bfloat16*>(
        cache_row + packed_nope_bytes + 2);  // after packed + FP16 norm
    __nv_bfloat16* rope_out = nope_out + d_c;

    for (int d = tid; d < d_rope; d += num_threads) {
        rope_out[d] = __ldg(rope_src + d);
    }
}

void run_tq_dequant_ckv_indexed(const TqDequantCKVIndexedParams& params, cudaStream_t stream) {
    int smem_bytes = (params.d_c + 16) * sizeof(float);
    tq_dequant_ckv_indexed_kernel<<<params.num_fetch, 256, smem_bytes, stream>>>(params);
}

}  // namespace sm120::prep
