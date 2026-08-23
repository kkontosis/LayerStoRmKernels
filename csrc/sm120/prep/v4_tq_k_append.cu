#pragma once
#include "v4_tq_k_append.h"

namespace sm120::prep {

// TQ quantize a single d-dimensional vector: normalize → rotate → searchsorted → pack.
// Writes packed 4-bit indices (d/2 bytes) to dst, returns L2 norm.
// Requires: s_unit[d] in shared memory (written by this function), s_centroids[16],
//           s_boundaries[15] already loaded.
__device__ float v4_tq_quantize_and_pack(
    const __nv_bfloat16* __restrict__ src, int d,
    const float* __restrict__ Pi,
    float* __restrict__ s_unit,
    const float* __restrict__ s_centroids,
    const float* __restrict__ s_boundaries,
    int num_centroids,
    uint8_t* __restrict__ dst,
    int tid
) {
    const int packed_bytes = d / 2;

    // Step 1: Load and compute L2 norm (single pass)
    float local_sq_sum = 0.0f;
    float cached[2] = {0.0f, 0.0f};
    int base = tid * 2;

    if (base < d) {
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(src) + tid);
        cached[0] = __low2float(pair);
        cached[1] = __high2float(pair);
        local_sq_sum = cached[0] * cached[0] + cached[1] * cached[1];
    }

    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sq_sum += __shfl_xor_sync(0xffffffff, local_sq_sum, offset);

    __shared__ float sWarpSum[8];
    if (tid % 32 == 0)
        sWarpSum[tid / 32] = local_sq_sum;
    __syncthreads();

    float norm_sq;
    if (tid == 0) {
        norm_sq = sWarpSum[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            norm_sq += sWarpSum[w];
        sWarpSum[0] = norm_sq;
    }
    __syncthreads();

    norm_sq = sWarpSum[0];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    // Step 2: Normalize to unit sphere → shared memory
    if (base < d) {
        s_unit[base]     = cached[0] * inv_norm;
        s_unit[base + 1] = cached[1] * inv_norm;
    }
    __syncthreads();

    // Step 3: Rotate → Quantize → Pack
    const int num_boundary = num_centroids - 1;

    if (tid < packed_bytes) {
        int out_j0 = tid * 2;
        int out_j1 = tid * 2 + 1;

        const float* pi_row0 = Pi + (int64_t)out_j0 * d;
        float y0 = 0.0f;
        for (int i = 0; i < d; i += 4) {
            y0 += s_unit[i]     * __ldg(pi_row0 + i);
            y0 += s_unit[i + 1] * __ldg(pi_row0 + i + 1);
            y0 += s_unit[i + 2] * __ldg(pi_row0 + i + 2);
            y0 += s_unit[i + 3] * __ldg(pi_row0 + i + 3);
        }

        float y1 = 0.0f;
        if (out_j1 < d) {
            const float* pi_row1 = Pi + (int64_t)out_j1 * d;
            for (int i = 0; i < d; i += 4) {
                y1 += s_unit[i]     * __ldg(pi_row1 + i);
                y1 += s_unit[i + 1] * __ldg(pi_row1 + i + 1);
                y1 += s_unit[i + 2] * __ldg(pi_row1 + i + 2);
                y1 += s_unit[i + 3] * __ldg(pi_row1 + i + 3);
            }
        }

        int idx0 = 0;
        for (int b = 0; b < num_boundary; ++b) {
            if (y0 >= s_boundaries[b]) idx0 = b + 1;
        }

        int idx1 = 0;
        if (out_j1 < d) {
            for (int b = 0; b < num_boundary; ++b) {
                if (y1 >= s_boundaries[b]) idx1 = b + 1;
            }
        }

        dst[tid] = (uint8_t)(idx0 & 0x0F) | (uint8_t)((idx1 & 0x0F) << 4);
    }
    __syncthreads();

    return norm;
}


__global__ void __launch_bounds__(256)
v4_tq_k_append_kernel(const V4TqKAppendParams params) {
    const int token_idx = blockIdx.x;
    if (token_idx >= params.num_tokens) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;

    // V4 TQ entry layout (644 bytes):
    //   K: [256B packed | 2B norm | 128B rope] = 386B
    //   V: [256B packed | 2B norm]              = 258B
    const int k_packed_bytes = hd / 2;       // 256
    const int k_norm_off = k_packed_bytes;   // 256
    const int k_rope_off = k_norm_off + 2;   // 258
    const int v_off = k_rope_off + rd * 2;   // 386
    const int v_packed_bytes = hd / 2;       // 256
    const int v_norm_off = v_off + v_packed_bytes;  // 642
    const int entry_bytes = v_norm_off + 2;  // 644

    int slot = __ldg(params.slot_mapping + token_idx);
    uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    // Shared memory: unit vector [hd] + centroids [16] + boundaries [15]
    extern __shared__ float smem[];
    float* s_unit = smem;
    float* s_centroids = smem + hd;
    float* s_boundaries = smem + hd + params.num_centroids;

    // Load codebook into shared memory
    if (tid < params.num_centroids) {
        s_centroids[tid] = params.centroids[tid];
    }
    if (tid < params.num_centroids - 1) {
        s_boundaries[tid] = params.decision_boundaries[tid];
    }
    __syncthreads();

    // --- K NOPE: TQ quantize → pack → write ---
    const __nv_bfloat16* k_in = params.k_nope + (int64_t)token_idx * hd;
    float k_norm = v4_tq_quantize_and_pack(
        k_in, hd, params.Pi, s_unit, s_centroids, s_boundaries,
        params.num_centroids, entry, tid);

    // K norm (FP16)
    if (tid == 0) {
        *reinterpret_cast<__half*>(entry + k_norm_off) = __float2half_rn(k_norm);
    }

    // K ROPE (copy as BF16)
    const __nv_bfloat16* rope_in = params.k_rope + (int64_t)token_idx * rd;
    __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
    for (int d = tid; d < rd; d += 256)
        rope_out[d] = __ldg(rope_in + d);

    __syncthreads();

    // --- V NOPE: TQ quantize → pack → write ---
    const __nv_bfloat16* v_in = params.v_nope + (int64_t)token_idx * hd;
    float v_norm = v4_tq_quantize_and_pack(
        v_in, hd, params.Pi, s_unit, s_centroids, s_boundaries,
        params.num_centroids, entry + v_off, tid);

    // V norm (FP16)
    if (tid == 0) {
        *reinterpret_cast<__half*>(entry + v_norm_off) = __float2half_rn(v_norm);
    }
}

void run_v4_tq_k_append(const V4TqKAppendParams& params, cudaStream_t stream) {
    if (params.num_tokens == 0) return;
    int smem_bytes = (params.head_dim + params.num_centroids + params.num_centroids - 1) * sizeof(float);
    v4_tq_k_append_kernel<<<params.num_tokens, 256, smem_bytes, stream>>>(params);
}


// ============================================================
// Decomposed kernels for GEMM-based rotation (V4K-16a)
// ============================================================

__global__ void __launch_bounds__(256)
v4_tq_normalize_kernel(const V4TqNormalizeParams params) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= params.num_vecs) return;

    const int tid = threadIdx.x;
    const int d = params.dim;
    const __nv_bfloat16* src = params.src + (int64_t)vec_idx * d;
    __nv_bfloat16* dst = params.dst_unit + (int64_t)vec_idx * d;

    float local_sq = 0.0f;
    float cached[2] = {0.0f, 0.0f};
    int base = tid * 2;

    if (base < d) {
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(src) + tid);
        cached[0] = __low2float(pair);
        cached[1] = __high2float(pair);
        local_sq = cached[0] * cached[0] + cached[1] * cached[1];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sq += __shfl_xor_sync(0xffffffff, local_sq, offset);

    __shared__ float sWarpSum[8];
    if (tid % 32 == 0)
        sWarpSum[tid / 32] = local_sq;
    __syncthreads();

    float norm_sq;
    if (tid == 0) {
        norm_sq = sWarpSum[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w)
            norm_sq += sWarpSum[w];
        sWarpSum[0] = norm_sq;
    }
    __syncthreads();

    norm_sq = sWarpSum[0];
    float norm = sqrtf(norm_sq);
    float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    if (base < d) {
        __nv_bfloat162 out;
        out.x = __float2bfloat16_rn(cached[0] * inv_norm);
        out.y = __float2bfloat16_rn(cached[1] * inv_norm);
        reinterpret_cast<__nv_bfloat162*>(dst)[tid] = out;
    }

    if (tid == 0)
        params.dst_norms[vec_idx] = norm;
}

void run_v4_tq_normalize(const V4TqNormalizeParams& params, cudaStream_t stream) {
    if (params.num_vecs == 0) return;
    v4_tq_normalize_kernel<<<params.num_vecs, 256, 0, stream>>>(params);
}


__global__ void __launch_bounds__(256)
v4_tq_quant_pack_write_kernel(const V4TqQuantPackWriteParams params) {
    const int token_idx = blockIdx.x;
    if (token_idx >= params.num_tokens) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;
    const int packed_bytes = hd / 2;
    const int num_boundary = params.num_centroids - 1;

    const int k_packed_bytes = hd / 2;
    const int k_norm_off = k_packed_bytes;
    const int k_rope_off = k_norm_off + 2;
    const int v_off = k_rope_off + rd * 2;
    const int v_packed_bytes = hd / 2;
    const int v_norm_off = v_off + v_packed_bytes;
    const int entry_bytes = v_norm_off + 2;

    int slot = __ldg(params.slot_mapping + token_idx);
    uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    __shared__ float s_boundaries[15];
    if (tid < num_boundary)
        s_boundaries[tid] = params.decision_boundaries[tid];
    __syncthreads();

    // --- K: searchsorted + pack ---
    if (tid < packed_bytes) {
        const __nv_bfloat16* k_row = params.k_rot + (int64_t)token_idx * hd;
        int j0 = tid * 2;
        int j1 = tid * 2 + 1;
        float y0 = __bfloat162float(__ldg(k_row + j0));
        float y1 = (j1 < hd) ? __bfloat162float(__ldg(k_row + j1)) : 0.0f;

        int idx0 = 0, idx1 = 0;
        #pragma unroll
        for (int b = 0; b < 15; ++b) {
            if (y0 >= s_boundaries[b]) idx0 = b + 1;
            if (y1 >= s_boundaries[b]) idx1 = b + 1;
        }
        entry[tid] = (uint8_t)(idx0 & 0x0F) | (uint8_t)((idx1 & 0x0F) << 4);
    }

    // K norm
    if (tid == 0)
        *reinterpret_cast<__half*>(entry + k_norm_off) = __float2half_rn(params.k_norms[token_idx]);

    // K rope
    const __nv_bfloat16* rope_in = params.k_rope + (int64_t)token_idx * rd;
    __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
    for (int d = tid; d < rd; d += 256)
        rope_out[d] = __ldg(rope_in + d);

    // --- V: searchsorted + pack ---
    if (tid < packed_bytes) {
        const __nv_bfloat16* v_row = params.v_rot + (int64_t)token_idx * hd;
        int j0 = tid * 2;
        int j1 = tid * 2 + 1;
        float y0 = __bfloat162float(__ldg(v_row + j0));
        float y1 = (j1 < hd) ? __bfloat162float(__ldg(v_row + j1)) : 0.0f;

        int idx0 = 0, idx1 = 0;
        #pragma unroll
        for (int b = 0; b < 15; ++b) {
            if (y0 >= s_boundaries[b]) idx0 = b + 1;
            if (y1 >= s_boundaries[b]) idx1 = b + 1;
        }
        (entry + v_off)[tid] = (uint8_t)(idx0 & 0x0F) | (uint8_t)((idx1 & 0x0F) << 4);
    }

    // V norm
    if (tid == 0)
        *reinterpret_cast<__half*>(entry + v_norm_off) = __float2half_rn(params.v_norms[token_idx]);
}

void run_v4_tq_quant_pack_write(const V4TqQuantPackWriteParams& params, cudaStream_t stream) {
    if (params.num_tokens == 0) return;
    v4_tq_quant_pack_write_kernel<<<params.num_tokens, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
