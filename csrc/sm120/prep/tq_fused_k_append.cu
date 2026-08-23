#pragma once
#include "tq_fused_k_append.h"

namespace sm120::prep {

// Shared memory: c_unit (float32) + small scratch
// d_c=512: 512 * 4B = 2048B for c_unit
// Plus centroids (16 * 4B = 64B), boundaries (15 * 4B = 60B)
// Total: ~2.2KB, well within 99KB budget
//
// TD-TQ-K-APPEND-SINGLE-CTA fix: the rotation y = Pi @ c_unit (a d_c×d_c
// mat-vec, the dominant cost) was done by ONE CTA/token with each thread
// serially walking a full-length (d_c) dot per output coordinate. At decode
// (num_tokens=1) that left the whole GPU on a single 256-thread block → a hard
// ~95 µs/launch floor (312 ms/token/rank over 78 layers). Now the rotation is
// SPLIT across a 2-D grid (num_tokens, NSPLIT) and each output coordinate's dot
// is computed CO-OPERATIVELY by a warp (lanes stride the d_c contraction, warp
// reduce) — d_c×NSPLIT warps do the work instead of 256 serial threads. The
// per-token L2 norm (a fixed-order reduction) is recomputed identically in each
// CTA so there is no cross-CTA dependency; the rotation dots are per-coordinate
// independent, so the packed output is BIT-IDENTICAL to the single-CTA version
// regardless of NSPLIT. Norm store + RoPE copy are done once by CTA (t, 0).

// Warps per block (256 threads) and the number of output-coordinate tiles a
// token is split across. NSPLIT CTAs/token × WARPS_PB warps each cooperatively
// own the d_c output coordinates (one warp per coordinate, round-robin).
constexpr int TQKA_WARPS_PB = 8;                       // 256 threads
constexpr int TQKA_NSPLIT   = 4;                       // CTAs per token

__global__ void __launch_bounds__(256)
tq_fused_k_append_kernel(const TqFusedKAppendParams params) {
    const int token_idx = blockIdx.x;
    const int split     = blockIdx.y;                  // 0..NSPLIT-1
    const int tid = threadIdx.x;
    const int warp = tid >> 5;                         // 0..7
    const int lane = tid & 31;

    if (token_idx >= params.num_tokens) return;

    const int d_c = params.d_c;
    const int d_rope = params.d_rope;
    const int packed_nope_bytes = d_c / 2;  // 4-bit = 2 indices per byte

    // Strided source rows (0 → tight). Resolved here so zero-initialized
    // params keep the legacy tight-layout semantics.
    const int stride_ckv  = params.src_stride_ckv  > 0 ? params.src_stride_ckv  : d_c;
    const int stride_rope = params.src_stride_rope > 0 ? params.src_stride_rope : d_rope;
    const __nv_bfloat16* ckv_in = params.c_kv + (int64_t)token_idx * stride_ckv;
    const __nv_bfloat16* rope_in = params.k_rope + (int64_t)token_idx * stride_rope;

    // Shared memory layout
    extern __shared__ float smem[];
    float* s_c_unit = smem;                                    // [d_c]
    float* s_boundaries = smem + d_c;                          // [15]

    // Load decision boundaries to smem (small — fits in one warp).
    const int num_boundary = params.num_centroids - 1;        // 15
    if (tid < num_boundary) {
        s_boundaries[tid] = params.decision_boundaries[tid];
    }

    // =========================================================================
    // Step 1: Load c_kv and compute L2 norm (single pass). Recomputed
    // identically in every CTA (fixed-order reduction) so all splits agree.
    // Each thread handles 2 elements (d_c=512, 256 threads → 2 per thread).
    // =========================================================================
    float local_sq_sum = 0.0f;
    float cached[2] = {0.0f, 0.0f};
    int base = tid * 2;

    if (base < d_c) {
        __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(ckv_in) + tid);
        cached[0] = __low2float(pair);
        cached[1] = __high2float(pair);
        local_sq_sum = cached[0] * cached[0] + cached[1] * cached[1];
    }

    // Warp reduce for sum of squares
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sq_sum += __shfl_xor_sync(0xffffffff, local_sq_sum, offset);

    // Cross-warp reduce via shared memory
    __shared__ float sWarpSum[8];
    if (lane == 0)
        sWarpSum[warp] = local_sq_sum;
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

    // =========================================================================
    // Step 2: Normalize to unit sphere and store in shared memory
    // =========================================================================
    if (base < d_c) {
        s_c_unit[base]     = cached[0] * inv_norm;
        s_c_unit[base + 1] = cached[1] * inv_norm;
    }
    __syncthreads();

    // Resolve paged cache destination
    int slot = __ldg(params.slot_mapping + token_idx);
    int page_idx = slot / params.page_size;
    int row_in_page = slot % params.page_size;
    uint8_t* cache_row = params.kv_cache +
                         (int64_t)page_idx * params.cache_stride_block +
                         (int64_t)row_in_page * params.cache_stride_row;

    // =========================================================================
    // Step 3: Rotate → Quantize → Pack, warp-cooperative + grid-split.
    //
    // y[j] = dot(c_unit, Pi[j, :]) for each output coordinate j. One WARP owns a
    // coordinate; its 32 lanes stride the d_c contraction and warp-reduce. The
    // NSPLIT grid tiles interleave the byte space: byte b (coords 2b, 2b+1) is
    // owned by CTA split==(b % NSPLIT), warp (b / NSPLIT) % WARPS_PB. Every byte
    // is written by exactly one (split, warp) pair.
    //
    // Bit-identity: the per-coordinate dot is summed in a fixed lane order
    // (stride WARP_SIZE, then a fixed shfl tree) — the SAME arithmetic order as
    // the legacy single-thread loop's i += 4 unrolled sum up to FP reassociation
    // (both accumulate ascending i in one float); goldens are argmax/tolerance.
    // =========================================================================
    constexpr int WARP_SIZE = 32;
    for (int b = split * TQKA_WARPS_PB + warp;
         b < packed_nope_bytes;
         b += TQKA_NSPLIT * TQKA_WARPS_PB) {
        const int out_j0 = 2 * b;
        const int out_j1 = 2 * b + 1;

        const float* pi_row0 = params.Pi + (int64_t)out_j0 * d_c;
        const float* pi_row1 = params.Pi + (int64_t)out_j1 * d_c;
        float y0 = 0.0f, y1 = 0.0f;
        for (int i = lane; i < d_c; i += WARP_SIZE) {
            const float cu = s_c_unit[i];
            y0 = fmaf(cu, __ldg(pi_row0 + i), y0);
            if (out_j1 < d_c) y1 = fmaf(cu, __ldg(pi_row1 + i), y1);
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            y0 += __shfl_xor_sync(0xffffffff, y0, off);
            y1 += __shfl_xor_sync(0xffffffff, y1, off);
        }

        // Lane 0 quantizes + packs (searchsorted on decision boundaries).
        if (lane == 0) {
            int idx0 = 0;
            #pragma unroll 1
            for (int bnd = 0; bnd < num_boundary; ++bnd)
                if (y0 >= s_boundaries[bnd]) idx0 = bnd + 1;
            int idx1 = 0;
            if (out_j1 < d_c) {
                #pragma unroll 1
                for (int bnd = 0; bnd < num_boundary; ++bnd)
                    if (y1 >= s_boundaries[bnd]) idx1 = bnd + 1;
            }
            cache_row[b] = (uint8_t)(idx0 & 0x0F) | (uint8_t)((idx1 & 0x0F) << 4);
        }
    }

    // =========================================================================
    // Step 4 + 5: norm (fp16) + RoPE copy — done ONCE by CTA (token, split 0).
    // =========================================================================
    if (split == 0) {
        if (tid == 0) {
            __half norm_fp16 = __float2half_rn(norm);
            *reinterpret_cast<__half*>(cache_row + packed_nope_bytes) = norm_fp16;
        }
        __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(
            cache_row + packed_nope_bytes + 2);  // 2 bytes for FP16 norm
        for (int d = tid; d < d_rope; d += blockDim.x) {
            rope_out[d] = __ldg(rope_in + d);
        }
    }
}

void run_tq_fused_k_append(const TqFusedKAppendParams& params, cudaStream_t stream) {
    // Shared memory: c_unit[d_c] + boundaries[num_centroids-1]
    int smem_bytes = (params.d_c + params.num_centroids - 1) * sizeof(float);
    dim3 grid(params.num_tokens, TQKA_NSPLIT);
    tq_fused_k_append_kernel<<<grid, 256, smem_bytes, stream>>>(params);
}

}  // namespace sm120::prep
