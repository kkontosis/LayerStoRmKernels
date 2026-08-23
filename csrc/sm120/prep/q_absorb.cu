#pragma once
#include "q_absorb.h"
#include "formats/gguf_dequant_one.h"

namespace sm120::prep {

// FP8 scale block size (matches kv_b_proj blockwise quantization, 128x128 tiles).
#ifndef SM120_PREP_Q_ABSORB_SCALE_BLOCK
#define SM120_PREP_Q_ABSORB_SCALE_BLOCK
static constexpr int Q_ABSORB_SCALE_BLOCK = 128;
static constexpr int Q_ABSORB_MAX_NOPE = 256;  // upper bound on d_nope_in (P), for shared-mem q_nope cache
#endif

// TD-QABSORB-OCCUPANCY: q_absorb was grid=(h_q, s_q, 1) = ONE CTA per (head,
// token). At decode (s_q=1, h_q=32) that is only 32 CTAs of 256 threads on a
// 170-SM GPU (~48 threads/SM) → severely occupancy-bound (34 µs/launch, 5.3
// ms/token/rank, the top untouched decode kernel). The L absorbed output
// columns are independent, so the grid is now split over L via grid.z
// (Q_ABSORB_LSPLIT CTAs/token/head) to raise the CTA count LSPLIT-fold, and
// inside each CTA the columns are computed WARP-COOPERATIVELY (one warp per
// column, its 32 lanes stride the P contraction + warp-reduce) so all 256
// threads stay busy even with few columns/CTA. q_nope is re-cached per CTA
// (cheap, P<=256 elems). Output is BIT-IDENTICAL up to FP reassociation of the
// per-column dot (the summation order changes from thread-serial to
// lane-strided+shfl-tree; goldens are argmax/tolerance — verify). The rope tail
// is written once by grid.z==0.
static constexpr int Q_ABSORB_LSPLIT = 8;
static constexpr int Q_ABSORB_WARPS  = 8;   // 256 threads = 8 warps (8 cols in flight)

// Concat (and optionally RoPE-rotate) the rope half: q_heads[P..P+R) → q_absorbed[L..L+R).
// Shared by the bf16/fp8 kernel and the GGUF kernel (the weight format only affects the
// absorbed-content contraction, not the rope tail).
__device__ __forceinline__
void q_absorb_write_rope(const QAbsorbParams& params, int s,
                         const __nv_bfloat16* q_in, __nv_bfloat16* q_out,
                         int P, int L, int R, int tid, int nthreads) {
    if (params.apply_rope) {
        int pos = __ldg(params.seqlens_k + s) - 1;
        if (pos < 0) pos = 0;
        if (pos >= params.max_pos) pos = params.max_pos - 1;
        const int half = R / 2;
        const float* cs = params.cos_sin + static_cast<int64_t>(pos) * R;
        for (int i = tid; i < half; i += nthreads) {
            const float c = __ldg(cs + i);
            const float sv = __ldg(cs + half + i);
            const float x0 = __bfloat162float(__ldg(q_in + P + 2 * i));
            const float x1 = __bfloat162float(__ldg(q_in + P + 2 * i + 1));
            q_out[L + 2 * i]     = __float2bfloat16_rn(x0 * c - x1 * sv);
            q_out[L + 2 * i + 1] = __float2bfloat16_rn(x0 * sv + x1 * c);
        }
    } else {
        for (int d = tid; d < R; d += nthreads)
            q_out[L + d] = __ldg(q_in + P + d);
    }
}

// grid.z-split column range [k0, k1) for this CTA over the L output columns.
__device__ __forceinline__ void q_absorb_col_range(int L, int split, int& k0, int& k1) {
    const int per = (L + Q_ABSORB_LSPLIT - 1) / Q_ABSORB_LSPLIT;
    k0 = split * per;
    k1 = k0 + per; if (k1 > L) k1 = L;
}

// One CTA per (head, query_token, L-split). 256 threads cooperatively compute
// this CTA's slice of the L absorbed output columns for (h, s) and copy the R
// rope dims (grid.z==0 only). q_nope[P] is cached in shared and reused across
// all output columns; W_UK is read coalesced across threads (consecutive output
// columns k → consecutive kv_b_proj addresses for a fixed contraction index d).
__global__ void __launch_bounds__(256)
q_absorb_kernel(const QAbsorbParams params) {
    const int h = blockIdx.x;   // head
    const int s = blockIdx.y;   // query token
    const int split = blockIdx.z;
    if (h >= params.h_q || s >= params.s_q) return;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int P = params.d_nope_in;   // contraction dim (qk_nope_head_dim, e.g. 128)
    const int L = params.d_c;         // absorbed output dim (kv_lora_rank, e.g. 512)
    const int R = params.d_rope;      // rope dim (e.g. 64)
    const int V = params.d_v;         // v_head_dim (e.g. 128)
    const int row_in  = P + R;        // q_heads inner stride (e.g. 192)
    const int row_out = L + R;        // q_absorbed inner stride (e.g. 576)
    const int kv_row  = P + V;        // kv_b_proj per-head row count (e.g. 256)

    const __nv_bfloat16* q_in  = params.q_heads    + static_cast<int64_t>(s * params.h_q + h) * row_in;
    __nv_bfloat16*       q_out = params.q_absorbed + static_cast<int64_t>(s * params.h_q + h) * row_out;

    // Cache q_nope[P] in shared memory (reused across every output column).
    __shared__ float sQ[Q_ABSORB_MAX_NOPE];
    for (int d = tid; d < P; d += nthreads)
        sQ[d] = __bfloat162float(__ldg(q_in + d));
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    int k0, k1; q_absorb_col_range(L, split, k0, k1);

    // FP8 scale: each head's K-half (P rows starting at h*kv_row) lies in one 128-row scale
    // block (n_block), so the scale is constant over d and depends only on the k-block.
    const int   n_block        = (h * kv_row) / Q_ABSORB_SCALE_BLOCK;
    const int   n_scale_blocks = (params.h_q * kv_row + Q_ABSORB_SCALE_BLOCK - 1) / Q_ABSORB_SCALE_BLOCK;
    const int64_t head_row0    = static_cast<int64_t>(h * kv_row);  // first W_UK row for this head

    // Warp-per-column: warp `warp` handles columns k0+warp, k0+warp+WARPS, ...
    // its 32 lanes stride the P contraction, then warp-reduce.
    if (params.weight_is_fp8) {
        const uint8_t* w = static_cast<const uint8_t*>(params.w_uk);
        for (int k = k0 + warp; k < k1; k += Q_ABSORB_WARPS) {
            float acc = 0.0f;
            for (int d = lane; d < P; d += 32) {
                const int64_t off = (head_row0 + d) * L + k;
                const __nv_fp8_e4m3 v = *reinterpret_cast<const __nv_fp8_e4m3*>(&w[off]);
                acc = fmaf(sQ[d], float(v), acc);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) {
                const float scale = __ldg(&params.w_uk_scales[
                    (k / Q_ABSORB_SCALE_BLOCK) * n_scale_blocks + n_block]);
                q_out[k] = __float2bfloat16_rn(acc * scale);
            }
        }
    } else {
        const __nv_bfloat16* w = static_cast<const __nv_bfloat16*>(params.w_uk);
        for (int k = k0 + warp; k < k1; k += Q_ABSORB_WARPS) {
            float acc = 0.0f;
            for (int d = lane; d < P; d += 32) {
                const int64_t off = (head_row0 + d) * L + k;
                acc = fmaf(sQ[d], __bfloat162float(__ldg(w + off)), acc);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) q_out[k] = __float2bfloat16_rn(acc);
        }
    }

    // Concat (+ optional RoPE) the rope half — once per (h, s), by grid.z==0.
    if (split == 0)
        q_absorb_write_rope(params, s, q_in, q_out, P, L, R, tid, nthreads);
}

// ---------------------------------------------------------------------------
// GGUF W_UK path: kv_b_proj is a packed GGUF weight, dequanted per element.
// Templated on a per-type policy (VALS values/block, BYTES/block, at(blk,p)).
// ---------------------------------------------------------------------------

namespace gguf_pol {
namespace fmt = layerstorm::formats;
struct Q2K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q2K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q2k_one(*(const fmt::block_q2_K*)b, p);} };
struct Q3K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q3K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q3k_one(*(const fmt::block_q3_K*)b, p);} };
struct Q4K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q4K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q4k_one(*(const fmt::block_q4_K*)b, p);} };
struct Q5K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q5K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q5k_one(*(const fmt::block_q5_K*)b, p);} };
struct Q6K { static constexpr int VALS = 256, BYTES = fmt::BLOCK_Q6K_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q6k_one(*(const fmt::block_q6_K*)b, p);} };
struct Q8_0 { static constexpr int VALS = 32, BYTES = fmt::BLOCK_Q8_0_SIZE;
    __device__ static float at(const void* b, int p){ return fmt::dequant_q8_0_one(*(const fmt::block_q8_0*)b, p);} };
}  // namespace gguf_pol

template <class P_>
__global__ void __launch_bounds__(256)
q_absorb_gguf_kernel(const QAbsorbParams params) {
    const int h = blockIdx.x, s = blockIdx.y;
    const int split = blockIdx.z;
    if (h >= params.h_q || s >= params.s_q) return;
    const int tid = threadIdx.x, nthreads = blockDim.x;

    const int P = params.d_nope_in, L = params.d_c, R = params.d_rope, V = params.d_v;
    const int row_in = P + R, row_out = L + R, kv_row = P + V;
    const int blocks_per_row = L / P_::VALS;

    const __nv_bfloat16* q_in  = params.q_heads    + static_cast<int64_t>(s * params.h_q + h) * row_in;
    __nv_bfloat16*       q_out = params.q_absorbed + static_cast<int64_t>(s * params.h_q + h) * row_out;

    __shared__ float sQ[Q_ABSORB_MAX_NOPE];
    for (int d = tid; d < P; d += nthreads)
        sQ[d] = __bfloat162float(__ldg(q_in + d));
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    int k0, k1; q_absorb_col_range(L, split, k0, k1);

    const uint8_t* w = static_cast<const uint8_t*>(params.w_uk);
    const int64_t head_row0 = static_cast<int64_t>(h * kv_row);

    // Warp-per-column, lanes stride the P contraction + warp-reduce.
    for (int k = k0 + warp; k < k1; k += Q_ABSORB_WARPS) {
        const int kb = k / P_::VALS;
        const int p  = k % P_::VALS;
        float acc = 0.0f;
        for (int d = lane; d < P; d += 32) {
            const int64_t row = head_row0 + d;
            const void* blk = w + (static_cast<int64_t>(row) * blocks_per_row + kb) * P_::BYTES;
            acc = fmaf(sQ[d], P_::at(blk, p), acc);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) q_out[k] = __float2bfloat16_rn(acc);
    }

    if (split == 0)
        q_absorb_write_rope(params, s, q_in, q_out, P, L, R, tid, nthreads);
}

void run_q_absorb(const QAbsorbParams& params, cudaStream_t stream) {
    if (params.s_q <= 0 || params.h_q <= 0 || params.d_c <= 0) return;
    dim3 grid(static_cast<unsigned>(params.h_q),
              static_cast<unsigned>(params.s_q),
              static_cast<unsigned>(Q_ABSORB_LSPLIT));

    if (params.gguf_type >= 0) {
        switch (params.gguf_type) {  // 0=Q2_K,1=Q3_K,2=Q4_K,3=Q5_K,4=Q6_K,5=Q8_0
            case 0: q_absorb_gguf_kernel<gguf_pol::Q2K> <<<grid, 256, 0, stream>>>(params); break;
            case 1: q_absorb_gguf_kernel<gguf_pol::Q3K> <<<grid, 256, 0, stream>>>(params); break;
            case 2: q_absorb_gguf_kernel<gguf_pol::Q4K> <<<grid, 256, 0, stream>>>(params); break;
            case 3: q_absorb_gguf_kernel<gguf_pol::Q5K> <<<grid, 256, 0, stream>>>(params); break;
            case 4: q_absorb_gguf_kernel<gguf_pol::Q6K> <<<grid, 256, 0, stream>>>(params); break;
            case 5: q_absorb_gguf_kernel<gguf_pol::Q8_0><<<grid, 256, 0, stream>>>(params); break;
        }
        return;
    }
    q_absorb_kernel<<<grid, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
