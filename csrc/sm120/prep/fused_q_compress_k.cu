#pragma once
#include "fused_q_compress_k.h"

namespace sm120::prep {

static constexpr float QCK_FP8_MAX = 448.0f;

__global__ void __launch_bounds__(256)
fused_q_compress_k_kernel(const FusedQCompressKParams params) {
    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (block_id < params.h_q) {
        // ============================================================
        // Q RMSNorm: normalize one head's d_qk dimensions
        // ============================================================
        __nv_bfloat16* q_head = params.q + (int64_t)block_id * params.d_qk;
        const int d = params.d_qk;

        // Compute sum of squares (2 elements per thread for d=576, need 288 threads)
        float local_ss = 0.0f;
        for (int i = tid * 2; i < d; i += 512) {
            float v0 = (i < d) ? __bfloat162float(q_head[i]) : 0.0f;
            float v1 = (i + 1 < d) ? __bfloat162float(q_head[i + 1]) : 0.0f;
            local_ss += v0 * v0 + v1 * v1;
        }

        // Warp reduce
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            local_ss += __shfl_xor_sync(0xffffffff, local_ss, offset);

        // Cross-warp reduce
        __shared__ float sWarpSum[8];
        if (tid % 32 == 0) sWarpSum[tid / 32] = local_ss;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 8; ++w) total += sWarpSum[w];
            sWarpSum[0] = total;
        }
        __syncthreads();

        float rms = rsqrtf(sWarpSum[0] / (float)d + params.rms_eps);

        // Normalize in-place
        for (int i = tid * 2; i < d; i += 512) {
            if (i < d)     q_head[i]     = __float2bfloat16_rn(__bfloat162float(q_head[i]) * rms);
            if (i + 1 < d) q_head[i + 1] = __float2bfloat16_rn(__bfloat162float(q_head[i + 1]) * rms);
        }

    } else if (block_id == params.h_q) {
        // ============================================================
        // Compressed K RoPE + FP8 quantize + cache insert
        // ============================================================
        const int hd = params.head_dim;
        const int rd = params.qk_rope_head_dim;
        const int k_scale_off = hd;
        const int k_rope_off  = hd + 4;
        const int v_nope_off  = hd + 4 + rd * 2;
        const int v_scale_off = v_nope_off + hd;
        const int entry_bytes = v_scale_off + 4;

        uint8_t* entry = params.kv_cache + (int64_t)params.slot * entry_bytes;

        // K_nope: FP8 quantize
        const int base = tid * 2;
        float k0 = 0.0f, k1 = 0.0f;
        if (base < hd) {
            __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(params.k_nope) + tid);
            k0 = __low2float(pair);
            k1 = __high2float(pair);
        }

        // amax reduce
        float local_amax = (base < hd) ? fmaxf(fabsf(k0), fabsf(k1)) : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));

        __shared__ float sWarpMax[8];
        if (tid % 32 == 0) sWarpMax[tid / 32] = local_amax;
        __syncthreads();
        if (tid == 0) {
            float m = sWarpMax[0];
            for (int w = 1; w < 8; ++w) m = fmaxf(m, sWarpMax[w]);
            sWarpMax[0] = m;
        }
        __syncthreads();

        float k_scale = sWarpMax[0] / QCK_FP8_MAX;
        float k_inv = (k_scale > 0.0f) ? (1.0f / k_scale) : 0.0f;
        __nv_fp8_e4m3* k_out = reinterpret_cast<__nv_fp8_e4m3*>(entry);
        if (base < hd) {
            k_out[base]     = __nv_fp8_e4m3(fmaxf(-QCK_FP8_MAX, fminf(QCK_FP8_MAX, k0 * k_inv)));
            k_out[base + 1] = __nv_fp8_e4m3(fmaxf(-QCK_FP8_MAX, fminf(QCK_FP8_MAX, k1 * k_inv)));
        }
        if (tid == 0) *reinterpret_cast<float*>(entry + k_scale_off) = k_scale;

        // K_rope: apply compressed RoPE, write BF16
        const int half_rope = rd / 2;
        if (tid < half_rope) {
            float xe = __bfloat162float(__ldg(params.k_rope_raw + 2 * tid));
            float xo = __bfloat162float(__ldg(params.k_rope_raw + 2 * tid + 1));
            float c = __ldg(params.compress_cos + params.rope_position * params.cos_sin_stride + tid);
            float s = __ldg(params.compress_sin + params.rope_position * params.cos_sin_stride + tid);

            __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
            rope_out[2 * tid]     = __float2bfloat16_rn(xe * c - xo * s);
            rope_out[2 * tid + 1] = __float2bfloat16_rn(xe * s + xo * c);
        }

        __syncthreads();

        // V_nope: FP8 quantize
        float v0 = 0.0f, v1 = 0.0f;
        if (base < hd) {
            __nv_bfloat162 pair = __ldg(reinterpret_cast<const __nv_bfloat162*>(params.v_nope) + tid);
            v0 = __low2float(pair);
            v1 = __high2float(pair);
        }

        local_amax = (base < hd) ? fmaxf(fabsf(v0), fabsf(v1)) : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));
        if (tid % 32 == 0) sWarpMax[tid / 32] = local_amax;
        __syncthreads();
        if (tid == 0) {
            float m = sWarpMax[0];
            for (int w = 1; w < 8; ++w) m = fmaxf(m, sWarpMax[w]);
            sWarpMax[0] = m;
        }
        __syncthreads();

        float v_scale = sWarpMax[0] / QCK_FP8_MAX;
        float v_inv = (v_scale > 0.0f) ? (1.0f / v_scale) : 0.0f;
        __nv_fp8_e4m3* v_out = reinterpret_cast<__nv_fp8_e4m3*>(entry + v_nope_off);
        if (base < hd) {
            v_out[base]     = __nv_fp8_e4m3(fmaxf(-QCK_FP8_MAX, fminf(QCK_FP8_MAX, v0 * v_inv)));
            v_out[base + 1] = __nv_fp8_e4m3(fmaxf(-QCK_FP8_MAX, fminf(QCK_FP8_MAX, v1 * v_inv)));
        }
        if (tid == 0) *reinterpret_cast<float*>(entry + v_scale_off) = v_scale;
    }
}

void run_fused_q_compress_k(const FusedQCompressKParams& params, cudaStream_t stream) {
    int grid = params.h_q + 1;
    fused_q_compress_k_kernel<<<grid, 256, 0, stream>>>(params);
}

}  // namespace sm120::prep
