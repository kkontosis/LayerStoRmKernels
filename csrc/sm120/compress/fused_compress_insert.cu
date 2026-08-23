#pragma once
#include "fused_compress_insert.h"

namespace sm120::compress {

static constexpr float FP8_E4M3_MAX = 448.0f;

// In-register FP8 quantize: compute amax across 256 threads (each holds 2 values),
// then quantize and write FP8 bytes to cache. Returns scale.
__device__ float quantize_and_write_fp8(
    float v0, float v1, int base, int head_dim,
    uint8_t* __restrict__ dst, int tid
) {
    float local_amax = (base < head_dim) ? fmaxf(fabsf(v0), fabsf(v1)) : 0.0f;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffff, local_amax, offset));

    __shared__ float sWarpMax[8];
    if (tid % 32 == 0)
        sWarpMax[tid / 32] = local_amax;
    __syncthreads();
    if (tid == 0) {
        float m = sWarpMax[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) m = fmaxf(m, sWarpMax[w]);
        sWarpMax[0] = m;
    }
    __syncthreads();

    float scale = sWarpMax[0] / FP8_E4M3_MAX;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    __nv_fp8_e4m3* out = reinterpret_cast<__nv_fp8_e4m3*>(dst);
    if (base < head_dim) {
        out[base]     = __nv_fp8_e4m3(fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, v0 * inv_scale)));
        out[base + 1] = __nv_fp8_e4m3(fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, v1 * inv_scale)));
    }
    return scale;
}

// ============================================================================
// CSA fused: window=8, stride=4
// ============================================================================
__global__ void __launch_bounds__(256)
fused_csa_compress_insert_kernel(const FusedCompressInsertParams params) {
    const int comp_idx = blockIdx.x;
    if (comp_idx >= params.num_compressed) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;
    const int window = params.window;
    const int stride = params.stride;
    const int win_start = comp_idx * stride;

    // V4 cache entry layout
    const int k_scale_off = hd;
    const int k_rope_off  = hd + 4;
    const int v_nope_off  = hd + 4 + rd * 2;
    const int v_scale_off = v_nope_off + hd;
    const int entry_bytes = v_scale_off + 4;  // 1160

    int slot = __ldg(params.slot_mapping + comp_idx);
    uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    // --- Step 1: Softmax gate weights ---
    __shared__ float s_weights[8];
    if (tid < window) {
        float gw = __bfloat162float(__ldg(params.gate_weights + tid));
        float pb = params.positional_bias
            ? __bfloat162float(__ldg(params.positional_bias + tid)) : 0.0f;
        s_weights[tid] = gw + pb;
    }
    __syncthreads();
    if (tid == 0) {
        float max_val = s_weights[0];
        for (int j = 1; j < window; ++j) max_val = fmaxf(max_val, s_weights[j]);
        float sum_exp = 0.0f;
        for (int j = 0; j < window; ++j) {
            s_weights[j] = expf(s_weights[j] - max_val);
            sum_exp += s_weights[j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < window; ++j) s_weights[j] *= inv_sum;
    }
    __syncthreads();

    float w[8];
    for (int j = 0; j < window; ++j) w[j] = s_weights[j];

    // --- Step 2+4: Weighted sum K_nope → FP8 quantize → cache write ---
    const int d0 = tid * 2, d1 = tid * 2 + 1;
    float acc_k0 = 0.0f, acc_k1 = 0.0f;
    float acc_v0 = 0.0f, acc_v1 = 0.0f;

    if (d0 < hd) {
        for (int j = 0; j < window; ++j) {
            int token = win_start + j;
            __nv_bfloat162 kp = __ldg(reinterpret_cast<const __nv_bfloat162*>(
                params.input_k_nope + (int64_t)token * hd) + tid);
            acc_k0 += w[j] * __low2float(kp);
            acc_k1 += w[j] * __high2float(kp);

            __nv_bfloat162 vp = __ldg(reinterpret_cast<const __nv_bfloat162*>(
                params.input_v + (int64_t)token * hd) + tid);
            acc_v0 += w[j] * __low2float(vp);
            acc_v1 += w[j] * __high2float(vp);
        }
    }

    // Quantize K_nope to FP8 and write to cache
    float k_scale = quantize_and_write_fp8(acc_k0, acc_k1, d0, hd, entry, tid);
    if (tid == 0) *reinterpret_cast<float*>(entry + k_scale_off) = k_scale;

    // --- Step 3: Weighted sum K_rope_raw + apply RoPE → BF16 to cache ---
    const int half_rope = rd / 2;
    if (tid < half_rope) {
        float acc_even = 0.0f, acc_odd = 0.0f;
        for (int j = 0; j < window; ++j) {
            int token = win_start + j;
            const __nv_bfloat16* row = params.input_k_rope_raw + (int64_t)token * rd;
            acc_even += w[j] * __bfloat162float(__ldg(row + 2 * tid));
            acc_odd  += w[j] * __bfloat162float(__ldg(row + 2 * tid + 1));
        }
        int rope_pos = win_start + window - 1;
        float cos_val = __ldg(params.compress_cos + rope_pos * params.cos_sin_stride + tid);
        float sin_val = __ldg(params.compress_sin + rope_pos * params.cos_sin_stride + tid);
        float out_even = acc_even * cos_val - acc_odd * sin_val;
        float out_odd  = acc_even * sin_val + acc_odd * cos_val;

        __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
        rope_out[2 * tid]     = __float2bfloat16_rn(out_even);
        rope_out[2 * tid + 1] = __float2bfloat16_rn(out_odd);
    }

    __syncthreads();

    // Quantize V_nope to FP8 and write to cache
    float v_scale = quantize_and_write_fp8(acc_v0, acc_v1, d0, hd, entry + v_nope_off, tid);
    if (tid == 0) *reinterpret_cast<float*>(entry + v_scale_off) = v_scale;
}

// ============================================================================
// HCA fused: window=128, stride=128
// Register-accumulation (128 tokens streamed from global, no smem for data)
// ============================================================================
__global__ void __launch_bounds__(256)
fused_hca_compress_insert_kernel(const FusedCompressInsertParams params) {
    const int comp_idx = blockIdx.x;
    if (comp_idx >= params.num_compressed) return;

    const int tid = threadIdx.x;
    const int hd = params.head_dim;
    const int rd = params.qk_rope_head_dim;
    const int window = params.window;
    const int stride = params.stride;
    const int win_start = comp_idx * stride;

    const int k_scale_off = hd;
    const int k_rope_off  = hd + 4;
    const int v_nope_off  = hd + 4 + rd * 2;
    const int v_scale_off = v_nope_off + hd;
    const int entry_bytes = v_scale_off + 4;

    int slot = __ldg(params.slot_mapping + comp_idx);
    uint8_t* entry = params.kv_cache + (int64_t)slot * entry_bytes;

    // --- Softmax gate weights (128 values) ---
    __shared__ float s_weights[128];
    for (int i = tid; i < window; i += 256) {
        float gw = __bfloat162float(__ldg(params.gate_weights + i));
        s_weights[i] = gw;
    }
    __syncthreads();
    if (tid == 0) {
        float max_val = s_weights[0];
        for (int j = 1; j < window; ++j) max_val = fmaxf(max_val, s_weights[j]);
        float sum_exp = 0.0f;
        for (int j = 0; j < window; ++j) {
            s_weights[j] = expf(s_weights[j] - max_val);
            sum_exp += s_weights[j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < window; ++j) s_weights[j] *= inv_sum;
    }
    __syncthreads();

    // --- Weighted sum K_nope + V (register accumulation over 128 tokens) ---
    const int d0 = tid * 2, d1 = tid * 2 + 1;
    float acc_k0 = 0.0f, acc_k1 = 0.0f;
    float acc_v0 = 0.0f, acc_v1 = 0.0f;

    if (d0 < hd) {
        for (int j = 0; j < window; ++j) {
            float wj = s_weights[j];
            int token = win_start + j;
            __nv_bfloat162 kp = __ldg(reinterpret_cast<const __nv_bfloat162*>(
                params.input_k_nope + (int64_t)token * hd) + tid);
            acc_k0 += wj * __low2float(kp);
            acc_k1 += wj * __high2float(kp);

            __nv_bfloat162 vp = __ldg(reinterpret_cast<const __nv_bfloat162*>(
                params.input_v + (int64_t)token * hd) + tid);
            acc_v0 += wj * __low2float(vp);
            acc_v1 += wj * __high2float(vp);
        }
    }

    float k_scale = quantize_and_write_fp8(acc_k0, acc_k1, d0, hd, entry, tid);
    if (tid == 0) *reinterpret_cast<float*>(entry + k_scale_off) = k_scale;

    // --- K_rope weighted sum + RoPE ---
    const int half_rope = rd / 2;
    if (tid < half_rope) {
        float acc_even = 0.0f, acc_odd = 0.0f;
        for (int j = 0; j < window; ++j) {
            float wj = s_weights[j];
            int token = win_start + j;
            const __nv_bfloat16* row = params.input_k_rope_raw + (int64_t)token * rd;
            acc_even += wj * __bfloat162float(__ldg(row + 2 * tid));
            acc_odd  += wj * __bfloat162float(__ldg(row + 2 * tid + 1));
        }
        int rope_pos = win_start + window - 1;
        float cos_val = __ldg(params.compress_cos + rope_pos * params.cos_sin_stride + tid);
        float sin_val = __ldg(params.compress_sin + rope_pos * params.cos_sin_stride + tid);

        __nv_bfloat16* rope_out = reinterpret_cast<__nv_bfloat16*>(entry + k_rope_off);
        rope_out[2 * tid]     = __float2bfloat16_rn(acc_even * cos_val - acc_odd * sin_val);
        rope_out[2 * tid + 1] = __float2bfloat16_rn(acc_even * sin_val + acc_odd * cos_val);
    }

    __syncthreads();

    float v_scale = quantize_and_write_fp8(acc_v0, acc_v1, d0, hd, entry + v_nope_off, tid);
    if (tid == 0) *reinterpret_cast<float*>(entry + v_scale_off) = v_scale;
}

// Launch wrappers
void run_fused_csa_compress_insert(const FusedCompressInsertParams& params, cudaStream_t stream) {
    if (params.num_compressed <= 0) return;
    fused_csa_compress_insert_kernel<<<params.num_compressed, 256, 0, stream>>>(params);
}

void run_fused_hca_compress_insert(const FusedCompressInsertParams& params, cudaStream_t stream) {
    if (params.num_compressed <= 0) return;
    fused_hca_compress_insert_kernel<<<params.num_compressed, 256, 0, stream>>>(params);
}

}  // namespace sm120::compress
