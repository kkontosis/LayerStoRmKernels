#pragma once
/***************************************************************************************************
 * Fused-Q-Quant: Per-token Q quantization + RoPE scale injection
 *
 * Single kernel that:
 * 1. Computes per-head amax over Q_NOPE dims → scale = amax / 448.0
 * 2. Quantizes Q_NOPE to FP8 e4m3: Q_nope_fp8 = Q_nope_bf16 / scale
 * 3. Pre-scales Q_ROPE by inverse content scale: Q_R' = Q_R / scale
 * 4. Stores Q_ROPE as BF16 (preserves ROPE precision, enables BF16 MMA for ROPE tile)
 * 5. Outputs:
 *    - q_nope_fp8: [s_q, h_q, d_nope] FP8
 *    - q_rope_bf16: [s_q, h_q, d_rope] BF16 (pre-scaled)
 *    - q_scales: [s_q, h_q] float32
 *
 * Grid: (s_q * h_q, 1, 1) — one CTA per (query_token, head) pair
 * Block: 256 threads
 *
 * Optimized: single-pass amax+quantize (reads NOPE data once, not twice),
 * vectorized __nv_bfloat162 loads (halves memory transactions).
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

#ifndef SM120_PREP_FP8_E4M3_MAX
#define SM120_PREP_FP8_E4M3_MAX
static constexpr float FP8_E4M3_MAX = 448.0f;
#endif

struct FusedQQuantParams {
    const __nv_bfloat16* __restrict__ q_bf16;    // [s_q, h_q, d_qk] input Q in bf16
    __nv_fp8_e4m3* __restrict__ q_nope_fp8;      // [s_q, h_q, d_nope] output Q_NOPE in FP8
    __nv_bfloat16* __restrict__ q_rope_bf16;     // [s_q, h_q, d_rope] output Q_ROPE in BF16 (pre-scaled)
    float* __restrict__ q_scales;                // [s_q, h_q] output per-head Q scales
    int s_q;
    int h_q;
    int d_qk;        // Total Q dim (e.g., 576 for V3.2, 512 for MODEL1)
    int d_nope;       // NOPE dims (e.g., 512 for V3.2, 448 for MODEL1)
    // d_rope = d_qk - d_nope (e.g., 64)
};

void run_fused_q_quant(const FusedQQuantParams& params, cudaStream_t stream);

}  // namespace sm120::prep
