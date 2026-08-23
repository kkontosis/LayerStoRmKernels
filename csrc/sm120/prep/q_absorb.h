#pragma once
/***************************************************************************************************
 * Q-Absorb: DeepSeek MLA W_UK query absorption + RoPE concat (SM120)
 *
 * Folds the K-up projection (W_UK, the K-half of kv_b_proj) into the query so that
 * attention can run in the compressed kv_lora latent space:
 *
 *     ql_nope[s, h, k] = sum_d  q_nope[s, h, d] * W_UK[h, d, k]      (d in [0,P), k in [0,L))
 *     q_absorbed[s, h, 0:L]   = ql_nope                              (absorbed content, L=kv_lora_rank)
 *     q_absorbed[s, h, L:L+R] = q_rope                              (rope half, copied as-is)
 *
 * Output q_absorbed = [s_q, h_q, L + R] (e.g. 512+64 = 576 for V3.2) is exactly the BF16
 * query the SnapMLA dense/sparse prefill kernels and the decode fused_q_quant consume.
 * No scaling is applied here — the softmax scale lives in the attention kernel and the
 * per-token FP8 amax is computed later by fused_q_quant (decode).
 *
 * Reference math (identical across vLLM mla_attention.py, TensorRT-LLM, SGLang, and the
 * kernel-lib sample generate_samples.py): einsum('shd,hdk->shk', q_nope, W_UK).
 *
 * W_UK source layout: the full kv_b_proj tensor, [h_q*(P+V), L] row-major (P=qk_nope_head_dim,
 * V=v_head_dim, L=kv_lora_rank). Per head h, rows [h*(P+V) .. h*(P+V)+P) are W_UK (K-up);
 * rows [h*(P+V)+P .. +V) are W_UV (V-up, used elsewhere). W_UK[h,d,k] = kv_b_proj[(h*(P+V)+d)*L + k].
 *
 * Weight dtype: BF16 (w_uk_scales == nullptr) or FP8 E4M3 with blockwise float32 scales laid
 * out [ceil(L/128), ceil(h_q*(P+V)/128)] K-major (same convention as kv_bv_extract_dequant).
 * For P==128 each head's K-half is exactly one 128-row scale block (n_block = (h*(P+V))/128),
 * so the FP8 scale is constant over the contraction dim and factors out per output column.
 *
 * Grid: (h_q, s_q, 1) — one CTA per (head, query_token). Block: 256 threads.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct QAbsorbParams {
    const __nv_bfloat16* __restrict__ q_heads;   // [s_q, h_q, d_nope_in + d_rope] BF16 (q_b_proj output)
    const void*          __restrict__ w_uk;      // kv_b_proj base: [h_q*(d_nope_in + d_v), d_c]; BF16 or FP8 E4M3
    const float*         __restrict__ w_uk_scales; // FP8 blockwise scales (K-major); nullptr if BF16
    __nv_bfloat16*       __restrict__ q_absorbed; // [s_q, h_q, d_c + d_rope] BF16 output
    int  s_q;
    int  h_q;
    int  d_nope_in;   // qk_nope_head_dim (P) — contraction dim
    int  d_c;         // kv_lora_rank (L) — absorbed output dim
    int  d_rope;      // qk_rope_head_dim (R)
    int  d_v;         // v_head_dim (V) — sets kv_b_proj per-head row stride (P+V)
    bool weight_is_fp8;

    // GGUF weight path: -1 = not GGUF (use bf16/fp8 branch above); otherwise a
    // layerstorm::compute::GgufType value (0=Q2_K,1=Q3_K,2=Q4_K,3=Q5_K,4=Q6_K,
    // 5=Q8_0). kv_b_proj is then a packed GGUF weight [h_q*(P+V), (L/QK)*bytes]
    // and W_UK is dequanted per element (dequant-only; accuracy-equal to a
    // load-time dequant to BF16). Requires L % QK == 0 (QK=256 k-quants, 32 Q8_0).
    int  gguf_type = -1;

    // Optional fused RoPE on the rope half (interleaved adjacent-pair, see rope_rotate.h).
    // When apply_rope is set, the rope copy rotates by pos = seqlens_k[s] − 1 using the
    // pure cos/sin table ([max_pos][d_rope]: d_rope/2 cos | d_rope/2 sin). Off by default
    // (rope half copied unchanged) so non-RoPE consumers keep the plain absorption.
    bool         apply_rope   = false;
    const int*   __restrict__ seqlens_k = nullptr;  // [s_q]
    const float* __restrict__ cos_sin   = nullptr;  // [max_pos][d_rope]
    int          max_pos      = 0;
};

void run_q_absorb(const QAbsorbParams& params, cudaStream_t stream);

}  // namespace sm120::prep
