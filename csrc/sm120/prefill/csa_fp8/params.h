#pragma once

//==============================================================================
// SM120 CSA FP8 Prefill — Parameters
//
// DeepSeek V4 Compressed Sparse Attention prefill with BF16 dequanted KV.
// Non-absorbed format: separate K and V tensors (K_NOPE ≠ V_NOPE in V4).
// Single KV head broadcast to all Q heads.
//
// Adapted from dense/fwd/head64/params.h — key differences:
//   - Separate K [s_kv, d_qk] and V [s_kv, d_v] pointers (not absorbed kv)
//   - Per-query causal masking via causal_seqlens array
//   - Head-group indexing for h_q > 64
//
// Source attribution: kernel adapted from FlashMLA dense prefill (MIT
// License, Copyright (c) 2025 DeepSeek — see THIRD_PARTY_NOTICES.md)
//==============================================================================

#include <cuda_runtime.h>
#include "cutlass/bfloat16.h"

namespace sm120::prefill::csa_fp8 {

struct CsaFp8PrefillParams {
    int s_q, s_kv, h_q, d_qk, d_v;
    float sm_scale, sm_scale_div_log2;

    // Q: BF16 [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ q;

    // K: BF16 [s_kv, d_qk] — single KV head, broadcast to all Q heads
    cutlass::bfloat16_t* __restrict__ k;

    // V: BF16 [s_kv, d_v] — single KV head
    cutlass::bfloat16_t* __restrict__ v;

    // Per-query causal mask: causal_seqlens[i] = #KV tokens visible to query i
    // nullptr = non-causal (all s_kv tokens visible to every query)
    const int* __restrict__ causal_seqlens;

    // Strides (elements, not bytes)
    int stride_q_s_q, stride_q_h_q;
    int stride_k_s_kv;   // = d_qk for contiguous K
    int stride_v_s_kv;   // = d_v for contiguous V

    // Output: BF16 [s_q, h_q, d_v], FP32 LSE [s_q, h_q]
    cutlass::bfloat16_t* __restrict__ out;
    float* __restrict__ lse;

    int num_sm;
    cudaStream_t stream;
};

template<int D_QK>
void run_csa_fp8_prefill_kernel(const CsaFp8PrefillParams& params);

}  // namespace sm120::prefill::csa_fp8
