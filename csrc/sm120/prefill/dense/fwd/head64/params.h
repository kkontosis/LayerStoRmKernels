#pragma once

//==============================================================================
// SM120 Dense Prefill - Parameters (Absorbed BF16 Path)
//
// Dense attention: attends to ALL tokens up to s_kv (no topk/indices).
// Same data types and pipeline as sparse prefill, minus index indirection.
//==============================================================================

#include <cuda_runtime.h>
#include "cutlass/bfloat16.h"

namespace sm120::prefill::dense::head64 {

struct DenseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v;
    float sm_scale, sm_scale_div_log2;

    // Optional per-query-row causal KV bound (device, [s_q]).
    // nullptr: every query row attends all s_kv staged rows (legacy flat mask).
    // non-null: query row i attends staged rows [0, min(s_kv_per_row[i], s_kv)).
    // CAUSAL-CHUNK CONTRACT: kv holds ONE contiguous prefix (the union
    // [0, s_kv)); s_kv_per_row[i] is row i's visible prefix length. Row i's
    // arithmetic (block count, mask, online-softmax order) is then IDENTICAL
    // to a batch_size==1 call with s_kv == s_kv_per_row[i] over the same
    // staging — batched-causal output is bit-equal to the per-row calls.
    const int* __restrict__ s_kv_per_row = nullptr;

    // Input tensors
    cutlass::bfloat16_t* __restrict__ q;    // [s_q, h_q, d_qk]
    cutlass::bfloat16_t* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    float* __restrict__ attn_sink;           // [h_q], may be nullptr

    // Strides
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;

    // Output tensors
    cutlass::bfloat16_t* __restrict__ out;   // [s_q, h_q, d_v]
    float* __restrict__ max_logits;          // [s_q, h_q]
    float* __restrict__ lse;                 // [s_q, h_q]

    int num_sm;
    cudaStream_t stream;

    // Optional gated deterministic (bit-reproducible) softmax-denominator
    // reduction. false (default) = legacy cross-warp atomicAdd path.
    bool deterministic_reduce = false;
};

template<int D_QK, bool DETERMINISTIC>
void run_dense_fwd_phase1_kernel(const DenseAttnFwdParams& params);

}  // namespace sm120::prefill::dense::head64
