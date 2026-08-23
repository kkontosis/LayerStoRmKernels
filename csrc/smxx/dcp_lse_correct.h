#pragma once

// DCP (Decode Context Parallelism) LSE Correction kernel.
//
// Corrects attention outputs when KV cache is sequence-sharded across
// multiple GPUs. Each GPU computes attention on its local KV shard,
// producing a partial output + LSE. After NCCL allgather of LSEs,
// this kernel reweights each rank's output by exp(local_lse - global_lse)
// to produce a properly normalized result.
//
// Mathematically identical to mla_combine (log-sum-exp rescaling), but
// simpler: fixed split count (dcp_size), no scheduling metadata, flat
// tensor layout. Arch-generic — no SM-specific instructions.
//
// Based on vLLM's _correct_attn_cp_out_kernel (Apache-2.0).
// ref/vllm/vllm/v1/attention/ops/common.py

#include <cuda_runtime.h>

struct DcpLseCorrectParams {
    // Input: this rank's partial attention output
    void* __restrict__ output;          // [B, H, D] BF16, corrected in-place
    // Input: all-gathered LSE from all DCP ranks
    const float* __restrict__ lses;     // [N, B, H] FP32 (N = dcp_size)
    // Output: final global LSE
    float* __restrict__ global_lse;     // [B, H] FP32

    int B;          // batch size (num tokens)
    int H;          // num heads
    int D;          // head dim (d_v, e.g. 512)
    int N;          // dcp_size (number of ranks)
    int rank;       // this rank's index in the DCP group

    // Strides (element counts, not bytes)
    int stride_o_B, stride_o_H, stride_o_D;
    int stride_lse_N, stride_lse_B, stride_lse_H;
};

void run_dcp_lse_correct_kernel(const DcpLseCorrectParams& params, cudaStream_t stream);
