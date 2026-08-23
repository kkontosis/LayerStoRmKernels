#pragma once
/***************************************************************************************************
 * V4 CSA FP8 Decode Graph Runner
 *
 * Captures CSA decode pipeline: splitkv_csa → mla_combine into a replayable graph.
 * Attend-only variant (no compression/indexing — those run outside the graph).
 *
 * Mirrors SnapMLA DecodeGraphRunner API: init/update/replay/get_output.
 *
 * Nodes captured:
 *   1. splitkv_csa_kernel — sparse compressed + SWA FP8 attention
 *   2. mla_combine_kernel<BF16> — merge split-KV partial results
 *
 * Fixed-address buffer strategy: pre-allocate at init, copy-update before replay.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../decode/csa_fp8/params.h"
#include "../../smxx/mla_combine.h"

namespace sm120::graph {

struct CsaFp8DecodeGraphConfig {
    int batch_size;
    int s_q;                        // 1 for decode
    int h_q;                        // 128 (Pro) or 64 (Flash)
    int topk;                       // 1024 (Pro) or 512 (Flash)
    float sm_scale;
    int num_sm_parts;

    // Compressed KV cache (FP8, V4 format)
    void* compressed_kv;            // stable base pointer
    int compressed_page_block_size;

    // SWA KV cache
    void* swa_kv;                   // stable base pointer
    int swa_page_block_size;
    int max_swa_blocks;             // max pages per batch item for SWA
};

class CsaFp8DecodeGraphRunner {
public:
    CsaFp8DecodeGraphRunner() = default;
    ~CsaFp8DecodeGraphRunner() { destroy(); }

    CsaFp8DecodeGraphRunner(const CsaFp8DecodeGraphRunner&) = delete;
    CsaFp8DecodeGraphRunner& operator=(const CsaFp8DecodeGraphRunner&) = delete;

    void init(const CsaFp8DecodeGraphConfig& cfg, cudaStream_t stream);

    void update(
        const void* q_nope_bf16,    // [b, s_q, h_q, 512] BF16
        const void* q_rope_bf16,    // [b, s_q, h_q, 64] BF16
        const int* sparse_indices,  // [b, s_q, topk] int32
        const int* swa_block_table, // [b, max_swa_blocks] int32
        const int* swa_seqlens,     // [b] int32
        cudaStream_t stream
    );

    void replay(cudaStream_t stream);

    __nv_bfloat16* out_ptr() { return static_cast<__nv_bfloat16*>(buf_out_); }
    float* lse_ptr() { return static_cast<float*>(buf_lse_); }
    void* sched_meta_ptr() { return buf_sched_meta_; }
    int* num_splits_ptr() { return static_cast<int*>(buf_num_splits_); }

    void destroy();

private:
    CsaFp8DecodeGraphConfig cfg_{};
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    // Fixed-address buffers
    void* buf_q_nope_ = nullptr;        // [b, s_q, h_q, 512] BF16
    void* buf_q_rope_ = nullptr;        // [b, s_q, h_q, 64] BF16
    void* buf_sparse_indices_ = nullptr; // [b, s_q, topk] int32
    void* buf_swa_block_table_ = nullptr;// [b, max_swa_blocks] int32
    void* buf_swa_seqlens_ = nullptr;    // [b] int32
    void* buf_out_ = nullptr;           // [b, s_q, h_q, 512] BF16
    void* buf_lse_ = nullptr;           // [b, s_q, h_q] float
    void* buf_lse_accum_ = nullptr;
    void* buf_o_accum_ = nullptr;
    void* buf_sched_meta_ = nullptr;
    void* buf_num_splits_ = nullptr;

    sm120::decode::csa_fp8::CsaFp8DecodeParams decode_params_{};
    MlaCombineParams combine_params_{};

    void allocate_buffers();
    void free_buffers();
    void build_params();
    void capture(cudaStream_t stream);
};

}  // namespace sm120::graph
