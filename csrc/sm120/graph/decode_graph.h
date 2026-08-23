#pragma once

/***************************************************************************************************
 * SM120 Decode CUDA Graph Runner
 *
 * Captures the decode pipeline into a replayable CUDA graph:
 *   1. fused_q_quant_kernel  — quantize Q BF16 → FP8 NOPE + BF16 ROPE + scales
 *   2. splitkv_mla_kernel    — sparse or dense FP8 attention
 *   3. mla_combine_kernel    — merge split-KV partial results
 *
 * Usage:
 *   DecodeGraphRunner runner;
 *   runner.init(cfg, stream);          // allocate fixed buffers, capture graph
 *   // per decode step:
 *   runner.update(q_bf16, seqlens_k, block_table, indices, stream);
 *   runner.replay(stream);
 *   // read from runner.out_ptr() / runner.lse_ptr()
 *   runner.destroy();
 *
 * Design: fixed-address buffer strategy (same as vLLM/SGLang-FluentLLM CUDA graph runners).
 * The graph captures kernel launches pointing to pre-allocated device buffers.
 * Before each replay, live data is copied into these buffers via cudaMemcpyAsync.
 * The graph reads updated data from the same addresses — no node patching needed.
 *
 * Requirements:
 *   - batch_size, s_q, h_q, num_sm_parts must be constant across replays
 *     (grid dimensions are baked into the graph)
 *   - kv_cache base pointer must be stable (same paged pool)
 *   - seqlens_k, block_table, indices change per step (copied into fixed buffers)
 *
 * Thread safety: NOT thread-safe. One runner per stream.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

#include "../components/sparse_config.h"
#include "../decode/sparse_fp8/params.h"
#include "../decode/dense_fp8/params.h"
#include "../prep/fused_q_quant.h"
#include "../../smxx/mla_combine.h"

namespace sm120::graph {

using sm120::sparse::ModelType;

//==============================================================================
// Configuration — fixed at graph capture time
//==============================================================================
struct DecodeGraphConfig {
    // Batch / head geometry (must be constant across replays)
    int batch_size;
    int s_q;                    // typically 1 for decode
    int h_q;                    // number of Q heads (e.g., 64)
    int h_kv;                   // number of KV heads (e.g., 1 for MLA)
    int d_qk;                   // total QK dim (576 for V3.2, 512 for MODEL1)
    int d_v;                    // V dim (512)
    int d_nope;                 // NOPE dims (512 for V3.2, 448 for MODEL1)

    // KV cache geometry
    int page_block_size;        // tokens per page (e.g., 64)
    int max_num_blocks_per_seq; // max pages per sequence
    int kv_stride_block;        // stride between page blocks (as FP8 element count)
    int kv_stride_row;          // stride between rows in page (as FP8 element count)
    void* kv_cache;             // paged KV cache base (stable across replays)

    // Attention config
    float sm_scale;
    ModelType model_type;

    // Split-KV
    int num_sm_parts;           // number of SM partitions for split-KV

    // Sparse-specific (set sparse=false for dense)
    bool sparse;
    int topk;                   // only used if sparse=true
    int extra_topk;             // only used if sparse=true, 0 if none

    // Optional gated deterministic (bit-reproducible) softmax-denominator
    // reduction (dense path). false (default) = legacy cross-warp atomicAdd.
    bool deterministic_reduce = false;
};

//==============================================================================
// Decode Graph Runner
//==============================================================================
class DecodeGraphRunner {
public:
    DecodeGraphRunner() = default;
    ~DecodeGraphRunner() { destroy(); }

    // Non-copyable
    DecodeGraphRunner(const DecodeGraphRunner&) = delete;
    DecodeGraphRunner& operator=(const DecodeGraphRunner&) = delete;

    //--------------------------------------------------------------------------
    // init: allocate fixed buffers, capture the decode pipeline as a CUDA graph
    //--------------------------------------------------------------------------
    void init(const DecodeGraphConfig& cfg, cudaStream_t stream) {
        cfg_ = cfg;
        allocate_fixed_buffers();
        build_param_structs();
        capture_graph(stream);
    }

    //--------------------------------------------------------------------------
    // update: copy live per-step data into graph's fixed buffers
    //
    // q_bf16:      [b, s_q, h_q, d_qk]  device ptr, BF16 query for this step
    // seqlens_k:   [b]                   device ptr, current sequence lengths
    // block_table: [b, max_blocks]       device ptr, page table
    // indices:     [b, s_q, topk]        device ptr, topk indices (sparse only, nullptr for dense)
    //--------------------------------------------------------------------------
    void update(
        const void* q_bf16,
        const int* seqlens_k,
        const int* block_table,
        const int* indices,
        cudaStream_t stream
    ) {
        // Copy Q input (consumed by fused_q_quant)
        size_t q_bytes = (size_t)cfg_.batch_size * cfg_.s_q * cfg_.h_q * cfg_.d_qk * sizeof(__nv_bfloat16);
        cudaMemcpyAsync(buf_q_bf16_, q_bf16, q_bytes, cudaMemcpyDeviceToDevice, stream);

        // Copy sequence lengths
        cudaMemcpyAsync(buf_seqlens_k_, seqlens_k,
                        cfg_.batch_size * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        // Copy block table (for dense decode page resolution)
        cudaMemcpyAsync(buf_block_table_, block_table,
                        (size_t)cfg_.batch_size * cfg_.max_num_blocks_per_seq * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Copy topk indices (sparse only)
        if (cfg_.sparse && indices) {
            size_t idx_bytes = (size_t)cfg_.batch_size * cfg_.s_q * cfg_.topk * sizeof(int);
            cudaMemcpyAsync(buf_indices_, indices, idx_bytes, cudaMemcpyDeviceToDevice, stream);
        }
    }

    //--------------------------------------------------------------------------
    // replay: launch the captured graph
    //--------------------------------------------------------------------------
    void replay(cudaStream_t stream) {
        cudaGraphLaunch(graph_exec_, stream);
    }

    //--------------------------------------------------------------------------
    // Output accessors (point to graph's fixed output buffers)
    //--------------------------------------------------------------------------
    __nv_bfloat16* out_ptr()        { return static_cast<__nv_bfloat16*>(buf_out_); }
    float*         lse_ptr()        { return static_cast<float*>(buf_lse_); }
    void*          sched_meta_ptr() { return buf_sched_meta_; }
    int*           num_splits_ptr() { return static_cast<int*>(buf_num_splits_); }

    //--------------------------------------------------------------------------
    // destroy: free all resources
    //--------------------------------------------------------------------------
    void destroy() {
        if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
        if (graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }
        free_buffers();
    }

private:
    DecodeGraphConfig cfg_{};

    // CUDA graph objects
    cudaGraph_t     graph_      = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    // Fixed-address device buffers (addresses baked into graph)
    // -- Q prep I/O --
    void* buf_q_bf16_       = nullptr;  // [b, s_q, h_q, d_qk] BF16 input
    void* buf_q_nope_fp8_   = nullptr;  // [b, s_q, h_q, d_nope] FP8 output
    void* buf_q_rope_bf16_  = nullptr;  // [b, s_q, h_q, d_rope] BF16 output
    void* buf_q_scales_     = nullptr;  // [b, s_q, h_q] float32

    // -- Attention I/O --
    void* buf_seqlens_k_    = nullptr;  // [b] int32
    void* buf_block_table_  = nullptr;  // [b, max_blocks] int32
    void* buf_indices_      = nullptr;  // [b, s_q, topk] int32 (sparse only)
    void* buf_out_          = nullptr;  // [b, s_q, h_q, d_v] BF16 output
    void* buf_lse_          = nullptr;  // [b, s_q, h_q] float32

    // -- Split-KV intermediates --
    void* buf_lse_accum_    = nullptr;  // [num_sm_parts, b, s_q, h_q]
    void* buf_o_accum_      = nullptr;  // [num_sm_parts, b, s_q, h_q, d_v]
    void* buf_sched_meta_   = nullptr;  // [num_sm_parts] DecodingSchedMeta
    void* buf_num_splits_   = nullptr;  // [b+1] cumulative split counts

    //--------------------------------------------------------------------------
    void allocate_fixed_buffers() {
        const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
        const int d_rope = cfg_.d_qk - cfg_.d_nope;

        // Q prep
        cudaMalloc(&buf_q_bf16_,      (size_t)b * sq * hq * cfg_.d_qk * sizeof(__nv_bfloat16));
        cudaMalloc(&buf_q_nope_fp8_,  (size_t)b * sq * hq * cfg_.d_nope * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&buf_q_rope_bf16_, (size_t)b * sq * hq * d_rope * sizeof(__nv_bfloat16));
        cudaMalloc(&buf_q_scales_,    (size_t)b * sq * hq * sizeof(float));

        // Control data
        cudaMalloc(&buf_seqlens_k_,   b * sizeof(int));
        cudaMalloc(&buf_block_table_, (size_t)b * cfg_.max_num_blocks_per_seq * sizeof(int));
        if (cfg_.sparse)
            cudaMalloc(&buf_indices_, (size_t)b * sq * cfg_.topk * sizeof(int));

        // Output
        cudaMalloc(&buf_out_, (size_t)b * sq * hq * cfg_.d_v * sizeof(__nv_bfloat16));
        cudaMalloc(&buf_lse_, (size_t)b * sq * hq * sizeof(float));

        // Split-KV intermediates
        const int nsp = cfg_.num_sm_parts;
        cudaMalloc(&buf_lse_accum_,  (size_t)nsp * b * sq * hq * sizeof(float));
        cudaMalloc(&buf_o_accum_,    (size_t)nsp * b * sq * hq * cfg_.d_v * sizeof(float));
        cudaMalloc(&buf_sched_meta_, nsp * sizeof(sm120::decode::sparse_fp8::DecodingSchedMeta));
        cudaMalloc(&buf_num_splits_, (b + 1) * sizeof(int));
    }

    //--------------------------------------------------------------------------
    void free_buffers() {
        auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
        safe_free(buf_q_bf16_);
        safe_free(buf_q_nope_fp8_);
        safe_free(buf_q_rope_bf16_);
        safe_free(buf_q_scales_);
        safe_free(buf_seqlens_k_);
        safe_free(buf_block_table_);
        safe_free(buf_indices_);
        safe_free(buf_out_);
        safe_free(buf_lse_);
        safe_free(buf_lse_accum_);
        safe_free(buf_o_accum_);
        safe_free(buf_sched_meta_);
        safe_free(buf_num_splits_);
    }

    //--------------------------------------------------------------------------
    void build_param_structs();   // defined in decode_graph.cu
    void capture_graph(cudaStream_t stream);  // defined in decode_graph.cu

    // Stored param structs (built once, used during capture)
    sm120::prep::FusedQQuantParams   q_quant_params_{};
    sm120::decode::sparse_fp8::SparseAttnDecodeParams sparse_params_{};
    sm120::decode::dense_fp8::DenseAttnDecodeParams   dense_params_{};
    MlaCombineParams                 combine_params_{};
};

}  // namespace sm120::graph
