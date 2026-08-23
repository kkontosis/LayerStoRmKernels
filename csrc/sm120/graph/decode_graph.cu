/***************************************************************************************************
 * SM120 Decode CUDA Graph Runner — Implementation
 *
 * Captures: fused_q_quant → splitkv_mla → mla_combine into a replayable graph.
 *
 * The graph is topology-fixed: grid dims are baked in at capture time.
 * Per-step data (Q, seqlens, block_table, indices) is copied into fixed-address
 * buffers before each replay — the graph reads updated data from stable addresses.
 **************************************************************************************************/

#include "decode_graph.h"

#include "../decode/sparse_fp8/params.h"
#include "../decode/dense_fp8/params.h"
#include "../prep/fused_q_quant.h"

// Forward declarations for kernel launch functions
namespace sm120::decode::sparse_fp8 {
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);
}
namespace sm120::decode::dense_fp8 {
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_dense_fp8_kernel(const DenseAttnDecodeParams &params);
}

namespace sm120::graph {

using namespace sm120::decode;

//==============================================================================
// Build param structs pointing to fixed-address buffers
//==============================================================================
void DecodeGraphRunner::build_param_structs() {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int d_rope = cfg_.d_qk - cfg_.d_nope;

    // --- FusedQQuant params ---
    q_quant_params_.q_bf16       = static_cast<const __nv_bfloat16*>(buf_q_bf16_);
    q_quant_params_.q_nope_fp8   = static_cast<__nv_fp8_e4m3*>(buf_q_nope_fp8_);
    q_quant_params_.q_rope_bf16  = static_cast<__nv_bfloat16*>(buf_q_rope_bf16_);
    q_quant_params_.q_scales     = static_cast<float*>(buf_q_scales_);
    q_quant_params_.s_q          = b * sq;  // grid spans all queries in batch
    q_quant_params_.h_q          = hq;
    q_quant_params_.d_qk         = cfg_.d_qk;
    q_quant_params_.d_nope       = cfg_.d_nope;

    // --- MlaCombine params (shared by sparse and dense) ---
    {
        auto& c = combine_params_;
        memset(&c, 0, sizeof(c));
        c.b = b;
        c.h_q = hq;
        c.h_k = cfg_.h_kv;
        c.q_seq_per_hk = (hq / cfg_.h_kv) * sq;
        c.d_v = cfg_.d_v;

        c.o_ptr = buf_out_;
        c.softmax_lse_ptr = buf_lse_;

        c.o_batch_stride = sq * hq * cfg_.d_v;
        c.o_head_stride = sq * cfg_.d_v;
        c.o_row_stride = cfg_.d_v;

        c.num_splits_ptr = static_cast<int*>(buf_num_splits_);
        c.num_sm_parts = cfg_.num_sm_parts;
        c.softmax_lseaccum_ptr = buf_lse_accum_;
        c.oaccum_ptr = buf_o_accum_;
    }

    if (cfg_.sparse) {
        // --- Sparse decode params ---
        auto& p = sparse_params_;
        p.b = b; p.s_q = sq;
        p.h_q = hq; p.h_kv = cfg_.h_kv;
        p.d_qk = cfg_.d_qk; p.d_v = cfg_.d_v;
        p.sm_scale = cfg_.sm_scale;
        p.sm_scale_div_log2 = cfg_.sm_scale / logf(2.0f);
        p.num_blocks = 0;  // unused for our kernel variant
        p.page_block_size = cfg_.page_block_size;
        p.topk = cfg_.topk;
        p.model_type = cfg_.model_type;

        // Fixed buffer pointers
        p.q      = static_cast<cutlass::bfloat16_t*>(buf_q_nope_fp8_);  // FP8 cast in kernel
        p.q_rope = static_cast<cutlass::bfloat16_t*>(buf_q_rope_bf16_);
        p.q_scales = static_cast<float*>(buf_q_scales_);
        p.kv     = static_cast<cutlass::bfloat16_t*>(cfg_.kv_cache);    // stable cache base
        p.indices = static_cast<int*>(buf_indices_);
        p.topk_length = nullptr;
        p.attn_sink = nullptr;

        p.out = static_cast<cutlass::bfloat16_t*>(buf_out_);
        p.lse = static_cast<float*>(buf_lse_);

        // Strides
        p.stride_q_b = sq * hq * cfg_.d_nope;
        p.stride_q_s_q = hq * cfg_.d_nope;
        p.stride_q_h_q = cfg_.d_nope;
        p.stride_kv_block = cfg_.kv_stride_block;
        p.stride_kv_row = cfg_.kv_stride_row;
        p.stride_indices_b = sq * cfg_.topk;
        p.stride_indices_s_q = cfg_.topk;
        p.stride_lse_b = sq * hq;
        p.stride_lse_s_q = hq;
        p.stride_o_b = sq * hq * cfg_.d_v;
        p.stride_o_s_q = hq * cfg_.d_v;
        p.stride_o_h_q = cfg_.d_v;

        // Extra KV (unused by default)
        p.extra_num_blocks = 0;
        p.extra_page_block_size = 0;
        p.extra_topk = cfg_.extra_topk;
        p.extra_kv = nullptr;
        p.extra_indices = nullptr;
        p.extra_topk_length = nullptr;
        p.stride_extra_kv_block = 0;
        p.stride_extra_kv_row = 0;
        p.stride_extra_indices_b = 0;
        p.stride_extra_indices_s_q = 0;

        // Split-KV
        p.lse_accum = static_cast<float*>(buf_lse_accum_);
        p.o_accum   = static_cast<float*>(buf_o_accum_);
        p.stride_lse_accum_split = b * sq * hq;
        p.stride_lse_accum_s_q = hq;
        p.stride_o_accum_split = b * sq * hq * cfg_.d_v;
        p.stride_o_accum_s_q = hq * cfg_.d_v;
        p.stride_o_accum_h_q = cfg_.d_v;
        p.tile_scheduler_metadata_ptr = static_cast<sparse_fp8::DecodingSchedMeta*>(buf_sched_meta_);
        p.num_splits_ptr = static_cast<int*>(buf_num_splits_);
        p.num_sm_parts = cfg_.num_sm_parts;
        p.stream = nullptr;  // set during capture
    } else {
        // --- Dense decode params ---
        auto& p = dense_params_;
        p.b = b; p.s_q = sq;
        p.h_q = hq; p.h_kv = cfg_.h_kv;
        p.d_qk = cfg_.d_qk; p.d_v = cfg_.d_v;
        p.d_nope = cfg_.d_nope;
        p.sm_scale = cfg_.sm_scale;

        p.q_nope = static_cast<cutlass::bfloat16_t*>(buf_q_nope_fp8_);
        p.q_rope = static_cast<cutlass::bfloat16_t*>(buf_q_rope_bf16_);
        p.q_scales = static_cast<float*>(buf_q_scales_);
        p.kv_cache = static_cast<cutlass::bfloat16_t*>(cfg_.kv_cache);
        p.stride_kv_block = cfg_.kv_stride_block;
        p.stride_kv_row = cfg_.kv_stride_row;

        p.block_table = static_cast<int*>(buf_block_table_);
        p.block_table_batch_stride = cfg_.max_num_blocks_per_seq;
        p.page_block_size = cfg_.page_block_size;
        p.seqlens_k = static_cast<int*>(buf_seqlens_k_);

        p.out = static_cast<cutlass::bfloat16_t*>(buf_out_);
        p.lse = static_cast<float*>(buf_lse_);
        p.stride_o_b = sq * hq * cfg_.d_v;
        p.stride_o_s_q = hq * cfg_.d_v;
        p.stride_o_h_q = cfg_.d_v;
        p.stride_lse_b = sq * hq;
        p.stride_lse_s_q = hq;

        p.lse_accum = static_cast<float*>(buf_lse_accum_);
        p.o_accum   = static_cast<float*>(buf_o_accum_);
        p.stride_lse_accum_split = b * sq * hq;
        p.stride_lse_accum_s_q = hq;
        p.stride_o_accum_split = b * sq * hq * cfg_.d_v;
        p.stride_o_accum_s_q = hq * cfg_.d_v;
        p.stride_o_accum_h_q = cfg_.d_v;
        p.tile_scheduler_metadata_ptr = static_cast<sparse_fp8::DecodingSchedMeta*>(buf_sched_meta_);
        p.num_splits_ptr = static_cast<int*>(buf_num_splits_);
        p.num_sm_parts = cfg_.num_sm_parts;
        p.deterministic_reduce = cfg_.deterministic_reduce;
        p.stream = nullptr;
    }
}

//==============================================================================
// Capture: record fused_q_quant → decode → combine into a CUDA graph
//==============================================================================
void DecodeGraphRunner::capture_graph(cudaStream_t stream) {
    // Use a dedicated capture stream
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    // --- Node 1: fused_q_quant ---
    sm120::prep::run_fused_q_quant(q_quant_params_, capture_stream);

    // --- Node 2: decode attention kernel ---
    // Dispatch by model type. The stream param inside the params struct is
    // ignored — the kernel is launched on capture_stream.
    if (cfg_.sparse) {
        sparse_params_.stream = capture_stream;
        if (cfg_.model_type == ModelType::V32)
            sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(sparse_params_);
        else
            sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(sparse_params_);
    } else {
        dense_params_.stream = capture_stream;
        if (dense_params_.deterministic_reduce) {
            if (cfg_.model_type == ModelType::V32)
                dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64, true>(dense_params_);
            else
                dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, true>(dense_params_);
        } else {
            if (cfg_.model_type == ModelType::V32)
                dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64, false>(dense_params_);
            else
                dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, false>(dense_params_);
        }
    }

    // --- Node 3: mla_combine ---
    // Merge split-KV partial results into final output. When num_sm_parts=1
    // (no split), the combine kernel early-returns (~1 µs overhead).
    run_mla_combine_kernel<cutlass::bfloat16_t>(combine_params_, capture_stream);

    cudaStreamEndCapture(capture_stream, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    cudaStreamDestroy(capture_stream);
}

}  // namespace sm120::graph
