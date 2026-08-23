/***************************************************************************************************
 * SM120 TQ Decode CUDA Graph Runner — Implementation
 *
 * Captures: tq_q_rotate → tq_dense_decode → mla_combine<float> → tq_v_rotate_back
 **************************************************************************************************/

#include "tq_decode_graph.h"

namespace sm120::graph {

void TqDecodeGraphRunner::init(const TqDecodeGraphConfig& cfg, cudaStream_t stream) {
    cfg_ = cfg;
    allocate_fixed_buffers();
    build_and_capture(stream);
}

void TqDecodeGraphRunner::update(
    const void* q_nope_bf16, const void* q_rope_bf16,
    const int* seqlens_k, const int* block_table,
    cudaStream_t stream
) {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;

    cudaMemcpyAsync(buf_q_nope_bf16_, q_nope_bf16,
        (size_t)b * sq * hq * cfg_.d_c * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_q_rope_bf16_, q_rope_bf16,
        (size_t)b * sq * hq * cfg_.d_rope * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_seqlens_k_, seqlens_k,
        b * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_block_table_, block_table,
        (size_t)b * cfg_.max_num_blocks_per_seq * sizeof(int),
        cudaMemcpyDeviceToDevice, stream);
}

void TqDecodeGraphRunner::replay(cudaStream_t stream) {
    cudaGraphLaunch(graph_exec_, stream);
}

void TqDecodeGraphRunner::destroy() {
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }
    free_buffers();
}

void TqDecodeGraphRunner::allocate_fixed_buffers() {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int nsp = cfg_.num_sm_parts;
    const int num_q_seqs = hq * sq;

    cudaMalloc(&buf_q_nope_bf16_, (size_t)b * sq * hq * cfg_.d_c * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_q_rope_bf16_, (size_t)b * sq * hq * cfg_.d_rope * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_q_rot_fp32_,  (size_t)b * sq * hq * cfg_.d_c * sizeof(float));
    cudaMalloc(&buf_out_rot_fp32_,(size_t)b * sq * hq * cfg_.d_c * sizeof(float));
    cudaMalloc(&buf_out_bf16_,    (size_t)b * sq * hq * cfg_.d_c * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_lse_,         (size_t)b * sq * hq * sizeof(float));
    cudaMalloc(&buf_seqlens_k_,   b * sizeof(int));
    cudaMalloc(&buf_block_table_, (size_t)b * cfg_.max_num_blocks_per_seq * sizeof(int));
    cudaMalloc(&buf_lse_accum_,   (size_t)nsp * b * num_q_seqs * sizeof(float));
    cudaMalloc(&buf_o_accum_,     (size_t)nsp * b * num_q_seqs * cfg_.d_c * sizeof(float));
    cudaMalloc(&buf_sched_meta_,  nsp * sizeof(sm120::decode::tq_dense::TqDecodingSchedMeta));
    cudaMalloc(&buf_num_splits_,  (b + 1) * sizeof(int));

    // Initialize lse_accum to -inf
    cudaMemset(buf_lse_accum_, 0xFF, (size_t)nsp * b * num_q_seqs * sizeof(float));
    cudaMemset(buf_o_accum_, 0, (size_t)nsp * b * num_q_seqs * cfg_.d_c * sizeof(float));
}

void TqDecodeGraphRunner::free_buffers() {
    auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free(buf_q_nope_bf16_);
    safe_free(buf_q_rope_bf16_);
    safe_free(buf_q_rot_fp32_);
    safe_free(buf_out_rot_fp32_);
    safe_free(buf_out_bf16_);
    safe_free(buf_lse_);
    safe_free(buf_seqlens_k_);
    safe_free(buf_block_table_);
    safe_free(buf_lse_accum_);
    safe_free(buf_o_accum_);
    safe_free(buf_sched_meta_);
    safe_free(buf_num_splits_);
}

void TqDecodeGraphRunner::build_and_capture(cudaStream_t stream) {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int num_q_seqs = hq * sq;
    const int nsp = cfg_.num_sm_parts;
    const int d_c = cfg_.d_c, d_rope = cfg_.d_rope;
    const int packed_bytes = d_c / 2;
    const int row_bytes = packed_bytes + 2 + d_rope * 2;

    // --- Build param structs ---

    // Node 1: tq_q_rotate
    sm120::prep::TqQRotateParams q_rot_params;
    q_rot_params.q_nope = static_cast<const __nv_bfloat16*>(buf_q_nope_bf16_);
    q_rot_params.Pi = cfg_.Pi;
    q_rot_params.q_rot = static_cast<float*>(buf_q_rot_fp32_);
    q_rot_params.batch_heads = b * sq * hq;
    q_rot_params.d_c = d_c;

    // Node 2: tq_dense_decode
    sm120::decode::tq_dense::TqDenseDecodeParams decode_params;
    memset(&decode_params, 0, sizeof(decode_params));
    decode_params.b = b; decode_params.s_q = sq; decode_params.h_q = hq; decode_params.h_kv = 1;
    decode_params.d_c = d_c; decode_params.d_rope = d_rope;
    decode_params.sm_scale = cfg_.sm_scale;
    decode_params.q_rot = static_cast<const float*>(buf_q_rot_fp32_);
    decode_params.q_rope = static_cast<const __nv_bfloat16*>(buf_q_rope_bf16_);
    decode_params.kv_cache = static_cast<const uint8_t*>(cfg_.kv_cache);
    decode_params.cache_stride_block = cfg_.page_block_size * row_bytes;
    decode_params.cache_stride_row = row_bytes;
    decode_params.block_table = static_cast<const int*>(buf_block_table_);
    decode_params.block_table_batch_stride = cfg_.max_num_blocks_per_seq;
    decode_params.page_block_size = cfg_.page_block_size;
    decode_params.seqlens_k = static_cast<const int*>(buf_seqlens_k_);
    decode_params.centroids = cfg_.centroids;
    decode_params.out = static_cast<float*>(buf_out_rot_fp32_);
    decode_params.lse = static_cast<float*>(buf_lse_);
    decode_params.stride_o_b = sq * hq * d_c;
    decode_params.stride_o_s_q = hq * d_c;
    decode_params.stride_o_h_q = d_c;
    decode_params.stride_lse_b = sq * hq;
    decode_params.stride_lse_s_q = hq;
    decode_params.num_sm_parts = nsp;
    decode_params.o_accum = static_cast<float*>(buf_o_accum_);
    decode_params.lse_accum = static_cast<float*>(buf_lse_accum_);
    decode_params.stride_o_accum_split = num_q_seqs * d_c;
    decode_params.stride_o_accum_s_q = hq * d_c;
    decode_params.stride_o_accum_h_q = d_c;
    decode_params.stride_lse_accum_split = num_q_seqs;
    decode_params.stride_lse_accum_s_q = hq;
    decode_params.tile_scheduler_metadata_ptr =
        static_cast<sm120::decode::tq_dense::TqDecodingSchedMeta*>(buf_sched_meta_);
    decode_params.num_splits_ptr = static_cast<int*>(buf_num_splits_);

    // Node 3: mla_combine<float>
    MlaCombineParams combine_params;
    memset(&combine_params, 0, sizeof(combine_params));
    combine_params.b = b; combine_params.h_q = hq; combine_params.h_k = 1;
    combine_params.q_seq_per_hk = num_q_seqs; combine_params.d_v = d_c;
    combine_params.o_ptr = buf_out_rot_fp32_;
    combine_params.softmax_lse_ptr = buf_lse_;
    combine_params.o_batch_stride = sq * hq * d_c;
    combine_params.o_head_stride = sq * d_c;
    combine_params.o_row_stride = d_c;
    combine_params.num_splits_ptr = static_cast<int*>(buf_num_splits_);
    combine_params.num_sm_parts = nsp;
    combine_params.softmax_lseaccum_ptr = buf_lse_accum_;
    combine_params.oaccum_ptr = buf_o_accum_;

    // Node 4: tq_v_rotate_back
    sm120::prep::TqVRotateBackParams vrot_params;
    vrot_params.out_rotated = static_cast<const float*>(buf_out_rot_fp32_);
    vrot_params.Pi = cfg_.Pi;
    vrot_params.out_final = static_cast<__nv_bfloat16*>(buf_out_bf16_);
    vrot_params.batch_heads = b * sq * hq;
    vrot_params.d_c = d_c;

    // --- Capture graph ---
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    // Node 1: q_rotate
    sm120::prep::run_tq_q_rotate(q_rot_params, capture_stream);

    // Node 2: decode
    decode_params.stream = capture_stream;
    sm120::decode::tq_dense::run_tq_dense_decode(decode_params);

    // Node 3: mla_combine (merges split-KV partials)
    run_mla_combine_kernel<float>(combine_params, capture_stream);

    // Node 4: v_rotate_back
    sm120::prep::run_tq_v_rotate_back(vrot_params, capture_stream);

    cudaStreamEndCapture(capture_stream, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    cudaStreamDestroy(capture_stream);
}

}  // namespace sm120::graph
