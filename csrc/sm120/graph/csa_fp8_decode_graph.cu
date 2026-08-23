#include "csa_fp8_decode_graph.h"
#include "../decode/csa_fp8/params.h"

namespace sm120::decode::csa_fp8 {
template<int NUM_HEADS>
void run_csa_fp8_decode_kernel(const CsaFp8DecodeParams& params);
}

namespace sm120::graph {

static constexpr int HEAD_DIM = 512;
static constexpr int ROPE_DIM = 64;
static constexpr int ENTRY_BYTES = 1160;

void CsaFp8DecodeGraphRunner::init(const CsaFp8DecodeGraphConfig& cfg, cudaStream_t stream) {
    cfg_ = cfg;
    allocate_buffers();
    build_params();
    capture(stream);
}

void CsaFp8DecodeGraphRunner::update(
    const void* q_nope, const void* q_rope,
    const int* sparse_indices, const int* swa_block_table,
    const int* swa_seqlens, cudaStream_t stream
) {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    cudaMemcpyAsync(buf_q_nope_, q_nope,
        (size_t)b * sq * hq * HEAD_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_q_rope_, q_rope,
        (size_t)b * sq * hq * ROPE_DIM * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_sparse_indices_, sparse_indices,
        (size_t)b * sq * cfg_.topk * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_swa_block_table_, swa_block_table,
        (size_t)b * cfg_.max_swa_blocks * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_swa_seqlens_, swa_seqlens,
        b * sizeof(int), cudaMemcpyDeviceToDevice, stream);
}

void CsaFp8DecodeGraphRunner::replay(cudaStream_t stream) {
    cudaGraphLaunch(graph_exec_, stream);
}

void CsaFp8DecodeGraphRunner::destroy() {
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
    free_buffers();
}

void CsaFp8DecodeGraphRunner::allocate_buffers() {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int nsp = cfg_.num_sm_parts;
    const int num_q_seqs = hq * sq;

    cudaMalloc(&buf_q_nope_, (size_t)b * sq * hq * HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_q_rope_, (size_t)b * sq * hq * ROPE_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_sparse_indices_, (size_t)b * sq * cfg_.topk * sizeof(int));
    cudaMalloc(&buf_swa_block_table_, (size_t)b * cfg_.max_swa_blocks * sizeof(int));
    cudaMalloc(&buf_swa_seqlens_, b * sizeof(int));
    cudaMalloc(&buf_out_, (size_t)b * sq * hq * HEAD_DIM * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_lse_, (size_t)b * sq * hq * sizeof(float));
    cudaMalloc(&buf_lse_accum_, (size_t)nsp * b * num_q_seqs * sizeof(float));
    cudaMalloc(&buf_o_accum_, (size_t)nsp * b * num_q_seqs * HEAD_DIM * sizeof(float));

    using Meta = sm120::decode::csa_fp8::DecodingSchedMeta;
    cudaMalloc(&buf_sched_meta_, nsp * sizeof(Meta));
    cudaMalloc(&buf_num_splits_, (b + 1) * sizeof(int));
}

void CsaFp8DecodeGraphRunner::free_buffers() {
    auto f = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    f(buf_q_nope_); f(buf_q_rope_); f(buf_sparse_indices_);
    f(buf_swa_block_table_); f(buf_swa_seqlens_);
    f(buf_out_); f(buf_lse_);
    f(buf_lse_accum_); f(buf_o_accum_);
    f(buf_sched_meta_); f(buf_num_splits_);
}

void CsaFp8DecodeGraphRunner::build_params() {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int num_q_seqs = hq * sq;

    auto& p = decode_params_;
    memset(&p, 0, sizeof(p));
    p.b = b; p.s_q = sq; p.h_q = hq;
    p.sm_scale = cfg_.sm_scale;
    p.sm_scale_log2 = cfg_.sm_scale * 1.4426950408889634f;

    p.q_nope = static_cast<cutlass::bfloat16_t*>(buf_q_nope_);
    p.q_rope = static_cast<cutlass::bfloat16_t*>(buf_q_rope_);
    p.q_scales = nullptr;

    p.compressed_kv = static_cast<const char*>(cfg_.compressed_kv);
    p.compressed_page_block_size = cfg_.compressed_page_block_size;
    p.stride_compressed_block = (int64_t)cfg_.compressed_page_block_size * ENTRY_BYTES;
    p.stride_compressed_row = ENTRY_BYTES;

    p.sparse_indices = static_cast<const int*>(buf_sparse_indices_);
    p.topk = cfg_.topk;
    p.stride_indices_b = sq * cfg_.topk;
    p.stride_indices_s_q = cfg_.topk;

    p.swa_kv = static_cast<const char*>(cfg_.swa_kv);
    p.swa_page_block_size = cfg_.swa_page_block_size;
    p.stride_swa_block = (int64_t)cfg_.swa_page_block_size * ENTRY_BYTES;
    p.stride_swa_row = ENTRY_BYTES;
    p.swa_block_table = static_cast<const int*>(buf_swa_block_table_);
    p.swa_block_table_stride = cfg_.max_swa_blocks;
    p.swa_seqlens = static_cast<const int*>(buf_swa_seqlens_);

    p.out = static_cast<cutlass::bfloat16_t*>(buf_out_);
    p.lse = static_cast<float*>(buf_lse_);

    p.stride_q_b = sq * hq * HEAD_DIM; p.stride_q_s_q = hq * HEAD_DIM; p.stride_q_h_q = HEAD_DIM;
    p.stride_o_b = sq * hq * HEAD_DIM; p.stride_o_s_q = hq * HEAD_DIM; p.stride_o_h_q = HEAD_DIM;
    p.stride_lse_b = sq * hq; p.stride_lse_s_q = hq;

    p.lse_accum = static_cast<float*>(buf_lse_accum_);
    p.o_accum = static_cast<float*>(buf_o_accum_);
    p.stride_lse_accum_split = num_q_seqs; p.stride_lse_accum_s_q = hq;
    p.stride_o_accum_split = num_q_seqs * HEAD_DIM; p.stride_o_accum_s_q = hq * HEAD_DIM; p.stride_o_accum_h_q = HEAD_DIM;
    p.tile_scheduler_metadata_ptr = static_cast<sm120::decode::csa_fp8::DecodingSchedMeta*>(buf_sched_meta_);
    p.num_splits_ptr = static_cast<int*>(buf_num_splits_);
    p.num_sm_parts = cfg_.num_sm_parts;

    // Combine params
    auto& c = combine_params_;
    memset(&c, 0, sizeof(c));
    c.b = b; c.h_q = hq; c.h_k = 1; c.q_seq_per_hk = hq * sq; c.d_v = HEAD_DIM;
    c.o_ptr = buf_out_; c.softmax_lse_ptr = buf_lse_;
    c.o_batch_stride = sq * hq * HEAD_DIM; c.o_head_stride = sq * HEAD_DIM; c.o_row_stride = HEAD_DIM;
    c.num_splits_ptr = static_cast<int*>(buf_num_splits_);
    c.num_sm_parts = cfg_.num_sm_parts;
    c.softmax_lseaccum_ptr = buf_lse_accum_;
    c.oaccum_ptr = buf_o_accum_;
}

void CsaFp8DecodeGraphRunner::capture(cudaStream_t stream) {
    cudaStream_t cs;
    cudaStreamCreate(&cs);
    cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);

    // Node 1: CSA FP8 decode
    decode_params_.stream = cs;
    if (cfg_.h_q <= 64)
        sm120::decode::csa_fp8::run_csa_fp8_decode_kernel<64>(decode_params_);
    else
        sm120::decode::csa_fp8::run_csa_fp8_decode_kernel<128>(decode_params_);

    // Node 2: mla_combine
    run_mla_combine_kernel<cutlass::bfloat16_t>(combine_params_, cs);

    cudaStreamEndCapture(cs, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    cudaStreamDestroy(cs);
}

}  // namespace sm120::graph
