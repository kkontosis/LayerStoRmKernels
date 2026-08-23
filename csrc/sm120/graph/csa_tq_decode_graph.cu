#include "csa_tq_decode_graph.h"

namespace sm120::decode::csa_tq {
void run_csa_tq_decode(const CsaTqDecodeParams& params);
}

namespace sm120::graph {

void CsaTqDecodeGraphRunner::init(const CsaTqDecodeGraphConfig& cfg, cudaStream_t stream) {
    cfg_ = cfg;
    allocate_buffers();
    build_and_capture(stream);
}

void CsaTqDecodeGraphRunner::update(
    const void* q_nope_bf16, const void* q_rope_bf16,
    const int* sparse_indices, cudaStream_t stream
) {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int hd = cfg_.head_dim, rd = cfg_.qk_rope_head_dim;

    cudaMemcpyAsync(buf_q_nope_bf16_, q_nope_bf16,
        (size_t)b * sq * hq * hd * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_q_rope_bf16_, q_rope_bf16,
        (size_t)b * sq * hq * rd * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(buf_sparse_indices_, sparse_indices,
        (size_t)b * sq * cfg_.topk * sizeof(int),
        cudaMemcpyDeviceToDevice, stream);
}

void CsaTqDecodeGraphRunner::replay(cudaStream_t stream) {
    cudaGraphLaunch(graph_exec_, stream);
}

void CsaTqDecodeGraphRunner::destroy() {
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }
    free_buffers();
}

void CsaTqDecodeGraphRunner::allocate_buffers() {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int hd = cfg_.head_dim, rd = cfg_.qk_rope_head_dim;

    cudaMalloc(&buf_q_nope_bf16_,    (size_t)b * sq * hq * hd * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_q_rope_bf16_,    (size_t)b * sq * hq * rd * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_q_rot_fp32_,     (size_t)b * sq * hq * hd * sizeof(float));
    cudaMalloc(&buf_sparse_indices_, (size_t)b * sq * cfg_.topk * sizeof(int));
    cudaMalloc(&buf_out_rot_fp32_,   (size_t)b * sq * hq * hd * sizeof(float));
    cudaMalloc(&buf_out_bf16_,       (size_t)b * sq * hq * hd * sizeof(__nv_bfloat16));
    cudaMalloc(&buf_lse_,            (size_t)b * sq * hq * sizeof(float));
}

void CsaTqDecodeGraphRunner::free_buffers() {
    auto f = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    f(buf_q_nope_bf16_); f(buf_q_rope_bf16_); f(buf_q_rot_fp32_);
    f(buf_sparse_indices_); f(buf_out_rot_fp32_); f(buf_out_bf16_);
    f(buf_lse_);
}

void CsaTqDecodeGraphRunner::build_and_capture(cudaStream_t stream) {
    const int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
    const int hd = cfg_.head_dim, rd = cfg_.qk_rope_head_dim;

    // Node 1: tq_q_rotate (BF16 q_nope → FP32 q_rot)
    sm120::prep::TqQRotateParams q_rot_params;
    q_rot_params.q_nope = static_cast<const __nv_bfloat16*>(buf_q_nope_bf16_);
    q_rot_params.Pi = cfg_.Pi;
    q_rot_params.q_rot = static_cast<float*>(buf_q_rot_fp32_);
    q_rot_params.batch_heads = b * sq * hq;
    q_rot_params.d_c = hd;

    // Node 2: csa_tq_decode (sparse TQ scoring, online softmax, rotated space)
    sm120::decode::csa_tq::CsaTqDecodeParams decode_params;
    memset(&decode_params, 0, sizeof(decode_params));
    decode_params.b = b; decode_params.s_q = sq; decode_params.h_q = hq;
    decode_params.head_dim = hd; decode_params.qk_rope_head_dim = rd;
    decode_params.sm_scale = cfg_.sm_scale;
    decode_params.q_rot = static_cast<const float*>(buf_q_rot_fp32_);
    decode_params.q_rope = static_cast<const __nv_bfloat16*>(buf_q_rope_bf16_);
    decode_params.kv_cache = static_cast<const uint8_t*>(cfg_.kv_cache);
    decode_params.indices = static_cast<const int*>(buf_sparse_indices_);
    decode_params.topk = cfg_.topk;
    decode_params.stride_indices_b = sq * cfg_.topk;
    decode_params.stride_indices_s_q = cfg_.topk;
    decode_params.centroids = cfg_.centroids;
    decode_params.out = static_cast<float*>(buf_out_rot_fp32_);
    decode_params.lse = static_cast<float*>(buf_lse_);
    decode_params.stride_o_b = sq * hq * hd;
    decode_params.stride_o_s_q = hq * hd;
    decode_params.stride_o_h_q = hd;
    decode_params.stride_lse_b = sq * hq;
    decode_params.stride_lse_s_q = hq;

    // Node 3: tq_v_rotate_back (FP32 rotated → BF16 original)
    sm120::prep::TqVRotateBackParams vrot_params;
    vrot_params.out_rotated = static_cast<const float*>(buf_out_rot_fp32_);
    vrot_params.Pi = cfg_.Pi;
    vrot_params.out_final = static_cast<__nv_bfloat16*>(buf_out_bf16_);
    vrot_params.batch_heads = b * sq * hq;
    vrot_params.d_c = hd;

    // Capture graph
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    sm120::prep::run_tq_q_rotate(q_rot_params, capture_stream);

    decode_params.stream = capture_stream;
    sm120::decode::csa_tq::run_csa_tq_decode(decode_params);

    sm120::prep::run_tq_v_rotate_back(vrot_params, capture_stream);

    cudaStreamEndCapture(capture_stream, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    cudaStreamDestroy(capture_stream);
}

}  // namespace sm120::graph
