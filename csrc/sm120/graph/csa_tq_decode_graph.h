#pragma once
/***************************************************************************************************
 * V4 CSA TQ Decode Graph Runner
 *
 * Captures the V4 CSA TQ decode pipeline into a replayable CUDA graph:
 *   1. tq_q_rotate_kernel       — rotate Q NOPE by Pi^T (BF16 → FP32)
 *   2. csa_tq_decode_kernel     — sparse TQ scoring (rotated space, separate K/V, online softmax)
 *   3. tq_v_rotate_back_kernel  — inverse rotation (FP32 → BF16)
 *
 * No mla_combine: CSA TQ decode uses online softmax in a single CTA per head.
 *
 * Usage:
 *   CsaTqDecodeGraphRunner runner;
 *   runner.init(cfg, stream);
 *   // per decode step:
 *   runner.update(q_nope, q_rope, sparse_indices, stream);
 *   runner.replay(stream);
 *   // read from runner.out_ptr() / runner.lse_ptr()
 *   runner.destroy();
 *
 * Fixed buffers: Pi, centroids, kv_cache base are constant across replays.
 * Per-step: q_nope, q_rope, sparse_indices copied into fixed buffers.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../decode/csa_tq/params.h"
#include "../prep/tq_q_rotate.h"
#include "../prep/tq_v_rotate_back.h"

namespace sm120::graph {

struct CsaTqDecodeGraphConfig {
    int batch_size;
    int s_q;                    // 1 for decode
    int h_q;                    // 128 (Pro) or 64 (Flash)
    int head_dim;               // 512
    int qk_rope_head_dim;       // 64
    int topk;                   // sparse indices count
    float sm_scale;

    // Stable device pointers (constant across replays)
    void* kv_cache;             // flat V4 TQ cache base
    const float* Pi;            // [head_dim, head_dim] rotation matrix
    const float* centroids;     // [16] codebook
};

class CsaTqDecodeGraphRunner {
public:
    CsaTqDecodeGraphRunner() = default;
    ~CsaTqDecodeGraphRunner() { destroy(); }

    CsaTqDecodeGraphRunner(const CsaTqDecodeGraphRunner&) = delete;
    CsaTqDecodeGraphRunner& operator=(const CsaTqDecodeGraphRunner&) = delete;

    void init(const CsaTqDecodeGraphConfig& cfg, cudaStream_t stream);

    void update(
        const void* q_nope_bf16,        // [b, s_q, h_q, head_dim] BF16
        const void* q_rope_bf16,        // [b, s_q, h_q, qk_rope_head_dim] BF16
        const int* sparse_indices,      // [b, s_q, topk] int32
        cudaStream_t stream
    );

    void replay(cudaStream_t stream);

    __nv_bfloat16* out_ptr()  { return static_cast<__nv_bfloat16*>(buf_out_bf16_); }
    float*         lse_ptr()  { return static_cast<float*>(buf_lse_); }

    void destroy();

private:
    CsaTqDecodeGraphConfig cfg_{};

    cudaGraph_t     graph_      = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    // Fixed-address device buffers
    void* buf_q_nope_bf16_      = nullptr;  // [b, s_q, h_q, head_dim] BF16
    void* buf_q_rope_bf16_      = nullptr;  // [b, s_q, h_q, qk_rope_head_dim] BF16
    void* buf_q_rot_fp32_       = nullptr;  // [b, s_q, h_q, head_dim] FP32
    void* buf_sparse_indices_   = nullptr;  // [b, s_q, topk] int32
    void* buf_out_rot_fp32_     = nullptr;  // [b, s_q, h_q, head_dim] FP32
    void* buf_out_bf16_         = nullptr;  // [b, s_q, h_q, head_dim] BF16
    void* buf_lse_              = nullptr;  // [b, s_q, h_q] FP32

    void allocate_buffers();
    void free_buffers();
    void build_and_capture(cudaStream_t stream);
};

}  // namespace sm120::graph
