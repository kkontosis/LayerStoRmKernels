#pragma once

/***************************************************************************************************
 * SM120 TQ Decode CUDA Graph Runner
 *
 * Captures the TQ decode pipeline into a replayable CUDA graph:
 *   1. tq_q_rotate_kernel      — rotate Q NOPE by Pi^T (BF16 → FP32)
 *   2. tq_dense_decode_kernel  — TQ attention (split-KV, scalar codebook, rotated-space PV)
 *   3. mla_combine_kernel<f32> — merge split-KV partials (FP32 for rotated space)
 *   4. tq_v_rotate_back_kernel — inverse rotation (FP32 → BF16)
 *
 * Usage:
 *   TqDecodeGraphRunner runner;
 *   runner.init(cfg, stream);
 *   // per decode step:
 *   runner.update(q_nope, q_rope, seqlens_k, block_table, stream);
 *   runner.replay(stream);
 *   // read from runner.out_ptr() / runner.lse_ptr()
 *   runner.destroy();
 *
 * Fixed buffers: Pi, centroids, kv_cache base are constant across replays.
 * Per-step: q_nope, q_rope, seqlens_k, block_table copied into fixed buffers.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../decode/tq_dense/params.h"
#include "../prep/tq_q_rotate.h"
#include "../prep/tq_v_rotate_back.h"
#include "../../smxx/mla_combine.h"

namespace sm120::graph {

//==============================================================================
// TQ Graph Config — fixed at capture time
//==============================================================================
struct TqDecodeGraphConfig {
    int batch_size;
    int s_q;
    int h_q;
    int d_c;
    int d_rope;
    int page_block_size;
    int max_num_blocks_per_seq;
    float sm_scale;
    int num_sm_parts;

    // Stable device pointers (constant across replays)
    void* kv_cache;             // paged TQ cache base
    const float* Pi;            // [d_c, d_c] rotation matrix
    const float* centroids;     // [16] codebook
};

//==============================================================================
// TQ Decode Graph Runner
//==============================================================================
class TqDecodeGraphRunner {
public:
    TqDecodeGraphRunner() = default;
    ~TqDecodeGraphRunner() { destroy(); }

    TqDecodeGraphRunner(const TqDecodeGraphRunner&) = delete;
    TqDecodeGraphRunner& operator=(const TqDecodeGraphRunner&) = delete;

    void init(const TqDecodeGraphConfig& cfg, cudaStream_t stream);
    void update(const void* q_nope_bf16, const void* q_rope_bf16,
                const int* seqlens_k, const int* block_table,
                cudaStream_t stream);
    void replay(cudaStream_t stream);

    __nv_bfloat16* out_ptr()   { return static_cast<__nv_bfloat16*>(buf_out_bf16_); }
    float*         lse_ptr()   { return static_cast<float*>(buf_lse_); }

    void destroy();

private:
    TqDecodeGraphConfig cfg_{};

    cudaGraph_t     graph_      = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    // Fixed-address device buffers
    void* buf_q_nope_bf16_  = nullptr;  // [b, s_q, h_q, d_c] BF16 input
    void* buf_q_rope_bf16_  = nullptr;  // [b, s_q, h_q, d_rope] BF16 input
    void* buf_q_rot_fp32_   = nullptr;  // [b, s_q, h_q, d_c] FP32 (q_rotate output)
    void* buf_out_rot_fp32_ = nullptr;  // [b, s_q, h_q, d_c] FP32 (decode output, rotated)
    void* buf_out_bf16_     = nullptr;  // [b, s_q, h_q, d_c] BF16 (final output)
    void* buf_lse_          = nullptr;  // [b, s_q, h_q] FP32
    void* buf_seqlens_k_    = nullptr;  // [b] int32
    void* buf_block_table_  = nullptr;  // [b, max_blocks] int32
    void* buf_lse_accum_    = nullptr;  // [nsp*b, num_q_seqs] FP32
    void* buf_o_accum_      = nullptr;  // [nsp*b, num_q_seqs, d_c] FP32
    void* buf_sched_meta_   = nullptr;  // [nsp] TqDecodingSchedMeta
    void* buf_num_splits_   = nullptr;  // [b+1] int32

    void allocate_fixed_buffers();
    void free_buffers();
    void build_and_capture(cudaStream_t stream);
};

}  // namespace sm120::graph
