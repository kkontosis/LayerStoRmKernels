#pragma once

/***************************************************************************************************
 * SM120 Projection GEMM CUDA Graph Runner — "segment (a)" of a whole-layer decode graph
 *
 * Captures a configurable LIST of linear-projection GEMMs (the matmuls that map a token's
 * hidden state through q_a / q_b / kv_a / o_proj / ... using quantized weights) into ONE
 * replayable CUDA graph with fixed-address I/O buffers. At decode (M=1) these GEMMs are tiny
 * and launch-bound; folding K projections (2K kernel launches) into a single cudaGraphLaunch
 * eliminates the per-token launch overhead.
 *
 * Scope-named, quant-parameterized
 * --------------------------------
 * A CUDA graph captures an op *sequence*. The quant only swaps which GEMM kernel runs inside
 * that sequence, so the runner is named for its SCOPE (projections), not for the quant. Each
 * ProjSpec carries a `ProjQuant` selecting the per-projection launch branch. Today only GGUF
 * is implemented; NVFP4 / FP8 slot in as additional branches in launch_projection() without
 * touching the capture/replay machinery.
 *
 * DCP-safe & composable
 * ---------------------
 * The captured region is purely local: projection matmuls only — no attention, no cross-
 * partition / cross-rank communication, no host round-trips. Capture happens on a caller-
 * provided stream and the cudaGraph_t / cudaGraphExec_t are exposed so a future whole-layer
 * (or DCP-segmented) runner can embed this graph as a child node and stitch it between other
 * segments.
 *
 * Fixed-address-buffer capture/replay (mirrors decode_graph.{h,cu})
 * ----------------------------------------------------------------
 *   init()    : allocate fixed device buffers (per-projection input/output + Q8_1 workspace),
 *               store the static weight pointers, then capture the GEMM sequence ONCE.
 *   set_input(): cudaMemcpyAsync the live activation into a fixed input buffer (OUTSIDE the
 *               captured graph).
 *   replay()  : one cudaGraphLaunch runs all projections + their activation-quant kernels.
 *   output()  : read results from the fixed output buffers.
 *
 * Chaining: a ProjSpec may set input_slot to a previous ProjSpec's output buffer, so e.g.
 * q_a → q_b is captured as two GEMMs reading/writing a fixed intermediate buffer with NO
 * host round-trip.
 *
 * Graph contract (REQUIRED)
 * -------------------------
 *   - M, N, K of every projection are CONSTANT across replays (grid dims are baked in at
 *     capture time). Changing any shape requires re-init.
 *   - Weight pointers are captured by value and must remain valid/stable for the graph's life.
 *   - NO cudaMalloc / cudaMemcpy(sync) / cudaStreamSynchronize inside the captured region —
 *     all buffers & workspaces are pre-allocated in init(). Capture uses
 *     cudaStreamCaptureModeGlobal.
 *
 * Thread safety: NOT thread-safe. One runner per stream.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sm120/gemm/gguf/gguf_mmvq.h"          // GgufMmvqParams, launch_gguf_mmvq, workspace
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"  // GgufType, gguf_block_values/bytes

namespace sm120::graph {

//==============================================================================
// Quant selector — extensible. GGUF now; NVFP4 / FP8 are future branches.
//==============================================================================
enum class ProjQuant { GGUF /*, NVFP4, FP8 */ };

//==============================================================================
// One projection GEMM: C[M,N] = A[M,K] @ dequant(weight)^T
//==============================================================================
struct ProjSpec {
    int M, N, K;

    ProjQuant quant = ProjQuant::GGUF;

    // Used when quant == GGUF. (Future quants would add their own type fields.)
    layerstorm::compute::GgufType gguf_type = layerstorm::compute::GgufType::Q8_0;

    // Device pointer to the packed weight, static — captured once by value into the graph.
    const void* weight = nullptr;

    // input_slot: -1  => this projection reads its OWN dedicated input buffer (an external
    //                    activation copied in via set_input(slot=this projection's index)).
    //             >=0 => read the OUTPUT buffer of the projection with that index (chaining).
    //                    That referenced projection must appear EARLIER in the list, and its
    //                    N must equal this projection's K.
    int input_slot = -1;

    // output_slot is informational / for symmetry; each projection owns one output buffer
    // indexed by its position in the spec list. Left here for future explicit remapping.
    int output_slot = -1;
};

//==============================================================================
// Projection Graph Runner
//==============================================================================
class ProjectionGraph {
public:
    ProjectionGraph() = default;
    ~ProjectionGraph() { destroy(); }

    // Non-copyable (owns device buffers + graph objects)
    ProjectionGraph(const ProjectionGraph&) = delete;
    ProjectionGraph& operator=(const ProjectionGraph&) = delete;

    //--------------------------------------------------------------------------
    // init: validate specs, allocate fixed buffers, capture the projection
    //       sequence as one CUDA graph on `stream`.
    //--------------------------------------------------------------------------
    void init(const std::vector<ProjSpec>& specs, cudaStream_t stream);

    //--------------------------------------------------------------------------
    // set_input: copy a live activation [M,K] BF16 into projection `slot`'s fixed
    //            input buffer (OUTSIDE the captured graph). Only valid for
    //            projections whose input_slot == -1 (own/external input). Issued
    //            on `stream`; replay() must run on the same stream (or be ordered
    //            after this copy) so the graph sees the updated data.
    //--------------------------------------------------------------------------
    void set_input(int slot, const __nv_bfloat16* src, cudaStream_t stream);

    //--------------------------------------------------------------------------
    // replay: launch the whole captured graph (all projections + quant kernels).
    //--------------------------------------------------------------------------
    void replay(cudaStream_t stream);

    //--------------------------------------------------------------------------
    // output: fixed output buffer of projection `slot` ([M,N] BF16, row-major).
    //--------------------------------------------------------------------------
    const __nv_bfloat16* output(int slot) const;

    // Mutable variant (e.g. for a downstream op to consume in place).
    __nv_bfloat16* output(int slot);

    // Fixed input buffer of projection `slot` ([M,K] BF16) — exposed for composition.
    __nv_bfloat16* input(int slot);

    //--------------------------------------------------------------------------
    // Graph handle accessors — for future whole-layer / DCP composition (embed as
    // a child node, or relaunch under an outer graph).
    //--------------------------------------------------------------------------
    cudaGraph_t     graph() const      { return graph_; }
    cudaGraphExec_t graph_exec() const { return graph_exec_; }

    int num_projections() const { return static_cast<int>(specs_.size()); }

    //--------------------------------------------------------------------------
    // destroy: free graph objects and all device buffers.
    //--------------------------------------------------------------------------
    void destroy();

private:
    std::vector<ProjSpec> specs_;

    // Per-projection fixed-address device buffers (index == projection index).
    std::vector<__nv_bfloat16*> in_bufs_;    // [M,K] BF16 input  (own buffer; may be unused if chained)
    std::vector<__nv_bfloat16*> out_bufs_;   // [M,N] BF16 output
    std::vector<void*>          ws_bufs_;     // Q8_1 activation workspace (GGUF)

    cudaGraph_t     graph_      = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;

    void allocate_fixed_buffers();
    void capture_graph(cudaStream_t stream);
    void launch_projection(int i, cudaStream_t stream);  // per-projection quant dispatch
    void free_buffers();

    // Resolve the input activation pointer for projection i (own buffer or a
    // previous projection's output buffer).
    const __nv_bfloat16* resolve_input_ptr(int i) const;
};

}  // namespace sm120::graph
