/***************************************************************************************************
 * SM120 Projection GEMM CUDA Graph Runner — Implementation
 *
 * Captures a list of quantized-weight projection GEMMs into one replayable CUDA graph.
 * See projection_graph.h for the full contract. Mirrors the fixed-address-buffer
 * capture/replay pattern of decode_graph.cu.
 **************************************************************************************************/

#include "projection_graph.h"

#include "../../smxx/utils.h"   // CHECK_CUDA, CHECK_CUDA_KERNEL_LAUNCH, FLASH_ASSERT

namespace sm120::graph {

using layerstorm::compute::GgufMmvqParams;
using layerstorm::compute::launch_gguf_mmvq;
using layerstorm::compute::gguf_mmvq_workspace_bytes;
using layerstorm::compute::gguf_block_values;

//==============================================================================
// init — validate, allocate fixed buffers, capture
//==============================================================================
void ProjectionGraph::init(const std::vector<ProjSpec>& specs, cudaStream_t stream) {
    FLASH_ASSERT(!specs.empty());
    specs_ = specs;

    // --- Validate every spec up-front so nothing throws/aborts inside capture ---
    for (int i = 0; i < num_projections(); ++i) {
        const ProjSpec& s = specs_[i];
        FLASH_ASSERT(s.M > 0 && s.N > 0 && s.K > 0);
        FLASH_ASSERT(s.weight != nullptr);

        if (s.quant == ProjQuant::GGUF) {
            const int qk = gguf_block_values(s.gguf_type);
            // launch_gguf_mmvq throws if K % qk != 0; the Q8_1 activation quant also needs K%32==0.
            FLASH_ASSERT(s.K % qk == 0);
            FLASH_ASSERT(s.K % 32 == 0);
        } else {
            // No other quant implemented yet.
            FLASH_ASSERT(false && "ProjectionGraph: unsupported ProjQuant");
        }

        // Validate chaining: referenced projection must be earlier and shape-compatible.
        if (s.input_slot >= 0) {
            FLASH_ASSERT(s.input_slot < i && "chained input_slot must refer to an earlier projection");
            FLASH_ASSERT(specs_[s.input_slot].N == s.K &&
                         "chained projection's K must equal the source projection's N");
            FLASH_ASSERT(specs_[s.input_slot].M == s.M &&
                         "chained projection's M must equal the source projection's M");
        }
    }

    allocate_fixed_buffers();
    capture_graph(stream);
}

//==============================================================================
// Fixed-address buffer allocation (init-time only; never inside capture)
//==============================================================================
void ProjectionGraph::allocate_fixed_buffers() {
    const int n = num_projections();
    in_bufs_.assign(n, nullptr);
    out_bufs_.assign(n, nullptr);
    ws_bufs_.assign(n, nullptr);

    for (int i = 0; i < n; ++i) {
        const ProjSpec& s = specs_[i];

        // Own input buffer only needed when not chained (input_slot < 0). Allocating
        // it unconditionally keeps indexing simple and costs M*K*2 bytes (tiny at M=1);
        // skip it for chained projections to avoid waste.
        if (s.input_slot < 0) {
            CHECK_CUDA(cudaMalloc(&in_bufs_[i],
                                  (size_t)s.M * s.K * sizeof(__nv_bfloat16)));
        }

        CHECK_CUDA(cudaMalloc(&out_bufs_[i],
                              (size_t)s.M * s.N * sizeof(__nv_bfloat16)));

        if (s.quant == ProjQuant::GGUF) {
            const size_t ws = gguf_mmvq_workspace_bytes(s.M, s.K);
            CHECK_CUDA(cudaMalloc(&ws_bufs_[i], ws));
        }
    }
}

//==============================================================================
// Resolve a projection's input activation pointer (own buffer or chained output)
//==============================================================================
const __nv_bfloat16* ProjectionGraph::resolve_input_ptr(int i) const {
    const ProjSpec& s = specs_[i];
    if (s.input_slot >= 0) {
        return out_bufs_[s.input_slot];   // chained: read source projection's output buffer
    }
    return in_bufs_[i];                    // own/external input
}

//==============================================================================
// Per-projection launch — quant dispatch. ALL pointers are fixed-address buffers
// (or the static weight). No malloc / sync here, so this captures cleanly.
//
// To add NVFP4 / FP8: add a `case ProjQuant::NVFP4:` branch that builds that
// kernel's params from the SAME fixed in/out/weight buffers (+ its own workspace,
// allocated in allocate_fixed_buffers) and calls its stream launcher.
//==============================================================================
void ProjectionGraph::launch_projection(int i, cudaStream_t stream) {
    const ProjSpec& s = specs_[i];

    switch (s.quant) {
        case ProjQuant::GGUF: {
            GgufMmvqParams p;
            p.M    = s.M;
            p.N    = s.N;
            p.K    = s.K;
            p.A    = resolve_input_ptr(i);
            p.B    = s.weight;
            p.C    = out_bufs_[i];
            p.type = s.gguf_type;
            // Two plain stream launches (quantize activation → Q8_1, then int8 mat-vec),
            // both into the pre-allocated workspace — no internal malloc/sync.
            launch_gguf_mmvq(p, ws_bufs_[i], stream);
            break;
        }
        default:
            // Validated against in init(); unreachable.
            FLASH_ASSERT(false && "ProjectionGraph: unsupported ProjQuant at launch");
    }
}

//==============================================================================
// Capture — record the whole projection sequence into one CUDA graph
//==============================================================================
void ProjectionGraph::capture_graph(cudaStream_t /*stream*/) {
    // Capture on a dedicated stream (Global mode), like decode_graph.cu. The caller's
    // stream is used at replay time; capture must not run on a stream with other work.
    cudaStream_t capture_stream;
    CHECK_CUDA(cudaStreamCreate(&capture_stream));

    CHECK_CUDA(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));

    for (int i = 0; i < num_projections(); ++i) {
        launch_projection(i, capture_stream);
    }

    // Sanity: ensure we are still actively capturing (no kernel silently invalidated it).
    cudaStreamCaptureStatus status;
    CHECK_CUDA(cudaStreamIsCapturing(capture_stream, &status));
    FLASH_ASSERT(status == cudaStreamCaptureStatusActive &&
                 "graph capture was invalidated (an op inside the region likely synced)");

    CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph_));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    CHECK_CUDA(cudaStreamDestroy(capture_stream));
}

//==============================================================================
// set_input — copy live activation into a projection's fixed input buffer
//             (OUTSIDE the captured graph).
//==============================================================================
void ProjectionGraph::set_input(int slot, const __nv_bfloat16* src, cudaStream_t stream) {
    FLASH_ASSERT(slot >= 0 && slot < num_projections());
    const ProjSpec& s = specs_[slot];
    FLASH_ASSERT(s.input_slot < 0 &&
                 "set_input on a chained projection: its input comes from another projection");
    FLASH_ASSERT(in_bufs_[slot] != nullptr);

    const size_t bytes = (size_t)s.M * s.K * sizeof(__nv_bfloat16);
    CHECK_CUDA(cudaMemcpyAsync(in_bufs_[slot], src, bytes,
                               cudaMemcpyDeviceToDevice, stream));
}

//==============================================================================
// replay — one launch covers all projections + their quantize kernels
//==============================================================================
void ProjectionGraph::replay(cudaStream_t stream) {
    CHECK_CUDA(cudaGraphLaunch(graph_exec_, stream));
}

//==============================================================================
// Accessors
//==============================================================================
const __nv_bfloat16* ProjectionGraph::output(int slot) const {
    FLASH_ASSERT(slot >= 0 && slot < num_projections());
    return out_bufs_[slot];
}

__nv_bfloat16* ProjectionGraph::output(int slot) {
    FLASH_ASSERT(slot >= 0 && slot < num_projections());
    return out_bufs_[slot];
}

__nv_bfloat16* ProjectionGraph::input(int slot) {
    FLASH_ASSERT(slot >= 0 && slot < num_projections());
    return in_bufs_[slot];
}

//==============================================================================
// Teardown
//==============================================================================
void ProjectionGraph::free_buffers() {
    auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    for (auto& p : in_bufs_)  { void* q = p; safe_free(q); p = static_cast<__nv_bfloat16*>(q); }
    for (auto& p : out_bufs_) { void* q = p; safe_free(q); p = static_cast<__nv_bfloat16*>(q); }
    for (auto& p : ws_bufs_)  { safe_free(p); }
    in_bufs_.clear();
    out_bufs_.clear();
    ws_bufs_.clear();
}

void ProjectionGraph::destroy() {
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_)      { cudaGraphDestroy(graph_);          graph_ = nullptr; }
    free_buffers();
}

}  // namespace sm120::graph
