// SM120 SnapMLA — single compilation unit.
//
// Kernel sources (csrc/sm120/, csrc/smxx/) are the C++ library, usable directly.
// Python wrappers (bindings_python.cu) are #included here for single-TU compilation
// to avoid NVCC linker issues with duplicate symbols from header-included .cu files.
//
// File organization:
//   bindings.cu          — this file (compilation driver, includes everything)
//   bindings_python.cu   — Python wrappers + pybind11 module (never compiled alone)
//   sm120/               — C++ kernels (the actual library)
//   smxx/                — arch-generic split-KV infrastructure

// Prep kernel sources (included — they define kernels inline in .cu files)
#include "sm120/prep/fused_q_quant.cu"
#include "sm120/prep/q_absorb.cu"
#include "sm120/prep/rope_rotate.cu"
#include "sm120/prep/fused_k_append.cu"
#include "sm120/prep/dequant_ckv_indexed.cu"

// V4 compressor kernels
#include "sm120/compress/csa_compressor.cu"
#include "sm120/compress/hca_compressor.cu"
#include "sm120/compress/fused_compress_insert.cu"

// V4 FP8 cache prep kernels
#include "sm120/prep/fused_inv_rope_fp8.cu"
#include "sm120/prep/fused_q_compress_k.cu"
#include "sm120/prep/v4_fp8_k_append.cu"
#include "sm120/prep/v4_fp8_dequant_indexed.cu"

// V4 Lightning Indexer kernels
#include "sm120/indexer/lightning_score.cu"
#include "sm120/indexer/lightning_score_mqa.cu"
#include "sm120/indexer/lightning_topk.cu"

// V4 TQ cache prep kernels
#include "sm120/prep/v4_tq_k_append.cu"
#include "sm120/prep/v4_tq_dequant_indexed.cu"

// TurboQuant prep kernels
#include "sm120/prep/tq_fused_k_append.cu"
#include "sm120/prep/tq_dequant_ckv_indexed.cu"
#include "sm120/prep/tq_q_rotate.cu"
#include "sm120/prep/tq_v_rotate_back.cu"

// V4 TQ decode kernels
#include "sm120/decode/csa_tq/splitkv_csa_tq.cu"

// TurboQuant decode kernels
#include "sm120/decode/tq_dense/splitkv_mla.cu"
#include "sm120/decode/tq_sparse/splitkv_mla.cu"

// GEMM kernels — from LayerStoRmGemmKernels submodule (resolved via include_dirs)
// Helper kernels (included inline)
#include "smxx/quant/bf16_to_nvfp4.cu"
#include "smxx/quant/reformat_scales.cu"

// GEMM declarations (implementations compiled as separate TUs from submodule)
#include "sm120/gemm/nvfp4/nvfp4_gemm.h"
#include "sm120/gemm/q4k/q4k_dequant_gemm.h"
#include "sm120/gemm/q4k/q4k_cutlass_gemm.h"
#include "sm120/gemm/gguf/gguf_dequant_gemm.h"
#include "sm120/gemm/gguf/gguf_mmvq.h"
#include "sm120/gemm/gguf/gguf_mmq.h"
#include "sm120/gemm/gguf/gguf_mmq_cute.h"
#include "sm120/gemm/gguf/mmq_mma/mmq_mma.h"

// CUDA graph runners (included — use prep kernel symbols directly)
#include "sm120/graph/decode_graph.cu"
#include "sm120/graph/tq_decode_graph.cu"
#include "sm120/graph/csa_fp8_decode_graph.cu"
#include "sm120/graph/csa_tq_decode_graph.cu"

// V4 mHC residual-stream kernels (included inline)
#include "smxx/mhc.cu"

// Split-KV + DCP infrastructure (headers — implementations compiled as separate TUs)
#include "smxx/get_mla_metadata.h"
#include "smxx/mla_combine.h"
#include "smxx/dcp_lse_correct.h"
#include "smxx/inverse_rope.h"
#include "smxx/params.h"

// Decode + prefill kernel headers (implementations in instantiation .cu files)
#include "sm120/decode/dense_fp8/params.h"
#include "sm120/decode/sparse_fp8/params.h"
#include "sm120/decode/csa_fp8/params.h"
#include "sm120/prefill/sparse/params.h"
#include "sm120/prefill/dense/fwd/head64/params.h"
#include "sm120/prefill/sparse/fwd/head64/phase1.h"
#include "sm120/prefill/dense/fwd/head64/phase1.h"
#include "sm120/prefill/csa_fp8/phase1.h"

// Python wrappers + pybind11 module definition
#include "bindings_python.cu"
