// SM120 Sparse FP8 Decode - V3.2 model, 64 heads
// HEAD_DIM_K=576, FP8 e4m3 KV cache with float32 scales

#include "../splitkv_mla.cu"

namespace sm120::decode::sparse_fp8 {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(
    const SparseAttnDecodeParams &params);

}  // namespace sm120::decode::sparse_fp8
