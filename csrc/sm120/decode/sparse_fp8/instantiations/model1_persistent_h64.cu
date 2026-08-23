// SM120 Sparse FP8 Decode - MODEL1, 64 heads
// HEAD_DIM_K=512, FP8 e4m3 KV cache with fp8_e8m0 scales

#include "../splitkv_mla.cu"

namespace sm120::decode::sparse_fp8 {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::MODEL1, 64>(
    const SparseAttnDecodeParams &params);

}  // namespace sm120::decode::sparse_fp8
