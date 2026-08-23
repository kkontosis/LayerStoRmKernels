// SM120 Dense FP8 Decode — MODEL1, 64 heads

#include "../splitkv_mla.cu"

namespace sm120::decode::dense_fp8 {

template void run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, false>(
    const DenseAttnDecodeParams &params);
template void run_flash_splitkv_mla_dense_fp8_kernel<ModelType::MODEL1, 64, true>(
    const DenseAttnDecodeParams &params);

}  // namespace sm120::decode::dense_fp8
