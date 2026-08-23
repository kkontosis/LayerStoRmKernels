#pragma once

#include "params.h"

namespace sm120::decode::sparse_fp8 {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}  // namespace sm120::decode::sparse_fp8
