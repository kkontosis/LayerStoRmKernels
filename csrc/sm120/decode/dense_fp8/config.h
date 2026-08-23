#pragma once

// Dense FP8 decode uses identical tiling/thread config as sparse FP8 decode.
// The smem plan, MMA config, and K-tile dimensions are the same.
#include "../sparse_fp8/config.h"

namespace sm120::decode::dense_fp8 {

using sm120::sparse::ModelType;

// Re-export sparse config — tiling is identical for dense
template<ModelType MODEL_TYPE>
using Config = sm120::decode::sparse_fp8::Config<MODEL_TYPE>;

}  // namespace sm120::decode::dense_fp8
