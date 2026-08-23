#pragma once

// Dense FP8 decode uses identical SharedMemoryPlan as sparse FP8 decode.
// Same dual-MMA layout: FP8 NOPE + BF16 ROPE + FP8 P/V.
#include "../sparse_fp8/traits.h"

namespace sm120::decode::dense_fp8 {

using sm120::sparse::ModelType;

// Re-export sparse traits — smem plan is identical for dense
template<ModelType MODEL_TYPE>
using Traits = sm120::decode::sparse_fp8::Traits<MODEL_TYPE>;

}  // namespace sm120::decode::dense_fp8
