#pragma once

// Re-export sparse prefill config — tiling is identical for dense.
#include "../../../sparse/fwd/head64/config.h"

namespace sm120::prefill::dense::head64 {
using namespace sm120::prefill::sparse::head64;
}  // namespace sm120::prefill::dense::head64
