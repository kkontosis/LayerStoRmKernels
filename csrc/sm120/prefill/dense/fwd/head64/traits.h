#pragma once

// Re-export sparse prefill traits — shared memory plan is identical for dense.
#include "../../../sparse/fwd/head64/traits.h"

namespace sm120::prefill::dense::head64 {

template<int D_QK>
using Traits = sm120::prefill::sparse::head64::Traits<D_QK>;

}  // namespace sm120::prefill::dense::head64
