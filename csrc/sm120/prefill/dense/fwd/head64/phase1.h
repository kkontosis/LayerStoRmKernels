#pragma once

#include "params.h"

namespace sm120::prefill::dense::head64 {

template<int D_QK, bool DETERMINISTIC>
void run_dense_fwd_phase1_kernel(const DenseAttnFwdParams& params);

}  // namespace sm120::prefill::dense::head64
