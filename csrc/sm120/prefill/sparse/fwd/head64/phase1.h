#pragma once

#include "../../params.h"

namespace sm120::prefill::sparse::head64 {

template<int D_QK, bool DETERMINISTIC>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

}  // namespace sm120::prefill::sparse::head64
