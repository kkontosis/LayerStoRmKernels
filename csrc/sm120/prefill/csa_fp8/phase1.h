#pragma once

#include "params.h"

namespace sm120::prefill::csa_fp8 {

template<int D_QK>
void run_csa_fp8_prefill_kernel(const CsaFp8PrefillParams& params);

}  // namespace sm120::prefill::csa_fp8
