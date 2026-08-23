// SM120 CSA FP8 Prefill - D_QK=576 (V4: 512 NOPE + 64 ROPE)

#include "../phase1.cuh"

namespace sm120::prefill::csa_fp8 {

template void run_csa_fp8_prefill_kernel<576>(const CsaFp8PrefillParams& params);

}  // namespace sm120::prefill::csa_fp8
