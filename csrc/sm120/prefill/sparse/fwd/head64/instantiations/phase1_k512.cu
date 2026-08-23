// SM120 Sparse Prefill - head64, D_QK=512 (MODEL1)

#include "../phase1.cuh"

namespace sm120::prefill::sparse::head64 {

template void run_fwd_phase1_kernel<512, false>(const SparseAttnFwdParams& params);
template void run_fwd_phase1_kernel<512, true>(const SparseAttnFwdParams& params);

}  // namespace sm120::prefill::sparse::head64
