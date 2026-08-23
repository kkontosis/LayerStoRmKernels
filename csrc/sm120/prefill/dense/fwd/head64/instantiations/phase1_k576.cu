// SM120 Dense Prefill - head64, D_QK=576 (V3.2)

#include "../phase1.cuh"

namespace sm120::prefill::dense::head64 {

template void run_dense_fwd_phase1_kernel<576, false>(const DenseAttnFwdParams& params);
template void run_dense_fwd_phase1_kernel<576, true>(const DenseAttnFwdParams& params);

}  // namespace sm120::prefill::dense::head64
