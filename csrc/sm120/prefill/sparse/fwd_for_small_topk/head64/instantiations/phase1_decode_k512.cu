// SM120 Sparse Prefill (Small TopK) - Decode mode with SplitKV, D_QK=512

#include "../phase1.cuh"

namespace sm120::prefill::sparse::small_topk::head64 {

template void run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(
    const SmallTopkArgT<SparseAttnFwdMode::DecodeWithSplitKV>& params);

}  // namespace sm120::prefill::sparse::small_topk::head64
