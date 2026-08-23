#pragma once

#include "../../params.h"
#include "../../../../decode/sparse_fp8/params.h"

namespace sm120::prefill::sparse::small_topk::head64 {

// Params unification: decode mode is SnapMLA FP8-native and takes the SnapMLA
// decode param struct (q_rope / q_scales / per-token KV scales), NOT the
// global-scope SparseAttnDecodeParams from prefill/sparse/params.h.
using SnapMlaDecodeParams = sm120::decode::sparse_fp8::SparseAttnDecodeParams;

template<SparseAttnFwdMode FWD_MODE>
using SmallTopkArgT = std::conditional_t<
    is_decode_v<FWD_MODE>, SnapMlaDecodeParams, SparseAttnFwdParams>;

template<SparseAttnFwdMode FWD_MODE, int D_QK>
void run_fwd_for_small_topk_phase1_kernel(const SmallTopkArgT<FWD_MODE>& params);

}  // namespace sm120::prefill::sparse::small_topk::head64
