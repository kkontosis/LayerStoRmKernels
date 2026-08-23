// SM120 CSA FP8 Decode - V4 Flash (64 Q heads)

#include "../splitkv_csa.cu"

namespace sm120::decode::csa_fp8 {

template void run_csa_fp8_decode_kernel<64>(
    const CsaFp8DecodeParams &params);

}  // namespace sm120::decode::csa_fp8
