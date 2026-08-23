#include "inverse_rope.h"

namespace smxx {

__global__ void __launch_bounds__(32)
inverse_rope_kernel(const InverseRopeParams params) {
    const int row = blockIdx.x;
    if (row >= params.N) return;

    const int half_dim = params.rope_dim / 2;
    const int tid = threadIdx.x;
    if (tid >= half_dim) return;

    const int pos = __ldg(params.positions + row);
    const float c = __ldg(params.cos_table + pos * half_dim + tid);
    const float s = __ldg(params.sin_table + pos * half_dim + tid);

    const __nv_bfloat16* x_row = params.x + row * params.rope_dim;
    float x_even = __bfloat162float(x_row[2 * tid]);
    float x_odd  = __bfloat162float(x_row[2 * tid + 1]);

    // Inverse RoPE: negate sin in even component
    float out_even = x_even * c + x_odd * s;
    float out_odd  = -x_even * s + x_odd * c;

    __nv_bfloat16* o_row = params.out + row * params.rope_dim;
    o_row[2 * tid]     = __float2bfloat16(out_even);
    o_row[2 * tid + 1] = __float2bfloat16(out_odd);
}

void run_inverse_rope(const InverseRopeParams& params, cudaStream_t stream) {
    if (params.N == 0) return;
    inverse_rope_kernel<<<params.N, 32, 0, stream>>>(params);
}

}  // namespace smxx
