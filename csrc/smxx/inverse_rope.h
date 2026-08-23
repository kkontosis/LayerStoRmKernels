#pragma once
/***************************************************************************************************
 * Inverse RoPE: remove positional coupling from attention output
 *
 * After attention with shared KV (num_kv_heads=1), the output carries positional
 * coupling via rotary embeddings. Inverse RoPE negates the angles to remove it:
 *   out[even] = x[even] * cos + x[odd] * sin
 *   out[odd]  = -x[even] * sin + x[odd] * cos
 *
 * Element-wise, arch-agnostic (no SM-specific instructions).
 * Standalone kernel for V4K-5b; fused variant in V4K-6b.
 *
 * Grid:  (N, 1, 1)  — one block per row
 * Block: 32 threads  — one thread per cos/sin pair (rope_dim/2)
 *
 * Reference: tests/test_v4_reference.py::ref_inverse_rope()
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace smxx {

struct InverseRopeParams {
    const __nv_bfloat16* __restrict__ x;      // [N, rope_dim] input (BF16)
    const float* __restrict__ cos_table;       // [max_pos, rope_dim/2] precomputed cos
    const float* __restrict__ sin_table;       // [max_pos, rope_dim/2] precomputed sin
    const int* __restrict__ positions;         // [N] position per row
    __nv_bfloat16* __restrict__ out;           // [N, rope_dim] output (BF16)
    int N;                                     // number of rows (batch * heads, or tokens)
    int rope_dim;                              // qk_rope_head_dim (64)
};

void run_inverse_rope(const InverseRopeParams& params, cudaStream_t stream);

}  // namespace smxx
