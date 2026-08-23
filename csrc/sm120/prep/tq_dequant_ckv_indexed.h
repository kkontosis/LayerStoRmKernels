#pragma once
/***************************************************************************************************
 * TQ Dequant-CKV-Indexed: Gather TQ-compressed tokens from paged cache, dequantize to BF16
 *
 * Per token: unpack 4-bit indices → codebook lookup → inverse rotation (y_hat @ Pi) → scale by norm
 * Output: contiguous BF16 [num_fetch, d_c + d_rope] for chunked prefill
 *
 * Grid: (num_fetch, 1, 1) — one CTA per fetched token
 * Block: 256 threads
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace sm120::prep {

struct TqDequantCKVIndexedParams {
    const uint8_t* __restrict__ kv_cache;          // Paged TQ cache base
    int64_t cache_stride_block;
    int cache_stride_row;
    int page_size;

    const int* __restrict__ indices;                // [num_fetch] → flat slot indices
    int num_fetch;

    __nv_bfloat16* __restrict__ k_out;             // [num_fetch, d_c + d_rope] output

    const float* __restrict__ Pi;                   // [d_c, d_c] rotation matrix
    const float* __restrict__ centroids;            // [16] codebook centroids

    int d_c;
    int d_rope;
};

void run_tq_dequant_ckv_indexed(const TqDequantCKVIndexedParams& params, cudaStream_t stream);

}  // namespace sm120::prep
