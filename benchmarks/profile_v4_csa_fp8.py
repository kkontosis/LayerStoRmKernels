"""
V4K-14a: Nsight Compute profiling launcher for CSA FP8 decode.

Runs a single CSA FP8 decode invocation at topk=1024, h_q=64 to produce
a clean ncu trace. Use with:

  ncu --set full -o profile_csa_fp8 python benchmarks/profile_v4_csa_fp8.py

Or for quick metrics:

  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
  python benchmarks/profile_v4_csa_fp8.py
"""

import torch
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE, V4_FP8_BYTES_PER_ENTRY,
)

import sm120_mla_kernels as K

TOPK = 1024
H_Q = 64
S_KV = 4096
B, S_Q = 1, 1
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
NUM_SM_PARTS = 1

def main():
    torch.manual_seed(42)

    num_pages = (S_KV + PAGE_SIZE - 1) // PAGE_SIZE
    cache = torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                        dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(S_KV, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(S_KV, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(S_KV, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot = torch.arange(S_KV, dtype=torch.int32, device='cuda')
    K.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)

    q_nope = torch.randn(B, S_Q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(B, S_Q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    indices = torch.randint(0, S_KV, (B, S_Q, TOPK), dtype=torch.int32, device='cuda')

    swa_cache = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                            dtype=torch.uint8, device='cuda')
    swa_bt = torch.zeros(B, 1, dtype=torch.int32, device='cuda')
    swa_sl = torch.zeros(B, dtype=torch.int32, device='cuda')

    # Warmup
    for _ in range(3):
        K.v4_csa_fp8_decode(q_nope, q_rope, cache, indices,
                            swa_cache, swa_bt, swa_sl,
                            SM_SCALE, TOPK, PAGE_SIZE, PAGE_SIZE, NUM_SM_PARTS)
    torch.cuda.synchronize()

    # Profiled invocation
    K.v4_csa_fp8_decode(q_nope, q_rope, cache, indices,
                        swa_cache, swa_bt, swa_sl,
                        SM_SCALE, TOPK, PAGE_SIZE, PAGE_SIZE, NUM_SM_PARTS)
    torch.cuda.synchronize()
    print("CSA FP8 decode profiling invocation complete.")

if __name__ == "__main__":
    main()
