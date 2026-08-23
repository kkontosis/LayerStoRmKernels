"""
V4K-15a: Nsight Compute profiling launcher for CSA TQ decode.

Usage:
  ncu --set full -o profile_csa_tq python benchmarks/profile_v4_csa_tq.py

Or quick metrics via sudo:
  sudo ncu --kernel-name csa_tq_decode --launch-skip 3 --launch-count 1 --set full \
    python benchmarks/profile_v4_csa_tq.py
"""

import torch
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE, V4_TQ_BYTES_PER_ENTRY,
    load_codebook, generate_rotation_matrix,
)

import sm120_mla_kernels as K

TOPK = 1024
H_Q = 64
S_KV = 4096
B, S_Q = 1, 1
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)

def main():
    torch.manual_seed(42)

    centroids, boundaries = load_codebook()
    Pi = generate_rotation_matrix()
    centroids_gpu = centroids.cuda()
    Pi_gpu = Pi.cuda()
    boundaries_gpu = boundaries[1:-1].cuda()

    num_pages = (S_KV + PAGE_SIZE - 1) // PAGE_SIZE
    cache = torch.zeros(num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY,
                        dtype=torch.uint8, device='cuda')
    k_nope = torch.randn(S_KV, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(S_KV, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(S_KV, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    slot = torch.arange(S_KV, dtype=torch.int32, device='cuda')
    K.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot, Pi_gpu, centroids_gpu, boundaries_gpu)

    q_nope = torch.randn(B, S_Q, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(B, S_Q, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    indices = torch.randint(0, S_KV, (B, S_Q, TOPK), dtype=torch.int32, device='cuda')

    q_rot = (q_nope.float() @ Pi_gpu.T).contiguous()

    # Warmup
    for _ in range(3):
        K.v4_csa_tq_decode(q_rot, q_rope, cache, indices, centroids_gpu, SM_SCALE)
    torch.cuda.synchronize()

    # Profiled invocation
    K.v4_csa_tq_decode(q_rot, q_rope, cache, indices, centroids_gpu, SM_SCALE)
    torch.cuda.synchronize()
    print("CSA TQ decode profiling invocation complete.")

if __name__ == "__main__":
    main()
