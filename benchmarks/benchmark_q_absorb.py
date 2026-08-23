"""Micro-benchmark for the q_absorb kernel (DeepSeek MLA W_UK query absorption).

Times ql_nope = einsum('shd,hdk->shk', q_nope, W_UK) + rope concat for a range of
s_q (token counts), BF16 and FP8 kv_b_proj. Run on one GPU:

    CUDA_VISIBLE_DEVICES=0 python benchmarks/benchmark_q_absorb.py
"""
import json
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_snapmla_reference import quantize_kv_b_fp8_blockwise  # noqa: E402

import sm120_mla_kernels as K  # noqa: E402

H_Q, P, L, R, V = 64, 128, 512, 64, 128
ITERS, WARMUP = 200, 30


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS  # ms


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.cuda.get_device_name(0)
    results = {"device": dev, "dims": {"h_q": H_Q, "P": P, "L": L, "R": R, "V": V}, "runs": []}

    kv_b = (torch.randn(H_Q * (P + V), L, dtype=torch.bfloat16, device="cuda") * 0.05)
    fp8, scales = quantize_kv_b_fp8_blockwise(kv_b.cpu())
    fp8, scales = fp8.cuda(), scales.cuda()

    for s_q in (1, 4, 16, 64, 256):
        q_heads = torch.randn(s_q, H_Q, P + R, dtype=torch.bfloat16, device="cuda")
        t_bf16 = bench(lambda: K.q_absorb(q_heads, kv_b, P, L, R, V))
        t_fp8 = bench(lambda: K.q_absorb(q_heads, fp8, P, L, R, V, scales))
        row = {"s_q": s_q, "bf16_ms": round(t_bf16, 5), "fp8_ms": round(t_fp8, 5)}
        results["runs"].append(row)
        print(f"s_q={s_q:4d}  BF16={t_bf16*1e3:8.2f} us   FP8={t_fp8*1e3:8.2f} us")

    out = os.path.join(os.path.dirname(__file__), "benchmark_q_absorb_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{dev}\nwrote {out}")


if __name__ == "__main__":
    main()
