# V4K-14a: CSA FP8 Decode Baseline Profile

**Date**: 2026-04-26
**GPU**: NVIDIA GeForce RTX 5090 (SM120, 170 SMs)
**Config**: topk=1024, h_q=64, s_kv=4096, batch=1, num_sm_parts=1

## Kernel Launch Config

| Parameter | Value |
|-----------|-------|
| Kernel | `csa_fp8_decode_sm120_kernel<64>` |
| Block Size | (256, 1, 1) — 8 warps |
| Grid Size | **(1, 1, 1)** — single CTA |
| Registers/Thread | **255** (maximum) |
| Dynamic Smem | 55.1 KB |
| Static Smem | 0 B |
| Duration | 3,963 us |
| Waves Per SM | 0.01 |

## Throughput

| Metric | % of Peak | Notes |
|--------|-----------|-------|
| Compute (SM) | 0.09% | Near zero — latency-bound |
| Memory | 0.09% | Near zero — not bandwidth-bound |
| DRAM | 0.02% | Almost no DRAM traffic |
| L1/TEX Cache | 14.71% | Modest L1 usage |
| L2 Cache | 0.06% | |
| FMA Pipe | 0.02% | FP32 pipe idle |
| Tensor Core (HMMA) | n/a | Not using tensor cores |

## Occupancy

| Metric | Value |
|--------|-------|
| Achieved Occupancy | 16.66% |
| Theoretical Occupancy | 16.67% |
| Active Warps/SM | 8.00 |
| Block Limit (Registers) | **1 block** (bottleneck) |
| Block Limit (Smem) | **1 block** (co-bottleneck) |
| Block Limit (Warps) | 6 blocks |
| Block Limit (Barriers) | 24 blocks |

## Warp Stall Breakdown (per issue-active instruction)

| Stall Reason | Ratio | Category |
|--------------|-------|----------|
| **Long Scoreboard** | **6.13** | Memory latency (global/local loads) |
| **Barrier** | **5.82** | __syncthreads / mbarrier waits |
| **Wait** | **2.00** | Instruction pipeline dependency |
| Short Scoreboard | 1.30 | Shared memory / L1 latency |
| MIO Throttle | 0.32 | Memory instruction queue full |
| No Instruction | 0.25 | Instruction cache miss / fetch |
| Not Selected | 0.14 | Eligible but scheduler picked another |
| Math Pipe Throttle | 0.12 | |
| LG Throttle | 0.05 | |
| No Eligible | 88.46% | % of cycles with NO eligible warps |

## Memory Traffic

| Category | Volume |
|----------|--------|
| Global Memory | 2.59 MB |
| Shared Memory | 11.21 MB |
| **Local Memory (spill)** | **1.90 MB** (1.36 MB load + 537 KB store) |
| L1/TEX Hit Rate | 61.77% |
| L2 Hit Rate | 89.70% |

## Shared Memory Bank Conflicts

| Operation | Count |
|-----------|-------|
| Load Bank Conflicts | 2 |
| **Store Bank Conflicts** | **143,360** |

## Scheduler Statistics

| Metric | Value |
|--------|-------|
| Active Warps/Scheduler | 2.00 |
| Eligible Warps/Scheduler | 0.13 |
| Issued Warp/Scheduler | 0.12 |
| One or More Eligible | 11.54% |

## Instruction Statistics

| Metric | Value |
|--------|-------|
| Executed Instructions | 3,682,355 |
| Issued Instructions | 3,683,728 |
| IPC Active | 0.46 |
| Branch Efficiency | 98.99% |

## Top 3 Bottlenecks

### 1. Single-CTA Launch (Grid = 1×1×1)

The kernel launches exactly **one thread block** on an RTX 5090 with **170 SMs**. This means 169 SMs sit completely idle. The kernel processes all 64 heads × 1024 tokens sequentially within a single CTA.

**Impact**: 99.4% of GPU SMs unused. This is the dominant bottleneck.
**Fix**: V4K-14f (split-KV scaling) — `num_sm_parts > 1` distributes work across CTAs. Also, the kernel should launch one CTA per head group (grid.x = h_q / BLOCK_SIZE_M).

### 2. Register Pressure → Local Memory Spilling (255 regs + 1.9MB spill)

The kernel maxes out at 255 registers/thread (the hardware maximum). The remaining state spills to local memory (1.9 MB total — 1.36 MB loads, 537 KB stores). This converts register accesses into L1/L2 cache accesses, adding latency.

**Impact**: Long scoreboard stalls (6.13 ratio) are partly caused by spill loads. 95,616 bytes of spill requests.
**Fix**: V4K-14d (warp specialization) — rebalance work to reduce peak register pressure. Also consider reducing per-thread state by restructuring accumulator layout.

### 3. Barrier Stalls + Smem Store Bank Conflicts

Barrier stalls (5.82 ratio) indicate warps spending significant time at `__syncthreads()`. Combined with 143K shared memory store bank conflicts, the producer-consumer pipeline has contention on smem writes.

**Impact**: 5.82 barrier stall ratio means ~36% of active warp cycles stalled at barriers.
**Fix**: V4K-14e (smem bank conflict elimination) — add swizzle/padding to smem layouts to eliminate the 143K store conflicts. This will reduce time spent at barriers.

## Secondary Issues

- **L1 hit rate 61.77%**: scatter-gather access pattern from sparse indices causes ~38% misses. TMA (V4K-14b) would help by issuing bulk descriptors.
- **No tensor core usage**: The FP8 NOPE scoring uses FMA pipe (0.02%) instead of HMMA. If the kernel intends to use tensor cores for FP8×BF16 MMA, they're not firing.
- **IPC = 0.46**: Low instruction-level parallelism, consistent with latency-bound profile.

---

## Post-Analysis: Profile at num_sm_parts=32 (Optimal Config)

**Duration**: 284 us (13.9x faster than nsp=1)

| Metric | nsp=1 | nsp=32 | Notes |
|--------|-------|--------|-------|
| Grid Size | (1,1,1) | (1,1,32) | 32 CTAs |
| Duration | 3963 us | 284 us | 13.9x |
| Compute Throughput | 0.09% | 1.31% | 14.5x |
| Memory Throughput | 0.09% | 1.37% | 15.2x |
| Occupancy | 16.67% | 16.64% | Same (register-limited) |
| IPC | 0.46 | 0.44 | Same |
| No Eligible | 88.46% | 88.93% | Same |
| Long Scoreboard | 6.13 | 6.55 | Same |
| Barrier | 5.82 | 5.51 | Same |
| Smem Store Conflicts | 143K | 143K | Same (per-CTA) |

**Remaining bottleneck at nsp=32**: Register pressure (255 regs/thread) limits occupancy to 1 block/SM (16.67%). Long scoreboard stalls (6.55) from register spill loads dominate. The 0.19 waves/SM means only 32 of 170 SMs are utilized.

## Phase 14 Summary

| Ticket | Status | Impact |
|--------|--------|--------|
| V4K-14a | ✅ Complete | Baseline profile |
| V4K-14b | ❌ Infeasible | TMA incompatible with sparse gather |
| V4K-14c | ⏭ Already done | Double-buffered K tiles + producer/consumer overlap |
| V4K-14d | ⏭ Hardware-fixed | 4+4 warp split dictated by MMA atom |
| V4K-14e | ⏭ Negligible | 143K conflicts but 0.21% est. speedup |
| V4K-14f | ✅ Complete | **10.99x at batch=1, 44.8x at batch=32** |

**V4K-14f (split-KV) is the only actionable optimization**, delivering massive speedups by parallelizing across SMs. The remaining per-CTA optimizations are blocked by fundamental constraints: register pressure (255/thread), scattered memory access patterns, and hardware MMA warp requirements.
