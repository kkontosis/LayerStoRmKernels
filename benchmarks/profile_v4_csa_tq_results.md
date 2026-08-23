# V4K-15a: CSA TQ Decode Baseline Profile

**Date**: 2026-04-26
**GPU**: NVIDIA GeForce RTX 5090 (SM120, 170 SMs)
**Config**: topk=1024, h_q=64, s_kv=4096, batch=1

## Kernel Launch Config

| Parameter | Value |
|-----------|-------|
| Kernel | `csa_tq_decode_kernel(CsaTqDecodeParams)` |
| Block Size | (256, 1, 1) — 8 warps |
| Grid Size | **(64, 1, 1)** — 1 CTA per head |
| Registers/Thread | **40** (light) |
| Dynamic Smem | 27.7 KB |
| Duration | 1,351 us |
| Waves Per SM | 0.13 |

## Throughput

| Metric | % of Peak | Notes |
|--------|-----------|-------|
| Compute (SM) | 5.17% | Low |
| Memory | 5.17% | Low |
| DRAM | 0.03% | Almost no DRAM traffic (L2 serves 98.5%) |
| L1/TEX Cache | 13.71% | Modest |
| FMA Pipe | 0.95% | Scalar FP32 ops barely used |

## Occupancy

| Metric | Value |
|--------|-------|
| Achieved Occupancy | 16.67% |
| Theoretical Occupancy | **50%** (much higher than FP8's 16.67%) |
| Achieved Active Warps/SM | 8.00 |
| Block Limit (Registers) | 6 blocks |
| Block Limit (Smem) | **3 blocks** (limiter for theoretical) |
| Block Limit (Warps) | 6 blocks |
| Waves Per SM | 0.13 (64 CTAs / 170 SMs) |

Note: Theoretical 50% achievable if grid had ≥510 CTAs (3 blocks × 170 SMs).

## Warp Stall Breakdown (per issue-active instruction)

| Stall Reason | Ratio | Category |
|--------------|-------|----------|
| **Long Scoreboard** | **11.12** | Global/L1 memory latency (59.9% of CPI) |
| **Wait** | **2.86** | Instruction pipeline dependency |
| **Short Scoreboard** | **2.57** | Shared memory / L1 latency |
| Barrier | 0.50 | __syncthreads waits |
| Not Selected | 0.05 | |
| Math Pipe Throttle | 0.03 | |
| MIO Throttle | 0.03 | |
| No Eligible | 89.18% | % cycles with NO eligible warps |

## Memory

| Metric | Value |
|--------|-------|
| L1/TEX Hit Rate | 43.00% |
| L2 Hit Rate | 98.46% |
| Uncoalesced Global Accesses | **48%** excessive sectors |
| Local Memory Spilling | **0** (none!) |
| Smem Bank Conflicts | **0** (none!) |

## Instruction Statistics

| Metric | Value |
|--------|-------|
| Executed Instructions | **75,144,640** (20x more than FP8's 3.7M) |
| IPC Active | 0.43 |
| FP32 FMA Instructions | 1,656,064 |
| FP32 Non-Fused Instructions | 4,222,720 |
| Branch Divergence | 99 divergent branches (1.98% of total) |

## Comparison: TQ vs FP8 Decode

| Metric | FP8 (nsp=1) | TQ | Ratio |
|--------|-------------|-----|-------|
| Duration | 3,963 us | 1,351 us | **2.9x faster** |
| Grid Size | (1,1,1) | (64,1,1) | 64 CTAs vs 1 |
| Registers/Thread | 255 | 40 | 6.4x fewer |
| Local Spill | 1.9 MB | 0 | Clean |
| Smem Bank Conflicts | 143K | 0 | Clean |
| Instructions | 3.7M | 75.1M | 20x more |
| Theoretical Occupancy | 16.67% | 50% | 3x better potential |

## Top 3 Bottlenecks

### 1. Memory Latency (Long Scoreboard = 11.12)

The kernel is **memory-latency-bound**, not compute-bound. 59.9% of CPI stalls are from L1TEX long scoreboard — waiting for global memory loads. The scattered cache access pattern (sparse indices → arbitrary entry addresses) causes 48% uncoalesced accesses and only 43% L1 hit rate.

**Impact**: Each thread does codebook lookup via global load → dependent FP32 compute. The load latency cannot be hidden with only 8 warps active.
**Fix**: V4K-15d (split-KV) would increase grid from 64 to 64×N CTAs, filling more SMs and increasing warp-level parallelism. But within each CTA, the sequential codebook scoring loop is the bottleneck.

### 2. Grid Underutilization (64 CTAs on 170 SMs)

64 CTAs on 170 SMs means 106 SMs sit idle. Theoretical occupancy is 50% (3 blocks/SM), but achieved is only 16.67% because 64/170 < 1 wave. Adding split-KV would multiply the grid by num_sm_parts.

**Impact**: 62% GPU underutilization (ncu estimate).
**Fix**: V4K-15d (split-KV for TQ decode).

### 3. Instruction Count (75M vs 3.7M for FP8)

The scalar codebook approach (per-byte 4-bit lookup → FP32 multiply) requires 20x more instructions than FP8's tensor-core MMA path. Despite being 2.9x faster due to smaller data (644B vs 1160B), TQ's instruction overhead limits how fast it can go.

**Impact**: 0.95% FMA pipe utilization — the scalar path doesn't saturate compute.
**Fix**: V4K-15b (vectorized codebook scoring) to reduce loop iterations. V4K-15c (PV vectorization) to reduce per-token instruction count.

## ncu Optimization Suggestions

1. **FP32 fusion opportunity**: 4.2M non-fused FP32 → fused FMA could give +36% FP32 perf (est. 0.34% overall)
2. **Uncoalesced access**: 48% excessive sectors — improve cache entry stride alignment
3. **Grid too small**: Needs ≥510 CTAs to saturate theoretical occupancy
