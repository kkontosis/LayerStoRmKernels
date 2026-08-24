# USAGE.md — SM120 SnapMLA Kernel Library

> **Dependencies note:** release archives don't carry submodule contents — clone CUTLASS (see `.gitmodules`) into `3rd-party/cutlass` before building.

## What This Is

CUDA kernel library for MLA (Multi-head Latent Attention) inference on SM120 GPUs (RTX 5090/5080). Provides the complete attention pipeline: FP8 decode, BF16 prefill, KV cache management, split-KV scheduling, DCP correction, and CUDA graph capture.

All kernels use the **SnapMLA interleaved FP8 KV cache format** — there is no BF16 KV decode path.

## SnapMLA KV Cache Format

Per-token row layout in the paged cache:

```
[d_c FP8 bytes] [1 float32 scale (4 bytes)] [d_rope BF16 bytes]
```

| Model | d_c (NOPE) | scale | d_rope (BF16) | Total bytes/token |
|-------|-----------|-------|---------------|-------------------|
| V3.2 (DeepSeek) | 512 | 4 | 128 (64 dims) | **644** |
| MODEL1 (Kimi) | 448 | 4 | 128 (64 dims) | **580** |

**Why this layout**: NOPE quantizes well to FP8 (tight dynamic range). ROPE cannot survive FP8 (spans +/-10^3, outlier tails) so stays BF16. The per-token float32 scale enables SnapMLA's dual-MMA pipeline where both NOPE and ROPE domains share a single dequantization factor.

**Write path** (`fused_k_append`): computes `scale = amax(c_kv) / 448`, stores `[c_kv/scale as FP8 | scale | k_rope/scale as BF16]`.

**Read path (decode)**: dual MMA reads FP8 NOPE and BF16 ROPE directly from cache — no dequant step.

**Read path (prefill)**: `dequant_ckv_indexed` recovers original values: `[fp8 * scale | bf16 * scale]` → contiguous BF16 `[d_c + d_rope]`.

## Kernel Inventory

| Kernel | File | Precision | Purpose |
|--------|------|-----------|---------|
| **fused_q_quant** | `prep/fused_q_quant.cu` | BF16→FP8+BF16 | Q quantization + ROPE pre-scaling |
| **fused_k_append** | `prep/fused_k_append.cu` | BF16→FP8 cache | KV cache write (per-token quant + page insert) |
| **dequant_ckv_indexed** | `prep/dequant_ckv_indexed.cu` | FP8 cache→BF16 | Indexed gather + dequant for chunked prefill |
| **dense_decode** | `decode/dense_fp8/splitkv_mla.cu` | FP8+BF16 dual MMA | All-token SnapMLA decode with split-KV |
| **sparse_decode** | `decode/sparse_fp8/splitkv_mla.cu` | FP8+BF16 dual MMA | Top-k SnapMLA decode with split-KV |
| **dense_prefill** | `prefill/dense/fwd/head64/phase1.cuh` | BF16 | Full-sequence absorbed BF16 prefill |
| **sparse_prefill** | `prefill/sparse/fwd/head64/phase1.cuh` | BF16 | Top-k absorbed BF16 prefill |
| **get_mla_metadata** | `smxx/get_mla_metadata.cu` | — | Split-KV scheduling (arch-generic) |
| **mla_combine** | `smxx/mla_combine.cu` | — | Merge split-KV partials via LSE (arch-generic) |
| **dcp_lse_correct** | `smxx/dcp_lse_correct.cu` | — | Multi-GPU DCP output correction (arch-generic) |
| **DecodeGraphRunner** | `graph/decode_graph.{h,cu}` | — | CUDA graph: q_quant → decode → combine |

All SM120 kernels: 256 threads (8 warps), producer/consumer architecture (warps 0-3 consume MMA, warps 4-7 produce loads).

## Inference Flows

### Flow 1: Initial Prefill

No prior cache exists. Model forward produces BF16 activations used directly.

```
Model forward → c_kv [N, d_c] BF16, k_rope [N, d_rope] BF16, q [N, h_q, d_qk] BF16
    │
    ├─ concat(c_kv, k_rope) → kv [N, d_qk] BF16
    │       │
    │       └─ sparse_prefill(q, kv, indices) → out [N, h_q, d_v], lse
    │          OR dense_prefill(q, kv) → out, lse
    │
    └─ fused_k_append(c_kv, k_rope, cache, slots) → FP8 paged cache
```

BF16 activations freed after this step. Only FP8 cache persists.

### Flow 2: Autoregressive Decode

Hot path — one new token per step.

```
fused_k_append(c_kv_new, k_rope_new, cache, slot)     ← append to FP8 cache
fused_q_quant(q_bf16)                                  ← Q → FP8 NOPE + BF16 ROPE + scales
get_mla_metadata(seqlens_k, num_sm_parts)              ← split-KV scheduling
dense_decode / sparse_decode(q_fp8, q_rope, scales,    ← FP8 attention
    cache, block_table/indices, ...)
mla_combine(o_accum, lse_accum, ...)                   ← merge if num_sm_parts > 1
```

With CUDA graph (production path):
```
get_mla_metadata(seqlens_k, ...)                       ← outside graph (per-step)
memcpy metadata → graph's fixed buffers
runner.update(q_bf16, seqlens_k, block_table, ...)     ← memcpy into fixed buffers
runner.replay()                                        ← fused_q_quant → decode → combine
read from runner.out_ptr(), runner.lse_ptr()
```

### Flow 3: Chunked Prefill

Extends cached sequence with new tokens. Past tokens dequanted in chunks to bound BF16 memory.

```
fused_k_append(c_kv_new, k_rope_new, cache, slots)    ← new tokens → FP8 cache FIRST

for each chunk of past tokens (default 16K):
    dequant_ckv_indexed(cache, chunk_slots)            ← FP8 → BF16 chunk
    sparse_prefill(q_new, chunk_kv, indices)           ← BF16 attention on chunk
    LSE-merge with accumulator                         ← numerically stable online merge

BF16 chunk buffer reused (18 MB at batch=1, d_qk=576)
```

### Flow 4: DCP (Decode Context Parallelism)

KV cache sequence-sharded across GPUs. Each GPU runs decode on its local shard.

```
[each GPU] decode(q, local_kv_shard) → partial_out, partial_lse
[NCCL]     allgather LSE → lses [dcp_size, B, H]
[each GPU] dcp_lse_correct(partial_out, lses, rank) → corrected_out, global_lse
[NCCL]     allreduce corrected_out → final_out
```

`dcp_lse_correct` is attention-backend-agnostic — works with any kernel that outputs (out, lse).

## C++ API

Include headers from `csrc/`. The kernels are organized as:
- **Param structs** define all inputs/outputs for each kernel
- **Launch functions** accept a param struct + CUDA stream

```cpp
// Example: dense decode
#include "sm120/decode/dense_fp8/params.h"
sm120::decode::dense_fp8::DenseAttnDecodeParams params;
// ... fill params ...
sm120::decode::dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<ModelType::V32, 64>(params);

// Example: prep kernel
#include "sm120/prep/fused_k_append.cu"  // defines struct + run function in namespace sm120::prep
sm120::prep::FusedKAppendParams params;
sm120::prep::run_fused_k_append(params, stream);

// Example: CUDA graph
#include "sm120/graph/decode_graph.h"
sm120::graph::DecodeGraphRunner runner;
runner.init(cfg, stream);
// per step:
runner.update(q_ptr, seqlens_ptr, block_table_ptr, indices_ptr, stream);
runner.replay(stream);
// read from runner.out_ptr(), runner.lse_ptr()
runner.destroy();
```

## Python API

```python
import sm120_mla_kernels as K

# Prep
q_nope_fp8, q_rope_bf16, q_scales = K.fused_q_quant(q_bf16, d_nope=512)
K.fused_k_append(c_kv, k_rope, kv_cache, slot_mapping, d_c, d_rope, page_size)
kv_bf16 = K.dequant_ckv_indexed(kv_cache, indices, d_c, d_rope, page_size)

# Decode (orchestrated: metadata + kernel + combine)
out, lse = K.dense_decode_v32(q_nope_fp8, q_rope_bf16, q_scales,
    kv_cache, block_table, seqlens_k, sm_scale, page_block_size, num_sm_parts)
out, lse = K.sparse_decode_v32(q_nope_fp8, q_rope_bf16, q_scales,
    kv_cache, indices, sm_scale, page_block_size, topk, num_sm_parts)

# Prefill
out, lse = K.sparse_prefill_v32(q_bf16, kv_bf16, indices, sm_scale, topk)
out, lse = K.dense_prefill_v32(q_bf16, kv_bf16, sm_scale)

# DCP
corrected_out, global_lse = K.dcp_lse_correct(partial_out, lses, rank)

# CUDA graph (persistent runner)
runner = K.DecodeGraphRunner()
runner.init(kv_cache, batch_size=1, s_q=1, h_q=64, h_kv=1,
    d_qk=576, d_v=512, d_nope=512, page_block_size=64,
    max_num_blocks_per_seq=512, sm_scale=0.0417, num_sm_parts=1)
runner.update_metadata(seqlens_k, num_sm_parts)
runner.update(q_bf16, seqlens_k, block_table)
runner.replay()
out, lse = runner.get_output(q_bf16)  # ref_tensor for device
runner.destroy()
```

## Model Dimensions

| Parameter | V3.2 (DeepSeek) | MODEL1 (Kimi) |
|-----------|-----------------|---------------|
| d_qk (Q/K total) | 576 | 512 |
| d_nope (content) | 512 | 448 |
| d_rope (positional) | 64 | 64 |
| d_v (value) | 512 | 512 |
| h_q (Q heads) | 64 | 64 |
| h_kv (KV heads) | 1 | 1 |
| Cache bytes/token | 644 | 580 |
| sm_scale | 1/sqrt(576) | 1/sqrt(512) |

## V4 Kernels (DeepSeek V4 CSA/HCA Sparse Attention)

V4 kernels implement CSA (Compressed Sparse Attention) and HCA (Heavily Compressed Attention) for DeepSeek V4 models with three KV storage backends: FP8, TQ 4-bit, and TQ-FP8-Mix.

### V4 API Constraints

**topk must be a multiple of 64.** All split-KV decode kernels (`v4_csa_fp8_decode`, `v4_hca_fp8_decode`, and the CSA FP8 graph runner) process sparse indices in tiles of TOPK_BLOCK_SIZE=64 tokens. This matches the FlashMLA and vLLM convention — both enforce `topk % 64 == 0`.

- DeepSeek V4 Pro uses `index_topk=1024` (1024/64 = 16 tiles) ✓
- DeepSeek V4 Flash uses `index_topk=512` (512/64 = 8 tiles) ✓
- The Python binding pads non-aligned topk with -1 sentinels and emits a `TORCH_WARN_ONCE`
- The graph runner (`CsaFp8DecodeGraphRunner`) enforces this with a hard `TORCH_CHECK`

### V4 Python API

```python
import sm120_mla_kernels as K

# Compressors (CSA 4:1, HCA 128:1)
k_nope, k_rope, v = K.v4_csa_compress(input_k_nope, input_k_rope, input_v,
    gate, pos_bias, cos, sin, head_dim=512, qk_rope_head_dim=64, window=8, stride=1)
k_nope, k_rope, v = K.v4_hca_compress(input_k_nope, input_k_rope, input_v,
    gate, cos, sin, head_dim=512, qk_rope_head_dim=64, window=128, stride=128)

# FP8 cache write/read
K.v4_fp8_k_append(k_nope, k_rope, v_nope, kv_cache, slot_mapping)
k_nope, k_rope, v_nope = K.v4_fp8_dequant_indexed(kv_cache, indices, 512, 64)

# TQ 4-bit cache write/read
K.v4_tq_k_append(k_nope, k_rope, v_nope, kv_cache, slot_mapping, Pi, centroids, boundaries)
k_nope, k_rope, v_nope = K.v4_tq_dequant_indexed(kv_cache, indices, Pi, centroids, 512, 64)

# Lightning Indexer
scores = K.v4_lightning_score(q_proj, indexer_k_fp8, k_scales, score_proj)
indices, scores, effective_k = K.v4_lightning_topk(scores, block_endpoints, query_pos, topk=1024)

# CSA FP8 decode (topk MUST be multiple of 64)
out, lse = K.v4_csa_fp8_decode(q_nope, q_rope, compressed_kv, sparse_indices,
    swa_kv, swa_block_table, swa_seqlens, sm_scale, topk=1024,
    compressed_page_block_size=64, swa_page_block_size=64, num_sm_parts=1)

# CSA TQ decode (no topk alignment constraint — single CTA per head)
out_rot, lse = K.v4_csa_tq_decode(q_rot, q_rope, tq_cache, indices, centroids, sm_scale)

# TQ graph runner
runner = K.CsaTqDecodeGraphRunner()
runner.init(tq_cache, Pi, centroids, batch_size=1, s_q=1, h_q=64, topk=1024, sm_scale=0.0417)
runner.update(q_nope_bf16, q_rope_bf16, sparse_indices)
runner.replay()
out, lse = runner.get_output(ref_tensor)
```

### V4 Cache Formats

| Backend | Bytes/entry | K layout | V layout |
|---------|-------------|----------|----------|
| **FP8** | 1160 | 512B FP8 nope + 4B scale + 128B BF16 rope | 512B FP8 nope + 4B scale |
| **TQ 4-bit** | 644 | 256B packed nope + 2B FP16 norm + 128B BF16 rope | 256B packed nope + 2B FP16 norm |

## Build

```bash
pip install -e . --no-build-isolation    # Python extension
```

Requires: CUDA 12.8+ (SM120), CUTLASS 3.x, PyTorch 2.x with FP8 support.

For C++ integration: include `csrc/` headers, compile `.cu` files with `-arch=sm_120 -std=c++17`. See `setup.py` for the full source list and compiler flags.

## License

Licensed under the Apache License 2.0 — see `LICENSE.md`. Third-party
attributions and license notices (FlashMLA, SGLang, vLLM, TensorRT-LLM,
CUTLASS, sample-data sources) are collected in `THIRD_PARTY_NOTICES.md`.
