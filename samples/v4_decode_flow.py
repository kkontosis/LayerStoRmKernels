"""
SM120 DeepSeek V4 — Decode Flow (Documentation Sample)

Shows how V4 CSA/HCA/SWA kernels are orchestrated for inference:
  - Initial prefill: compress raw tokens → FP8 cache + BF16 prefill
  - Autoregressive decode: per-layer-type dispatch (CSA, HCA, SWA)
  - TQ variant: TQ cache + CUDA graph runner

DeepSeek V4 config:
  head_dim     = 512     (compressed latent, NOPE)
  qk_rope_dim  = 64      (RoPE)
  h_q          = 128 (Pro) / 64 (Flash)
  num_kv_heads = 1       (single KV head, broadcast to all Q heads)
  page_size    = 64

Three layer types determined by compress_ratios[layer_idx]:
  4   → CSA: 4:1 compression + Lightning Indexer + sparse decode
  128 → HCA: 128:1 compression + dense decode (no indexer)
  0   → SWA: 128-token sliding window only

Three KV storage backends (user-selectable):
  FP8       — 1160 bytes/entry (default)
  TQ 4-bit  — 644 bytes/entry (1.8x compression, 3.2x faster decode)
  TQ-FP8 Mix — TQ on CSA (96.9%), FP8 on HCA (3.1%)

Usage:
  python samples/v4_decode_flow.py

This sample runs real CUDA kernels on an RTX 5090/5080.
"""

import torch
import math
import sm120_mla_kernels as K
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_v4_reference import (
    HEAD_DIM, QK_ROPE_HEAD_DIM, PAGE_SIZE,
    V4_FP8_BYTES_PER_ENTRY, V4_TQ_BYTES_PER_ENTRY,
    load_codebook, generate_rotation_matrix,
)

# ============================================================================
# Config
# ============================================================================

H_Q = 64            # Flash config (use 128 for Pro)
INDEX_N_HEADS = 4
INDEX_HEAD_DIM = 128
TOPK = 64           # Must be multiple of 64
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM + QK_ROPE_HEAD_DIM)
SWA_WINDOW = 128

NUM_TOKENS = 256    # Simulated prompt length


def alloc_fp8_cache(num_entries):
    """Allocate paged FP8 KV cache."""
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    return torch.zeros(num_pages * PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY,
                       dtype=torch.uint8, device='cuda')


def alloc_tq_cache(num_entries):
    """Allocate paged TQ KV cache."""
    num_pages = (num_entries + PAGE_SIZE - 1) // PAGE_SIZE
    return torch.zeros(num_pages * PAGE_SIZE * V4_TQ_BYTES_PER_ENTRY,
                       dtype=torch.uint8, device='cuda')


# ============================================================================
# Flow 1: CSA Layer (compress_ratio=4) — FP8 Backend
# ============================================================================

def demo_csa_fp8():
    """Full CSA pipeline: compress → FP8 cache → index → sparse decode."""
    print("\n--- CSA FP8 Decode Flow ---")
    torch.manual_seed(42)

    # Simulate raw token activations from model forward pass
    raw_k_nope = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    raw_k_rope = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    raw_v = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    gate = torch.randn(NUM_TOKENS, 8, dtype=torch.bfloat16, device='cuda')
    pos_bias = torch.randn(NUM_TOKENS, 8, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
    sin = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

    # Step 1: CSA compression (4:1 gated softmax pooling)
    k_nope, k_rope, v_nope = K.v4_csa_compress(
        raw_k_nope, raw_k_rope, raw_v, gate, pos_bias, cos, sin,
        HEAD_DIM, QK_ROPE_HEAD_DIM, 8, 1)
    num_compressed = k_nope.shape[0]
    print(f"  Compressed {NUM_TOKENS} tokens → {num_compressed} entries (4:1)")

    # Step 2: FP8 quantize + cache write
    cache = alloc_fp8_cache(num_compressed)
    slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
    K.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)
    print(f"  Cache: {cache.numel()} bytes ({cache.numel()/1024:.1f} KB)")

    # Step 3: Lightning Indexer (score all compressed blocks, select top-k)
    num_blocks = (num_compressed + 7) // 8
    indexer_k = torch.randn(num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM,
                            dtype=torch.float32, device='cuda').to(torch.float8_e4m3fn)
    k_scales = torch.ones(num_blocks, dtype=torch.float32, device='cuda')
    q_proj = torch.randn(INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    score_proj = torch.ones(INDEX_N_HEADS, dtype=torch.float32, device='cuda')
    block_endpoints = torch.arange(1, num_blocks + 1, dtype=torch.int32, device='cuda') * 8

    scores = K.v4_lightning_score(q_proj, indexer_k, k_scales, score_proj)
    topk_result = K.v4_lightning_topk(scores, block_endpoints, num_compressed - 1, TOPK)
    sparse_indices = topk_result[0].unsqueeze(0).unsqueeze(0)
    print(f"  Indexer: scored {num_blocks} blocks → selected top-{TOPK}")

    # Step 4: CSA FP8 sparse decode (split-KV attention)
    q_nope = torch.randn(1, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_bt = torch.zeros(1, 1, dtype=torch.int32, device='cuda')
    swa_sl = torch.zeros(1, dtype=torch.int32, device='cuda')

    out, lse = K.v4_csa_fp8_decode(
        q_nope, q_rope, cache, sparse_indices, swa, swa_bt, swa_sl,
        SM_SCALE, TOPK, PAGE_SIZE, PAGE_SIZE, 1)
    print(f"  Output: {out.shape}, finite={torch.isfinite(out).all().item()}")


# ============================================================================
# Flow 2: HCA Layer (compress_ratio=128) — FP8 Backend
# ============================================================================

def demo_hca_fp8():
    """HCA pipeline: compress → FP8 cache → dense decode (no indexer)."""
    print("\n--- HCA FP8 Decode Flow ---")
    torch.manual_seed(42)

    raw_k_nope = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    raw_k_rope = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    raw_v = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    gate = torch.randn(NUM_TOKENS, 128, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
    sin = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

    # HCA compression (128:1 averaging)
    k_nope, k_rope, v_nope = K.v4_hca_compress(
        raw_k_nope, raw_k_rope, raw_v, gate, cos, sin,
        HEAD_DIM, QK_ROPE_HEAD_DIM, 128, 128)
    num_compressed = k_nope.shape[0]
    print(f"  Compressed {NUM_TOKENS} tokens → {num_compressed} entries (128:1)")

    cache = alloc_fp8_cache(num_compressed)
    slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
    K.v4_fp8_k_append(k_nope, k_rope, v_nope, cache, slot)

    # Dense decode over ALL compressed entries (no indexer needed)
    q_nope = torch.randn(1, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    swa = torch.zeros(PAGE_SIZE * V4_FP8_BYTES_PER_ENTRY, dtype=torch.uint8, device='cuda')
    swa_bt = torch.zeros(1, 1, dtype=torch.int32, device='cuda')
    swa_sl = torch.zeros(1, dtype=torch.int32, device='cuda')

    out, lse = K.v4_hca_fp8_decode(
        q_nope, q_rope, cache, num_compressed,
        swa, swa_bt, swa_sl, SM_SCALE, PAGE_SIZE, PAGE_SIZE, 1)
    print(f"  Output: {out.shape}, finite={torch.isfinite(out).all().item()}")


# ============================================================================
# Flow 3: SWA Layer (compress_ratio=0) — Sliding Window Only
# ============================================================================

def demo_swa():
    """SWA pipeline: direct FP8 cache write → sliding window decode."""
    print("\n--- SWA Decode Flow ---")
    torch.manual_seed(42)
    swa_len = min(NUM_TOKENS, SWA_WINDOW)

    k_nope = torch.randn(swa_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    k_rope = torch.randn(swa_len, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    v_nope = torch.randn(swa_len, HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    swa_cache = alloc_fp8_cache(swa_len)
    slot = torch.arange(swa_len, dtype=torch.int32, device='cuda')
    K.v4_fp8_k_append(k_nope, k_rope, v_nope, swa_cache, slot)
    print(f"  SWA cache: {swa_len} tokens ({swa_cache.numel()} bytes)")

    num_pages = (swa_len + PAGE_SIZE - 1) // PAGE_SIZE
    swa_bt = torch.arange(num_pages, dtype=torch.int32, device='cuda').unsqueeze(0)
    swa_sl = torch.tensor([swa_len], dtype=torch.int32, device='cuda')

    q_nope = torch.randn(1, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')

    out, lse = K.v4_swa_decode(
        q_nope, q_rope, swa_cache, swa_bt, swa_sl, SM_SCALE, PAGE_SIZE, 1)
    print(f"  Output: {out.shape}, finite={torch.isfinite(out).all().item()}")


# ============================================================================
# Flow 4: CSA TQ (TurboQuant 4-bit) — Graph Runner
# ============================================================================

def demo_csa_tq():
    """CSA TQ pipeline via CUDA graph runner (1.8x cache compression)."""
    print("\n--- CSA TQ Decode Flow (CUDA Graph) ---")
    torch.manual_seed(42)

    centroids, boundaries = load_codebook()
    Pi = generate_rotation_matrix()
    centroids_gpu = centroids.cuda()
    Pi_gpu = Pi.cuda()
    boundaries_gpu = boundaries[1:-1].cuda()

    # Compress + TQ quantize
    raw_k = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    raw_kr = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    raw_v = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    gate = torch.randn(NUM_TOKENS, 8, dtype=torch.bfloat16, device='cuda')
    pos_bias = torch.randn(NUM_TOKENS, 8, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')
    sin = torch.randn(NUM_TOKENS, QK_ROPE_HEAD_DIM // 2, dtype=torch.float32, device='cuda')

    k_nope, k_rope, v_nope = K.v4_csa_compress(
        raw_k, raw_kr, raw_v, gate, pos_bias, cos, sin,
        HEAD_DIM, QK_ROPE_HEAD_DIM, 8, 1)
    num_compressed = k_nope.shape[0]

    cache = alloc_tq_cache(num_compressed)
    slot = torch.arange(num_compressed, dtype=torch.int32, device='cuda')
    K.v4_tq_k_append(k_nope, k_rope, v_nope, cache, slot,
                      Pi_gpu, centroids_gpu, boundaries_gpu)
    print(f"  TQ cache: {num_compressed} entries, {cache.numel()} bytes "
          f"(vs FP8: {num_compressed * V4_FP8_BYTES_PER_ENTRY} bytes, "
          f"{V4_FP8_BYTES_PER_ENTRY/V4_TQ_BYTES_PER_ENTRY:.1f}x smaller)")

    # Capture CUDA graph (once per config)
    runner = K.CsaTqDecodeGraphRunner()
    runner.init(cache, Pi_gpu, centroids_gpu, 1, 1, H_Q, TOPK, SM_SCALE)
    print(f"  Graph captured: q_rotate → TQ decode → v_rotate_back")

    # Decode step (repeated per token)
    q_nope = torch.randn(1, 1, H_Q, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    q_rope = torch.randn(1, 1, H_Q, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    indices = torch.arange(TOPK, dtype=torch.int32, device='cuda').unsqueeze(0).unsqueeze(0)

    runner.update(q_nope, q_rope, indices)
    runner.replay()
    out, lse = runner.get_output(cache)
    print(f"  Output: {out.shape}, finite={torch.isfinite(out).all().item()}")
    runner.destroy()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DeepSeek V4 Decode Flow — SM120 Kernel Library")
    print("=" * 60)

    demo_csa_fp8()
    demo_hca_fp8()
    demo_swa()
    demo_csa_tq()

    print("\n" + "=" * 60)
    print("All flows completed successfully.")
    print("=" * 60)
