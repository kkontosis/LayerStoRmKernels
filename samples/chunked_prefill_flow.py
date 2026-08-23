"""
SM120 SnapMLA — Chunked Prefill + Decode Flow (Documentation Sample)

Shows how all SM120 kernels are orchestrated for inference:
  - Initial prompt prefill (BF16 ragged attention on model activations)
  - Autoregressive decode (FP8 SnapMLA with CUDA graph)
  - Chunked prefill mid-generation (dequant chunk from FP8 → BF16 prefill)

VRAM design: SnapMLA does NOT require a full-sequence BF16 KV cache buffer.
  - Initial prefill: uses model's BF16 activations directly (inherent, not extra)
  - Chunked prefill: dequants only chunk-sized subsets from FP8 cache (default 16K)
  - Decode: FP8-native, no BF16 dequant at all
  - Only FP8 paged cache is persistent (644 bytes/token for V3.2)
  Ref: SGLang-FluentLLM chunker.py (mla_max_chunk_capacity = 16K tokens)

This is a documentation sample — kernel launches are represented as function
calls matching the actual CUDA kernel interfaces. Replace `launch_*` calls
with your PyTorch custom op / ctypes / pybind11 bindings.

DeepSeek V3.2 config (adjust for MODEL1):
  d_c    = 512   (compressed latent, NOPE portion)
  d_rope = 64    (RoPE portion)
  d_qk   = 576   (d_c + d_rope)
  d_v    = 512
  h_q    = 64    (Q heads)
  h_kv   = 1     (MLA: single KV head, GQA groups expand in Q)
  n_group = 8    (GQA groups for sparse attention scoring)
  page_block_size = 64

SnapMLA KV cache row layout (per token):
  [d_c FP8 bytes | 1 float32 scale (4 bytes) | d_rope BF16 bytes]
  V3.2:   512 + 4 + 128 = 644 bytes/token
  MODEL1: 448 + 4 + 128 = 580 bytes/token
"""

import torch
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """DeepSeek V3.2 MLA configuration."""
    d_c: int = 512          # compressed latent dim (NOPE)
    d_rope: int = 64        # RoPE dim
    d_qk: int = 576         # d_c + d_rope
    d_v: int = 512           # value dim (same as d_c for MLA)
    h_q: int = 64            # number of Q heads
    h_kv: int = 1            # KV heads (1 for MLA)
    n_group: int = 8         # GQA groups for sparse scoring
    n_layers: int = 61       # number of transformer layers
    page_block_size: int = 64
    topk: int = 64           # tokens per topk block (sparse attention)


@dataclass
class CacheConfig:
    """Paged KV cache geometry."""
    num_pages: int           # total pages in pool
    page_block_size: int     # tokens per page
    bytes_per_token: int     # row stride in cache

    # Cache row layout offsets
    nope_offset: int = 0                           # FP8 NOPE starts at byte 0
    scale_offset: int = 0                          # float32 scale at byte d_c
    rope_offset: int = 0                           # BF16 ROPE at byte d_c + 4

    @staticmethod
    def for_v32() -> "CacheConfig":
        d_c, d_rope = 512, 64
        return CacheConfig(
            num_pages=4096,
            page_block_size=64,
            bytes_per_token=d_c + 4 + d_rope * 2,  # 644
            nope_offset=0,
            scale_offset=d_c,
            rope_offset=d_c + 4,
        )


# ---------------------------------------------------------------------------
# Kernel launch stubs
# ---------------------------------------------------------------------------
# These match the actual CUDA kernel param structs.
# Replace with your bindings (torch custom ops, ctypes, pybind11).

def launch_fused_q_quant(
    q_bf16: torch.Tensor,      # [s_q, h_q, d_qk] BF16
    q_nope_fp8: torch.Tensor,  # [s_q, h_q, d_nope] FP8 (output)
    q_rope_bf16: torch.Tensor, # [s_q, h_q, d_rope] BF16 (output, pre-scaled)
    q_scales: torch.Tensor,    # [s_q, h_q] float32 (output)
    d_nope: int,
) -> None:
    """sm120::prep::fused_q_quant_kernel — per-head Q quantization + RoPE pre-scaling.

    Grid: (s_q * h_q, 1, 1), Block: 256
    Steps:
      1. amax = max(|Q_NOPE|) over d_nope dims
      2. scale = amax / 448.0  (FP8 e4m3 max)
      3. Q_NOPE_FP8 = Q_NOPE / scale
      4. Q_ROPE_out = Q_ROPE / scale  (pre-scale into content domain)
      5. Store scale for post-QK dequant
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def launch_fused_k_append(
    c_kv: torch.Tensor,           # [num_tokens, d_c] BF16 compressed latent
    k_rope: torch.Tensor,         # [num_tokens, d_rope] BF16 RoPE embeddings
    kv_cache: torch.Tensor,       # paged cache base (FP8)
    slot_mapping: torch.Tensor,   # [num_tokens] int32 → flat slot in cache
    cache_stride_block: int,
    cache_stride_row: int,
    d_c: int,
    d_rope: int,
    page_size: int,
) -> None:
    """sm120::prep::fused_k_append_kernel — per-token KV quant + cache write.

    Grid: (num_tokens, 1, 1), Block: 256
    Steps:
      1. amax = max(|c_KV|) over d_c dims
      2. scale = amax / 448.0
      3. Write [c_KV/scale as FP8 | scale as f32 | k_rope/scale as BF16]
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def launch_dequant_ckv_indexed(
    kv_cache: torch.Tensor,        # paged FP8 cache base
    indices: torch.Tensor,         # [num_fetch] int32 → flat slot indices
    k_out: torch.Tensor,           # [num_fetch, d_qk] BF16 (output)
    cache_stride_block: int,
    cache_stride_row: int,
    page_size: int,
    d_c: int,
    d_rope: int,
) -> None:
    """sm120::prep::dequant_ckv_fused_indexed_kernel — gather + dequant for chunked prefill.

    Grid: (num_fetch, 1, 1), Block: 128
    Reads from interleaved cache, outputs contiguous BF16:
      NOPE: FP8 × scale → BF16
      ROPE: BF16 × scale → BF16 (un-scales the pre-scaling applied by fused_k_append)
    Matches SGLang-FluentLLM's dequantize_ckv_fused_indexed behavior.
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def launch_sparse_prefill(
    q: torch.Tensor,       # [s_q, h_q, d_qk] BF16
    kv: torch.Tensor,      # [s_kv, h_kv, d_qk] BF16  (contiguous, NOT from cache)
    indices: torch.Tensor,  # [s_q, h_kv, topk] int32
    out: torch.Tensor,     # [s_q, h_q, d_v] BF16 (output)
    lse: torch.Tensor,     # [s_q, h_q] float32 (output)
    sm_scale: float,
    d_qk: int,
) -> None:
    """sm120::prefill::sparse::head64::run_fwd_phase1_kernel<D_QK>.

    BF16 absorbed sparse prefill. KV is a contiguous BF16 buffer (NOT FP8 cache).
    Grid: (s_q, 1, 1), Block: 256
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def launch_sparse_decode(
    q_nope_fp8: torch.Tensor,  # [b, s_q, h_q, d_nope] FP8
    q_rope_bf16: torch.Tensor, # [b, s_q, h_q, d_rope] BF16
    q_scales: torch.Tensor,    # [b, s_q, h_q] float32
    kv_cache: torch.Tensor,    # paged FP8 cache base
    indices: torch.Tensor,     # [b, s_q, topk] int32
    out: torch.Tensor,         # [b, s_q, h_q, d_v] BF16
    lse: torch.Tensor,         # [b, s_q, h_q] float32
    # split-KV buffers
    lse_accum: torch.Tensor,
    o_accum: torch.Tensor,
    sched_meta: torch.Tensor,
    num_splits: torch.Tensor,
    num_sm_parts: int,
    sm_scale: float,
    kv_stride_block: int,
    kv_stride_row: int,
    page_block_size: int,
    topk: int,
) -> None:
    """sm120::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, 64>.

    SnapMLA FP8-native sparse decode. Dual MMA (FP8 NOPE + BF16 ROPE).
    Grid: (NUM_M_BLOCKS, s_q, num_sm_parts), Block: 256
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def launch_dense_decode(
    q_nope_fp8: torch.Tensor,
    q_rope_bf16: torch.Tensor,
    q_scales: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,  # [b, max_blocks] int32
    seqlens_k: torch.Tensor,    # [b] int32
    out: torch.Tensor,
    lse: torch.Tensor,
    lse_accum: torch.Tensor,
    o_accum: torch.Tensor,
    sched_meta: torch.Tensor,
    num_splits: torch.Tensor,
    num_sm_parts: int,
    sm_scale: float,
    kv_stride_block: int,
    kv_stride_row: int,
    page_block_size: int,
) -> None:
    """sm120::decode::dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<MODEL_TYPE, 64>.

    SnapMLA FP8-native dense decode. Attends to ALL tokens (no topk).
    Grid: (NUM_M_BLOCKS, s_q, num_sm_parts), Block: 256
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def launch_mla_combine(
    lse_accum: torch.Tensor,
    o_accum: torch.Tensor,
    num_splits: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """smxx::mla_combine — merge split-KV partial results.

    Arch-independent kernel from FlashMLA. Weighted-sum of partial outputs
    using LSE values for numerically stable merging.
    """
    raise NotImplementedError("Replace with CUDA kernel binding")


def compute_topk_indices(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    n_group: int,
    topk: int,
) -> torch.Tensor:
    """Compute sparse attention topk indices.

    Typically done via a lightweight scoring kernel or from the routing layer.
    Returns [s_q, h_kv, topk] int32 indices into the KV cache.
    """
    raise NotImplementedError("Application-specific scoring logic")


# ---------------------------------------------------------------------------
# Paged Cache Manager (simplified)
# ---------------------------------------------------------------------------

class PagedCacheManager:
    """Simplified paged KV cache for documentation.

    Real implementations use vLLM/SGLang-FluentLLM block managers.
    """
    def __init__(self, cfg: CacheConfig, device: str = "cuda"):
        self.cfg = cfg
        # Allocate cache pool as raw bytes (interpreted as FP8 elements)
        self.pool = torch.zeros(
            cfg.num_pages, cfg.page_block_size, cfg.bytes_per_token,
            dtype=torch.uint8, device=device
        )
        self.free_pages = list(range(cfg.num_pages))

    def allocate_pages(self, n: int) -> list[int]:
        pages = self.free_pages[:n]
        self.free_pages = self.free_pages[n:]
        return pages

    def get_slot_mapping(self, block_table: list[int], seqlen: int) -> torch.Tensor:
        """Map token positions to flat slot indices in the cache pool."""
        slots = []
        for pos in range(seqlen):
            page_idx = block_table[pos // self.cfg.page_block_size]
            row_in_page = pos % self.cfg.page_block_size
            slots.append(page_idx * self.cfg.page_block_size + row_in_page)
        return torch.tensor(slots, dtype=torch.int32, device=self.pool.device)


# ===========================================================================
#  FLOW 1: Initial Prompt Prefill
# ===========================================================================

def initial_prefill(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    cache: PagedCacheManager,
    q: torch.Tensor,       # [prompt_len, h_q, d_qk] BF16  (after RoPE application)
    c_kv: torch.Tensor,    # [prompt_len, d_c] BF16         (compressed latent from down-projection)
    k_rope: torch.Tensor,  # [prompt_len, d_rope] BF16      (RoPE embeddings for K)
    indices: torch.Tensor,  # [prompt_len, h_kv, topk] int32 (from routing)
):
    """
    PHASE: Initial prompt processing (first forward pass, no prior cache).

    The model's linear layers produce BF16 KV as inherent computation output.
    These activations are used directly for BF16 ragged attention — no dequant
    from FP8 cache needed. After attention, KV is quantized into the FP8 paged
    cache for future decode steps. The BF16 activations are freed afterward.

    VRAM note: the BF16 KV buffer here is the model's OWN activation memory
    (c_kv from down-projection). It would exist regardless of SnapMLA — it's
    the computation output, not an extra cache copy. No full-sequence BF16
    "cache" is allocated.

    This matches SGLang-FluentLLM's use_ragged=True path (flashinfer_mla_backend.py:432):
    when extend_prefix_lens=0, SGLang-FluentLLM runs ragged BF16 attention directly on
    model activations, bypassing the paged FP8 cache entirely for the read side.

    SnapMLA does NOT change the prefill attention kernel. Prefill always uses
    BF16 attention. SnapMLA only affects how KV is stored afterward (FP8 with
    per-token scales via fused_k_append).

    Steps:
      1. BF16 sparse prefill attention on model activations (ragged, not paged)
      2. Quantize KV into FP8 paged cache via fused_k_append (for future decode)
    """
    prompt_len = q.shape[0]
    device = q.device

    # --- Step 1: Allocate pages for this sequence ---
    num_pages_needed = (prompt_len + cache_cfg.page_block_size - 1) // cache_cfg.page_block_size
    block_table = cache.allocate_pages(num_pages_needed)

    # --- Step 2: BF16 sparse prefill attention ---
    # KV buffer is contiguous BF16 — NOT from FP8 cache
    # Shape: [prompt_len, h_kv, d_qk] — constructed by concatenating c_kv and k_rope
    kv_bf16 = torch.cat([
        c_kv.unsqueeze(1).expand(-1, model_cfg.h_kv, -1),     # [prompt_len, h_kv, d_c]
        k_rope.unsqueeze(1).expand(-1, model_cfg.h_kv, -1),   # [prompt_len, h_kv, d_rope]
    ], dim=-1)  # [prompt_len, h_kv, d_qk]

    out = torch.empty(prompt_len, model_cfg.h_q, model_cfg.d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(prompt_len, model_cfg.h_q, dtype=torch.float32, device=device)

    sm_scale = 1.0 / (model_cfg.d_qk ** 0.5)

    launch_sparse_prefill(
        q=q,
        kv=kv_bf16,
        indices=indices,
        out=out,
        lse=lse,
        sm_scale=sm_scale,
        d_qk=model_cfg.d_qk,
    )

    # --- Step 3: Quantize KV into FP8 paged cache ---
    # This stores [c_kv FP8 | scale f32 | k_rope/scale BF16] per token
    slot_mapping = cache.get_slot_mapping(block_table, prompt_len)

    launch_fused_k_append(
        c_kv=c_kv,
        k_rope=k_rope,
        kv_cache=cache.pool.view(-1),  # flat view
        slot_mapping=slot_mapping,
        cache_stride_block=cache_cfg.page_block_size * cache_cfg.bytes_per_token,
        cache_stride_row=cache_cfg.bytes_per_token,
        d_c=model_cfg.d_c,
        d_rope=model_cfg.d_rope,
        page_size=cache_cfg.page_block_size,
    )

    return out, lse, block_table


# ===========================================================================
#  FLOW 2: Autoregressive Decode (with optional CUDA graph)
# ===========================================================================

def decode_step(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    cache: PagedCacheManager,
    # Per-step inputs
    q_bf16: torch.Tensor,     # [batch, 1, h_q, d_qk] BF16 (new query token)
    c_kv_new: torch.Tensor,   # [batch, d_c] BF16 (new token's compressed latent)
    k_rope_new: torch.Tensor,  # [batch, d_rope] BF16 (new token's RoPE)
    # Persistent state
    block_tables: list[list[int]],  # [batch] list of page lists
    seqlens_k: torch.Tensor,       # [batch] int32
    # Sparse attention
    use_sparse: bool = True,
    indices: Optional[torch.Tensor] = None,  # [batch, 1, topk] int32
):
    """
    PHASE: Single autoregressive decode step.

    This is the hot path — runs once per generated token.
    Suitable for CUDA graph capture (see DecodeGraphRunner in graph/decode_graph.h).

    Steps:
      1. Append new KV token to FP8 cache (fused_k_append)
      2. Quantize Q to FP8 (fused_q_quant)
      3. Run FP8 attention (sparse or dense)
      4. Merge split-KV results (mla_combine)
    """
    batch = q_bf16.shape[0]
    device = q_bf16.device

    # --- Step 1: Append new KV to cache ---
    # Compute slot mapping for the new token in each batch element
    new_slots = []
    for b in range(batch):
        seq_pos = seqlens_k[b].item()
        page_idx = block_tables[b][seq_pos // cache_cfg.page_block_size]
        row = seq_pos % cache_cfg.page_block_size
        new_slots.append(page_idx * cache_cfg.page_block_size + row)
    slot_mapping = torch.tensor(new_slots, dtype=torch.int32, device=device)

    launch_fused_k_append(
        c_kv=c_kv_new,
        k_rope=k_rope_new,
        kv_cache=cache.pool.view(-1),
        slot_mapping=slot_mapping,
        cache_stride_block=cache_cfg.page_block_size * cache_cfg.bytes_per_token,
        cache_stride_row=cache_cfg.bytes_per_token,
        d_c=model_cfg.d_c,
        d_rope=model_cfg.d_rope,
        page_size=cache_cfg.page_block_size,
    )

    # Update sequence lengths (now includes the new token)
    seqlens_k = seqlens_k + 1

    # --- Step 2: Quantize Q → FP8 NOPE + BF16 ROPE ---
    q_flat = q_bf16.view(-1, model_cfg.h_q, model_cfg.d_qk)  # [batch*s_q, h_q, d_qk]
    q_nope_fp8 = torch.empty(batch, 1, model_cfg.h_q, model_cfg.d_c,
                             dtype=torch.float8_e4m3fn, device=device)
    q_rope_bf16 = torch.empty(batch, 1, model_cfg.h_q, model_cfg.d_rope,
                              dtype=torch.bfloat16, device=device)
    q_scales = torch.empty(batch, 1, model_cfg.h_q, dtype=torch.float32, device=device)

    launch_fused_q_quant(
        q_bf16=q_flat,
        q_nope_fp8=q_nope_fp8.view(-1, model_cfg.h_q, model_cfg.d_c),
        q_rope_bf16=q_rope_bf16.view(-1, model_cfg.h_q, model_cfg.d_rope),
        q_scales=q_scales.view(-1, model_cfg.h_q),
        d_nope=model_cfg.d_c,
    )

    # --- Step 3: FP8 Attention ---
    sm_scale = 1.0 / (model_cfg.d_qk ** 0.5)
    num_sm_parts = 8  # tunable: number of split-KV partitions

    out = torch.empty(batch, 1, model_cfg.h_q, model_cfg.d_v,
                      dtype=torch.bfloat16, device=device)
    lse = torch.empty(batch, 1, model_cfg.h_q, dtype=torch.float32, device=device)

    # Split-KV intermediates
    lse_accum = torch.empty(num_sm_parts, batch, 1, model_cfg.h_q,
                            dtype=torch.float32, device=device)
    o_accum = torch.empty(num_sm_parts, batch, 1, model_cfg.h_q, model_cfg.d_v,
                          dtype=torch.float32, device=device)
    sched_meta = torch.empty(num_sm_parts, 8, dtype=torch.int32, device=device)  # DecodingSchedMeta
    num_splits = torch.empty(batch, dtype=torch.int32, device=device)

    if use_sparse:
        assert indices is not None, "Sparse decode requires topk indices"
        launch_sparse_decode(
            q_nope_fp8=q_nope_fp8,
            q_rope_bf16=q_rope_bf16,
            q_scales=q_scales,
            kv_cache=cache.pool.view(-1),
            indices=indices,
            out=out,
            lse=lse,
            lse_accum=lse_accum,
            o_accum=o_accum,
            sched_meta=sched_meta,
            num_splits=num_splits,
            num_sm_parts=num_sm_parts,
            sm_scale=sm_scale,
            kv_stride_block=cache_cfg.page_block_size * cache_cfg.bytes_per_token,
            kv_stride_row=cache_cfg.bytes_per_token,
            page_block_size=cache_cfg.page_block_size,
            topk=model_cfg.topk,
        )
    else:
        # Build block_table tensor
        max_blocks = max(len(bt) for bt in block_tables)
        bt_tensor = torch.zeros(batch, max_blocks, dtype=torch.int32, device=device)
        for b, bt in enumerate(block_tables):
            bt_tensor[b, :len(bt)] = torch.tensor(bt, dtype=torch.int32)

        launch_dense_decode(
            q_nope_fp8=q_nope_fp8,
            q_rope_bf16=q_rope_bf16,
            q_scales=q_scales,
            kv_cache=cache.pool.view(-1),
            block_table=bt_tensor,
            seqlens_k=seqlens_k,
            out=out,
            lse=lse,
            lse_accum=lse_accum,
            o_accum=o_accum,
            sched_meta=sched_meta,
            num_splits=num_splits,
            num_sm_parts=num_sm_parts,
            sm_scale=sm_scale,
            kv_stride_block=cache_cfg.page_block_size * cache_cfg.bytes_per_token,
            kv_stride_row=cache_cfg.bytes_per_token,
            page_block_size=cache_cfg.page_block_size,
        )

    # --- Step 4: Merge split-KV results ---
    launch_mla_combine(
        lse_accum=lse_accum,
        o_accum=o_accum,
        num_splits=num_splits,
        out=out,
        lse=lse,
    )

    return out, lse


# ===========================================================================
#  FLOW 3: Chunked Prefill (new prompt arrives mid-generation)
# ===========================================================================

MLA_MAX_CHUNK_CAPACITY = 16 * 1024  # SGLang-FluentLLM default (chunker.py)


def chunked_prefill(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    cache: PagedCacheManager,
    # Existing sequence state
    block_table: list[int],        # existing pages for this sequence
    past_seqlen: int,              # tokens already in FP8 cache
    # New tokens from model forward pass
    q_new: torch.Tensor,           # [new_len, h_q, d_qk] BF16
    c_kv_new: torch.Tensor,        # [new_len, d_c] BF16
    k_rope_new: torch.Tensor,      # [new_len, d_rope] BF16
    indices: torch.Tensor,         # [new_len, h_kv, topk] int32
    batch_size: int = 1,
):
    """
    PHASE: Chunked prefill — process new tokens while past tokens are
    already stored in the FP8 cache.

    The key insight: past tokens are dequantized from FP8 in CHUNKS, not
    all at once. Only a small BF16 working buffer is ever allocated.

    This matches SGLang-FluentLLM's chunker.py (mla_max_chunk_capacity = 16K tokens):
      chunk_len = capacity // batch_size
      for each chunk of past tokens:
        dequant chunk from FP8 → BF16 buffer (small, reused)
        run BF16 attention over [chunk KV + new tokens]
        accumulate partial output via log-sum-exp merging

    VRAM: chunk_len × d_qk × 2 bytes of BF16 at any time.
    For batch=1, 16K chunk at d_qk=576: 16K × 576 × 2 = 18 MB.
    Compare to dequanting all 128K past tokens: 128K × 576 × 2 = 141 MB.
    """
    new_len = q_new.shape[0]
    total_len = past_seqlen + new_len
    device = q_new.device
    sm_scale = 1.0 / (model_cfg.d_qk ** 0.5)

    # --- Step 0: Append new tokens to FP8 cache FIRST ---
    # (SGLang-FluentLLM does this before attention — set_mla_kv_buffer in forward_absorbed)
    current_pages = len(block_table)
    pages_needed = (total_len + cache_cfg.page_block_size - 1) // cache_cfg.page_block_size
    if pages_needed > current_pages:
        new_pages = cache.allocate_pages(pages_needed - current_pages)
        block_table.extend(new_pages)

    new_slots = cache.get_slot_mapping(
        block_table, total_len
    )[past_seqlen:]  # slots for new tokens only

    launch_fused_k_append(
        c_kv=c_kv_new,
        k_rope=k_rope_new,
        kv_cache=cache.pool.view(-1),
        slot_mapping=new_slots,
        cache_stride_block=cache_cfg.page_block_size * cache_cfg.bytes_per_token,
        cache_stride_row=cache_cfg.bytes_per_token,
        d_c=model_cfg.d_c,
        d_rope=model_cfg.d_rope,
        page_size=cache_cfg.page_block_size,
    )

    # --- Step 1: Compute chunk boundaries ---
    # SGLang-FluentLLM: chunk_len = mla_max_chunk_capacity // batch_size
    chunk_len = MLA_MAX_CHUNK_CAPACITY // batch_size
    num_chunks = (past_seqlen + chunk_len - 1) // chunk_len
    num_chunks = max(num_chunks, 1)  # at least 1 chunk even if past_seqlen=0

    # Accumulated output and LSE across chunks (for log-sum-exp merging)
    out_accum = torch.zeros(new_len, model_cfg.h_q, model_cfg.d_v,
                            dtype=torch.float32, device=device)
    lse_accum = torch.full((new_len, model_cfg.h_q), float('-inf'),
                           dtype=torch.float32, device=device)

    # Reusable BF16 buffer — only chunk_len tokens, NOT full past_seqlen
    chunk_kv_buf = torch.empty(chunk_len, model_cfg.d_qk,
                               dtype=torch.bfloat16, device=device)

    # --- Step 2: Loop over chunks of past tokens ---
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_len
        chunk_end = min(chunk_start + chunk_len, past_seqlen)
        this_chunk_len = chunk_end - chunk_start

        if this_chunk_len == 0:
            continue

        # Dequant this chunk from FP8 cache → reusable BF16 buffer
        chunk_slots = cache.get_slot_mapping(block_table, past_seqlen)[chunk_start:chunk_end]

        launch_dequant_ckv_indexed(
            kv_cache=cache.pool.view(-1),
            indices=chunk_slots,
            k_out=chunk_kv_buf[:this_chunk_len],
            cache_stride_block=cache_cfg.page_block_size * cache_cfg.bytes_per_token,
            cache_stride_row=cache_cfg.bytes_per_token,
            page_size=cache_cfg.page_block_size,
            d_c=model_cfg.d_c,
            d_rope=model_cfg.d_rope,
        )

        # Build KV for this chunk: [chunk_past + new_tokens]
        new_kv_bf16 = torch.cat([c_kv_new, k_rope_new], dim=-1)
        chunk_kv = torch.cat([
            chunk_kv_buf[:this_chunk_len],
            new_kv_bf16,
        ], dim=0)
        chunk_kv = chunk_kv.unsqueeze(1).expand(-1, model_cfg.h_kv, -1)

        # Run BF16 prefill attention over [chunk_past + new_tokens]
        chunk_out = torch.empty(new_len, model_cfg.h_q, model_cfg.d_v,
                                dtype=torch.bfloat16, device=device)
        chunk_lse = torch.empty(new_len, model_cfg.h_q, dtype=torch.float32, device=device)

        launch_sparse_prefill(
            q=q_new,
            kv=chunk_kv,
            indices=indices,
            out=chunk_out,
            lse=chunk_lse,
            sm_scale=sm_scale,
            d_qk=model_cfg.d_qk,
        )

        # --- Log-sum-exp merge with accumulated output ---
        # This is numerically stable online accumulation:
        #   new_lse = log(exp(old_lse) + exp(chunk_lse))
        #   out = (exp(old_lse - new_lse) * old_out + exp(chunk_lse - new_lse) * chunk_out)
        new_lse = torch.logaddexp(lse_accum, chunk_lse)
        old_weight = torch.exp(lse_accum - new_lse).unsqueeze(-1)  # [new_len, h_q, 1]
        new_weight = torch.exp(chunk_lse - new_lse).unsqueeze(-1)
        out_accum = old_weight * out_accum + new_weight * chunk_out.float()
        lse_accum = new_lse

    # --- Final output ---
    out = out_accum.to(torch.bfloat16)
    lse = lse_accum

    return out, lse, block_table


# ===========================================================================
#  FLOW 4: CUDA Graph Decode (amortize CPU launch overhead)
# ===========================================================================

def cuda_graph_decode_example(
    model_cfg: ModelConfig,
    cache_cfg: CacheConfig,
    cache: PagedCacheManager,
    batch_size: int = 32,
    max_gen_tokens: int = 512,
):
    """
    PHASE: Decode loop with CUDA graph (amortizes kernel launch overhead).

    Uses the DecodeGraphRunner from csrc/sm120/graph/decode_graph.h.
    The graph captures the fixed kernel sequence:
      fused_q_quant → splitkv_mla → mla_combine

    Between replays, only the data changes (via cudaMemcpyAsync into
    fixed-address buffers). Grid dimensions stay constant.

    Requirements for graph replay:
      - batch_size must be constant (baked into grid dims)
      - s_q = 1 (always for decode)
      - kv_cache base pointer is stable (same paged pool)
      - num_sm_parts is constant
    """
    device = "cuda"

    # -----------------------------------------------------------------------
    # Graph capture phase (once per batch_size)
    # -----------------------------------------------------------------------

    # In C++, this is:
    #   sm120::graph::DecodeGraphRunner runner;
    #   runner.init(cfg, stream);
    #
    # In Python with ctypes/pybind11:
    #   runner = DecodeGraphRunner(
    #       batch_size=batch_size,
    #       s_q=1,
    #       h_q=model_cfg.h_q,
    #       h_kv=model_cfg.h_kv,
    #       d_qk=model_cfg.d_qk,
    #       d_v=model_cfg.d_v,
    #       d_nope=model_cfg.d_c,
    #       page_block_size=cache_cfg.page_block_size,
    #       max_num_blocks_per_seq=max_gen_tokens // cache_cfg.page_block_size + 1,
    #       kv_cache=cache.pool.data_ptr(),
    #       sm_scale=1.0 / (model_cfg.d_qk ** 0.5),
    #       model_type="V32",
    #       num_sm_parts=8,
    #       sparse=True,
    #       topk=model_cfg.topk,
    #   )

    print(f"Graph captured for batch_size={batch_size}")

    # -----------------------------------------------------------------------
    # Decode loop (per generated token)
    # -----------------------------------------------------------------------

    for step in range(max_gen_tokens):
        # 1. Model forward pass produces Q, c_kv, k_rope for new token
        # q_bf16     = model.forward(...)  → [batch, 1, h_q, d_qk]
        # c_kv_new   = down_proj(...)      → [batch, d_c]
        # k_rope_new = rope_embed(...)     → [batch, d_rope]

        # 2. Append new KV to cache (outside graph — changes page allocation)
        # launch_fused_k_append(c_kv_new, k_rope_new, ...)

        # 3. Compute topk indices (outside graph — changes per step)
        # indices = compute_topk_indices(q, kv_cache, ...)

        # 4. Update graph inputs and replay
        # runner.update(q_bf16, seqlens_k, block_table, indices, stream)
        # runner.replay(stream)

        # 5. Read output from graph's fixed buffers
        # out = runner.out_ptr()  → [batch, 1, h_q, d_v]
        # lse = runner.lse_ptr()  → [batch, 1, h_q]

        pass

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    # runner.destroy()


# ===========================================================================
#  Main: end-to-end example
# ===========================================================================

if __name__ == "__main__":
    """
    End-to-end inference flow for DeepSeek V3.2 on SM120 (RTX 5090).

    Timeline for a single request:
      ┌─────────────────────────────────────────────────────────────────┐
      │ 1. Initial prefill (BF16 absorbed attention)                   │
      │    - KV comes from model's down-projection (contiguous BF16)   │
      │    - Output computed, KV quantized into FP8 cache              │
      ├─────────────────────────────────────────────────────────────────┤
      │ 2. Decode loop (FP8 SnapMLA, CUDA graph replay)               │
      │    - fused_q_quant: Q BF16 → FP8 NOPE + BF16 ROPE             │
      │    - fused_k_append: new KV → FP8 cache                       │
      │    - sparse/dense FP8 decode: dual MMA + split-KV              │
      │    - mla_combine: merge partial results                        │
      │    Repeat until EOS or max_tokens                              │
      ├─────────────────────────────────────────────────────────────────┤
      │ 3. [Optional] Chunked prefill (if new prompt arrives)          │
      │    - dequant_ckv_indexed: past FP8 → BF16 buffer               │
      │    - BF16 prefill over [past + new chunk]                      │
      │    - fused_k_append: new chunk → FP8 cache                    │
      └─────────────────────────────────────────────────────────────────┘

    Kernel summary:
      ┌──────────────────────────┬──────────────┬──────────────────────┐
      │ Kernel                   │ Phase        │ Data type            │
      ├──────────────────────────┼──────────────┼──────────────────────┤
      │ fused_q_quant            │ Decode       │ BF16 → FP8+BF16     │
      │ fused_k_append           │ Both         │ BF16 → FP8 cache    │
      │ dequant_ckv_indexed      │ Chunk prefill│ FP8 cache → BF16    │
      │ sparse_prefill (phase1)  │ Prefill      │ BF16 × BF16         │
      │ sparse_decode (splitkv)  │ Decode       │ FP8+BF16 dual MMA   │
      │ dense_decode (splitkv)   │ Decode       │ FP8+BF16 dual MMA   │
      │ mla_combine              │ Decode       │ float32 merge       │
      └──────────────────────────┴──────────────┴──────────────────────┘
    """
    print("SM120 SnapMLA — Chunked Prefill + Decode Flow")
    print("This is a documentation sample. See kernel stubs for API reference.")
    print()

    cfg = ModelConfig()
    cache_cfg = CacheConfig.for_v32()

    print(f"Model: DeepSeek V3.2")
    print(f"  d_c={cfg.d_c}, d_rope={cfg.d_rope}, d_qk={cfg.d_qk}, d_v={cfg.d_v}")
    print(f"  h_q={cfg.h_q}, h_kv={cfg.h_kv}, n_group={cfg.n_group}")
    print(f"Cache: {cache_cfg.bytes_per_token} bytes/token, {cache_cfg.page_block_size} tokens/page")
    print()

    print("Flows:")
    print("  1. initial_prefill()    — BF16 absorbed prefill + FP8 cache write")
    print("  2. decode_step()        — FP8 SnapMLA decode (sparse or dense)")
    print("  3. chunked_prefill()    — dequant past + BF16 prefill + FP8 append")
    print("  4. cuda_graph_decode()  — graph-captured decode for amortized launch overhead")
