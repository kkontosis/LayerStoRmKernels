# Real-Data Test Results

All cosines are vs Dense BF16 (full-sequence ground truth) unless noted.
Data sources: random (`torch.randn` projections), tokenized (real text embeddings), forward (layers 0-1).

## Dense Decode (vs BF16)

### Kimi K2.5 (dense kernel)

| Data | s_kv | Rough | Naive FP8 | SnapMLA FP8 | GPU Dense |
|------|------|-------|-----------|-------------|-----------|
| random | 256 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| random | 1024 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| random | 4096 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| random | 32768 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| tokenized | 1024 | 0.9999 | 0.9991 | 0.9994 | 0.9992 |
| tokenized | 4096 | 0.9999 | 0.9993 | 0.9996 | 0.9995 |
| tokenized | 32768 | 0.9999 | 0.9996 | 0.9999 | 0.9999 |
| forward | 1024 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| forward | 4096 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| forward | 32768 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |

### DeepSeek V3.2 (sparse kernel)

| Data | s_kv | Rough | Naive FP8 | SnapMLA FP8 | GPU Sparse |
|------|------|-------|-----------|-------------|------------|
| random | 1024 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| random | 4096 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| random | 32768 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| tokenized | 1024 | 1.0000 | 0.9992 | 0.9997 | 0.9996 |
| tokenized | 4096 | 1.0000 | 0.9995 | 0.9998 | 0.9998 |
| tokenized | 32768 | 1.0000 | 0.9994 | 0.9999 | 0.9999 |
| forward | 1024 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| forward | 4096 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |
| forward | 32768 | 1.0000 | 0.9999 | 0.9999 | 0.9999 |

## Sparse Topk Decode — Oracle 25% (DeepSeek V3.2)

cos(sparse) = vs Sparse BF16 baseline (same token subset, isolates kernel accuracy).
cos(dense) = vs Dense BF16 (combined indexer + kernel gap).

| Data | s_kv | topk | Sparse BF16 | SnapMLA cos(s) | SnapMLA cos(d) | GPU cos(s) | GPU cos(d) | NSA cos(d) |
|------|------|------|-------------|----------------|----------------|------------|------------|------------|
| random | 256 | 64 | 0.738 | 0.9999 | 0.738 | 0.9999 | 0.738 | — |
| random | 1024 | 256 | 0.769 | 0.9999 | 0.769 | 0.9999 | 0.769 | — |
| random | 4096 | 1024 | 0.771 | 0.9999 | 0.771 | 0.9999 | 0.771 | — |
| random | 32768 | 8192 | 0.734 | 0.9999 | 0.733 | 0.9999 | 0.734 | — |
| tokenized | 256 | 64 | 0.746 | 0.9995 | 0.746 | 0.9992 | 0.745 | 0.463 |
| tokenized | 1024 | 256 | 0.788 | 0.9996 | 0.787 | 0.9994 | 0.787 | 0.709 |
| tokenized | 4096 | 1024 | 0.733 | 0.9998 | 0.733 | 0.9998 | 0.732 | 0.797 |
| tokenized | 32768 | 8192 | 0.850 | 0.9999 | 0.850 | 0.9999 | 0.850 | **0.865** |
| forward | 256 | 64 | 0.757 | 0.9998 | 0.757 | 0.9997 | 0.757 | 0.711 |
| forward | 1024 | 256 | 0.875 | 0.9999 | 0.874 | 0.9999 | 0.874 | 0.673 |
| forward | 4096 | 1024 | 0.932 | 0.9999 | 0.932 | 0.9999 | 0.932 | 0.677 |
| forward | 32768 | 8192 | 0.622 | 0.9998 | 0.624 | 0.9999 | 0.624 | **0.770** |

## Dense Prefill s_q=128 (vs BF16)

### Kimi K2.5 (dense kernel)

| Data | s_kv | Rough | Naive FP8 | SnapMLA FP8 | GPU Prefill |
|------|------|-------|-----------|-------------|-------------|
| random | 1024 | 1.0003 | 1.0002 | 1.0002 | 1.0001 |
| random | 32768 | 1.0002 | 1.0002 | 1.0002 | 1.0000 |
| tokenized | 1024 | 1.0002 | 0.9994 | 0.9997 | 1.0000 |
| tokenized | 32768 | 1.0002 | 0.9998 | 1.0001 | 1.0001 |
| forward | 1024 | 1.0001 | 1.0000 | 1.0000 | 1.0000 |
| forward | 32768 | 1.0002 | 1.0001 | 1.0001 | 1.0001 |

### DeepSeek V3.2 (sparse kernel, topk=all)

| Data | s_kv | Rough | Naive FP8 | SnapMLA FP8 | GPU Prefill |
|------|------|-------|-----------|-------------|-------------|
| random | 1024 | 1.0005 | 1.0005 | 1.0005 | 1.0004 |
| random | 32768 | 1.0005 | 1.0005 | 1.0005 | 1.0004 |
| tokenized | 1024 | 1.0006 | 0.9994 | 1.0003 | 1.0005 |
| tokenized | 32768 | 1.0006 | 1.0000 | 1.0005 | 1.0006 |
| forward | 1024 | 1.0006 | 1.0006 | 1.0006 | 1.0005 |
| forward | 32768 | 1.0006 | 1.0006 | 1.0006 | 1.0006 |

## Sparse Topk Prefill s_q=128 — Oracle 25% (DeepSeek V3.2)

| Data | s_kv | topk | Sparse BF16 | SnapMLA cos(s) | SnapMLA cos(d) | GPU cos(s) | GPU cos(d) | NSA cos(d) |
|------|------|------|-------------|----------------|----------------|------------|------------|------------|
| random | 1024 | 256 | 0.732 | 1.0005 | 0.732 | 1.0004 | 0.732 | — |
| random | 32768 | 8192 | 0.766 | 1.0005 | 0.766 | 1.0004 | 0.766 | — |
| tokenized | 1024 | 256 | 0.612 | 1.0002 | 0.612 | 1.0005 | 0.612 | 0.612 |
| tokenized | 32768 | 8192 | 0.893 | 1.0005 | 0.893 | 1.0006 | 0.893 | 0.893 |
| forward | 1024 | 256 | 0.975 | 1.0005 | 0.975 | 1.0005 | 0.975 | 0.969 |
| forward | 32768 | 8192 | 0.993 | 1.0007 | 0.993 | 1.0006 | 0.993 | **0.994** |

## Indexer Comparison: Oracle vs NSA vs Random (DeepSeek V3.2, s_kv=1024, topk=256)

All cosines vs Dense BF16.

| Data | Oracle | NSA | Random |
|------|--------|-----|--------|
| random | 0.769 | — | 0.993 |
| tokenized | 0.788 | 0.709 | — |
| forward | 0.875 | 0.673 | — |

## Indexer Comparison at Scale (DeepSeek V3.2, s_kv=32768, topk=8192)

| Data | Oracle | NSA |
|------|--------|-----|
| tokenized | 0.850 | **0.865** |
| forward | 0.622 | **0.770** |

NSA outperforms oracle at 32K context where learned importance gating captures
long-range dependencies better than naive head-averaging.
