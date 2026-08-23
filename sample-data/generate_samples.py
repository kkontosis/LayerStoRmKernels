"""
Generate sample activations from real MLA weights for SnapMLA kernel testing.

Three modes:
  1. Random: torch.randn hidden states → weight projections (original)
  2. Tokenized: real text → tokenizer → embedding lookup → weight projections
  3. Forward: tokenized embeddings → transformer layer(s) → weight projections
     (produces most realistic peaked attention patterns)

The tokenized mode produces naturally peaked attention because real token
embeddings have learned structure, unlike Gaussian noise. The forward mode
further improves this by running through actual transformer layers (with
NVFP4 dequant for DeepSeek MLP weights).

Usage:
    python sample-data/generate_samples.py              # random + tokenized
    python sample-data/generate_samples.py --random-only
    python sample-data/generate_samples.py --tok-only
    python sample-data/generate_samples.py --fwd-only   # forward pass mode

Dependencies:
    pip install safetensors transformers tiktoken
"""

import math
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Constants (shared by both models)
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 7168
KV_LORA_RANK = 512         # D_C
QK_ROPE_HEAD_DIM = 64      # D_ROPE
Q_LORA_RANK = 1536
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
TARGET_H_Q = 64            # Kernel's H_Q

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR     # sample-data/
TEXTS_DIR = SCRIPT_DIR / 'texts'

MODELS = {
    'kimi_k2.5': {
        'shard': PROJECT_DIR / 'test-data' / 'Kimi-K2.5-huggingface' / 'model-00003-of-000064.safetensors',
        'prefix': 'language_model.model.layers.2.self_attn',
        'h_q': 64,
        'h_kv': 64,
        'rms_norm_eps': 1e-5,
        'model_dir': PROJECT_DIR / 'test-data' / 'Kimi-K2.5-huggingface',
        'embed_shard': PROJECT_DIR / 'test-data' / 'Kimi-K2.5-huggingface' / 'model-00062-of-000064.safetensors',
        'embed_key': 'language_model.model.embed_tokens.weight',
        # Forward pass: layer 0 only (layer 1 is MoE with 384 experts across shards)
        'fwd_shard': PROJECT_DIR / 'test-data' / 'Kimi-K2.5-huggingface' / 'model-00001-of-000064.safetensors',
        'fwd_layers': [0],
        'layer_prefix': 'language_model.model.layers',
        'has_nvfp4_mlp': False,
    },
    'deepseek_v32': {
        'shard': PROJECT_DIR / 'test-data' / 'DeepSeek-V3.2-NVFP4' / 'model-00001-of-000163.safetensors',
        'prefix': 'model.layers.2.self_attn',
        'h_q': 128,
        'h_kv': 128,
        'rms_norm_eps': 1e-6,
        'model_dir': PROJECT_DIR / 'test-data' / 'DeepSeek-V3.2-NVFP4',
        'embed_shard': PROJECT_DIR / 'test-data' / 'DeepSeek-V3.2-NVFP4' / 'model-00001-of-000163.safetensors',
        'embed_key': 'model.embed_tokens.weight',
        # Forward pass: layers 0-1 (first_k_dense_replace=3, all in shard 1)
        # Attention projections are BF16; o_proj + MLP are NVFP4
        'fwd_shard': PROJECT_DIR / 'test-data' / 'DeepSeek-V3.2-NVFP4' / 'model-00001-of-000163.safetensors',
        'fwd_layers': [0, 1],
        'layer_prefix': 'model.layers',
        'has_nvfp4_mlp': True,
        # NSA indexer: index_topk=2048, index_n_heads=64, index_head_dim=128
        'has_nsa': True,
        'nsa_n_heads': 64,
        'nsa_head_dim': 128,
    },
}

# (s_kv, s_q, suffix)
CONFIGS = [
    (256,   1,   's256'),
    (1024,  1,   's1024'),
    (4096,  1,   's4096'),
    (16384, 1,   's16k'),
    (32768, 1,   's32k'),
    (1024,  128, 'prefill_s1024'),
    (16384, 128, 'prefill_s16k'),
    (32768, 128, 'prefill_s32k'),
]


def rms_norm(x, weight, eps=1e-5):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    x_f = x.float()
    rms = (x_f.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x_f * rms * weight.float())


# ---------------------------------------------------------------------------
# NVFP4 Dequantization
# ---------------------------------------------------------------------------

# FP4 E2M1 lookup table: 4-bit index → float value
# Format: 1 sign bit, 2 exponent bits, 1 mantissa bit (bias=1)
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def nvfp4_dequant_linear(weight_uint8, weight_scale_e4m3, weight_scale_2, x):
    """Dequant NVFP4 weight and apply as linear: x @ dequant(W).T

    Args:
        weight_uint8: [out, in/2] packed FP4 (2 values per byte)
        weight_scale_e4m3: [out, in/16] per-group FP8 E4M3 scales
        weight_scale_2: scalar float32 global scale
        x: [*, in] float32 input
    """
    out_features, packed_in = weight_uint8.shape
    in_features = packed_in * 2

    # Unpack: low nibble = even index, high nibble = odd index
    lo = (weight_uint8 & 0x0F).to(torch.int64)
    hi = (weight_uint8 >> 4).to(torch.int64)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(out_features, in_features)

    # FP4 → float via lookup
    fp4_float = FP4_E2M1_TABLE[unpacked]

    # Per-group scale (group_size=16) * global scale
    scale = weight_scale_e4m3.float().repeat_interleave(16, dim=1)
    w = fp4_float * scale * weight_scale_2.float().item()

    return x.float() @ w.T


def linear_or_nvfp4(w, name, x, has_nvfp4):
    """Apply linear projection — BF16 matmul or NVFP4 dequant+matmul."""
    if has_nvfp4:
        return nvfp4_dequant_linear(
            w[f'{name}_weight'], w[f'{name}_scale'], w[f'{name}_scale_2'], x)
    else:
        return x.float() @ w[name].float().T


def load_attn_weights(cfg):
    """Load layer-2 attention weights and build absorption matrix."""
    shard_path = cfg['shard']
    prefix = cfg['prefix']

    f = safe_open(str(shard_path), framework='pt')
    kv_a_proj = f.get_tensor(f'{prefix}.kv_a_proj_with_mqa.weight').float()
    kv_a_ln   = f.get_tensor(f'{prefix}.kv_a_layernorm.weight').float()
    kv_b_proj = f.get_tensor(f'{prefix}.kv_b_proj.weight').float()
    q_a_proj  = f.get_tensor(f'{prefix}.q_a_proj.weight').float()
    q_a_ln    = f.get_tensor(f'{prefix}.q_a_layernorm.weight').float()
    q_b_proj  = f.get_tensor(f'{prefix}.q_b_proj.weight').float()

    kv_b = kv_b_proj.view(cfg['h_kv'], QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
    kv_b_nope = kv_b[:, :QK_NOPE_HEAD_DIM, :]

    result = dict(kv_a_proj=kv_a_proj, kv_a_ln=kv_a_ln, kv_b_nope=kv_b_nope,
                  q_a_proj=q_a_proj, q_a_ln=q_a_ln, q_b_proj=q_b_proj)

    # NSA indexer weights (DeepSeek V3.2 only — same shard)
    if cfg.get('has_nsa'):
        nsa_prefix = f'{prefix}.indexer'
        result['nsa_wq_b'] = f.get_tensor(f'{nsa_prefix}.wq_b.weight').float()
        result['nsa_wk'] = f.get_tensor(f'{nsa_prefix}.wk.weight').float()
        result['nsa_k_norm_w'] = f.get_tensor(f'{nsa_prefix}.k_norm.weight').float()
        result['nsa_k_norm_b'] = f.get_tensor(f'{nsa_prefix}.k_norm.bias').float()
        result['nsa_weights_proj'] = f.get_tensor(f'{nsa_prefix}.weights_proj.weight').float()
        print(f"  Loaded NSA indexer weights: wq_b{list(result['nsa_wq_b'].shape)}, "
              f"wk{list(result['nsa_wk'].shape)}, weights_proj{list(result['nsa_weights_proj'].shape)}")

    return result


def load_fwd_layer_weights(cfg, layer_idx):
    """Load all weights for a single transformer layer forward pass.

    Returns dict with attention weights (BF16), layernorms, and MLP weights
    (BF16 for Kimi, NVFP4 packed for DeepSeek).
    """
    shard_path = cfg['fwd_shard']
    lp = f"{cfg['layer_prefix']}.{layer_idx}"
    has_nvfp4 = cfg['has_nvfp4_mlp']

    f = safe_open(str(shard_path), framework='pt')

    w = {
        'input_layernorm': f.get_tensor(f'{lp}.input_layernorm.weight'),
        'post_attn_layernorm': f.get_tensor(f'{lp}.post_attention_layernorm.weight'),
        # Attention projections (always BF16 — excluded from NVFP4)
        'kv_a_proj': f.get_tensor(f'{lp}.self_attn.kv_a_proj_with_mqa.weight'),
        'kv_a_ln': f.get_tensor(f'{lp}.self_attn.kv_a_layernorm.weight'),
        'kv_b_proj': f.get_tensor(f'{lp}.self_attn.kv_b_proj.weight'),
        'q_a_proj': f.get_tensor(f'{lp}.self_attn.q_a_proj.weight'),
        'q_a_ln': f.get_tensor(f'{lp}.self_attn.q_a_layernorm.weight'),
        'q_b_proj': f.get_tensor(f'{lp}.self_attn.q_b_proj.weight'),
    }

    # MLP + o_proj: NVFP4 for DeepSeek, BF16 for Kimi
    mlp_projs = [
        ('o_proj', f'{lp}.self_attn.o_proj'),
        ('gate_proj', f'{lp}.mlp.gate_proj'),
        ('up_proj', f'{lp}.mlp.up_proj'),
        ('down_proj', f'{lp}.mlp.down_proj'),
    ]
    for name, key_prefix in mlp_projs:
        if has_nvfp4:
            w[f'{name}_weight'] = f.get_tensor(f'{key_prefix}.weight')
            w[f'{name}_scale'] = f.get_tensor(f'{key_prefix}.weight_scale')
            w[f'{name}_scale_2'] = f.get_tensor(f'{key_prefix}.weight_scale_2')
        else:
            w[name] = f.get_tensor(f'{key_prefix}.weight')

    return w


def run_mla_pipeline(hidden, s_kv, s_q, weights, cfg):
    """Run MLA absorption pipeline on hidden states, return (q, c_kv, k_rope)."""
    eps = cfg['rms_norm_eps']
    h_q = cfg['h_q']
    w = weights

    # KV path
    h_kv_states = hidden[:s_kv]
    kv_comp = h_kv_states @ w['kv_a_proj'].T
    c_kv = rms_norm(kv_comp[:, :KV_LORA_RANK], w['kv_a_ln'], eps)
    k_rope = kv_comp[:, KV_LORA_RANK:]

    # Q path
    h_q_states = hidden[s_kv:s_kv + s_q]
    q_comp = rms_norm(h_q_states @ w['q_a_proj'].T, w['q_a_ln'], eps)
    head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
    q_per_head = (q_comp @ w['q_b_proj'].T).view(s_q, h_q, head_dim)
    q_nope = q_per_head[:, :, :QK_NOPE_HEAD_DIM]
    q_rope = q_per_head[:, :, QK_NOPE_HEAD_DIM:]

    # Absorption
    q_nope_abs = torch.einsum('shd,hdk->shk', q_nope, w['kv_b_nope'])
    q_final = torch.cat([q_nope_abs, q_rope], dim=-1)

    if h_q > TARGET_H_Q:
        q_final = q_final[:, :TARGET_H_Q, :]

    return q_final, c_kv, k_rope


# ---------------------------------------------------------------------------
# Transformer layer forward pass
# ---------------------------------------------------------------------------

def layer_forward_pass(hidden, layer_weights, cfg):
    """Run a single transformer layer: attention + o_proj + MLP.

    Uses full MLA attention (unabsorbed form) with per-head K/V expansion.
    Processes heads one at a time to manage memory for large sequences.
    """
    s = hidden.shape[0]
    h_q = cfg['h_q']
    eps = cfg['rms_norm_eps']
    has_nvfp4 = cfg['has_nvfp4_mlp']
    w = layer_weights

    # 1. Input layernorm
    h = rms_norm(hidden, w['input_layernorm'], eps)

    # 2. MLA Attention
    # KV path (shared across heads)
    kv_comp = h @ w['kv_a_proj'].float().T                              # [s, 576]
    c_kv = rms_norm(kv_comp[:, :KV_LORA_RANK], w['kv_a_ln'].float(), eps)  # [s, 512]
    k_rope = kv_comp[:, KV_LORA_RANK:]                                  # [s, 64]

    # Q path
    q_comp = rms_norm(h @ w['q_a_proj'].float().T, w['q_a_ln'].float(), eps)
    head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM                      # 192
    q = (q_comp @ w['q_b_proj'].float().T).view(s, h_q, head_dim)       # [s, h_q, 192]

    # Expand KV per head via kv_b_proj
    kv_b = w['kv_b_proj'].float().view(h_q, QK_NOPE_HEAD_DIM + V_HEAD_DIM, KV_LORA_RANK)
    kv_b_nope = kv_b[:, :QK_NOPE_HEAD_DIM, :]                          # [h_q, 128, 512]
    kv_b_v = kv_b[:, QK_NOPE_HEAD_DIM:, :]                             # [h_q, 128, 512]

    # Attention per head (memory-efficient: one head at a time)
    sm_scale = 1.0 / math.sqrt(head_dim)
    attn_out = torch.zeros(s, h_q, V_HEAD_DIM, dtype=torch.float32)

    for head_idx in range(h_q):
        k_nope_h = c_kv @ kv_b_nope[head_idx].T                        # [s, 128]
        k_h = torch.cat([k_nope_h, k_rope], dim=-1)                    # [s, 192]
        v_h = c_kv @ kv_b_v[head_idx].T                                # [s, 128]
        q_h = q[:, head_idx, :]                                         # [s, 192]

        # SDPA with causal mask
        out_h = F.scaled_dot_product_attention(
            q_h.unsqueeze(0).unsqueeze(0),
            k_h.unsqueeze(0).unsqueeze(0),
            v_h.unsqueeze(0).unsqueeze(0),
            is_causal=True, scale=sm_scale,
        )
        attn_out[:, head_idx, :] = out_h.squeeze(0).squeeze(0)

        if (head_idx + 1) % 16 == 0:
            print(f"      head {head_idx + 1}/{h_q}", end='\r')
    if h_q > 16:
        print(f"      head {h_q}/{h_q}")

    # 3. O projection + residual
    attn_flat = attn_out.reshape(s, h_q * V_HEAD_DIM)
    hidden = hidden + linear_or_nvfp4(w, 'o_proj', attn_flat, has_nvfp4)

    # 4. Post-attention layernorm + MLP (SwiGLU)
    h = rms_norm(hidden, w['post_attn_layernorm'].float(), eps)
    gate = linear_or_nvfp4(w, 'gate_proj', h, has_nvfp4)
    up = linear_or_nvfp4(w, 'up_proj', h, has_nvfp4)
    hidden = hidden + linear_or_nvfp4(w, 'down_proj', F.silu(gate) * up, has_nvfp4)

    return hidden


def compute_nsa_fields(hidden, s_kv, s_q, weights, cfg):
    """Compute NSA indexer pre-projections. Returns dict or None."""
    if 'nsa_wq_b' not in weights:
        return None
    w = weights
    eps = cfg['rms_norm_eps']
    n_heads = cfg['nsa_n_heads']
    head_dim = cfg['nsa_head_dim']

    h_kv = hidden[:s_kv].float()
    h_q = hidden[s_kv:s_kv + s_q].float()

    # K index: hidden → wk → layer_norm
    k_index = F.layer_norm(
        h_kv @ w['nsa_wk'].T, [head_dim],
        w['nsa_k_norm_w'], w['nsa_k_norm_b'])  # [s_kv, 128]

    # Q index: q_compressed → wq_b (recomputes q_compressed — cheap)
    q_comp = rms_norm(h_q @ w['q_a_proj'].T, w['q_a_ln'], eps)  # [s_q, 1536]
    q_index = (q_comp @ w['nsa_wq_b'].T).reshape(s_q, n_heads, head_dim)  # [s_q, 64, 128]

    # Importance weights: query hidden → weights_proj
    importance = h_q @ w['nsa_weights_proj'].T  # [s_q, 64]

    return {'k_index': k_index, 'q_index': q_index, 'importance': importance}


def save_sample(q, c_kv, k_rope, name, s_kv, s_q, suffix, nsa_fields=None):
    """Save a sample .pt file and print info."""
    out_path = OUTPUT_DIR / f'{name}_{suffix}.pt'
    data = {
        'q': q.bfloat16(),
        'c_kv': c_kv.bfloat16(),
        'k_rope': k_rope.bfloat16(),
        'model': name,
        's_kv': s_kv,
        's_q': s_q,
    }
    if nsa_fields is not None:
        data['k_index'] = nsa_fields['k_index'].bfloat16()
        data['q_index'] = nsa_fields['q_index'].bfloat16()
        data['importance'] = nsa_fields['importance'].bfloat16()
    torch.save(data, str(out_path))
    mb = out_path.stat().st_size / (1024 * 1024)
    nsa_tag = " +NSA" if nsa_fields else ""
    print(f"    {out_path.name}: q{list(q.shape)} "
          f"c_kv{list(c_kv.shape)} k_rope{list(k_rope.shape)} "
          f"({mb:.1f} MB){nsa_tag}")


# ---------------------------------------------------------------------------
# Random mode (original)
# ---------------------------------------------------------------------------

def generate_random(name, cfg, weights):
    """Generate samples from torch.randn hidden states."""
    max_tokens = max(s_kv + s_q for s_kv, s_q, _ in CONFIGS)
    torch.manual_seed(42)
    hidden = torch.randn(max_tokens, HIDDEN_SIZE, dtype=torch.float32)

    for s_kv, s_q, suffix in CONFIGS:
        q, c_kv, k_rope = run_mla_pipeline(hidden, s_kv, s_q, weights, cfg)
        nsa = compute_nsa_fields(hidden, s_kv, s_q, weights, cfg)
        save_sample(q, c_kv, k_rope, name, s_kv, s_q, suffix, nsa_fields=nsa)


# ---------------------------------------------------------------------------
# Tokenized mode
# ---------------------------------------------------------------------------

def load_all_texts():
    """Concatenate all .txt files from sample-data/texts/ into one string."""
    if not TEXTS_DIR.exists():
        return None
    texts = []
    for txt_file in sorted(TEXTS_DIR.glob('*.txt')):
        texts.append(txt_file.read_text(encoding='utf-8', errors='replace'))
    if not texts:
        return None
    return '\n\n'.join(texts)


def tokenize_and_embed(cfg, text, max_tokens):
    """Tokenize text and look up embeddings. Returns [n_tokens, hidden_size]."""
    from transformers import AutoTokenizer

    model_dir = str(cfg['model_dir'])
    print(f"    Loading tokenizer from {Path(model_dir).name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    token_ids = tokenizer.encode(text)
    n_available = len(token_ids)
    print(f"    Tokenized: {n_available} tokens")

    # Repeat if needed to reach max_tokens
    if n_available < max_tokens:
        repeats = (max_tokens // n_available) + 1
        token_ids = (token_ids * repeats)[:max_tokens]
        print(f"    Repeated to {len(token_ids)} tokens")
    else:
        token_ids = token_ids[:max_tokens]

    token_ids_t = torch.tensor(token_ids, dtype=torch.long)

    # Load embedding weight
    embed_path = cfg['embed_shard']
    print(f"    Loading embeddings from {embed_path.name} ...")
    ef = safe_open(str(embed_path), framework='pt')
    embed_weight = ef.get_tensor(cfg['embed_key'])  # [vocab, hidden]

    hidden = embed_weight[token_ids_t].float()  # [n_tokens, hidden_size]
    return hidden


def generate_tokenized(name, cfg, weights):
    """Generate samples from tokenized real text → embedding hidden states."""
    text = load_all_texts()
    if text is None:
        print(f"    SKIP tokenized: no text files in {TEXTS_DIR}")
        return

    max_tokens = max(s_kv + s_q for s_kv, s_q, _ in CONFIGS)
    hidden = tokenize_and_embed(cfg, text, max_tokens)

    for s_kv, s_q, suffix in CONFIGS:
        tok_suffix = f'tok_{suffix}'
        q, c_kv, k_rope = run_mla_pipeline(hidden, s_kv, s_q, weights, cfg)
        nsa = compute_nsa_fields(hidden, s_kv, s_q, weights, cfg)
        save_sample(q, c_kv, k_rope, name, s_kv, s_q, tok_suffix, nsa_fields=nsa)


# ---------------------------------------------------------------------------
# Forward pass mode
# ---------------------------------------------------------------------------

def generate_fwd(name, cfg, weights):
    """Generate samples from tokenized text → transformer layer forward pass.

    Passes token embeddings through dense transformer layers before the
    MLA pipeline on layer 2. This produces more realistic peaked attention
    patterns than raw embeddings.
    """
    text = load_all_texts()
    if text is None:
        print(f"    SKIP fwd: no text files in {TEXTS_DIR}")
        return

    fwd_shard = cfg.get('fwd_shard')
    if fwd_shard is None or not fwd_shard.exists():
        print(f"    SKIP fwd: shard not found")
        return

    max_tokens = max(s_kv + s_q for s_kv, s_q, _ in CONFIGS)
    hidden = tokenize_and_embed(cfg, text, max_tokens)

    # Run forward pass through each dense layer
    for layer_idx in cfg['fwd_layers']:
        print(f"    Loading layer {layer_idx} weights ...")
        layer_w = load_fwd_layer_weights(cfg, layer_idx)
        print(f"    Running layer {layer_idx} forward (s={hidden.shape[0]}) ...")
        hidden = layer_forward_pass(hidden, layer_w, cfg)
        del layer_w  # free memory
        print(f"    Layer {layer_idx} done.")

    # Now run MLA pipeline on the forward-passed hidden states
    for s_kv, s_q, suffix in CONFIGS:
        fwd_suffix = f'fwd_{suffix}'
        q, c_kv, k_rope = run_mla_pipeline(hidden, s_kv, s_q, weights, cfg)
        nsa = compute_nsa_fields(hidden, s_kv, s_q, weights, cfg)
        save_sample(q, c_kv, k_rope, name, s_kv, s_q, fwd_suffix, nsa_fields=nsa)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate SnapMLA sample activations')
    parser.add_argument('--random-only', action='store_true', help='Only generate random samples')
    parser.add_argument('--tok-only', action='store_true', help='Only generate tokenized samples')
    parser.add_argument('--fwd-only', action='store_true', help='Only generate forward-pass samples')
    args = parser.parse_args()

    # Default: run all modes. If a specific flag is set, only that mode.
    exclusive = args.random_only or args.tok_only or args.fwd_only
    do_random = args.random_only if exclusive else True
    do_tok = args.tok_only if exclusive else True
    do_fwd = args.fwd_only if exclusive else False  # fwd is opt-in by default (slow)

    print("Generating SnapMLA sample activations from real model weights")
    print("=" * 60)

    for name, cfg in MODELS.items():
        shard_path = cfg['shard']
        if not shard_path.exists():
            print(f"\n{name}: SKIP — {shard_path} not found")
            continue

        print(f"\n{name}:")
        print(f"  Loading attention weights from {shard_path.name} ...")
        weights = load_attn_weights(cfg)

        if do_random:
            print(f"  Random samples:")
            generate_random(name, cfg, weights)

        if do_tok:
            print(f"  Tokenized samples:")
            if not cfg.get('embed_shard', Path('_')).exists():
                print(f"    SKIP: embedding shard not found")
            else:
                generate_tokenized(name, cfg, weights)

        if do_fwd:
            print(f"  Forward-pass samples:")
            generate_fwd(name, cfg, weights)

    print(f"\nDone. Files saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
