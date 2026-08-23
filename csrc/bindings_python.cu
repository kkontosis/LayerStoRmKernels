// Python bindings for SM120 SnapMLA kernels (pybind11).
//
// All Python wrapper functions live here. They accept torch::Tensor,
// extract pointers, build param structs, and call C++ kernel launch functions.
// C++ users call the kernel functions directly — they never need this file.
//
// This file is #included by bindings.cu (single TU compilation) — it is NOT
// compiled as a standalone translation unit. All kernel headers and symbols
// are already visible from the includes in bindings.cu.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


// ===========================================================================
// Prep Kernels
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_q_quant(torch::Tensor q_bf16, int64_t d_nope) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16, "q_bf16 must be BF16 on CUDA");
    TORCH_CHECK(q_bf16.dim() == 3 && q_bf16.is_contiguous(), "q_bf16 must be contiguous [s_q, h_q, d_qk]");

    int s_q = q_bf16.size(0), h_q = q_bf16.size(1), d_qk = q_bf16.size(2);
    int d_rope = d_qk - d_nope;
    TORCH_CHECK(d_rope > 0, "d_nope must be less than d_qk");

    auto q_nope_fp8 = torch::empty({s_q, h_q, d_nope},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(q_bf16.device()));
    auto q_rope_bf16 = torch::empty({s_q, h_q, d_rope}, q_bf16.options());
    auto q_scales = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));

    sm120::prep::FusedQQuantParams params;
    params.q_bf16 = reinterpret_cast<const __nv_bfloat16*>(q_bf16.data_ptr());
    params.q_nope_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(q_nope_fp8.data_ptr());
    params.q_rope_bf16 = reinterpret_cast<__nv_bfloat16*>(q_rope_bf16.data_ptr());
    params.q_scales = reinterpret_cast<float*>(q_scales.data_ptr());
    params.s_q = s_q; params.h_q = h_q; params.d_qk = d_qk; params.d_nope = d_nope;

    sm120::prep::run_fused_q_quant(params, at::cuda::getCurrentCUDAStream());
    return {q_nope_fp8, q_rope_bf16, q_scales};
}

// Q-absorb: ql_nope = einsum('shd,hdk->shk', q_nope, W_UK); concat q_rope → [s_q, h_q, d_c+d_rope].
// kv_b_proj is the full [h_q*(d_nope_in+d_v), d_c] tensor (BF16) or its FP8 E4M3 form + scales.
// Map a GGUF quant_type string to the QAbsorbParams.gguf_type code
// (0=Q2_K,1=Q3_K,2=Q4_K,3=Q5_K,4=Q6_K,5=Q8_0). Matches GgufType enum order.
static int gguf_type_code(const std::string& t) {
    if (t == "q2_k") return 0;
    if (t == "q3_k") return 1;
    if (t == "q4_k") return 2;
    if (t == "q5_k") return 3;
    if (t == "q6_k") return 4;
    if (t == "q8_0") return 5;
    TORCH_CHECK(false, "unknown gguf quant_type '" + t + "' for q_absorb");
    return -1;
}

torch::Tensor q_absorb(
    torch::Tensor q_heads, torch::Tensor kv_b_proj,
    int64_t d_nope_in, int64_t d_c, int64_t d_rope, int64_t d_v,
    c10::optional<torch::Tensor> w_uk_scales,
    c10::optional<torch::Tensor> seqlens_k,   // [s_q] int32 — enables fused RoPE
    c10::optional<torch::Tensor> cos_sin,     // [max_pos, d_rope] float32 (cos|sin halves)
    c10::optional<std::string> gguf_quant_type  // q2_k..q8_0 — kv_b_proj is packed GGUF
) {
    TORCH_CHECK(q_heads.is_cuda() && q_heads.dtype() == torch::kBFloat16,
                "q_heads must be BF16 on CUDA");
    TORCH_CHECK(q_heads.dim() == 3 && q_heads.is_contiguous(),
                "q_heads must be contiguous [s_q, h_q, d_nope_in+d_rope]");
    TORCH_CHECK(q_heads.size(2) == d_nope_in + d_rope,
                "q_heads last dim must equal d_nope_in + d_rope");
    TORCH_CHECK(kv_b_proj.is_cuda() && kv_b_proj.is_contiguous(), "kv_b_proj must be contiguous CUDA");
    TORCH_CHECK(d_nope_in <= 256, "d_nope_in exceeds shared-mem cache (256)");

    const int s_q = q_heads.size(0), h_q = q_heads.size(1);
    auto q_absorbed = torch::empty({s_q, h_q, d_c + d_rope}, q_heads.options());

    const bool is_gguf = gguf_quant_type.has_value();
    const bool is_fp8 = !is_gguf && (kv_b_proj.dtype() == torch::kFloat8_e4m3fn);
    int gguf_type = -1;
    if (is_gguf) {
        gguf_type = gguf_type_code(*gguf_quant_type);
        TORCH_CHECK(kv_b_proj.dtype() == torch::kUInt8,
                    "GGUF kv_b_proj must be uint8 packed");
        const int qk = (gguf_type == 5) ? 32 : 256;
        TORCH_CHECK(d_c % qk == 0,
                    "GGUF q_absorb requires d_c divisible by " + std::to_string(qk));
    } else if (is_fp8) {
        TORCH_CHECK(w_uk_scales.has_value() && w_uk_scales->is_cuda()
                        && w_uk_scales->dtype() == torch::kFloat32,
                    "FP8 kv_b_proj requires float32 w_uk_scales on CUDA");
    } else {
        TORCH_CHECK(kv_b_proj.dtype() == torch::kBFloat16, "kv_b_proj must be BF16 or FP8 E4M3");
    }

    sm120::prep::QAbsorbParams params;
    params.q_heads = reinterpret_cast<const __nv_bfloat16*>(q_heads.data_ptr());
    params.w_uk = kv_b_proj.data_ptr();
    params.w_uk_scales = (is_fp8 && w_uk_scales.has_value())
        ? reinterpret_cast<const float*>(w_uk_scales->data_ptr()) : nullptr;
    params.q_absorbed = reinterpret_cast<__nv_bfloat16*>(q_absorbed.data_ptr());
    params.s_q = s_q; params.h_q = h_q;
    params.d_nope_in = d_nope_in; params.d_c = d_c; params.d_rope = d_rope; params.d_v = d_v;
    params.weight_is_fp8 = is_fp8;
    params.gguf_type = gguf_type;

    if (seqlens_k.has_value() && cos_sin.has_value()) {
        TORCH_CHECK(seqlens_k->is_cuda() && seqlens_k->dtype() == torch::kInt32
                        && seqlens_k->numel() == s_q,
                    "seqlens_k must be [s_q] int32 on CUDA");
        TORCH_CHECK(cos_sin->is_cuda() && cos_sin->dtype() == torch::kFloat32
                        && cos_sin->dim() == 2 && cos_sin->size(1) == d_rope
                        && cos_sin->is_contiguous(),
                    "cos_sin must be contiguous [max_pos, d_rope] float32 on CUDA");
        params.apply_rope = true;
        params.seqlens_k = reinterpret_cast<const int*>(seqlens_k->data_ptr());
        params.cos_sin = reinterpret_cast<const float*>(cos_sin->data_ptr());
        params.max_pos = cos_sin->size(0);
    }

    sm120::prep::run_q_absorb(params, at::cuda::getCurrentCUDAStream());
    return q_absorbed;
}

// In-place interleaved-pair RoPE on a strided rope slice (see rope_rotate.h).
// x: BF16 view whose LAST dim is d_rope and whose rows are spaced row_stride elements
// apart in the underlying storage; rows_per_token rows share each token's position.
void rope_rotate(
    torch::Tensor x, torch::Tensor seqlens_k, torch::Tensor cos_sin,
    int64_t num_tokens, int64_t rows_per_token, int64_t row_stride
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16, "x must be BF16 on CUDA");
    TORCH_CHECK(seqlens_k.is_cuda() && seqlens_k.dtype() == torch::kInt32
                    && seqlens_k.numel() == num_tokens,
                "seqlens_k must be [num_tokens] int32 on CUDA");
    const int d_rope = x.size(-1);
    TORCH_CHECK(d_rope % 2 == 0, "d_rope must be even");
    TORCH_CHECK(cos_sin.is_cuda() && cos_sin.dtype() == torch::kFloat32
                    && cos_sin.dim() == 2 && cos_sin.size(1) == d_rope
                    && cos_sin.is_contiguous(),
                "cos_sin must be contiguous [max_pos, d_rope] float32 on CUDA");

    sm120::prep::RopeRotateParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.seqlens_k = reinterpret_cast<const int*>(seqlens_k.data_ptr());
    params.cos_sin = reinterpret_cast<const float*>(cos_sin.data_ptr());
    params.num_tokens = num_tokens;
    params.rows_per_token = rows_per_token;
    params.row_stride = row_stride;
    params.d_rope = d_rope;
    params.max_pos = cos_sin.size(0);
    sm120::prep::run_rope_rotate(params, at::cuda::getCurrentCUDAStream());
}

void fused_k_append(
    torch::Tensor c_kv, torch::Tensor k_rope, torch::Tensor kv_cache,
    torch::Tensor slot_mapping, int64_t d_c, int64_t d_rope, int64_t page_size
) {
    TORCH_CHECK(c_kv.is_cuda() && k_rope.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(c_kv.dtype() == torch::kBFloat16 && k_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt32);
    TORCH_CHECK(c_kv.is_contiguous() && k_rope.is_contiguous());

    int row_bytes = d_c + 4 + d_rope * 2;
    sm120::prep::FusedKAppendParams params;
    params.c_kv = reinterpret_cast<const __nv_bfloat16*>(c_kv.data_ptr());
    params.k_rope = reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr());
    params.kv_cache = reinterpret_cast<__nv_fp8_e4m3*>(kv_cache.data_ptr());
    params.cache_stride_block = page_size * row_bytes;
    params.cache_stride_row = row_bytes;
    params.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
    params.num_tokens = c_kv.size(0); params.d_c = d_c; params.d_rope = d_rope; params.page_size = page_size;

    sm120::prep::run_fused_k_append(params, at::cuda::getCurrentCUDAStream());
}

torch::Tensor dequant_ckv_indexed(
    torch::Tensor kv_cache, torch::Tensor indices, int64_t d_c, int64_t d_rope, int64_t page_size
) {
    TORCH_CHECK(kv_cache.is_cuda() && indices.is_cuda() && indices.dtype() == torch::kInt32);

    int num_fetch = indices.size(0), d_qk = d_c + d_rope, row_bytes = d_c + 4 + d_rope * 2;
    auto k_out = torch::empty({num_fetch, d_qk},
        torch::TensorOptions().dtype(torch::kBFloat16).device(kv_cache.device()));

    sm120::prep::DequantCKVIndexedParams params;
    params.kv_cache = reinterpret_cast<const __nv_fp8_e4m3*>(kv_cache.data_ptr());
    params.cache_stride_block = page_size * row_bytes; params.cache_stride_row = row_bytes;
    params.page_size = page_size;
    params.indices = reinterpret_cast<const int*>(indices.data_ptr());
    params.num_fetch = num_fetch;
    params.k_out = reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr());
    params.d_c = d_c; params.d_rope = d_rope;

    sm120::prep::run_dequant_ckv_fused_indexed(params, at::cuda::getCurrentCUDAStream());
    return k_out;
}


// ===========================================================================
// TurboQuant Prep Kernels
// ===========================================================================

void tq_fused_k_append(
    torch::Tensor c_kv, torch::Tensor k_rope, torch::Tensor kv_cache,
    torch::Tensor slot_mapping, torch::Tensor Pi, torch::Tensor centroids,
    torch::Tensor boundaries, int64_t d_c, int64_t d_rope, int64_t page_size
) {
    TORCH_CHECK(c_kv.is_cuda() && k_rope.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(c_kv.dtype() == torch::kBFloat16 && k_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32, "Pi must be float32 on CUDA");
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(boundaries.is_cuda() && boundaries.dtype() == torch::kFloat32);
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt32);

    int num_centroids = centroids.size(0);
    int packed_nope_bytes = d_c / 2;
    int row_bytes = packed_nope_bytes + 2 + d_rope * 2;  // packed + fp16_norm + bf16_rope

    sm120::prep::TqFusedKAppendParams params;
    params.c_kv = reinterpret_cast<const __nv_bfloat16*>(c_kv.data_ptr());
    params.k_rope = reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr());
    params.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
    params.cache_stride_block = page_size * row_bytes;
    params.cache_stride_row = row_bytes;
    params.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
    params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
    params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    // boundaries tensor has [n+1] entries; we pass interior boundaries [1:-1] = n-1 entries
    params.decision_boundaries = reinterpret_cast<const float*>(boundaries.data_ptr()) + 1;
    params.num_tokens = c_kv.size(0);
    params.d_c = d_c; params.d_rope = d_rope; params.page_size = page_size;
    params.num_centroids = num_centroids;

    sm120::prep::run_tq_fused_k_append(params, at::cuda::getCurrentCUDAStream());
}

torch::Tensor tq_dequant_ckv_indexed(
    torch::Tensor kv_cache, torch::Tensor indices, torch::Tensor Pi,
    torch::Tensor centroids, int64_t d_c, int64_t d_rope, int64_t page_size
) {
    TORCH_CHECK(kv_cache.is_cuda() && indices.is_cuda() && indices.dtype() == torch::kInt32);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);

    int num_fetch = indices.size(0);
    int d_qk = d_c + d_rope;
    int packed_nope_bytes = d_c / 2;
    int row_bytes = packed_nope_bytes + 2 + d_rope * 2;

    auto k_out = torch::empty({num_fetch, d_qk},
        torch::TensorOptions().dtype(torch::kBFloat16).device(kv_cache.device()));

    sm120::prep::TqDequantCKVIndexedParams params;
    params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
    params.cache_stride_block = page_size * row_bytes;
    params.cache_stride_row = row_bytes;
    params.page_size = page_size;
    params.indices = reinterpret_cast<const int*>(indices.data_ptr());
    params.num_fetch = num_fetch;
    params.k_out = reinterpret_cast<__nv_bfloat16*>(k_out.data_ptr());
    params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
    params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    params.d_c = d_c; params.d_rope = d_rope;

    sm120::prep::run_tq_dequant_ckv_indexed(params, at::cuda::getCurrentCUDAStream());
    return k_out;
}

torch::Tensor tq_q_rotate(
    torch::Tensor q_nope, torch::Tensor Pi
) {
    TORCH_CHECK(q_nope.is_cuda() && q_nope.dtype() == torch::kBFloat16, "q_nope must be BF16 CUDA");
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(q_nope.is_contiguous());

    // q_nope can be [s_q, h_q, d_c] or [batch_heads, d_c]
    int64_t total_elements = q_nope.numel();
    int d_c = q_nope.size(-1);
    int batch_heads = total_elements / d_c;

    auto q_rot = torch::empty(q_nope.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32).device(q_nope.device()));

    sm120::prep::TqQRotateParams params;
    params.q_nope = reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr());
    params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
    params.q_rot = reinterpret_cast<float*>(q_rot.data_ptr());
    params.batch_heads = batch_heads;
    params.d_c = d_c;

    sm120::prep::run_tq_q_rotate(params, at::cuda::getCurrentCUDAStream());
    return q_rot;
}

torch::Tensor tq_v_rotate_back(
    torch::Tensor out_rotated, torch::Tensor Pi
) {
    TORCH_CHECK(out_rotated.is_cuda() && out_rotated.dtype() == torch::kFloat32);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(out_rotated.is_contiguous());

    int64_t total_elements = out_rotated.numel();
    int d_c = out_rotated.size(-1);
    int batch_heads = total_elements / d_c;

    auto out_final = torch::empty(out_rotated.sizes(),
        torch::TensorOptions().dtype(torch::kBFloat16).device(out_rotated.device()));

    sm120::prep::TqVRotateBackParams params;
    params.out_rotated = reinterpret_cast<const float*>(out_rotated.data_ptr());
    params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
    params.out_final = reinterpret_cast<__nv_bfloat16*>(out_final.data_ptr());
    params.batch_heads = batch_heads;
    params.d_c = d_c;

    sm120::prep::run_tq_v_rotate_back(params, at::cuda::getCurrentCUDAStream());
    return out_final;
}


// ===========================================================================
// V4 Compressor Kernels
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
v4_csa_compress(
    torch::Tensor input_k_nope, torch::Tensor input_k_rope_raw, torch::Tensor input_v,
    torch::Tensor gate_weights, torch::Tensor positional_bias,
    torch::Tensor compress_cos, torch::Tensor compress_sin,
    int64_t head_dim, int64_t qk_rope_head_dim, int64_t window, int64_t stride
) {
    TORCH_CHECK(input_k_nope.is_cuda() && input_k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(input_k_rope_raw.is_cuda() && input_k_rope_raw.dtype() == torch::kBFloat16);
    TORCH_CHECK(input_v.is_cuda() && input_v.dtype() == torch::kBFloat16);
    TORCH_CHECK(gate_weights.is_cuda() && gate_weights.dtype() == torch::kBFloat16);
    TORCH_CHECK(positional_bias.is_cuda() && positional_bias.dtype() == torch::kBFloat16);
    TORCH_CHECK(compress_cos.is_cuda() && compress_cos.dtype() == torch::kFloat32);
    TORCH_CHECK(compress_sin.is_cuda() && compress_sin.dtype() == torch::kFloat32);
    TORCH_CHECK(input_k_nope.is_contiguous() && input_k_rope_raw.is_contiguous() && input_v.is_contiguous());

    int num_tokens = input_k_nope.size(0);
    int num_compressed = (num_tokens >= window) ? (num_tokens - window) / stride : 0;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(input_k_nope.device());
    auto out_k_nope = torch::empty({num_compressed, head_dim}, opts_bf16);
    auto out_k_rope = torch::empty({num_compressed, qk_rope_head_dim}, opts_bf16);
    auto out_v = torch::empty({num_compressed, head_dim}, opts_bf16);

    if (num_compressed > 0) {
        sm120::compress::CsaCompressorParams params;
        params.input_k_nope = reinterpret_cast<const __nv_bfloat16*>(input_k_nope.data_ptr());
        params.input_k_rope_raw = reinterpret_cast<const __nv_bfloat16*>(input_k_rope_raw.data_ptr());
        params.input_v = reinterpret_cast<const __nv_bfloat16*>(input_v.data_ptr());
        params.gate_weights = reinterpret_cast<const __nv_bfloat16*>(gate_weights.data_ptr());
        params.positional_bias = reinterpret_cast<const __nv_bfloat16*>(positional_bias.data_ptr());
        params.compress_cos = reinterpret_cast<const float*>(compress_cos.data_ptr());
        params.compress_sin = reinterpret_cast<const float*>(compress_sin.data_ptr());
        params.cos_sin_stride = qk_rope_head_dim / 2;
        params.out_k_nope = reinterpret_cast<__nv_bfloat16*>(out_k_nope.data_ptr());
        params.out_k_rope = reinterpret_cast<__nv_bfloat16*>(out_k_rope.data_ptr());
        params.out_v = reinterpret_cast<__nv_bfloat16*>(out_v.data_ptr());
        params.num_tokens = num_tokens;
        params.num_compressed = num_compressed;
        params.head_dim = head_dim;
        params.qk_rope_head_dim = qk_rope_head_dim;
        params.window = window;
        params.stride = stride;

        sm120::compress::run_csa_compressor(params, at::cuda::getCurrentCUDAStream());
    }

    return {out_k_nope, out_k_rope, out_v};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
v4_hca_compress(
    torch::Tensor input_k_nope, torch::Tensor input_k_rope_raw, torch::Tensor input_v,
    torch::Tensor gate_weights,
    torch::Tensor compress_cos, torch::Tensor compress_sin,
    int64_t head_dim, int64_t qk_rope_head_dim, int64_t window, int64_t stride
) {
    TORCH_CHECK(input_k_nope.is_cuda() && input_k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(input_k_rope_raw.is_cuda() && input_k_rope_raw.dtype() == torch::kBFloat16);
    TORCH_CHECK(input_v.is_cuda() && input_v.dtype() == torch::kBFloat16);
    TORCH_CHECK(gate_weights.is_cuda() && gate_weights.dtype() == torch::kBFloat16);
    TORCH_CHECK(compress_cos.is_cuda() && compress_cos.dtype() == torch::kFloat32);
    TORCH_CHECK(compress_sin.is_cuda() && compress_sin.dtype() == torch::kFloat32);
    TORCH_CHECK(input_k_nope.is_contiguous() && input_k_rope_raw.is_contiguous() && input_v.is_contiguous());

    int num_tokens = input_k_nope.size(0);
    int num_compressed = num_tokens / stride;

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(input_k_nope.device());
    auto out_k_nope = torch::empty({num_compressed, head_dim}, opts_bf16);
    auto out_k_rope = torch::empty({num_compressed, qk_rope_head_dim}, opts_bf16);
    auto out_v = torch::empty({num_compressed, head_dim}, opts_bf16);

    if (num_compressed > 0) {
        sm120::compress::HcaCompressorParams params;
        params.input_k_nope = reinterpret_cast<const __nv_bfloat16*>(input_k_nope.data_ptr());
        params.input_k_rope_raw = reinterpret_cast<const __nv_bfloat16*>(input_k_rope_raw.data_ptr());
        params.input_v = reinterpret_cast<const __nv_bfloat16*>(input_v.data_ptr());
        params.gate_weights = reinterpret_cast<const __nv_bfloat16*>(gate_weights.data_ptr());
        params.compress_cos = reinterpret_cast<const float*>(compress_cos.data_ptr());
        params.compress_sin = reinterpret_cast<const float*>(compress_sin.data_ptr());
        params.cos_sin_stride = qk_rope_head_dim / 2;
        params.out_k_nope = reinterpret_cast<__nv_bfloat16*>(out_k_nope.data_ptr());
        params.out_k_rope = reinterpret_cast<__nv_bfloat16*>(out_k_rope.data_ptr());
        params.out_v = reinterpret_cast<__nv_bfloat16*>(out_v.data_ptr());
        params.num_tokens = num_tokens;
        params.num_compressed = num_compressed;
        params.head_dim = head_dim;
        params.qk_rope_head_dim = qk_rope_head_dim;
        params.window = window;
        params.stride = stride;

        sm120::compress::run_hca_compressor(params, at::cuda::getCurrentCUDAStream());
    }

    return {out_k_nope, out_k_rope, out_v};
}


// ===========================================================================
// V4 Fused Q Norm + Compressed K RoPE + K Insert
// ===========================================================================

void
v4_fused_q_compress_k(
    torch::Tensor q_bf16,                // [h_q, d_qk] BF16, modified in-place
    torch::Tensor k_nope,                // [1, head_dim] BF16
    torch::Tensor k_rope_raw,            // [1, rope_dim] BF16
    torch::Tensor v_nope,                // [1, head_dim] BF16
    torch::Tensor compress_cos,          // [max_pos, rope_dim/2] FP32
    torch::Tensor compress_sin,          // [max_pos, rope_dim/2] FP32
    torch::Tensor kv_cache,              // uint8 paged cache
    int64_t slot,                        // target cache slot
    int64_t rope_position,               // compressed RoPE position
    double rms_eps
) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_bf16.dim() == 2 && q_bf16.is_contiguous());
    TORCH_CHECK(k_nope.is_cuda() && k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);

    int h_q = q_bf16.size(0), d_qk = q_bf16.size(1);
    int head_dim = k_nope.size(1);
    int rope_dim = k_rope_raw.size(1);

    sm120::prep::FusedQCompressKParams p;
    p.q = reinterpret_cast<__nv_bfloat16*>(q_bf16.data_ptr());
    p.h_q = h_q; p.d_qk = d_qk;
    p.rms_eps = static_cast<float>(rms_eps);
    p.k_nope = reinterpret_cast<const __nv_bfloat16*>(k_nope.data_ptr());
    p.k_rope_raw = reinterpret_cast<const __nv_bfloat16*>(k_rope_raw.data_ptr());
    p.v_nope = reinterpret_cast<const __nv_bfloat16*>(v_nope.data_ptr());
    p.compress_cos = reinterpret_cast<const float*>(compress_cos.data_ptr());
    p.compress_sin = reinterpret_cast<const float*>(compress_sin.data_ptr());
    p.cos_sin_stride = rope_dim / 2;
    p.rope_position = rope_position;
    p.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
    p.slot = slot;
    p.head_dim = head_dim; p.qk_rope_head_dim = rope_dim;
    sm120::prep::run_fused_q_compress_k(p, at::cuda::getCurrentCUDAStream());
}


// ===========================================================================
// V4 Fused Compress + Insert (compressor + RoPE + FP8 quant + cache write)
// Mirrors SnapMLA fused_k_append pattern — writes directly to paged cache.
// ===========================================================================

void
v4_fused_csa_compress_insert(
    torch::Tensor input_k_nope, torch::Tensor input_k_rope_raw, torch::Tensor input_v,
    torch::Tensor gate_weights, torch::Tensor positional_bias,
    torch::Tensor compress_cos, torch::Tensor compress_sin,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t head_dim, int64_t qk_rope_head_dim, int64_t window, int64_t stride
) {
    TORCH_CHECK(input_k_nope.is_cuda() && input_k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt32);

    int num_tokens = input_k_nope.size(0);
    int num_compressed = (num_tokens >= window) ? (num_tokens - window) / stride : 0;

    if (num_compressed > 0) {
        sm120::compress::FusedCompressInsertParams p;
        p.input_k_nope = reinterpret_cast<const __nv_bfloat16*>(input_k_nope.data_ptr());
        p.input_k_rope_raw = reinterpret_cast<const __nv_bfloat16*>(input_k_rope_raw.data_ptr());
        p.input_v = reinterpret_cast<const __nv_bfloat16*>(input_v.data_ptr());
        p.gate_weights = reinterpret_cast<const __nv_bfloat16*>(gate_weights.data_ptr());
        p.positional_bias = reinterpret_cast<const __nv_bfloat16*>(positional_bias.data_ptr());
        p.compress_cos = reinterpret_cast<const float*>(compress_cos.data_ptr());
        p.compress_sin = reinterpret_cast<const float*>(compress_sin.data_ptr());
        p.cos_sin_stride = qk_rope_head_dim / 2;
        p.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
        p.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
        p.num_tokens = num_tokens; p.num_compressed = num_compressed;
        p.head_dim = head_dim; p.qk_rope_head_dim = qk_rope_head_dim;
        p.window = window; p.stride = stride;
        sm120::compress::run_fused_csa_compress_insert(p, at::cuda::getCurrentCUDAStream());
    }
}

void
v4_fused_hca_compress_insert(
    torch::Tensor input_k_nope, torch::Tensor input_k_rope_raw, torch::Tensor input_v,
    torch::Tensor gate_weights,
    torch::Tensor compress_cos, torch::Tensor compress_sin,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t head_dim, int64_t qk_rope_head_dim, int64_t window, int64_t stride
) {
    TORCH_CHECK(input_k_nope.is_cuda() && input_k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt32);

    int num_tokens = input_k_nope.size(0);
    int num_compressed = num_tokens / stride;

    if (num_compressed > 0) {
        sm120::compress::FusedCompressInsertParams p;
        p.input_k_nope = reinterpret_cast<const __nv_bfloat16*>(input_k_nope.data_ptr());
        p.input_k_rope_raw = reinterpret_cast<const __nv_bfloat16*>(input_k_rope_raw.data_ptr());
        p.input_v = reinterpret_cast<const __nv_bfloat16*>(input_v.data_ptr());
        p.gate_weights = reinterpret_cast<const __nv_bfloat16*>(gate_weights.data_ptr());
        p.positional_bias = nullptr;
        p.compress_cos = reinterpret_cast<const float*>(compress_cos.data_ptr());
        p.compress_sin = reinterpret_cast<const float*>(compress_sin.data_ptr());
        p.cos_sin_stride = qk_rope_head_dim / 2;
        p.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
        p.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
        p.num_tokens = num_tokens; p.num_compressed = num_compressed;
        p.head_dim = head_dim; p.qk_rope_head_dim = qk_rope_head_dim;
        p.window = window; p.stride = stride;
        sm120::compress::run_fused_hca_compress_insert(p, at::cuda::getCurrentCUDAStream());
    }
}


// ===========================================================================
// V4 Fused Inverse RoPE + FP8 Quantization
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
v4_fused_inv_rope_fp8(
    torch::Tensor x_bf16,           // [N, head_dim] BF16 attention output
    torch::Tensor cos_table,        // [max_pos, rope_dim/2] FP32
    torch::Tensor sin_table,        // [max_pos, rope_dim/2] FP32
    torch::Tensor positions,        // [N] int32
    int64_t qk_rope_head_dim
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous());
    TORCH_CHECK(cos_table.is_cuda() && cos_table.dtype() == torch::kFloat32);
    TORCH_CHECK(positions.is_cuda() && positions.dtype() == torch::kInt32);

    int N = x_bf16.size(0), hd = x_bf16.size(1);
    auto device = x_bf16.device();

    auto out_fp8 = torch::empty({N, hd},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));
    auto out_scales = torch::empty({N},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    sm120::prep::FusedInvRopeFp8Params p;
    p.x = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    p.cos_table = reinterpret_cast<const float*>(cos_table.data_ptr());
    p.sin_table = reinterpret_cast<const float*>(sin_table.data_ptr());
    p.positions = reinterpret_cast<const int*>(positions.data_ptr());
    p.out_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(out_fp8.data_ptr());
    p.out_scales = reinterpret_cast<float*>(out_scales.data_ptr());
    p.N = N; p.head_dim = hd; p.qk_rope_head_dim = qk_rope_head_dim;
    sm120::prep::run_fused_inv_rope_fp8(p, at::cuda::getCurrentCUDAStream());

    return {out_fp8, out_scales};
}


// ===========================================================================
// V4 Lightning Indexer
// ===========================================================================

torch::Tensor
v4_lightning_score(
    torch::Tensor q_proj, torch::Tensor indexer_k_cache_fp8,
    torch::Tensor k_scales, torch::Tensor score_proj
) {
    TORCH_CHECK(q_proj.is_cuda() && q_proj.dtype() == torch::kBFloat16);
    TORCH_CHECK(indexer_k_cache_fp8.is_cuda() && indexer_k_cache_fp8.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(k_scales.is_cuda() && k_scales.dtype() == torch::kFloat32);
    TORCH_CHECK(score_proj.is_cuda() && score_proj.dtype() == torch::kFloat32);
    TORCH_CHECK(q_proj.is_contiguous() && indexer_k_cache_fp8.is_contiguous());
    TORCH_CHECK(k_scales.is_contiguous() && score_proj.is_contiguous());
    TORCH_CHECK(q_proj.dim() == 2, "q_proj must be [INDEX_N_HEADS, INDEX_HEAD_DIM]");
    TORCH_CHECK(indexer_k_cache_fp8.dim() == 3, "indexer_k_cache must be [num_blocks, INDEX_N_HEADS, INDEX_HEAD_DIM]");
    TORCH_CHECK(score_proj.dim() == 1, "score_proj must be [INDEX_N_HEADS]");

    int index_n_heads = q_proj.size(0);
    int index_head_dim = q_proj.size(1);
    int num_blocks = indexer_k_cache_fp8.size(0);

    TORCH_CHECK(indexer_k_cache_fp8.size(1) == index_n_heads && indexer_k_cache_fp8.size(2) == index_head_dim);
    TORCH_CHECK(k_scales.size(0) == num_blocks, "k_scales must be [num_blocks]");
    TORCH_CHECK(score_proj.size(0) == index_n_heads);

    auto scores_out = torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(q_proj.device()));

    if (num_blocks > 0) {
        sm120::indexer::LightningScoreParams params;
        params.q_proj = reinterpret_cast<const __nv_bfloat16*>(q_proj.data_ptr());
        params.indexer_k_cache = reinterpret_cast<const __nv_fp8_e4m3*>(indexer_k_cache_fp8.data_ptr());
        params.k_scales = reinterpret_cast<const float*>(k_scales.data_ptr());
        params.score_proj = reinterpret_cast<const float*>(score_proj.data_ptr());
        params.scores_out = reinterpret_cast<float*>(scores_out.data_ptr());
        params.num_blocks = num_blocks;
        params.index_n_heads = index_n_heads;
        params.index_head_dim = index_head_dim;

        sm120::indexer::run_lightning_score(params, at::cuda::getCurrentCUDAStream());
    }

    return scores_out;
}


// MQA variant: K cache stores ONE shared key per block [num_blocks, INDEX_HEAD_DIM]
// (DeepSeek-V3.2 / GLM-5.2 indexer wk → single head), broadcast over query heads.
torch::Tensor
v4_lightning_score_mqa(
    torch::Tensor q_proj, torch::Tensor indexer_k_cache_fp8,
    torch::Tensor k_scales, torch::Tensor score_proj
) {
    TORCH_CHECK(q_proj.is_cuda() && q_proj.dtype() == torch::kBFloat16);
    TORCH_CHECK(indexer_k_cache_fp8.is_cuda() && indexer_k_cache_fp8.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(k_scales.is_cuda() && k_scales.dtype() == torch::kFloat32);
    TORCH_CHECK(score_proj.is_cuda() && score_proj.dtype() == torch::kFloat32);
    TORCH_CHECK(q_proj.is_contiguous() && indexer_k_cache_fp8.is_contiguous());
    TORCH_CHECK(k_scales.is_contiguous() && score_proj.is_contiguous());
    TORCH_CHECK(q_proj.dim() == 2, "q_proj must be [INDEX_N_HEADS, INDEX_HEAD_DIM]");
    TORCH_CHECK(indexer_k_cache_fp8.dim() == 2, "MQA indexer_k_cache must be [num_blocks, INDEX_HEAD_DIM]");
    TORCH_CHECK(score_proj.dim() == 1, "score_proj must be [INDEX_N_HEADS]");

    int index_n_heads = q_proj.size(0);
    int index_head_dim = q_proj.size(1);
    int num_blocks = indexer_k_cache_fp8.size(0);

    TORCH_CHECK(indexer_k_cache_fp8.size(1) == index_head_dim, "MQA K cache dim must be INDEX_HEAD_DIM");
    TORCH_CHECK(k_scales.size(0) == num_blocks, "k_scales must be [num_blocks]");
    TORCH_CHECK(score_proj.size(0) == index_n_heads);

    auto scores_out = torch::empty({num_blocks}, torch::TensorOptions().dtype(torch::kFloat32).device(q_proj.device()));

    if (num_blocks > 0) {
        sm120::indexer::LightningScoreMqaParams params;
        params.q_proj = reinterpret_cast<const __nv_bfloat16*>(q_proj.data_ptr());
        params.indexer_k_cache = reinterpret_cast<const __nv_fp8_e4m3*>(indexer_k_cache_fp8.data_ptr());
        params.k_scales = reinterpret_cast<const float*>(k_scales.data_ptr());
        params.score_proj = reinterpret_cast<const float*>(score_proj.data_ptr());
        params.scores_out = reinterpret_cast<float*>(scores_out.data_ptr());
        params.num_blocks = num_blocks;
        params.index_n_heads = index_n_heads;
        params.index_head_dim = index_head_dim;

        sm120::indexer::run_lightning_score_mqa(params, at::cuda::getCurrentCUDAStream());
    }

    return scores_out;
}


std::vector<torch::Tensor>
v4_lightning_topk(
    torch::Tensor scores, torch::Tensor block_endpoints,
    int64_t query_position, int64_t topk
) {
    TORCH_CHECK(scores.is_cuda() && scores.dtype() == torch::kFloat32);
    TORCH_CHECK(block_endpoints.is_cuda() && block_endpoints.dtype() == torch::kInt32);
    TORCH_CHECK(scores.is_contiguous() && block_endpoints.is_contiguous());
    TORCH_CHECK(scores.dim() == 1, "scores must be [num_blocks]");
    TORCH_CHECK(block_endpoints.dim() == 1, "block_endpoints must be [num_blocks]");

    int num_blocks = scores.size(0);
    TORCH_CHECK(block_endpoints.size(0) == num_blocks);
    TORCH_CHECK(topk > 0 && topk <= 2048, "topk must be in [1, 2048]");

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(scores.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(scores.device());

    auto output_indices = torch::full({topk}, -1, opts_i32);
    auto output_scores = torch::full({topk}, -std::numeric_limits<float>::infinity(), opts_f32);
    auto effective_k = torch::zeros({1}, opts_i32);

    if (num_blocks > 0) {
        sm120::indexer::LightningTopkParams params;
        params.scores = reinterpret_cast<const float*>(scores.data_ptr());
        params.block_endpoints = reinterpret_cast<const int*>(block_endpoints.data_ptr());
        params.output_indices = reinterpret_cast<int*>(output_indices.data_ptr());
        params.output_scores = reinterpret_cast<float*>(output_scores.data_ptr());
        params.effective_k_out = reinterpret_cast<int*>(effective_k.data_ptr());
        params.num_blocks = num_blocks;
        params.topk = (int)topk;
        params.query_position = (int)query_position;

        sm120::indexer::run_lightning_topk(params, at::cuda::getCurrentCUDAStream());
    }

    return {output_indices, output_scores, effective_k};
}


// ===========================================================================
// V4 Inverse RoPE
// ===========================================================================

torch::Tensor
v4_inverse_rope(
    torch::Tensor x, torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor positions
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(cos_table.is_cuda() && cos_table.dtype() == torch::kFloat32);
    TORCH_CHECK(sin_table.is_cuda() && sin_table.dtype() == torch::kFloat32);
    TORCH_CHECK(positions.is_cuda() && positions.dtype() == torch::kInt32);
    TORCH_CHECK(x.is_contiguous() && cos_table.is_contiguous());
    TORCH_CHECK(sin_table.is_contiguous() && positions.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "x must be [N, rope_dim]");
    TORCH_CHECK(cos_table.dim() == 2 && sin_table.dim() == 2);
    TORCH_CHECK(positions.dim() == 1);

    int N = x.size(0);
    int rope_dim = x.size(1);
    int half_dim = rope_dim / 2;

    TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
    TORCH_CHECK(positions.size(0) == N);
    TORCH_CHECK(cos_table.size(1) == half_dim && sin_table.size(1) == half_dim);

    auto out = torch::empty_like(x);

    if (N > 0) {
        smxx::InverseRopeParams params;
        params.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
        params.cos_table = reinterpret_cast<const float*>(cos_table.data_ptr());
        params.sin_table = reinterpret_cast<const float*>(sin_table.data_ptr());
        params.positions = reinterpret_cast<const int*>(positions.data_ptr());
        params.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
        params.N = N;
        params.rope_dim = rope_dim;

        smxx::run_inverse_rope(params, at::cuda::getCurrentCUDAStream());
    }

    return out;
}


// ===========================================================================
// V4 mHC (hyper-connection residual streams)
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mhc_pre(
    torch::Tensor residual,   // [num_tokens, hc, hidden] BF16
    torch::Tensor fn,         // [(2+hc)*hc, hc*hidden] FP32
    torch::Tensor hc_scale,   // [3] FP32
    torch::Tensor hc_base,    // [(2+hc)*hc] FP32
    double rms_eps, double hc_eps, double post_mult, int64_t sinkhorn_iters
) {
    TORCH_CHECK(residual.is_cuda() && residual.dtype() == torch::kBFloat16);
    TORCH_CHECK(residual.is_contiguous() && residual.dim() == 3);
    TORCH_CHECK(fn.is_cuda() && fn.dtype() == torch::kFloat32 && fn.is_contiguous());
    TORCH_CHECK(hc_scale.is_cuda() && hc_scale.dtype() == torch::kFloat32);
    TORCH_CHECK(hc_base.is_cuda() && hc_base.dtype() == torch::kFloat32);

    const int num_tokens = residual.size(0);
    const int hc = residual.size(1);
    const int hidden = residual.size(2);
    const int hc_mix = (2 + hc) * hc;
    TORCH_CHECK(hc == 4, "mhc kernels are instantiated for hc_mult=4");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == hc_mix && fn.size(1) == (int64_t)hc * hidden);
    TORCH_CHECK(hc_scale.numel() == 3 && hc_base.numel() == hc_mix);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(residual.device());
    auto post = torch::empty({num_tokens, hc}, opts_f32);
    auto comb = torch::empty({num_tokens, hc, hc}, opts_f32);
    auto x = torch::empty({num_tokens, hidden},
                          torch::TensorOptions().dtype(torch::kBFloat16).device(residual.device()));

    smxx::mhc::MhcPreParams p{};
    p.residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
    p.residual_row_stride = (int64_t)hc * hidden;
    p.fn = reinterpret_cast<const float*>(fn.data_ptr());
    p.hc_scale = reinterpret_cast<const float*>(hc_scale.data_ptr());
    p.hc_base = reinterpret_cast<const float*>(hc_base.data_ptr());
    p.rms_eps = static_cast<float>(rms_eps);
    p.hc_eps = static_cast<float>(hc_eps);
    p.post_mult = static_cast<float>(post_mult);
    p.sinkhorn_iters = static_cast<int>(sinkhorn_iters);
    p.post_out = reinterpret_cast<float*>(post.data_ptr());
    p.comb_out = reinterpret_cast<float*>(comb.data_ptr());
    p.x_out = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    p.x_out_row_stride = hidden;
    p.num_tokens = num_tokens;
    p.hc = hc;
    p.hidden = hidden;

    smxx::mhc::run_mhc_pre(p, at::cuda::getCurrentCUDAStream());
    return {post, comb, x};
}

torch::Tensor
mhc_post(
    torch::Tensor y,          // [num_tokens, hidden] BF16
    torch::Tensor residual,   // [num_tokens, hc, hidden] BF16
    torch::Tensor post,       // [num_tokens, hc] FP32
    torch::Tensor comb        // [num_tokens, hc, hc] FP32 ([src][dst])
) {
    TORCH_CHECK(y.is_cuda() && y.dtype() == torch::kBFloat16 && y.is_contiguous());
    TORCH_CHECK(residual.is_cuda() && residual.dtype() == torch::kBFloat16 &&
                residual.is_contiguous() && residual.dim() == 3);
    TORCH_CHECK(post.is_cuda() && post.dtype() == torch::kFloat32 && post.is_contiguous());
    TORCH_CHECK(comb.is_cuda() && comb.dtype() == torch::kFloat32 && comb.is_contiguous());

    const int num_tokens = residual.size(0);
    const int hc = residual.size(1);
    const int hidden = residual.size(2);
    TORCH_CHECK(hc == 4, "mhc kernels are instantiated for hc_mult=4");
    TORCH_CHECK(y.dim() == 2 && y.size(0) == num_tokens && y.size(1) == hidden);
    TORCH_CHECK(post.numel() == (int64_t)num_tokens * hc);
    TORCH_CHECK(comb.numel() == (int64_t)num_tokens * hc * hc);

    auto out = torch::empty_like(residual);

    smxx::mhc::MhcPostParams p{};
    p.y = reinterpret_cast<const __nv_bfloat16*>(y.data_ptr());
    p.y_row_stride = hidden;
    p.residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
    p.residual_row_stride = (int64_t)hc * hidden;
    p.post = reinterpret_cast<const float*>(post.data_ptr());
    p.comb = reinterpret_cast<const float*>(comb.data_ptr());
    p.residual_out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.residual_out_row_stride = (int64_t)hc * hidden;
    p.num_tokens = num_tokens;
    p.hc = hc;
    p.hidden = hidden;

    smxx::mhc::run_mhc_post(p, at::cuda::getCurrentCUDAStream());
    return out;
}

torch::Tensor
mhc_head(
    torch::Tensor residual,   // [num_tokens, hc, hidden] BF16
    torch::Tensor fn,         // [hc, hc*hidden] FP32
    torch::Tensor hc_scale,   // [1] FP32
    torch::Tensor hc_base,    // [hc] FP32
    double rms_eps, double hc_eps
) {
    TORCH_CHECK(residual.is_cuda() && residual.dtype() == torch::kBFloat16 &&
                residual.is_contiguous() && residual.dim() == 3);
    TORCH_CHECK(fn.is_cuda() && fn.dtype() == torch::kFloat32 && fn.is_contiguous());
    TORCH_CHECK(hc_scale.is_cuda() && hc_scale.dtype() == torch::kFloat32);
    TORCH_CHECK(hc_base.is_cuda() && hc_base.dtype() == torch::kFloat32);

    const int num_tokens = residual.size(0);
    const int hc = residual.size(1);
    const int hidden = residual.size(2);
    TORCH_CHECK(hc == 4, "mhc kernels are instantiated for hc_mult=4");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == hc && fn.size(1) == (int64_t)hc * hidden);
    TORCH_CHECK(hc_scale.numel() == 1 && hc_base.numel() == hc);

    auto x = torch::empty({num_tokens, hidden},
                          torch::TensorOptions().dtype(torch::kBFloat16).device(residual.device()));

    smxx::mhc::MhcHeadParams p{};
    p.residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
    p.residual_row_stride = (int64_t)hc * hidden;
    p.fn = reinterpret_cast<const float*>(fn.data_ptr());
    p.hc_scale = reinterpret_cast<const float*>(hc_scale.data_ptr());
    p.hc_base = reinterpret_cast<const float*>(hc_base.data_ptr());
    p.rms_eps = static_cast<float>(rms_eps);
    p.hc_eps = static_cast<float>(hc_eps);
    p.x_out = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    p.x_out_row_stride = hidden;
    p.num_tokens = num_tokens;
    p.hc = hc;
    p.hidden = hidden;

    smxx::mhc::run_mhc_head(p, at::cuda::getCurrentCUDAStream());
    return x;
}


// ===========================================================================
// V4 FP8 Cache Prep
// ===========================================================================

void
v4_fp8_k_append(
    torch::Tensor k_nope, torch::Tensor k_rope, torch::Tensor v_nope,
    torch::Tensor kv_cache, torch::Tensor slot_mapping
) {
    TORCH_CHECK(k_nope.is_cuda() && k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(k_rope.is_cuda() && k_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(v_nope.is_cuda() && v_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt32);
    TORCH_CHECK(k_nope.is_contiguous() && k_rope.is_contiguous() && v_nope.is_contiguous());
    TORCH_CHECK(kv_cache.is_contiguous() && slot_mapping.is_contiguous());
    TORCH_CHECK(k_nope.dim() == 2 && k_rope.dim() == 2 && v_nope.dim() == 2);

    int num_tokens = k_nope.size(0);
    int head_dim = k_nope.size(1);
    int qk_rope_head_dim = k_rope.size(1);

    TORCH_CHECK(k_rope.size(0) == num_tokens);
    TORCH_CHECK(v_nope.size(0) == num_tokens && v_nope.size(1) == head_dim);
    TORCH_CHECK(slot_mapping.size(0) == num_tokens);
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for vectorized BF16 loads");

    if (num_tokens > 0) {
        sm120::prep::V4Fp8KAppendParams params;
        params.k_nope = reinterpret_cast<const __nv_bfloat16*>(k_nope.data_ptr());
        params.k_rope = reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr());
        params.v_nope = reinterpret_cast<const __nv_bfloat16*>(v_nope.data_ptr());
        params.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
        params.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
        params.num_tokens = num_tokens;
        params.head_dim = head_dim;
        params.qk_rope_head_dim = qk_rope_head_dim;

        sm120::prep::run_v4_fp8_k_append(params, at::cuda::getCurrentCUDAStream());
    }
}

void
v4_tq_k_append(
    torch::Tensor k_nope, torch::Tensor k_rope, torch::Tensor v_nope,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    torch::Tensor Pi, torch::Tensor centroids, torch::Tensor boundaries
) {
    TORCH_CHECK(k_nope.is_cuda() && k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(k_rope.is_cuda() && k_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(v_nope.is_cuda() && v_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt32);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(boundaries.is_cuda() && boundaries.dtype() == torch::kFloat32);
    TORCH_CHECK(k_nope.is_contiguous() && k_rope.is_contiguous() && v_nope.is_contiguous());
    TORCH_CHECK(kv_cache.is_contiguous() && slot_mapping.is_contiguous());
    TORCH_CHECK(Pi.is_contiguous() && centroids.is_contiguous() && boundaries.is_contiguous());
    TORCH_CHECK(k_nope.dim() == 2 && k_rope.dim() == 2 && v_nope.dim() == 2);

    int num_tokens = k_nope.size(0);
    int head_dim = k_nope.size(1);
    int qk_rope_head_dim = k_rope.size(1);
    int num_centroids = centroids.size(0);

    TORCH_CHECK(k_rope.size(0) == num_tokens);
    TORCH_CHECK(v_nope.size(0) == num_tokens && v_nope.size(1) == head_dim);
    TORCH_CHECK(slot_mapping.size(0) == num_tokens);
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even for vectorized BF16 loads");
    TORCH_CHECK(Pi.size(0) == head_dim && Pi.size(1) == head_dim);
    TORCH_CHECK(boundaries.size(0) == num_centroids - 1, "boundaries must be interior only (15)");

    if (num_tokens > 0) {
        sm120::prep::V4TqKAppendParams params;
        params.k_nope = reinterpret_cast<const __nv_bfloat16*>(k_nope.data_ptr());
        params.k_rope = reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr());
        params.v_nope = reinterpret_cast<const __nv_bfloat16*>(v_nope.data_ptr());
        params.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
        params.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
        params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
        params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
        params.decision_boundaries = reinterpret_cast<const float*>(boundaries.data_ptr());
        params.num_tokens = num_tokens;
        params.head_dim = head_dim;
        params.qk_rope_head_dim = qk_rope_head_dim;
        params.num_centroids = num_centroids;

        sm120::prep::run_v4_tq_k_append(params, at::cuda::getCurrentCUDAStream());
    }
}

void
v4_tq_k_append_gemm(
    torch::Tensor k_nope, torch::Tensor k_rope, torch::Tensor v_nope,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    torch::Tensor Pi_bf16, torch::Tensor centroids, torch::Tensor boundaries
) {
    TORCH_CHECK(k_nope.is_cuda() && k_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(k_rope.is_cuda() && k_rope.dtype() == torch::kBFloat16);
    TORCH_CHECK(v_nope.is_cuda() && v_nope.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(slot_mapping.is_cuda() && slot_mapping.dtype() == torch::kInt32);
    TORCH_CHECK(Pi_bf16.is_cuda() && Pi_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(boundaries.is_cuda() && boundaries.dtype() == torch::kFloat32);
    TORCH_CHECK(k_nope.is_contiguous() && k_rope.is_contiguous() && v_nope.is_contiguous());
    TORCH_CHECK(kv_cache.is_contiguous() && slot_mapping.is_contiguous());
    TORCH_CHECK(Pi_bf16.is_contiguous() && centroids.is_contiguous() && boundaries.is_contiguous());
    TORCH_CHECK(k_nope.dim() == 2 && k_rope.dim() == 2 && v_nope.dim() == 2);

    int num_tokens = k_nope.size(0);
    int head_dim = k_nope.size(1);
    int qk_rope_head_dim = k_rope.size(1);
    int num_centroids = centroids.size(0);

    TORCH_CHECK(Pi_bf16.size(0) == head_dim && Pi_bf16.size(1) == head_dim);
    TORCH_CHECK(boundaries.size(0) == num_centroids - 1);

    if (num_tokens == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto bf16_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(k_nope.device());
    auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(k_nope.device());

    // Batch K+V into [2N, D] for single normalize + GEMM
    auto kv_cat = torch::cat({k_nope, v_nope}, /*dim=*/0);  // [2N, D] BF16
    auto kv_unit = torch::empty({2 * num_tokens, head_dim}, bf16_opts);
    auto kv_norms = torch::empty({2 * num_tokens}, f32_opts);

    // 1. Normalize K+V
    {
        sm120::prep::V4TqNormalizeParams params;
        params.src = reinterpret_cast<const __nv_bfloat16*>(kv_cat.data_ptr());
        params.dst_unit = reinterpret_cast<__nv_bfloat16*>(kv_unit.data_ptr());
        params.dst_norms = reinterpret_cast<float*>(kv_norms.data_ptr());
        params.num_vecs = 2 * num_tokens;
        params.dim = head_dim;
        sm120::prep::run_v4_tq_normalize(params, stream);
    }

    // 2. Rotate via cuBLAS BF16 GEMM: [2N, D] @ Pi^T → [2N, D]
    auto kv_rot = torch::mm(kv_unit, Pi_bf16.t());

    // 3. Quantize + pack + write to cache
    {
        sm120::prep::V4TqQuantPackWriteParams params;
        params.k_rot = reinterpret_cast<const __nv_bfloat16*>(kv_rot.data_ptr());
        params.k_norms = reinterpret_cast<const float*>(kv_norms.data_ptr());
        params.v_rot = reinterpret_cast<const __nv_bfloat16*>(kv_rot.data_ptr()) + (int64_t)num_tokens * head_dim;
        params.v_norms = reinterpret_cast<const float*>(kv_norms.data_ptr()) + num_tokens;
        params.k_rope = reinterpret_cast<const __nv_bfloat16*>(k_rope.data_ptr());
        params.kv_cache = reinterpret_cast<uint8_t*>(kv_cache.data_ptr());
        params.slot_mapping = reinterpret_cast<const int*>(slot_mapping.data_ptr());
        params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
        params.decision_boundaries = reinterpret_cast<const float*>(boundaries.data_ptr());
        params.num_tokens = num_tokens;
        params.head_dim = head_dim;
        params.qk_rope_head_dim = qk_rope_head_dim;
        params.num_centroids = num_centroids;
        sm120::prep::run_v4_tq_quant_pack_write(params, stream);
    }
}

std::vector<torch::Tensor>
v4_fp8_dequant_indexed(
    torch::Tensor kv_cache, torch::Tensor indices,
    int64_t head_dim, int64_t qk_rope_head_dim
) {
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(indices.is_cuda() && indices.dtype() == torch::kInt32);
    TORCH_CHECK(kv_cache.is_contiguous() && indices.is_contiguous());
    TORCH_CHECK(indices.dim() == 1);
    TORCH_CHECK(head_dim % 2 == 0);

    int num_fetch = indices.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(kv_cache.device());

    auto k_nope_out = torch::empty({num_fetch, head_dim}, opts);
    auto k_rope_out = torch::empty({num_fetch, qk_rope_head_dim}, opts);
    auto v_nope_out = torch::empty({num_fetch, head_dim}, opts);

    if (num_fetch > 0) {
        sm120::prep::V4Fp8DequantIndexedParams params;
        params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
        params.indices = reinterpret_cast<const int*>(indices.data_ptr());
        params.k_nope_out = reinterpret_cast<__nv_bfloat16*>(k_nope_out.data_ptr());
        params.k_rope_out = reinterpret_cast<__nv_bfloat16*>(k_rope_out.data_ptr());
        params.v_nope_out = reinterpret_cast<__nv_bfloat16*>(v_nope_out.data_ptr());
        params.num_fetch = num_fetch;
        params.head_dim = (int)head_dim;
        params.qk_rope_head_dim = (int)qk_rope_head_dim;

        sm120::prep::run_v4_fp8_dequant_indexed(params, at::cuda::getCurrentCUDAStream());
    }

    return {k_nope_out, k_rope_out, v_nope_out};
}

std::vector<torch::Tensor>
v4_tq_dequant_indexed(
    torch::Tensor kv_cache, torch::Tensor indices,
    torch::Tensor Pi, torch::Tensor centroids,
    int64_t head_dim, int64_t qk_rope_head_dim
) {
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(indices.is_cuda() && indices.dtype() == torch::kInt32);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(kv_cache.is_contiguous() && indices.is_contiguous());
    TORCH_CHECK(Pi.is_contiguous() && centroids.is_contiguous());
    TORCH_CHECK(indices.dim() == 1);
    TORCH_CHECK(head_dim % 2 == 0);
    TORCH_CHECK(Pi.size(0) == head_dim && Pi.size(1) == head_dim);

    int num_fetch = indices.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(kv_cache.device());

    auto k_nope_out = torch::empty({num_fetch, head_dim}, opts);
    auto k_rope_out = torch::empty({num_fetch, qk_rope_head_dim}, opts);
    auto v_nope_out = torch::empty({num_fetch, head_dim}, opts);

    if (num_fetch > 0) {
        sm120::prep::V4TqDequantIndexedParams params;
        params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
        params.indices = reinterpret_cast<const int*>(indices.data_ptr());
        params.k_nope_out = reinterpret_cast<__nv_bfloat16*>(k_nope_out.data_ptr());
        params.k_rope_out = reinterpret_cast<__nv_bfloat16*>(k_rope_out.data_ptr());
        params.v_nope_out = reinterpret_cast<__nv_bfloat16*>(v_nope_out.data_ptr());
        params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
        params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
        params.num_fetch = num_fetch;
        params.head_dim = (int)head_dim;
        params.qk_rope_head_dim = (int)qk_rope_head_dim;

        sm120::prep::run_v4_tq_dequant_indexed(params, at::cuda::getCurrentCUDAStream());
    }

    return {k_nope_out, k_rope_out, v_nope_out};
}


// ===========================================================================
// V4 CSA FP8 Decode — sparse compressed + SWA combine
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
v4_csa_fp8_decode(
    torch::Tensor q_nope_bf16, torch::Tensor q_rope_bf16,
    torch::Tensor compressed_kv, torch::Tensor sparse_indices,
    torch::Tensor swa_kv, torch::Tensor swa_block_table, torch::Tensor swa_seqlens,
    double sm_scale, int64_t topk,
    int64_t compressed_page_block_size, int64_t swa_page_block_size,
    int64_t num_sm_parts
) {
    TORCH_CHECK(q_nope_bf16.is_cuda() && q_nope_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_nope_bf16.dim() == 4, "q_nope_bf16 must be [b, s_q, h_q, head_dim]");
    TORCH_CHECK(sparse_indices.is_cuda() && sparse_indices.dtype() == torch::kInt32);

    int b = q_nope_bf16.size(0), s_q = q_nope_bf16.size(1), h_q = q_nope_bf16.size(2);
    int head_dim = q_nope_bf16.size(3);   // 512
    int rope_dim = q_rope_bf16.size(3);    // 64
    int d_v = head_dim;

    int topk_block_size = 64;
    int compressed_entry_bytes = sm120::decode::csa_fp8::V4CacheLayout::ENTRY_BYTES;

    auto device = q_nope_bf16.device();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // The kernel processes indices in TOPK_BLOCK_SIZE (64) tiles.
    // topk SHOULD be a multiple of 64 for best performance (matches FlashMLA/vLLM convention).
    // If not, we pad with -1 sentinels to avoid OOB reads.
    if (topk % topk_block_size != 0) {
        TORCH_WARN_ONCE(
            "v4_csa_fp8_decode: topk (", topk, ") is not a multiple of TOPK_BLOCK_SIZE (", topk_block_size,
            "). Padding indices to ", ((topk + topk_block_size - 1) / topk_block_size) * topk_block_size,
            ". For best performance, use topk that is a multiple of 64.");
    }
    int padded_topk = ((topk + topk_block_size - 1) / topk_block_size) * topk_block_size;
    torch::Tensor padded_indices;
    if (padded_topk > topk) {
        padded_indices = torch::full({b, s_q, padded_topk}, -1, opts_i32);
        padded_indices.slice(2, 0, topk).copy_(sparse_indices);
    } else {
        padded_indices = sparse_indices.contiguous();
    }

    // Scheduler: split-KV over sparse compressed blocks
    auto sched_meta = torch::zeros({num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
    auto num_splits_tensor = torch::zeros({b + 1}, opts_i32);
    {
        auto topk_seqlens = torch::full({b}, (int)topk, opts_i32);
        GetMlaMetadataParams mp;
        mp.seqlens_k_ptr = reinterpret_cast<int*>(topk_seqlens.data_ptr());
        mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
        mp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        mp.batch_size = b; mp.block_size_n = topk_block_size;
        mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts; mp.topk = topk;
        run_get_mla_metadata_kernel(mp, at::cuda::getCurrentCUDAStream());
    }

    auto out = torch::empty({b, s_q, h_q, d_v}, opts_bf16);
    auto lse = torch::empty({b, s_q, h_q}, opts_f32);
    int num_q_seqs = h_q * s_q;
    auto o_accum = torch::zeros({num_sm_parts * b, num_q_seqs, d_v}, opts_f32);
    auto lse_accum = torch::full({num_sm_parts * b, num_q_seqs}, -INFINITY, opts_f32);

    {
        sm120::decode::csa_fp8::CsaFp8DecodeParams p;
        memset(&p, 0, sizeof(p));
        p.b = b; p.s_q = s_q; p.h_q = h_q;
        p.sm_scale = static_cast<float>(sm_scale);
        p.sm_scale_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;

        p.q_nope = reinterpret_cast<cutlass::bfloat16_t*>(q_nope_bf16.data_ptr());
        p.q_rope = reinterpret_cast<cutlass::bfloat16_t*>(q_rope_bf16.data_ptr());
        p.q_scales = nullptr;  // BF16 Q for now (no FP8 Q quantization)

        p.compressed_kv = reinterpret_cast<const char*>(compressed_kv.data_ptr());
        p.compressed_page_block_size = compressed_page_block_size;
        p.stride_compressed_block = compressed_page_block_size * compressed_entry_bytes;
        p.stride_compressed_row = compressed_entry_bytes;

        p.sparse_indices = reinterpret_cast<const int*>(padded_indices.data_ptr());
        p.topk = topk;
        p.stride_indices_b = s_q * padded_topk; p.stride_indices_s_q = padded_topk;

        p.swa_kv = reinterpret_cast<const char*>(swa_kv.data_ptr());
        p.swa_page_block_size = swa_page_block_size;
        p.stride_swa_block = swa_page_block_size * compressed_entry_bytes;
        p.stride_swa_row = compressed_entry_bytes;
        p.swa_block_table = reinterpret_cast<const int*>(swa_block_table.data_ptr());
        p.swa_block_table_stride = swa_block_table.size(1);
        p.swa_seqlens = reinterpret_cast<const int*>(swa_seqlens.data_ptr());

        p.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
        p.lse = reinterpret_cast<float*>(lse.data_ptr());

        p.stride_q_b = s_q * h_q * head_dim; p.stride_q_s_q = h_q * head_dim; p.stride_q_h_q = head_dim;
        p.stride_o_b = s_q * h_q * d_v; p.stride_o_s_q = h_q * d_v; p.stride_o_h_q = d_v;
        p.stride_lse_b = s_q * h_q; p.stride_lse_s_q = h_q;
        p.stream = at::cuda::getCurrentCUDAStream();

        p.lse_accum = reinterpret_cast<float*>(lse_accum.data_ptr());
        p.o_accum = reinterpret_cast<float*>(o_accum.data_ptr());
        p.stride_lse_accum_split = num_q_seqs; p.stride_lse_accum_s_q = h_q;
        p.stride_o_accum_split = num_q_seqs * d_v; p.stride_o_accum_s_q = h_q * d_v; p.stride_o_accum_h_q = d_v;
        p.tile_scheduler_metadata_ptr = reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(sched_meta.data_ptr());
        p.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        p.num_sm_parts = num_sm_parts;

        if (h_q <= 64) {
            sm120::decode::csa_fp8::run_csa_fp8_decode_kernel<64>(p);
        } else {
            sm120::decode::csa_fp8::run_csa_fp8_decode_kernel<128>(p);
        }
    }

    if (num_sm_parts > 1) {
        MlaCombineParams cp; memset(&cp, 0, sizeof(cp));
        cp.b = b; cp.h_q = h_q; cp.h_k = 1; cp.q_seq_per_hk = h_q * s_q; cp.d_v = d_v;
        cp.o_ptr = out.data_ptr(); cp.softmax_lse_ptr = lse.data_ptr();
        cp.o_batch_stride = s_q * h_q * d_v; cp.o_head_stride = s_q * d_v; cp.o_row_stride = d_v;
        cp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        cp.num_sm_parts = num_sm_parts;
        cp.softmax_lseaccum_ptr = lse_accum.data_ptr(); cp.oaccum_ptr = o_accum.data_ptr();
        run_mla_combine_kernel<cutlass::bfloat16_t>(cp, at::cuda::getCurrentCUDAStream());
    }
    return {out, lse};
}


// ===========================================================================
// V4 SWA-Only Decode — pure sliding window (no compression)
//
// Reuses CSA decode kernel with topk=0 (no compressed blocks).
// Only the SWA attention path runs. For compress_ratios=0 layers + MTP.
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
v4_swa_decode(
    torch::Tensor q_nope_bf16, torch::Tensor q_rope_bf16,
    torch::Tensor swa_kv, torch::Tensor swa_block_table, torch::Tensor swa_seqlens,
    double sm_scale,
    int64_t swa_page_block_size,
    int64_t num_sm_parts
) {
    TORCH_CHECK(q_nope_bf16.is_cuda() && q_nope_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_nope_bf16.dim() == 4, "q_nope_bf16 must be [b, s_q, h_q, head_dim]");

    int b = q_nope_bf16.size(0), s_q = q_nope_bf16.size(1);
    auto device = q_nope_bf16.device();
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);

    // Empty compressed cache (1 page, 0 entries)
    int entry_bytes = sm120::decode::csa_fp8::V4CacheLayout::ENTRY_BYTES;
    auto empty_compressed = torch::zeros({1, swa_page_block_size * entry_bytes}, opts_u8);
    auto empty_indices = torch::zeros({b, s_q, 0}, opts_i32);

    return v4_csa_fp8_decode(
        q_nope_bf16, q_rope_bf16,
        empty_compressed, empty_indices,
        swa_kv, swa_block_table, swa_seqlens,
        sm_scale, /*topk=*/0,
        /*compressed_page_block_size=*/swa_page_block_size,
        swa_page_block_size,
        num_sm_parts
    );
}


// ===========================================================================
// V4 HCA FP8 Decode — dense attention over ALL compressed blocks + SWA
//
// Reuses CSA decode kernel with topk = num_compressed (no sparse selection).
// HCA compressed sequence is short (~7800 entries at 1M context with 128:1),
// so dense attention is fast. Frameworks do the same (topk=all).
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
v4_hca_fp8_decode(
    torch::Tensor q_nope_bf16, torch::Tensor q_rope_bf16,
    torch::Tensor compressed_kv,
    int64_t num_compressed,
    torch::Tensor swa_kv, torch::Tensor swa_block_table, torch::Tensor swa_seqlens,
    double sm_scale,
    int64_t compressed_page_block_size, int64_t swa_page_block_size,
    int64_t num_sm_parts
) {
    TORCH_CHECK(q_nope_bf16.is_cuda() && q_nope_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_nope_bf16.dim() == 4, "q_nope_bf16 must be [b, s_q, h_q, head_dim]");

    int b = q_nope_bf16.size(0), s_q = q_nope_bf16.size(1);

    // Generate sequential indices [0, 1, ..., num_compressed-1] for dense access
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(q_nope_bf16.device());
    auto sparse_indices = torch::arange(num_compressed, opts_i32)
        .unsqueeze(0).unsqueeze(0)
        .expand({b, s_q, num_compressed})
        .contiguous();

    // Delegate to CSA decode with topk=num_compressed
    return v4_csa_fp8_decode(
        q_nope_bf16, q_rope_bf16,
        compressed_kv, sparse_indices,
        swa_kv, swa_block_table, swa_seqlens,
        sm_scale, num_compressed,
        compressed_page_block_size, swa_page_block_size,
        num_sm_parts
    );
}


// ===========================================================================
// V4 CSA FP8 Prefill — non-absorbed dense attention with causal masking
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
v4_csa_fp8_prefill(
    torch::Tensor q_bf16,          // [s_q, h_q, d_qk] BF16
    torch::Tensor k_bf16,          // [s_kv, d_qk] BF16 (single KV head)
    torch::Tensor v_bf16,          // [s_kv, d_v] BF16 (single KV head)
    double sm_scale,
    std::optional<torch::Tensor> causal_seqlens  // [s_q] int32, or nullopt
) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(k_bf16.is_cuda() && k_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(v_bf16.is_cuda() && v_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_bf16.dim() == 3 && q_bf16.is_contiguous(), "q must be [s_q, h_q, d_qk]");
    TORCH_CHECK(k_bf16.dim() == 2 && k_bf16.is_contiguous(), "k must be [s_kv, d_qk]");
    TORCH_CHECK(v_bf16.dim() == 2 && v_bf16.is_contiguous(), "v must be [s_kv, d_v]");

    int s_q = q_bf16.size(0), h_q = q_bf16.size(1), d_qk = q_bf16.size(2);
    int s_kv = k_bf16.size(0), d_v = v_bf16.size(1);
    TORCH_CHECK(k_bf16.size(1) == d_qk, "k d_qk must match q d_qk");
    TORCH_CHECK(v_bf16.size(0) == s_kv, "v s_kv must match k s_kv");

    auto device = q_bf16.device();
    auto out = torch::empty({s_q, h_q, d_v},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto lse = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    sm120::prefill::csa_fp8::CsaFp8PrefillParams p;
    memset(&p, 0, sizeof(p));
    p.s_q = s_q; p.s_kv = s_kv; p.h_q = h_q; p.d_qk = d_qk; p.d_v = d_v;
    p.sm_scale = static_cast<float>(sm_scale);
    p.sm_scale_div_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
    p.q = reinterpret_cast<cutlass::bfloat16_t*>(q_bf16.data_ptr());
    p.k = reinterpret_cast<cutlass::bfloat16_t*>(k_bf16.data_ptr());
    p.v = reinterpret_cast<cutlass::bfloat16_t*>(v_bf16.data_ptr());
    p.causal_seqlens = causal_seqlens.has_value()
        ? reinterpret_cast<const int*>(causal_seqlens->data_ptr()) : nullptr;
    p.stride_q_s_q = h_q * d_qk; p.stride_q_h_q = d_qk;
    p.stride_k_s_kv = d_qk;
    p.stride_v_s_kv = d_v;
    p.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
    p.lse = reinterpret_cast<float*>(lse.data_ptr());
    p.stream = at::cuda::getCurrentCUDAStream();

    sm120::prefill::csa_fp8::run_csa_fp8_prefill_kernel<576>(p);
    return {out, lse};
}


// ===========================================================================
// TurboQuant Decode Kernels
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
tq_dense_decode_v32(
    torch::Tensor q_rot, torch::Tensor q_rope,
    torch::Tensor kv_cache, torch::Tensor block_table,
    torch::Tensor seqlens_k, torch::Tensor centroids,
    double sm_scale, int64_t page_block_size, int64_t num_sm_parts
) {
    TORCH_CHECK(q_rot.is_cuda() && q_rot.dtype() == torch::kFloat32, "q_rot must be float32 CUDA");
    TORCH_CHECK(q_rope.is_cuda() && q_rope.dtype() == torch::kBFloat16, "q_rope must be BF16 CUDA");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(block_table.dtype() == torch::kInt32 && seqlens_k.dtype() == torch::kInt32);
    TORCH_CHECK(q_rot.is_contiguous() && q_rope.is_contiguous());

    int b = q_rot.size(0), s_q = q_rot.size(1), h_q = q_rot.size(2), d_c = q_rot.size(3);
    int d_rope = q_rope.size(3);
    int packed_bytes_per_token = d_c / 2;
    int row_bytes = packed_bytes_per_token + 2 + d_rope * 2;

    auto device = q_rot.device();
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    auto out = torch::zeros({b, s_q, h_q, d_c}, opts_f32);
    auto lse = torch::empty({b, s_q, h_q}, opts_f32);

    sm120::decode::tq_dense::TqDenseDecodeParams params;
    memset(&params, 0, sizeof(params));
    params.b = b; params.s_q = s_q; params.h_q = h_q; params.h_kv = 1;
    params.d_c = d_c; params.d_rope = d_rope;
    params.sm_scale = (float)sm_scale;
    params.q_rot = reinterpret_cast<const float*>(q_rot.data_ptr());
    params.q_rope = reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr());
    params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
    params.cache_stride_block = page_block_size * row_bytes;
    params.cache_stride_row = row_bytes;
    params.block_table = reinterpret_cast<const int*>(block_table.data_ptr());
    params.block_table_batch_stride = block_table.size(1);
    params.page_block_size = page_block_size;
    params.seqlens_k = reinterpret_cast<const int*>(seqlens_k.data_ptr());
    params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    params.out = reinterpret_cast<float*>(out.data_ptr());
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.stride_o_b = s_q * h_q * d_c;
    params.stride_o_s_q = h_q * d_c;
    params.stride_o_h_q = d_c;
    params.stride_lse_b = s_q * h_q;
    params.stride_lse_s_q = h_q;
    params.num_sm_parts = num_sm_parts;
    params.stream = at::cuda::getCurrentCUDAStream();

    if (num_sm_parts > 1) {
        // Split-KV: compute tile scheduling + allocate accumulators
        auto sched_meta = torch::zeros({(int)num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
        auto num_splits_tensor = torch::zeros({b + 1}, opts_i32);
        {
            GetMlaMetadataParams mp;
            mp.seqlens_k_ptr = reinterpret_cast<int*>(seqlens_k.data_ptr());
            mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
            mp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
            mp.batch_size = b; mp.block_size_n = page_block_size;
            mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts; mp.topk = -1;
            run_get_mla_metadata_kernel(mp, at::cuda::getCurrentCUDAStream());
        }

        int num_q_seqs = h_q * s_q;
        auto o_accum = torch::zeros({(int)num_sm_parts * b, num_q_seqs, d_c}, opts_f32);
        auto lse_accum = torch::full({(int)num_sm_parts * b, num_q_seqs}, -INFINITY, opts_f32);

        params.o_accum = reinterpret_cast<float*>(o_accum.data_ptr());
        params.lse_accum = reinterpret_cast<float*>(lse_accum.data_ptr());
        params.stride_o_accum_split = num_q_seqs * d_c;
        params.stride_o_accum_s_q = h_q * d_c;
        params.stride_o_accum_h_q = d_c;
        params.stride_lse_accum_split = num_q_seqs;
        params.stride_lse_accum_s_q = h_q;
        params.tile_scheduler_metadata_ptr = reinterpret_cast<sm120::decode::tq_dense::TqDecodingSchedMeta*>(sched_meta.data_ptr());
        params.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());

        sm120::decode::tq_dense::run_tq_dense_decode(params);

        // mla_combine: merge split partials → FP32 output
        MlaCombineParams cp; memset(&cp, 0, sizeof(cp));
        cp.b = b; cp.h_q = h_q; cp.h_k = 1; cp.q_seq_per_hk = h_q * s_q; cp.d_v = d_c;
        cp.o_ptr = out.data_ptr(); cp.softmax_lse_ptr = lse.data_ptr();
        cp.o_batch_stride = s_q * h_q * d_c; cp.o_head_stride = s_q * d_c; cp.o_row_stride = d_c;
        cp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        cp.num_sm_parts = num_sm_parts;
        cp.softmax_lseaccum_ptr = lse_accum.data_ptr();
        cp.oaccum_ptr = o_accum.data_ptr();
        run_mla_combine_kernel<float>(cp, at::cuda::getCurrentCUDAStream());
    } else {
        // Simple mode: no split-KV
        sm120::decode::tq_dense::run_tq_dense_decode(params);
    }

    return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor>
tq_sparse_decode_v32(
    torch::Tensor q_rot, torch::Tensor q_rope,
    torch::Tensor kv_cache, torch::Tensor indices,
    torch::Tensor centroids,
    double sm_scale, int64_t page_block_size
) {
    TORCH_CHECK(q_rot.is_cuda() && q_rot.dtype() == torch::kFloat32, "q_rot must be float32 CUDA");
    TORCH_CHECK(q_rope.is_cuda() && q_rope.dtype() == torch::kBFloat16, "q_rope must be BF16 CUDA");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(indices.is_cuda() && indices.dtype() == torch::kInt32);
    TORCH_CHECK(q_rot.is_contiguous() && q_rope.is_contiguous() && indices.is_contiguous());

    int b = q_rot.size(0), s_q = q_rot.size(1), h_q = q_rot.size(2), d_c = q_rot.size(3);
    int d_rope = q_rope.size(3);
    int topk = indices.size(-1);  // [b, s_q, topk] or [b, topk]
    int packed_bytes_per_token = d_c / 2;
    int row_bytes = packed_bytes_per_token + 2 + d_rope * 2;

    auto out = torch::zeros({b, s_q, h_q, d_c},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_rot.device()));
    auto lse = torch::empty({b, s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_rot.device()));

    sm120::decode::tq_sparse::TqSparseDecodeParams params;
    memset(&params, 0, sizeof(params));
    params.b = b; params.s_q = s_q; params.h_q = h_q; params.h_kv = 1;
    params.d_c = d_c; params.d_rope = d_rope;
    params.sm_scale = (float)sm_scale;
    params.q_rot = reinterpret_cast<const float*>(q_rot.data_ptr());
    params.q_rope = reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr());
    params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
    params.cache_stride_block = page_block_size * row_bytes;
    params.cache_stride_row = row_bytes;
    params.page_block_size = page_block_size;
    params.indices = reinterpret_cast<const int*>(indices.data_ptr());
    params.topk = topk;
    // indices shape: [b, s_q, topk] — compute strides
    if (indices.dim() == 3) {
        params.stride_indices_b = indices.size(1) * indices.size(2);
        params.stride_indices_s_q = indices.size(2);
    } else {
        // [b, topk] — no s_q dim, all queries share same indices
        params.stride_indices_b = indices.size(1);
        params.stride_indices_s_q = 0;
    }
    params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    params.out = reinterpret_cast<float*>(out.data_ptr());
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.stride_o_b = s_q * h_q * d_c;
    params.stride_o_s_q = h_q * d_c;
    params.stride_o_h_q = d_c;
    params.stride_lse_b = s_q * h_q;
    params.stride_lse_s_q = h_q;
    params.stream = at::cuda::getCurrentCUDAStream();

    sm120::decode::tq_sparse::run_tq_sparse_decode(params);
    return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor>
v4_csa_tq_decode(
    torch::Tensor q_rot, torch::Tensor q_rope,
    torch::Tensor kv_cache, torch::Tensor indices,
    torch::Tensor centroids,
    double sm_scale,
    int64_t num_sm_parts
) {
    TORCH_CHECK(q_rot.is_cuda() && q_rot.dtype() == torch::kFloat32, "q_rot must be float32 CUDA");
    TORCH_CHECK(q_rope.is_cuda() && q_rope.dtype() == torch::kBFloat16, "q_rope must be BF16 CUDA");
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(indices.is_cuda() && indices.dtype() == torch::kInt32);
    TORCH_CHECK(q_rot.is_contiguous() && q_rope.is_contiguous() && indices.is_contiguous());

    int b = q_rot.size(0), s_q = q_rot.size(1), h_q = q_rot.size(2), hd = q_rot.size(3);
    int rd = q_rope.size(3);
    int topk = indices.size(-1);
    int num_q_seqs = s_q * h_q;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q_rot.device());
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(q_rot.device());

    auto out = torch::zeros({b, s_q, h_q, hd}, opts_f32);
    auto lse = torch::empty({b, s_q, h_q}, opts_f32);

    sm120::decode::csa_tq::CsaTqDecodeParams params;
    memset(&params, 0, sizeof(params));
    params.b = b; params.s_q = s_q; params.h_q = h_q;
    params.head_dim = hd; params.qk_rope_head_dim = rd;
    params.sm_scale = (float)sm_scale;
    params.q_rot = reinterpret_cast<const float*>(q_rot.data_ptr());
    params.q_rope = reinterpret_cast<const __nv_bfloat16*>(q_rope.data_ptr());
    params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
    params.indices = reinterpret_cast<const int*>(indices.data_ptr());
    params.topk = topk;
    if (indices.dim() == 3) {
        params.stride_indices_b = indices.size(1) * indices.size(2);
        params.stride_indices_s_q = indices.size(2);
    } else {
        params.stride_indices_b = indices.size(1);
        params.stride_indices_s_q = 0;
    }
    params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    params.out = reinterpret_cast<float*>(out.data_ptr());
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    params.stride_o_b = s_q * h_q * hd;
    params.stride_o_s_q = h_q * hd;
    params.stride_o_h_q = hd;
    params.stride_lse_b = s_q * h_q;
    params.stride_lse_s_q = h_q;
    params.num_sm_parts = (int)num_sm_parts;
    params.stream = at::cuda::getCurrentCUDAStream();

    if (num_sm_parts > 1) {
        constexpr int TileSchedulerMetaDataSize = 8;
        auto sched_meta = torch::zeros({(int)num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
        auto num_splits = torch::zeros({b + 1}, opts_i32);
        {
            auto topk_seqlens = torch::full({b}, (int)topk, opts_i32);
            GetMlaMetadataParams mp;
            mp.seqlens_k_ptr = reinterpret_cast<int*>(topk_seqlens.data_ptr());
            mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
            mp.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
            mp.batch_size = b; mp.block_size_n = 64;
            mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts; mp.topk = topk;
            run_get_mla_metadata_kernel(mp, at::cuda::getCurrentCUDAStream());
        }

        auto o_accum = torch::zeros({(int)num_sm_parts * b, num_q_seqs, hd}, opts_f32);
        auto lse_accum = torch::full({(int)num_sm_parts * b, num_q_seqs}, -INFINITY, opts_f32);

        params.o_accum = reinterpret_cast<float*>(o_accum.data_ptr());
        params.lse_accum = reinterpret_cast<float*>(lse_accum.data_ptr());
        params.stride_o_accum_split = num_q_seqs * hd;
        params.stride_o_accum_s_q = h_q * hd;
        params.stride_o_accum_h_q = hd;
        params.stride_lse_accum_split = num_q_seqs;
        params.stride_lse_accum_s_q = h_q;
        params.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
        params.tile_scheduler_metadata_ptr = sched_meta.data_ptr();

        sm120::decode::csa_tq::run_csa_tq_decode(params);

        // mla_combine: merge split partials → FP32 output
        MlaCombineParams cp; memset(&cp, 0, sizeof(cp));
        cp.b = b; cp.h_q = h_q; cp.h_k = 1; cp.q_seq_per_hk = h_q * s_q; cp.d_v = hd;
        cp.o_ptr = out.data_ptr(); cp.softmax_lse_ptr = lse.data_ptr();
        cp.o_batch_stride = s_q * h_q * hd; cp.o_head_stride = s_q * hd; cp.o_row_stride = hd;
        cp.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
        cp.num_sm_parts = num_sm_parts;
        cp.softmax_lseaccum_ptr = lse_accum.data_ptr();
        cp.oaccum_ptr = o_accum.data_ptr();
        run_mla_combine_kernel<float>(cp, at::cuda::getCurrentCUDAStream());
    } else {
        sm120::decode::csa_tq::run_csa_tq_decode(params);
    }

    return {out, lse};
}


// ===========================================================================
// TurboQuant Prefill (dequant + existing BF16 prefill)
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
tq_dense_prefill_v32(
    torch::Tensor q_bf16, torch::Tensor kv_cache,
    torch::Tensor Pi, torch::Tensor centroids,
    int64_t s_kv, double sm_scale,
    int64_t d_c, int64_t d_rope, int64_t page_size
) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(q_bf16.dim() == 3 && q_bf16.is_contiguous());

    // Step 1: Dequant all s_kv tokens from TQ cache to BF16
    auto indices = torch::arange(s_kv, torch::TensorOptions().dtype(torch::kInt32).device(kv_cache.device()));

    int d_qk = d_c + d_rope;
    int packed_nope_bytes = d_c / 2;
    int row_bytes = packed_nope_bytes + 2 + d_rope * 2;

    auto kv_bf16 = torch::empty({s_kv, d_qk},
        torch::TensorOptions().dtype(torch::kBFloat16).device(kv_cache.device()));

    sm120::prep::TqDequantCKVIndexedParams dq_params;
    dq_params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
    dq_params.cache_stride_block = page_size * row_bytes;
    dq_params.cache_stride_row = row_bytes;
    dq_params.page_size = page_size;
    dq_params.indices = reinterpret_cast<const int*>(indices.data_ptr());
    dq_params.num_fetch = s_kv;
    dq_params.k_out = reinterpret_cast<__nv_bfloat16*>(kv_bf16.data_ptr());
    dq_params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
    dq_params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    dq_params.d_c = d_c; dq_params.d_rope = d_rope;

    sm120::prep::run_tq_dequant_ckv_indexed(dq_params, at::cuda::getCurrentCUDAStream());

    // Step 2: Reshape to [s_kv, 1, d_qk] and call existing dense prefill
    auto kv_3d = kv_bf16.unsqueeze(1);  // [s_kv, 1, d_qk]

    int s_q = q_bf16.size(0), h_q = q_bf16.size(1);
    int d_v = d_c;

    auto out = torch::empty({s_q, h_q, d_v},
        torch::TensorOptions().dtype(torch::kBFloat16).device(q_bf16.device()));
    auto lse = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));
    auto max_logits = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));

    sm120::prefill::dense::head64::DenseAttnFwdParams pf_params;
    memset(&pf_params, 0, sizeof(pf_params));
    pf_params.s_q = s_q; pf_params.s_kv = s_kv; pf_params.h_q = h_q; pf_params.h_kv = 1;
    pf_params.d_qk = d_qk; pf_params.d_v = d_v;
    pf_params.sm_scale = static_cast<float>(sm_scale);
    pf_params.sm_scale_div_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
    pf_params.q = reinterpret_cast<cutlass::bfloat16_t*>(q_bf16.data_ptr());
    pf_params.kv = reinterpret_cast<cutlass::bfloat16_t*>(kv_3d.data_ptr());
    pf_params.attn_sink = nullptr;
    pf_params.stride_q_s_q = h_q * d_qk; pf_params.stride_q_h_q = d_qk;
    pf_params.stride_kv_s_kv = 1 * d_qk; pf_params.stride_kv_h_kv = d_qk;
    pf_params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
    pf_params.max_logits = reinterpret_cast<float*>(max_logits.data_ptr());
    pf_params.lse = reinterpret_cast<float*>(lse.data_ptr());
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    pf_params.num_sm = prop.multiProcessorCount;
    pf_params.stream = at::cuda::getCurrentCUDAStream();

    sm120::prefill::dense::head64::run_dense_fwd_phase1_kernel<576, false>(pf_params);
    return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor>
tq_sparse_prefill_v32(
    torch::Tensor q_bf16, torch::Tensor kv_cache,
    torch::Tensor sparse_indices, torch::Tensor Pi, torch::Tensor centroids,
    double sm_scale, int64_t topk,
    int64_t d_c, int64_t d_rope, int64_t page_size
) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_cache.is_cuda() && kv_cache.dtype() == torch::kUInt8);
    TORCH_CHECK(sparse_indices.is_cuda() && sparse_indices.dtype() == torch::kInt32);
    TORCH_CHECK(Pi.is_cuda() && Pi.dtype() == torch::kFloat32);
    TORCH_CHECK(centroids.is_cuda() && centroids.dtype() == torch::kFloat32);
    TORCH_CHECK(q_bf16.dim() == 3 && q_bf16.is_contiguous());

    int d_qk = d_c + d_rope;
    int packed_nope_bytes = d_c / 2;
    int row_bytes = packed_nope_bytes + 2 + d_rope * 2;
    int s_q = q_bf16.size(0), h_q = q_bf16.size(1);
    int d_v = d_c;

    // Step 1: Dequant only the topk tokens from TQ cache
    // sparse_indices: [s_q, h_kv, topk] or [topk] — flatten to get unique tokens
    // For simplicity, dequant all unique tokens referenced by sparse_indices
    auto flat_indices = sparse_indices.flatten().contiguous();
    // Filter out -1 (invalid) and find unique
    auto valid_mask = flat_indices >= 0;
    auto valid_indices = flat_indices.index({valid_mask});
    auto unique_result = at::_unique2(valid_indices, /*sorted=*/true, /*return_inverse=*/false, /*return_counts=*/false);
    auto unique_indices = std::get<0>(unique_result);
    int num_unique = unique_indices.size(0);

    auto kv_bf16 = torch::empty({num_unique, d_qk},
        torch::TensorOptions().dtype(torch::kBFloat16).device(kv_cache.device()));

    sm120::prep::TqDequantCKVIndexedParams dq_params;
    dq_params.kv_cache = reinterpret_cast<const uint8_t*>(kv_cache.data_ptr());
    dq_params.cache_stride_block = page_size * row_bytes;
    dq_params.cache_stride_row = row_bytes;
    dq_params.page_size = page_size;
    dq_params.indices = reinterpret_cast<const int*>(unique_indices.data_ptr());
    dq_params.num_fetch = num_unique;
    dq_params.k_out = reinterpret_cast<__nv_bfloat16*>(kv_bf16.data_ptr());
    dq_params.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
    dq_params.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
    dq_params.d_c = d_c; dq_params.d_rope = d_rope;

    sm120::prep::run_tq_dequant_ckv_indexed(dq_params, at::cuda::getCurrentCUDAStream());

    // Step 2: Remap sparse_indices from global token IDs to local indices in kv_bf16
    // Build a mapping: unique_indices[i] -> i
    // Since unique_indices is sorted, use searchsorted
    auto remapped_indices = torch::searchsorted(unique_indices, sparse_indices.to(torch::kInt64).clamp_min(0));
    // Re-apply the -1 mask
    remapped_indices = remapped_indices.where(sparse_indices >= 0, torch::full_like(remapped_indices, 0));
    auto remapped_int32 = remapped_indices.to(torch::kInt32).contiguous();

    // Step 3: Call existing sparse prefill
    auto kv_3d = kv_bf16.unsqueeze(1);  // [num_unique, 1, d_qk]

    auto out = torch::empty({s_q, h_q, d_v},
        torch::TensorOptions().dtype(torch::kBFloat16).device(q_bf16.device()));
    auto lse = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));
    auto max_logits = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));

    SparseAttnFwdParams pf_params; memset(&pf_params, 0, sizeof(pf_params));
    pf_params.s_q = s_q; pf_params.s_kv = num_unique; pf_params.h_q = h_q; pf_params.h_kv = 1;
    pf_params.d_qk = d_qk; pf_params.d_v = d_v; pf_params.topk = topk;
    pf_params.sm_scale = static_cast<float>(sm_scale);
    pf_params.sm_scale_div_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
    pf_params.q = reinterpret_cast<cutlass::bfloat16_t*>(q_bf16.data_ptr());
    pf_params.kv = reinterpret_cast<cutlass::bfloat16_t*>(kv_3d.data_ptr());
    pf_params.indices = reinterpret_cast<int*>(remapped_int32.data_ptr());
    pf_params.attn_sink = nullptr; pf_params.topk_length = nullptr;
    pf_params.stride_q_s_q = h_q * d_qk; pf_params.stride_q_h_q = d_qk;
    pf_params.stride_kv_s_kv = 1 * d_qk; pf_params.stride_kv_h_kv = d_qk;
    pf_params.stride_indices_s_q = 1 * topk; pf_params.stride_indices_h_kv = topk;
    pf_params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
    pf_params.max_logits = reinterpret_cast<float*>(max_logits.data_ptr());
    pf_params.lse = reinterpret_cast<float*>(lse.data_ptr());
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    pf_params.num_sm = prop.multiProcessorCount;
    pf_params.stream = at::cuda::getCurrentCUDAStream();

    // DET-REDUCE (TD-SPARSE-PREFILL-DETREDUCE): dispatch on the runtime flag
    // (defaults false = legacy atomic path, byte-identical).
    if (pf_params.deterministic_reduce)
        sm120::prefill::sparse::head64::run_fwd_phase1_kernel<576, true>(pf_params);
    else
        sm120::prefill::sparse::head64::run_fwd_phase1_kernel<576, false>(pf_params);
    return {out, lse};
}


// ===========================================================================
// Split-KV Infrastructure
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
get_mla_metadata(torch::Tensor seqlens_k, int64_t num_sm_parts, int64_t block_size_n, int64_t topk) {
    TORCH_CHECK(seqlens_k.is_cuda() && seqlens_k.dtype() == torch::kInt32);
    int batch_size = seqlens_k.size(0);
    auto device = seqlens_k.device();
    auto sched_meta = torch::zeros({num_sm_parts, TileSchedulerMetaDataSize},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_splits = torch::zeros({batch_size + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    GetMlaMetadataParams params;
    params.seqlens_k_ptr = reinterpret_cast<int*>(seqlens_k.data_ptr());
    params.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
    params.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
    params.batch_size = batch_size; params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = 1; params.num_sm_parts = num_sm_parts; params.topk = topk;

    run_get_mla_metadata_kernel(params, at::cuda::getCurrentCUDAStream());
    return {sched_meta, num_splits};
}

void mla_combine(
    torch::Tensor o_accum, torch::Tensor lse_accum, torch::Tensor num_splits,
    torch::Tensor out, torch::Tensor lse, int64_t num_sm_parts,
    int64_t batch_size, int64_t s_q, int64_t h_q, int64_t d_v
) {
    MlaCombineParams params;
    params.b = batch_size; params.h_q = h_q; params.h_k = 1;
    params.q_seq_per_hk = h_q * s_q; params.d_v = d_v;
    params.o_ptr = out.data_ptr(); params.softmax_lse_ptr = lse.data_ptr();
    params.o_batch_stride = s_q * h_q * d_v; params.o_head_stride = s_q * d_v; params.o_row_stride = d_v;
    params.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
    params.num_sm_parts = num_sm_parts;
    params.softmax_lseaccum_ptr = lse_accum.data_ptr();
    params.oaccum_ptr = o_accum.data_ptr();

    run_mla_combine_kernel<cutlass::bfloat16_t>(params, at::cuda::getCurrentCUDAStream());
}


// ===========================================================================
// Dense Decode — orchestrates metadata + decode + combine
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
dense_decode_v32(
    torch::Tensor q_nope_fp8, torch::Tensor q_rope_bf16, torch::Tensor q_scales,
    torch::Tensor kv_cache, torch::Tensor block_table, torch::Tensor seqlens_k,
    double sm_scale, int64_t page_block_size, int64_t num_sm_parts
) {
    TORCH_CHECK(q_nope_fp8.is_cuda() && q_nope_fp8.dim() == 4);
    int b = q_nope_fp8.size(0), s_q = q_nope_fp8.size(1), h_q = q_nope_fp8.size(2);
    int d_nope = q_nope_fp8.size(3), d_rope = q_rope_bf16.size(3);
    int d_qk = d_nope + d_rope, d_v = d_nope, row_bytes = d_nope + 4 + d_rope * 2;

    auto device = q_nope_fp8.device();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    auto sched_meta = torch::zeros({num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
    auto num_splits_tensor = torch::zeros({b + 1}, opts_i32);
    {
        GetMlaMetadataParams mp; mp.seqlens_k_ptr = reinterpret_cast<int*>(seqlens_k.data_ptr());
        mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
        mp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        mp.batch_size = b; mp.block_size_n = page_block_size;
        mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts; mp.topk = -1;
        run_get_mla_metadata_kernel(mp, at::cuda::getCurrentCUDAStream());
    }

    auto out = torch::empty({b, s_q, h_q, d_v}, opts_bf16);
    auto lse = torch::empty({b, s_q, h_q}, opts_f32);
    int num_q_seqs = h_q * s_q;
    auto o_accum = torch::zeros({num_sm_parts * b, num_q_seqs, d_v}, opts_f32);
    auto lse_accum = torch::full({num_sm_parts * b, num_q_seqs}, -INFINITY, opts_f32);

    {
        sm120::decode::dense_fp8::DenseAttnDecodeParams p;
        memset(&p, 0, sizeof(p));
        p.b = b; p.s_q = s_q; p.h_q = h_q; p.h_kv = 1; p.d_qk = d_qk; p.d_v = d_v; p.d_nope = d_nope;
        p.sm_scale = static_cast<float>(sm_scale);
        p.q_nope = reinterpret_cast<cutlass::bfloat16_t*>(q_nope_fp8.data_ptr());
        p.q_rope = reinterpret_cast<cutlass::bfloat16_t*>(q_rope_bf16.data_ptr());
        p.q_scales = reinterpret_cast<float*>(q_scales.data_ptr());
        p.kv_cache = reinterpret_cast<cutlass::bfloat16_t*>(kv_cache.data_ptr());
        p.stride_kv_block = page_block_size * row_bytes; p.stride_kv_row = row_bytes;
        p.block_table = reinterpret_cast<int*>(block_table.data_ptr());
        p.block_table_batch_stride = block_table.size(1); p.page_block_size = page_block_size;
        p.seqlens_k = reinterpret_cast<int*>(seqlens_k.data_ptr());
        p.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
        p.lse = reinterpret_cast<float*>(lse.data_ptr());
        p.stride_o_b = s_q*h_q*d_v; p.stride_o_s_q = h_q*d_v; p.stride_o_h_q = d_v;
        p.stride_lse_b = s_q*h_q; p.stride_lse_s_q = h_q;
        p.stream = at::cuda::getCurrentCUDAStream();
        p.lse_accum = reinterpret_cast<float*>(lse_accum.data_ptr());
        p.o_accum = reinterpret_cast<float*>(o_accum.data_ptr());
        p.stride_lse_accum_split = num_q_seqs; p.stride_lse_accum_s_q = h_q;
        p.stride_o_accum_split = num_q_seqs*d_v; p.stride_o_accum_s_q = h_q*d_v; p.stride_o_accum_h_q = d_v;
        p.tile_scheduler_metadata_ptr = reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(sched_meta.data_ptr());
        p.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        p.num_sm_parts = num_sm_parts;
        sm120::decode::dense_fp8::run_flash_splitkv_mla_dense_fp8_kernel<sm120::sparse::ModelType::V32, 64, false>(p);
    }

    if (num_sm_parts > 1) {
        MlaCombineParams cp; memset(&cp, 0, sizeof(cp));
        cp.b = b; cp.h_q = h_q; cp.h_k = 1; cp.q_seq_per_hk = h_q*s_q; cp.d_v = d_v;
        cp.o_ptr = out.data_ptr(); cp.softmax_lse_ptr = lse.data_ptr();
        cp.o_batch_stride = s_q*h_q*d_v; cp.o_head_stride = s_q*d_v; cp.o_row_stride = d_v;
        cp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        cp.num_sm_parts = num_sm_parts;
        cp.softmax_lseaccum_ptr = lse_accum.data_ptr(); cp.oaccum_ptr = o_accum.data_ptr();
        run_mla_combine_kernel<cutlass::bfloat16_t>(cp, at::cuda::getCurrentCUDAStream());
    }
    return {out, lse};
}


// ===========================================================================
// Sparse Decode — orchestrates metadata + decode + combine
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
sparse_decode_v32(
    torch::Tensor q_nope_fp8, torch::Tensor q_rope_bf16, torch::Tensor q_scales,
    torch::Tensor kv_cache, torch::Tensor indices, double sm_scale,
    int64_t page_block_size, int64_t topk, int64_t num_sm_parts
) {
    TORCH_CHECK(q_nope_fp8.is_cuda() && q_nope_fp8.dim() == 4);
    TORCH_CHECK(indices.is_cuda() && indices.dtype() == torch::kInt32);
    int b = q_nope_fp8.size(0), s_q = q_nope_fp8.size(1), h_q = q_nope_fp8.size(2);
    int d_nope = q_nope_fp8.size(3), d_rope = q_rope_bf16.size(3);
    int d_qk = d_nope + d_rope, d_v = d_nope, row_bytes = d_nope + 4 + d_rope * 2;
    int topk_block_size = 64, num_blocks = (topk + topk_block_size - 1) / topk_block_size;

    auto device = q_nope_fp8.device();
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

    auto sched_meta = torch::zeros({num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
    auto num_splits_tensor = torch::zeros({b + 1}, opts_i32);
    {
        auto topk_seqlens = torch::full({b}, (int)topk, opts_i32);
        GetMlaMetadataParams mp; mp.seqlens_k_ptr = reinterpret_cast<int*>(topk_seqlens.data_ptr());
        mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
        mp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        mp.batch_size = b; mp.block_size_n = topk_block_size;
        mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts; mp.topk = topk;
        run_get_mla_metadata_kernel(mp, at::cuda::getCurrentCUDAStream());
    }

    auto out = torch::empty({b, s_q, h_q, d_v}, opts_bf16);
    auto lse = torch::empty({b, s_q, h_q}, opts_f32);
    int num_q_seqs = h_q * s_q;
    auto o_accum = torch::zeros({num_sm_parts * b, num_q_seqs, d_v}, opts_f32);
    auto lse_accum = torch::full({num_sm_parts * b, num_q_seqs}, -INFINITY, opts_f32);

    {
        sm120::decode::sparse_fp8::SparseAttnDecodeParams p;
        memset(&p, 0, sizeof(p));
        p.b = b; p.s_q = s_q; p.h_q = h_q; p.h_kv = 1; p.d_qk = d_qk; p.d_v = d_v;
        p.sm_scale = static_cast<float>(sm_scale);
        p.sm_scale_div_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
        p.num_blocks = num_blocks; p.page_block_size = page_block_size; p.topk = topk;
        p.model_type = sm120::sparse::ModelType::V32;
        p.q = reinterpret_cast<cutlass::bfloat16_t*>(q_nope_fp8.data_ptr());
        p.q_rope = reinterpret_cast<cutlass::bfloat16_t*>(q_rope_bf16.data_ptr());
        p.q_scales = reinterpret_cast<float*>(q_scales.data_ptr());
        p.kv = reinterpret_cast<cutlass::bfloat16_t*>(kv_cache.data_ptr());
        p.stride_kv_block = page_block_size * row_bytes; p.stride_kv_row = row_bytes;
        p.indices = reinterpret_cast<int*>(indices.data_ptr());
        p.topk_length = nullptr; p.attn_sink = nullptr;
        p.stride_q_b = s_q*h_q*d_nope; p.stride_q_s_q = h_q*d_nope; p.stride_q_h_q = d_nope;
        p.stride_indices_b = s_q*topk; p.stride_indices_s_q = topk;
        p.stride_lse_b = s_q*h_q; p.stride_lse_s_q = h_q;
        p.stride_o_b = s_q*h_q*d_v; p.stride_o_s_q = h_q*d_v; p.stride_o_h_q = d_v;
        p.extra_num_blocks = 0; p.extra_page_block_size = 0; p.extra_topk = 0;
        p.extra_kv = nullptr; p.extra_indices = nullptr; p.extra_topk_length = nullptr;
        p.stride_extra_kv_block = 0; p.stride_extra_kv_row = 0;
        p.stride_extra_indices_b = 0; p.stride_extra_indices_s_q = 0;
        p.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
        p.lse = reinterpret_cast<float*>(lse.data_ptr());
        p.stream = at::cuda::getCurrentCUDAStream();
        p.lse_accum = reinterpret_cast<float*>(lse_accum.data_ptr());
        p.o_accum = reinterpret_cast<float*>(o_accum.data_ptr());
        p.stride_lse_accum_split = num_q_seqs; p.stride_lse_accum_s_q = h_q;
        p.stride_o_accum_split = num_q_seqs*d_v; p.stride_o_accum_s_q = h_q*d_v; p.stride_o_accum_h_q = d_v;
        p.tile_scheduler_metadata_ptr = reinterpret_cast<sm120::decode::sparse_fp8::DecodingSchedMeta*>(sched_meta.data_ptr());
        p.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        p.num_sm_parts = num_sm_parts;
        sm120::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<sm120::sparse::ModelType::V32, 64>(p);
    }

    if (num_sm_parts > 1) {
        MlaCombineParams cp; memset(&cp, 0, sizeof(cp));
        cp.b = b; cp.h_q = h_q; cp.h_k = 1; cp.q_seq_per_hk = h_q*s_q; cp.d_v = d_v;
        cp.o_ptr = out.data_ptr(); cp.softmax_lse_ptr = lse.data_ptr();
        cp.o_batch_stride = s_q*h_q*d_v; cp.o_head_stride = s_q*d_v; cp.o_row_stride = d_v;
        cp.num_splits_ptr = reinterpret_cast<int*>(num_splits_tensor.data_ptr());
        cp.num_sm_parts = num_sm_parts;
        cp.softmax_lseaccum_ptr = lse_accum.data_ptr(); cp.oaccum_ptr = o_accum.data_ptr();
        run_mla_combine_kernel<cutlass::bfloat16_t>(cp, at::cuda::getCurrentCUDAStream());
    }
    return {out, lse};
}


// ===========================================================================
// Prefill Kernels
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
sparse_prefill_v32(
    torch::Tensor q_bf16, torch::Tensor kv_bf16, torch::Tensor indices,
    double sm_scale, int64_t topk
) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_bf16.is_cuda() && kv_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.is_cuda() && indices.dtype() == torch::kInt32);
    TORCH_CHECK(q_bf16.dim() == 3 && kv_bf16.dim() == 3 && indices.dim() == 3);
    TORCH_CHECK(q_bf16.is_contiguous() && kv_bf16.is_contiguous() && indices.is_contiguous());

    int s_q = q_bf16.size(0), h_q = q_bf16.size(1), d_qk = q_bf16.size(2);
    int s_kv = kv_bf16.size(0), h_kv = kv_bf16.size(1), d_v = d_qk - 64;

    auto out = torch::empty({s_q, h_q, d_v},
        torch::TensorOptions().dtype(torch::kBFloat16).device(q_bf16.device()));
    auto lse = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));
    auto max_logits = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));

    SparseAttnFwdParams params; memset(&params, 0, sizeof(params));
    params.s_q = s_q; params.s_kv = s_kv; params.h_q = h_q; params.h_kv = h_kv;
    params.d_qk = d_qk; params.d_v = d_v; params.topk = topk;
    params.sm_scale = static_cast<float>(sm_scale);
    params.sm_scale_div_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
    params.q = reinterpret_cast<cutlass::bfloat16_t*>(q_bf16.data_ptr());
    params.kv = reinterpret_cast<cutlass::bfloat16_t*>(kv_bf16.data_ptr());
    params.indices = reinterpret_cast<int*>(indices.data_ptr());
    params.attn_sink = nullptr; params.topk_length = nullptr;
    params.stride_q_s_q = h_q*d_qk; params.stride_q_h_q = d_qk;
    params.stride_kv_s_kv = h_kv*d_qk; params.stride_kv_h_kv = d_qk;
    params.stride_indices_s_q = h_kv*topk; params.stride_indices_h_kv = topk;
    params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
    params.max_logits = reinterpret_cast<float*>(max_logits.data_ptr());
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    params.num_sm = prop.multiProcessorCount;
    params.stream = at::cuda::getCurrentCUDAStream();

    // DET-REDUCE (TD-SPARSE-PREFILL-DETREDUCE): dispatch on the runtime flag
    // (defaults false = legacy atomic path, byte-identical).
    if (params.deterministic_reduce)
        sm120::prefill::sparse::head64::run_fwd_phase1_kernel<576, true>(params);
    else
        sm120::prefill::sparse::head64::run_fwd_phase1_kernel<576, false>(params);
    return {out, lse};
}

std::tuple<torch::Tensor, torch::Tensor>
dense_prefill_v32(torch::Tensor q_bf16, torch::Tensor kv_bf16, double sm_scale) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(kv_bf16.is_cuda() && kv_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(q_bf16.dim() == 3 && kv_bf16.dim() == 3);
    TORCH_CHECK(q_bf16.is_contiguous() && kv_bf16.is_contiguous());

    int s_q = q_bf16.size(0), h_q = q_bf16.size(1), d_qk = q_bf16.size(2);
    int s_kv = kv_bf16.size(0), h_kv = kv_bf16.size(1), d_v = d_qk - 64;

    auto out = torch::empty({s_q, h_q, d_v},
        torch::TensorOptions().dtype(torch::kBFloat16).device(q_bf16.device()));
    auto lse = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));
    auto max_logits = torch::empty({s_q, h_q},
        torch::TensorOptions().dtype(torch::kFloat32).device(q_bf16.device()));

    sm120::prefill::dense::head64::DenseAttnFwdParams params; memset(&params, 0, sizeof(params));
    params.s_q = s_q; params.s_kv = s_kv; params.h_q = h_q; params.h_kv = h_kv;
    params.d_qk = d_qk; params.d_v = d_v;
    params.sm_scale = static_cast<float>(sm_scale);
    params.sm_scale_div_log2 = static_cast<float>(sm_scale) * 1.4426950408889634f;
    params.q = reinterpret_cast<cutlass::bfloat16_t*>(q_bf16.data_ptr());
    params.kv = reinterpret_cast<cutlass::bfloat16_t*>(kv_bf16.data_ptr());
    params.attn_sink = nullptr;
    params.stride_q_s_q = h_q*d_qk; params.stride_q_h_q = d_qk;
    params.stride_kv_s_kv = h_kv*d_qk; params.stride_kv_h_kv = d_qk;
    params.out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr());
    params.max_logits = reinterpret_cast<float*>(max_logits.data_ptr());
    params.lse = reinterpret_cast<float*>(lse.data_ptr());
    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    params.num_sm = prop.multiProcessorCount;
    params.stream = at::cuda::getCurrentCUDAStream();

    sm120::prefill::dense::head64::run_dense_fwd_phase1_kernel<576, false>(params);
    return {out, lse};
}


// ===========================================================================
// DCP LSE Correction — corrects attention output across DCP ranks
// ===========================================================================

std::tuple<torch::Tensor, torch::Tensor>
dcp_lse_correct(
    torch::Tensor output,      // [B, H, D] BF16 — this rank's partial attention output (modified in-place)
    torch::Tensor lses,        // [N, B, H] FP32 — all-gathered LSE from all DCP ranks
    int64_t rank               // this rank's index in the DCP group
) {
    TORCH_CHECK(output.is_cuda() && output.dtype() == torch::kBFloat16, "output must be BF16 on CUDA");
    TORCH_CHECK(lses.is_cuda() && lses.dtype() == torch::kFloat32, "lses must be FP32 on CUDA");
    TORCH_CHECK(output.dim() == 3, "output must be [B, H, D]");
    TORCH_CHECK(lses.dim() == 3, "lses must be [N, B, H]");
    TORCH_CHECK(output.is_contiguous() && lses.is_contiguous());

    int B = output.size(0), H = output.size(1), D = output.size(2);
    int N = lses.size(0);
    TORCH_CHECK(rank >= 0 && rank < N, "rank must be in [0, N)");

    auto global_lse = torch::empty({B, H},
        torch::TensorOptions().dtype(torch::kFloat32).device(output.device()));

    DcpLseCorrectParams params;
    params.output = output.data_ptr();
    params.lses = reinterpret_cast<const float*>(lses.data_ptr());
    params.global_lse = reinterpret_cast<float*>(global_lse.data_ptr());
    params.B = B; params.H = H; params.D = D; params.N = N; params.rank = rank;
    params.stride_o_B = H * D; params.stride_o_H = D; params.stride_o_D = 1;
    params.stride_lse_N = B * H; params.stride_lse_B = H; params.stride_lse_H = 1;

    run_dcp_lse_correct_kernel(params, at::cuda::getCurrentCUDAStream());
    return {output, global_lse};
}


// ===========================================================================
// CUDA Graph Runner — Python wrapper
// ===========================================================================

class PyDecodeGraphRunner {
public:
    void init(
        torch::Tensor kv_cache, int64_t batch_size, int64_t s_q, int64_t h_q, int64_t h_kv,
        int64_t d_qk, int64_t d_v, int64_t d_nope, int64_t page_block_size,
        int64_t max_num_blocks_per_seq, double sm_scale, int64_t num_sm_parts,
        bool sparse, int64_t topk, int64_t extra_topk
    ) {
        int row_bytes = d_nope + 4 + (d_qk - d_nope) * 2;
        sm120::graph::DecodeGraphConfig cfg{};
        cfg.batch_size = batch_size; cfg.s_q = s_q; cfg.h_q = h_q; cfg.h_kv = h_kv;
        cfg.d_qk = d_qk; cfg.d_v = d_v; cfg.d_nope = d_nope;
        cfg.page_block_size = page_block_size; cfg.max_num_blocks_per_seq = max_num_blocks_per_seq;
        cfg.kv_stride_block = page_block_size * row_bytes; cfg.kv_stride_row = row_bytes;
        cfg.kv_cache = kv_cache.data_ptr();
        cfg.sm_scale = static_cast<float>(sm_scale);
        cfg.model_type = sm120::sparse::ModelType::V32;
        cfg.num_sm_parts = num_sm_parts;
        cfg.sparse = sparse; cfg.topk = topk; cfg.extra_topk = extra_topk;
        cfg_ = cfg;
        runner_.init(cfg, at::cuda::getCurrentCUDAStream());
    }

    void update_metadata(torch::Tensor seqlens_k, int64_t num_sm_parts) {
        auto stream = at::cuda::getCurrentCUDAStream();
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(seqlens_k.device());
        auto sched_meta = torch::zeros({(int)num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
        auto num_splits = torch::zeros({(int)cfg_.batch_size + 1}, opts_i32);
        GetMlaMetadataParams mp;
        mp.seqlens_k_ptr = reinterpret_cast<int*>(seqlens_k.data_ptr());
        mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
        mp.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
        mp.batch_size = cfg_.batch_size; mp.block_size_n = cfg_.page_block_size;
        mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts;
        mp.topk = cfg_.sparse ? cfg_.topk : -1;
        run_get_mla_metadata_kernel(mp, stream);
        cudaMemcpyAsync(runner_.sched_meta_ptr(), sched_meta.data_ptr(),
            num_sm_parts * TileSchedulerMetaDataSize * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(runner_.num_splits_ptr(), num_splits.data_ptr(),
            (cfg_.batch_size + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }

    void update(torch::Tensor q_bf16, torch::Tensor seqlens_k, torch::Tensor block_table) {
        runner_.update(q_bf16.data_ptr(), reinterpret_cast<const int*>(seqlens_k.data_ptr()),
            reinterpret_cast<const int*>(block_table.data_ptr()), nullptr, at::cuda::getCurrentCUDAStream());
    }

    void update_with_indices(torch::Tensor q_bf16, torch::Tensor seqlens_k,
                             torch::Tensor block_table, torch::Tensor indices) {
        runner_.update(q_bf16.data_ptr(), reinterpret_cast<const int*>(seqlens_k.data_ptr()),
            reinterpret_cast<const int*>(block_table.data_ptr()),
            reinterpret_cast<const int*>(indices.data_ptr()), at::cuda::getCurrentCUDAStream());
    }

    void replay() { runner_.replay(at::cuda::getCurrentCUDAStream()); }

    std::tuple<torch::Tensor, torch::Tensor> get_output(torch::Tensor ref_tensor) {
        auto stream = at::cuda::getCurrentCUDAStream();
        int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q, dv = cfg_.d_v;
        auto out = torch::empty({b, sq, hq, dv},
            torch::TensorOptions().dtype(torch::kBFloat16).device(ref_tensor.device()));
        auto lse = torch::empty({b, sq, hq},
            torch::TensorOptions().dtype(torch::kFloat32).device(ref_tensor.device()));
        cudaMemcpyAsync(out.data_ptr(), runner_.out_ptr(),
            b*sq*hq*dv*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(lse.data_ptr(), runner_.lse_ptr(),
            b*sq*hq*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        return {out, lse};
    }

    void destroy() { runner_.destroy(); }

private:
    sm120::graph::DecodeGraphRunner runner_;
    sm120::graph::DecodeGraphConfig cfg_{};
};


class PyTqDecodeGraphRunner {
public:
    void init(
        torch::Tensor kv_cache, torch::Tensor Pi, torch::Tensor centroids,
        int64_t batch_size, int64_t s_q, int64_t h_q,
        int64_t d_c, int64_t d_rope, int64_t page_block_size,
        int64_t max_num_blocks_per_seq, double sm_scale, int64_t num_sm_parts
    ) {
        sm120::graph::TqDecodeGraphConfig cfg{};
        cfg.batch_size = batch_size; cfg.s_q = s_q; cfg.h_q = h_q;
        cfg.d_c = d_c; cfg.d_rope = d_rope;
        cfg.page_block_size = page_block_size;
        cfg.max_num_blocks_per_seq = max_num_blocks_per_seq;
        cfg.sm_scale = static_cast<float>(sm_scale);
        cfg.num_sm_parts = num_sm_parts;
        cfg.kv_cache = kv_cache.data_ptr();
        cfg.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
        cfg.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
        cfg_ = cfg;
        runner_.init(cfg, at::cuda::getCurrentCUDAStream());
    }

    void update_metadata(torch::Tensor seqlens_k, int64_t num_sm_parts) {
        auto stream = at::cuda::getCurrentCUDAStream();
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(seqlens_k.device());
        auto sched_meta = torch::zeros({(int)num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
        auto num_splits = torch::zeros({(int)cfg_.batch_size + 1}, opts_i32);
        GetMlaMetadataParams mp;
        mp.seqlens_k_ptr = reinterpret_cast<int*>(seqlens_k.data_ptr());
        mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
        mp.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
        mp.batch_size = cfg_.batch_size; mp.block_size_n = cfg_.page_block_size;
        mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts; mp.topk = -1;
        run_get_mla_metadata_kernel(mp, stream);
        // Copy scheduling data to graph's fixed buffers
        // Note: the graph runner stores sched_meta and num_splits internally
        // We need to copy them via the runner's buffer pointers
        // For now, we update by memcpy into the runner's pre-allocated buffers
        // The graph reads from these stable addresses
    }

    void update(torch::Tensor q_nope_bf16, torch::Tensor q_rope_bf16,
                torch::Tensor seqlens_k, torch::Tensor block_table) {
        runner_.update(q_nope_bf16.data_ptr(), q_rope_bf16.data_ptr(),
            reinterpret_cast<const int*>(seqlens_k.data_ptr()),
            reinterpret_cast<const int*>(block_table.data_ptr()),
            at::cuda::getCurrentCUDAStream());
    }

    void replay() { runner_.replay(at::cuda::getCurrentCUDAStream()); }

    std::tuple<torch::Tensor, torch::Tensor> get_output(torch::Tensor ref_tensor) {
        auto stream = at::cuda::getCurrentCUDAStream();
        int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q, dc = cfg_.d_c;
        auto out = torch::empty({b, sq, hq, dc},
            torch::TensorOptions().dtype(torch::kBFloat16).device(ref_tensor.device()));
        auto lse = torch::empty({b, sq, hq},
            torch::TensorOptions().dtype(torch::kFloat32).device(ref_tensor.device()));
        cudaMemcpyAsync(out.data_ptr(), runner_.out_ptr(),
            b*sq*hq*dc*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(lse.data_ptr(), runner_.lse_ptr(),
            b*sq*hq*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        return {out, lse};
    }

    void destroy() { runner_.destroy(); }

private:
    sm120::graph::TqDecodeGraphRunner runner_;
    sm120::graph::TqDecodeGraphConfig cfg_{};
};


// ===========================================================================
// V4 CSA FP8 Decode Graph Runner (mirrors SnapMLA DecodeGraphRunner API)
// ===========================================================================

class PyCsaFp8DecodeGraphRunner {
public:
    void init(
        torch::Tensor compressed_kv, torch::Tensor swa_kv,
        int64_t batch_size, int64_t s_q, int64_t h_q, int64_t topk,
        double sm_scale, int64_t num_sm_parts,
        int64_t compressed_page_block_size, int64_t swa_page_block_size,
        int64_t max_swa_blocks
    ) {
        TORCH_CHECK(topk % 64 == 0,
            "CsaFp8DecodeGraphRunner: topk (", topk, ") must be a multiple of 64 "
            "(TOPK_BLOCK_SIZE). Graph runners do not pad internally.");
        sm120::graph::CsaFp8DecodeGraphConfig cfg{};
        cfg.batch_size = batch_size; cfg.s_q = s_q; cfg.h_q = h_q;
        cfg.topk = topk; cfg.sm_scale = static_cast<float>(sm_scale);
        cfg.num_sm_parts = num_sm_parts;
        cfg.compressed_kv = compressed_kv.data_ptr();
        cfg.compressed_page_block_size = compressed_page_block_size;
        cfg.swa_kv = swa_kv.data_ptr();
        cfg.swa_page_block_size = swa_page_block_size;
        cfg.max_swa_blocks = max_swa_blocks;
        cfg_ = cfg;
        runner_.init(cfg, at::cuda::getCurrentCUDAStream());
    }

    void update_metadata(torch::Tensor topk_seqlens, int64_t num_sm_parts) {
        auto stream = at::cuda::getCurrentCUDAStream();
        auto device = topk_seqlens.device();
        auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
        auto sched_meta = torch::zeros({(int)num_sm_parts, TileSchedulerMetaDataSize}, opts_i32);
        auto num_splits = torch::zeros({(int)cfg_.batch_size + 1}, opts_i32);
        GetMlaMetadataParams mp;
        mp.seqlens_k_ptr = reinterpret_cast<int*>(topk_seqlens.data_ptr());
        mp.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(sched_meta.data_ptr());
        mp.num_splits_ptr = reinterpret_cast<int*>(num_splits.data_ptr());
        mp.batch_size = cfg_.batch_size; mp.block_size_n = 64;
        mp.fixed_overhead_num_blocks = 1; mp.num_sm_parts = num_sm_parts;
        mp.topk = cfg_.topk;
        run_get_mla_metadata_kernel(mp, stream);
        cudaMemcpyAsync(runner_.sched_meta_ptr(), sched_meta.data_ptr(),
            num_sm_parts * TileSchedulerMetaDataSize * sizeof(int),
            cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(runner_.num_splits_ptr(), num_splits.data_ptr(),
            (cfg_.batch_size + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }

    void update(torch::Tensor q_nope, torch::Tensor q_rope,
                torch::Tensor sparse_indices,
                torch::Tensor swa_block_table, torch::Tensor swa_seqlens) {
        runner_.update(q_nope.data_ptr(), q_rope.data_ptr(),
            reinterpret_cast<const int*>(sparse_indices.data_ptr()),
            reinterpret_cast<const int*>(swa_block_table.data_ptr()),
            reinterpret_cast<const int*>(swa_seqlens.data_ptr()),
            at::cuda::getCurrentCUDAStream());
    }

    void replay() { runner_.replay(at::cuda::getCurrentCUDAStream()); }

    std::tuple<torch::Tensor, torch::Tensor> get_output(torch::Tensor ref) {
        auto stream = at::cuda::getCurrentCUDAStream();
        int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q;
        auto out = torch::empty({b, sq, hq, 512},
            torch::TensorOptions().dtype(torch::kBFloat16).device(ref.device()));
        auto lse = torch::empty({b, sq, hq},
            torch::TensorOptions().dtype(torch::kFloat32).device(ref.device()));
        cudaMemcpyAsync(out.data_ptr(), runner_.out_ptr(),
            b*sq*hq*512*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(lse.data_ptr(), runner_.lse_ptr(),
            b*sq*hq*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        return {out, lse};
    }

    void destroy() { runner_.destroy(); }

private:
    sm120::graph::CsaFp8DecodeGraphRunner runner_;
    sm120::graph::CsaFp8DecodeGraphConfig cfg_{};
};


// ===========================================================================
// V4 CSA TQ Decode Graph Runner
// ===========================================================================

class PyCsaTqDecodeGraphRunner {
public:
    void init(
        torch::Tensor kv_cache, torch::Tensor Pi, torch::Tensor centroids,
        int64_t batch_size, int64_t s_q, int64_t h_q,
        int64_t topk, double sm_scale
    ) {
        sm120::graph::CsaTqDecodeGraphConfig cfg{};
        cfg.batch_size = batch_size; cfg.s_q = s_q; cfg.h_q = h_q;
        cfg.head_dim = 512; cfg.qk_rope_head_dim = 64;
        cfg.topk = topk; cfg.sm_scale = static_cast<float>(sm_scale);
        cfg.kv_cache = kv_cache.data_ptr();
        cfg.Pi = reinterpret_cast<const float*>(Pi.data_ptr());
        cfg.centroids = reinterpret_cast<const float*>(centroids.data_ptr());
        cfg_ = cfg;
        runner_.init(cfg, at::cuda::getCurrentCUDAStream());
    }

    void update(torch::Tensor q_nope_bf16, torch::Tensor q_rope_bf16,
                torch::Tensor sparse_indices) {
        runner_.update(q_nope_bf16.data_ptr(), q_rope_bf16.data_ptr(),
            reinterpret_cast<const int*>(sparse_indices.data_ptr()),
            at::cuda::getCurrentCUDAStream());
    }

    void replay() { runner_.replay(at::cuda::getCurrentCUDAStream()); }

    std::tuple<torch::Tensor, torch::Tensor> get_output(torch::Tensor ref) {
        auto stream = at::cuda::getCurrentCUDAStream();
        int b = cfg_.batch_size, sq = cfg_.s_q, hq = cfg_.h_q, hd = cfg_.head_dim;
        auto out = torch::empty({b, sq, hq, hd},
            torch::TensorOptions().dtype(torch::kBFloat16).device(ref.device()));
        auto lse = torch::empty({b, sq, hq},
            torch::TensorOptions().dtype(torch::kFloat32).device(ref.device()));
        cudaMemcpyAsync(out.data_ptr(), runner_.out_ptr(),
            b*sq*hq*hd*sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(lse.data_ptr(), runner_.lse_ptr(),
            b*sq*hq*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        return {out, lse};
    }

    void destroy() { runner_.destroy(); }

private:
    sm120::graph::CsaTqDecodeGraphRunner runner_;
    sm120::graph::CsaTqDecodeGraphConfig cfg_{};
};


// ===========================================================================
// NVFP4 Attention Weight GEMM
// ===========================================================================

// Preprocess weight scales: reformat [N, K/16] float32 → Sm1xx interleaved UE4M3.
// Call once per weight matrix; pass result to nvfp4_gemm() to skip per-call reformatting.
torch::Tensor nvfp4_gemm_preprocess(
    torch::Tensor weight_scales,
    int64_t N,
    int64_t K
) {
    TORCH_CHECK(weight_scales.is_cuda(), "weight_scales must be on CUDA");
    TORCH_CHECK(weight_scales.dim() == 2 && weight_scales.is_contiguous(),
        "weight_scales must be contiguous [N, K/16]");
    TORCH_CHECK(weight_scales.size(0) == N && weight_scales.size(1) == K / 16,
        "weight_scales shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto weight_scales_f32 = weight_scales.to(torch::kFloat32);

    // Use M=1 for SF buffer query — SFB layout depends on N,K only
    size_t wt_sf_size = layerstorm::compute::query_sf_buffer_size_b(1, N, K);
    auto wt_scales_buf = torch::zeros({static_cast<int64_t>(wt_sf_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight_scales.device()));

    layerstorm::compute::ReformatScalesParams reformat_params;
    reformat_params.src_scales = reinterpret_cast<const float*>(weight_scales_f32.data_ptr());
    reformat_params.dst_scales = reinterpret_cast<uint8_t*>(wt_scales_buf.data_ptr());
    reformat_params.rows = N;
    reformat_params.groups_per_row = K / 16;
    reformat_params.M = 1;
    reformat_params.N = N;
    reformat_params.K = K;
    reformat_params.is_scale_a = false;
    layerstorm::compute::launch_reformat_scales(reformat_params, stream);

    return wt_scales_buf;
}

torch::Tensor nvfp4_gemm(
    torch::Tensor x_bf16,
    torch::Tensor weight_packed_uint8,
    torch::Tensor weight_scales,
    double weight_scale_global
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_packed_uint8.is_cuda() && weight_packed_uint8.dtype() == torch::kUInt8,
        "weight_packed_uint8 must be uint8 on CUDA");
    TORCH_CHECK(weight_packed_uint8.dim() == 2 && weight_packed_uint8.is_contiguous(),
        "weight_packed_uint8 must be contiguous [N, K/2]");
    TORCH_CHECK(weight_scales.is_cuda(),
        "weight_scales must be on CUDA");

    int M = x_bf16.size(0);
    int K = x_bf16.size(1);
    int N = weight_packed_uint8.size(0);
    TORCH_CHECK(weight_packed_uint8.size(1) == K / 2,
        "weight shape mismatch: expected [N, K/2]");
    TORCH_CHECK(K % 32 == 0, "K must be divisible by 32, got K=" + std::to_string(K));
    TORCH_CHECK(N % 32 == 0, "N must be divisible by 32, got N=" + std::to_string(N));

    auto stream = at::cuda::getCurrentCUDAStream();

    // Detect if weight_scales are pre-reformatted (1D uint8 from nvfp4_gemm_preprocess)
    bool scales_preprocessed = (weight_scales.dtype() == torch::kUInt8 && weight_scales.dim() == 1);

    if (!scales_preprocessed) {
        TORCH_CHECK(weight_scales.dim() == 2 && weight_scales.is_contiguous(),
            "weight_scales must be contiguous [N, K/16] or preprocessed 1D uint8");
        TORCH_CHECK(weight_scales.size(0) == N && weight_scales.size(1) == K / 16,
            "weight_scales shape mismatch: expected [N, K/16]");
    }

    // Compute buffer sizes
    size_t act_packed_bytes = static_cast<size_t>(M) * (K / 2);
    size_t act_sf_size = layerstorm::compute::query_sf_buffer_size_a(M, N, K);
    size_t wt_sf_size = scales_preprocessed ? 0 : layerstorm::compute::query_sf_buffer_size_b(M, N, K);
    size_t ws_size = layerstorm::compute::query_nvfp4_gemm_workspace_size(M, N, K,
        layerstorm::compute::GemmOutputDtype::kBFloat16);

    // Single allocation for all temp buffers (reduces malloc overhead from 4→1)
    size_t total_temp = act_packed_bytes + act_sf_size + wt_sf_size + ws_size;
    auto temp_buf = torch::empty({static_cast<int64_t>(total_temp)},
        torch::TensorOptions().dtype(torch::kUInt8).device(x_bf16.device()));
    uint8_t* temp_ptr = reinterpret_cast<uint8_t*>(temp_buf.data_ptr());

    uint8_t* act_packed_ptr  = temp_ptr;
    uint8_t* act_scales_ptr  = temp_ptr + act_packed_bytes;
    uint8_t* wt_scales_ptr   = scales_preprocessed
        ? reinterpret_cast<uint8_t*>(weight_scales.data_ptr())
        : temp_ptr + act_packed_bytes + act_sf_size;
    void*    workspace_ptr   = (ws_size > 0)
        ? temp_ptr + act_packed_bytes + act_sf_size + wt_sf_size
        : nullptr;

    // Zero the activation scale buffer (Sm1xx layout requires zero-init for padding)
    cudaMemsetAsync(act_scales_ptr, 0, act_sf_size, stream);

    // Step 1: Quantize BF16 activation → NVFP4
    layerstorm::compute::Bf16ToNvfp4Params quant_params;
    quant_params.input = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    quant_params.output_packed = act_packed_ptr;
    quant_params.output_scales = act_scales_ptr;
    quant_params.M = M;
    quant_params.N = N;
    quant_params.K = K;
    layerstorm::compute::launch_bf16_to_nvfp4(quant_params, stream);

    // Step 2: Reformat weight scales (skip if preprocessed)
    if (!scales_preprocessed) {
        auto weight_scales_f32 = weight_scales.to(torch::kFloat32);
        layerstorm::compute::ReformatScalesParams reformat_params;
        reformat_params.src_scales = reinterpret_cast<const float*>(weight_scales_f32.data_ptr());
        reformat_params.dst_scales = wt_scales_ptr;
        reformat_params.rows = N;
        reformat_params.groups_per_row = K / 16;
        reformat_params.M = M;
        reformat_params.N = N;
        reformat_params.K = K;
        reformat_params.is_scale_a = false;
        layerstorm::compute::launch_reformat_scales(reformat_params, stream);
    }

    // Step 3: Allocate output
    auto output = torch::empty({M, N}, x_bf16.options());

    // Step 4: Launch GEMM
    layerstorm::compute::Nvfp4GemmParams gemm_params;
    gemm_params.M = M;
    gemm_params.N = N;
    gemm_params.K = K;
    gemm_params.A = act_packed_ptr;
    gemm_params.B = weight_packed_uint8.data_ptr();
    gemm_params.D = output.data_ptr();
    gemm_params.scale_A = act_scales_ptr;
    gemm_params.scale_B = wt_scales_ptr;
    gemm_params.alpha = static_cast<float>(weight_scale_global);
    gemm_params.output_dtype = layerstorm::compute::GemmOutputDtype::kBFloat16;

    layerstorm::compute::launch_nvfp4_gemm(gemm_params, workspace_ptr, stream);

    return output;
}


// ===========================================================================
// Q4_K Dequant-GEMM
// ===========================================================================

torch::Tensor q4k_dequant_gemm(
    torch::Tensor x_bf16,
    torch::Tensor weight_q4k_packed
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_q4k_packed.is_cuda() && weight_q4k_packed.dtype() == torch::kUInt8,
        "weight_q4k_packed must be uint8 on CUDA");
    TORCH_CHECK(weight_q4k_packed.dim() == 2 && weight_q4k_packed.is_contiguous(),
        "weight_q4k_packed must be contiguous [N, K*9/16]");

    int M = x_bf16.size(0);
    int K = x_bf16.size(1);
    int N = weight_q4k_packed.size(0);
    int expected_row_bytes = (K / 256) * 144;  // K/256 blocks × 144 bytes each

    TORCH_CHECK(K % 256 == 0, "K must be divisible by 256, got K=" + std::to_string(K));
    TORCH_CHECK(weight_q4k_packed.size(1) == expected_row_bytes,
        "weight row bytes mismatch: expected " + std::to_string(expected_row_bytes) +
        ", got " + std::to_string((int)weight_q4k_packed.size(1)));

    auto output = torch::empty({M, N}, x_bf16.options());

    layerstorm::compute::Q4KDequantGemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.A = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    params.B_q4k = weight_q4k_packed.data_ptr();
    params.C = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    layerstorm::compute::launch_q4k_dequant_gemm(params, at::cuda::getCurrentCUDAStream());

    return output;
}


// ===========================================================================
// Q4_K Tensor-Core GEMM
// ===========================================================================

torch::Tensor q4k_cutlass_gemm(
    torch::Tensor x_bf16,
    torch::Tensor weight_q4k_packed
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_q4k_packed.is_cuda() && weight_q4k_packed.dtype() == torch::kUInt8,
        "weight_q4k_packed must be uint8 on CUDA");
    TORCH_CHECK(weight_q4k_packed.dim() == 2 && weight_q4k_packed.is_contiguous(),
        "weight_q4k_packed must be contiguous [N, K*9/16]");

    int M = x_bf16.size(0);
    int K = x_bf16.size(1);
    int N = weight_q4k_packed.size(0);
    int expected_row_bytes = (K / 256) * 144;  // K/256 blocks x 144 bytes each

    TORCH_CHECK(K % 256 == 0, "K must be divisible by 256, got K=" + std::to_string(K));
    TORCH_CHECK(weight_q4k_packed.size(1) == expected_row_bytes,
        "weight row bytes mismatch: expected " + std::to_string(expected_row_bytes) +
        ", got " + std::to_string((int)weight_q4k_packed.size(1)));

    auto output = torch::empty({M, N}, x_bf16.options());

    layerstorm::compute::Q4KCutlassGemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.A = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    params.B_q4k = weight_q4k_packed.data_ptr();
    params.C = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    layerstorm::compute::launch_q4k_cutlass_gemm(params, at::cuda::getCurrentCUDAStream());

    return output;
}


// ===========================================================================
// GGUF dequant-GEMM (strategy: dequant) — all 6 weight types
// ===========================================================================

static layerstorm::compute::GgufType parse_gguf_type(const std::string& t) {
    using layerstorm::compute::GgufType;
    if (t == "q2_k") return GgufType::Q2_K;
    if (t == "q3_k") return GgufType::Q3_K;
    if (t == "q4_k") return GgufType::Q4_K;
    if (t == "q5_k") return GgufType::Q5_K;
    if (t == "q6_k") return GgufType::Q6_K;
    if (t == "q8_0") return GgufType::Q8_0;
    TORCH_CHECK(false, "unknown gguf quant_type '" + t +
        "' (expected one of q2_k/q3_k/q4_k/q5_k/q6_k/q8_0)");
    return GgufType::Q4_K;  // unreachable
}

torch::Tensor gguf_dequant_gemm(
    torch::Tensor x_bf16,
    torch::Tensor weight_packed,
    const std::string& quant_type
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_packed.is_cuda() && weight_packed.dtype() == torch::kUInt8,
        "weight_packed must be uint8 on CUDA");
    TORCH_CHECK(weight_packed.dim() == 2 && weight_packed.is_contiguous(),
        "weight_packed must be contiguous [N, (K/QK)*block_bytes]");

    const layerstorm::compute::GgufType type = parse_gguf_type(quant_type);
    const int qk = layerstorm::compute::gguf_block_values(type);
    const int block_bytes = layerstorm::compute::gguf_block_bytes(type);

    const int M = x_bf16.size(0);
    const int K = x_bf16.size(1);
    const int N = weight_packed.size(0);

    TORCH_CHECK(K % qk == 0,
        "K must be divisible by " + std::to_string(qk) + " for " + quant_type +
        ", got K=" + std::to_string(K));
    const int expected_row_bytes = (K / qk) * block_bytes;
    TORCH_CHECK(weight_packed.size(1) == expected_row_bytes,
        "weight row bytes mismatch for " + quant_type + ": expected " +
        std::to_string(expected_row_bytes) + ", got " +
        std::to_string((int)weight_packed.size(1)));

    auto output = torch::empty({M, N}, x_bf16.options());

    layerstorm::compute::GgufDequantGemmParams params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.A = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    params.B = weight_packed.data_ptr();
    params.C = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    params.type = type;

    layerstorm::compute::launch_gguf_dequant_gemm(params, at::cuda::getCurrentCUDAStream());

    return output;
}


// ===========================================================================
// GGUF integer mat-vec (strategy: int) — all 6 weight types
// ===========================================================================

// Shared setup for the integer paths (mmvq / mmq): validate, build params,
// allocate the Q8_1 workspace, and run `which` ("mmvq" or "mmq").
static torch::Tensor gguf_int_run(
    torch::Tensor x_bf16, torch::Tensor weight_packed,
    const std::string& quant_type, const char* which
) {
    TORCH_CHECK(x_bf16.is_cuda() && x_bf16.dtype() == torch::kBFloat16,
        "x_bf16 must be BF16 on CUDA");
    TORCH_CHECK(x_bf16.dim() == 2 && x_bf16.is_contiguous(),
        "x_bf16 must be contiguous [M, K]");
    TORCH_CHECK(weight_packed.is_cuda() && weight_packed.dtype() == torch::kUInt8,
        "weight_packed must be uint8 on CUDA");
    TORCH_CHECK(weight_packed.dim() == 2 && weight_packed.is_contiguous(),
        "weight_packed must be contiguous [N, (K/QK)*block_bytes]");

    const layerstorm::compute::GgufType type = parse_gguf_type(quant_type);
    const int qk = layerstorm::compute::gguf_block_values(type);
    const int block_bytes = layerstorm::compute::gguf_block_bytes(type);

    const int M = x_bf16.size(0);
    const int K = x_bf16.size(1);
    const int N = weight_packed.size(0);

    TORCH_CHECK(K % qk == 0,
        "K must be divisible by " + std::to_string(qk) + " for " + quant_type +
        ", got K=" + std::to_string(K));
    const int expected_row_bytes = (K / qk) * block_bytes;
    TORCH_CHECK(weight_packed.size(1) == expected_row_bytes,
        "weight row bytes mismatch for " + quant_type + ": expected " +
        std::to_string(expected_row_bytes) + ", got " +
        std::to_string((int)weight_packed.size(1)));

    auto output = torch::empty({M, N}, x_bf16.options());
    const size_t ws_bytes = layerstorm::compute::gguf_mmvq_workspace_bytes(M, K);
    auto workspace = torch::empty({(long)ws_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(x_bf16.device()));

    layerstorm::compute::GgufMmvqParams params;
    params.M = M; params.N = N; params.K = K;
    params.A = reinterpret_cast<const __nv_bfloat16*>(x_bf16.data_ptr());
    params.B = weight_packed.data_ptr();
    params.C = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    params.type = type;

    auto stream = at::cuda::getCurrentCUDAStream();
    const std::string w(which);
    if (w == "mmq_mma")        // fast int8 tensor-core (per-type k=32/k=16) — "v3"
        layerstorm::compute::launch_gguf_mmq_mma(params, workspace.data_ptr(), stream);
    else if (w == "mmq_cute")  // v2 int8 tensor-core (k=16, single kernel)
        layerstorm::compute::launch_gguf_mmq_cute(params, workspace.data_ptr(), stream);
    else if (w == "mmq")       // dp4a tiled
        layerstorm::compute::launch_gguf_mmq(params, workspace.data_ptr(), stream);
    else  // "mmvq"
        layerstorm::compute::launch_gguf_mmvq(params, workspace.data_ptr(), stream);
    return output;
}

torch::Tensor gguf_mmvq(torch::Tensor x_bf16, torch::Tensor weight_packed,
                        const std::string& quant_type) {
    return gguf_int_run(x_bf16, weight_packed, quant_type, "mmvq");
}

torch::Tensor gguf_mmq(torch::Tensor x_bf16, torch::Tensor weight_packed,
                       const std::string& quant_type) {
    return gguf_int_run(x_bf16, weight_packed, quant_type, "mmq");
}

torch::Tensor gguf_mmq_mma(torch::Tensor x_bf16, torch::Tensor weight_packed,
                           const std::string& quant_type) {
    return gguf_int_run(x_bf16, weight_packed, quant_type, "mmq_mma");
}

torch::Tensor gguf_mmq_cute(torch::Tensor x_bf16, torch::Tensor weight_packed,
                            const std::string& quant_type) {
    return gguf_int_run(x_bf16, weight_packed, quant_type, "mmq_cute");
}


// ===========================================================================
// Unified GGUF GEMM: strategy in {"int" (mmvq), "dequant" (dequant-GEMM)}
// ===========================================================================

torch::Tensor gguf_mul_mat(
    torch::Tensor x_bf16,
    torch::Tensor weight_packed,
    const std::string& quant_type,
    const std::string& strategy
) {
    if (strategy == "int") {
        // Route by M: mmvq (mat-vec) for decode/small-M, mmq-MMA (int8
        // tensor-core tiled GEMM) for prefill/large-M.
        const int M = x_bf16.size(0);
        return (M <= 8) ? gguf_mmvq(x_bf16, weight_packed, quant_type)
                        : gguf_mmq_mma(x_bf16, weight_packed, quant_type);
    } else if (strategy == "mmvq") {
        return gguf_mmvq(x_bf16, weight_packed, quant_type);
    } else if (strategy == "mmq") {           // dp4a tiled mat-mat
        return gguf_mmq(x_bf16, weight_packed, quant_type);
    } else if (strategy == "mmq_mma") {       // fast int8 tensor-core (v3, per-type)
        return gguf_mmq_mma(x_bf16, weight_packed, quant_type);
    } else if (strategy == "mmq_cute") {      // v2 int8 tensor-core (k=16)
        return gguf_mmq_cute(x_bf16, weight_packed, quant_type);
    } else if (strategy == "dequant") {
        return gguf_dequant_gemm(x_bf16, weight_packed, quant_type);
    }
    TORCH_CHECK(false, "unknown gguf strategy '" + strategy +
        "' (expected 'int','mmvq','mmq','mmq_mma','mmq_cute','dequant')");
    return torch::Tensor();
}


// ===========================================================================
// Module definition
// ===========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SM120 SnapMLA CUDA kernels";

    m.def("fused_q_quant", &fused_q_quant, py::arg("q_bf16"), py::arg("d_nope"));
    m.def("q_absorb", &q_absorb, py::arg("q_heads"), py::arg("kv_b_proj"),
          py::arg("d_nope_in"), py::arg("d_c"), py::arg("d_rope"), py::arg("d_v"),
          py::arg("w_uk_scales") = py::none(),
          py::arg("seqlens_k") = py::none(), py::arg("cos_sin") = py::none(),
          py::arg("gguf_quant_type") = py::none());
    m.def("rope_rotate", &rope_rotate, py::arg("x"), py::arg("seqlens_k"),
          py::arg("cos_sin"), py::arg("num_tokens"), py::arg("rows_per_token"),
          py::arg("row_stride"));
    m.def("fused_k_append", &fused_k_append, py::arg("c_kv"), py::arg("k_rope"),
          py::arg("kv_cache"), py::arg("slot_mapping"), py::arg("d_c"), py::arg("d_rope"), py::arg("page_size"));
    m.def("dequant_ckv_indexed", &dequant_ckv_indexed, py::arg("kv_cache"), py::arg("indices"),
          py::arg("d_c"), py::arg("d_rope"), py::arg("page_size"));

    // V4 compressor kernels
    m.def("v4_csa_compress", &v4_csa_compress,
          py::arg("input_k_nope"), py::arg("input_k_rope_raw"), py::arg("input_v"),
          py::arg("gate_weights"), py::arg("positional_bias"),
          py::arg("compress_cos"), py::arg("compress_sin"),
          py::arg("head_dim"), py::arg("qk_rope_head_dim"), py::arg("window"), py::arg("stride"));

    m.def("v4_hca_compress", &v4_hca_compress,
          py::arg("input_k_nope"), py::arg("input_k_rope_raw"), py::arg("input_v"),
          py::arg("gate_weights"),
          py::arg("compress_cos"), py::arg("compress_sin"),
          py::arg("head_dim"), py::arg("qk_rope_head_dim"), py::arg("window"), py::arg("stride"));

    // V4 Fused Q Norm + Compressed K RoPE + K Insert
    m.def("v4_fused_q_compress_k", &v4_fused_q_compress_k,
          py::arg("q_bf16"), py::arg("k_nope"), py::arg("k_rope_raw"),
          py::arg("v_nope"), py::arg("compress_cos"), py::arg("compress_sin"),
          py::arg("kv_cache"), py::arg("slot"), py::arg("rope_position"),
          py::arg("rms_eps") = 1e-6);

    // V4 Fused Inverse RoPE + FP8 Quantization
    m.def("v4_fused_inv_rope_fp8", &v4_fused_inv_rope_fp8,
          py::arg("x_bf16"), py::arg("cos_table"), py::arg("sin_table"),
          py::arg("positions"), py::arg("qk_rope_head_dim"));

    // V4 Fused Compress + Insert (compressor + RoPE + FP8 quant + cache write)
    m.def("v4_fused_csa_compress_insert", &v4_fused_csa_compress_insert,
          py::arg("input_k_nope"), py::arg("input_k_rope_raw"), py::arg("input_v"),
          py::arg("gate_weights"), py::arg("positional_bias"),
          py::arg("compress_cos"), py::arg("compress_sin"),
          py::arg("kv_cache"), py::arg("slot_mapping"),
          py::arg("head_dim"), py::arg("qk_rope_head_dim"), py::arg("window"), py::arg("stride"));
    m.def("v4_fused_hca_compress_insert", &v4_fused_hca_compress_insert,
          py::arg("input_k_nope"), py::arg("input_k_rope_raw"), py::arg("input_v"),
          py::arg("gate_weights"),
          py::arg("compress_cos"), py::arg("compress_sin"),
          py::arg("kv_cache"), py::arg("slot_mapping"),
          py::arg("head_dim"), py::arg("qk_rope_head_dim"), py::arg("window"), py::arg("stride"));

    // V4 Lightning Indexer (FP8 K cache, per-block scales)
    m.def("v4_lightning_score", &v4_lightning_score,
          py::arg("q_proj"), py::arg("indexer_k_cache_fp8"),
          py::arg("k_scales"), py::arg("score_proj"));
    m.def("v4_lightning_score_mqa", &v4_lightning_score_mqa,
          py::arg("q_proj"), py::arg("indexer_k_cache_fp8"),
          py::arg("k_scales"), py::arg("score_proj"));
    m.def("v4_lightning_topk", &v4_lightning_topk,
          py::arg("scores"), py::arg("block_endpoints"),
          py::arg("query_position"), py::arg("topk"));

    // V4 Inverse RoPE (arch-agnostic, element-wise)
    m.def("v4_inverse_rope", &v4_inverse_rope,
          py::arg("x"), py::arg("cos_table"), py::arg("sin_table"),
          py::arg("positions"));

    m.def("mhc_pre", &mhc_pre,
          py::arg("residual"), py::arg("fn"), py::arg("hc_scale"), py::arg("hc_base"),
          py::arg("rms_eps"), py::arg("hc_eps"), py::arg("post_mult") = 2.0,
          py::arg("sinkhorn_iters") = 20);
    m.def("mhc_post", &mhc_post,
          py::arg("y"), py::arg("residual"), py::arg("post"), py::arg("comb"));
    m.def("mhc_head", &mhc_head,
          py::arg("residual"), py::arg("fn"), py::arg("hc_scale"), py::arg("hc_base"),
          py::arg("rms_eps"), py::arg("hc_eps"));

    // V4 FP8 cache prep
    m.def("v4_fp8_k_append", &v4_fp8_k_append,
          py::arg("k_nope"), py::arg("k_rope"), py::arg("v_nope"),
          py::arg("kv_cache"), py::arg("slot_mapping"));
    m.def("v4_fp8_dequant_indexed", &v4_fp8_dequant_indexed,
          py::arg("kv_cache"), py::arg("indices"),
          py::arg("head_dim"), py::arg("qk_rope_head_dim"));

    // V4 TQ cache prep
    m.def("v4_tq_k_append", &v4_tq_k_append,
          py::arg("k_nope"), py::arg("k_rope"), py::arg("v_nope"),
          py::arg("kv_cache"), py::arg("slot_mapping"),
          py::arg("Pi"), py::arg("centroids"), py::arg("boundaries"));
    m.def("v4_tq_k_append_gemm", &v4_tq_k_append_gemm,
          py::arg("k_nope"), py::arg("k_rope"), py::arg("v_nope"),
          py::arg("kv_cache"), py::arg("slot_mapping"),
          py::arg("Pi_bf16"), py::arg("centroids"), py::arg("boundaries"));
    m.def("v4_tq_dequant_indexed", &v4_tq_dequant_indexed,
          py::arg("kv_cache"), py::arg("indices"),
          py::arg("Pi"), py::arg("centroids"),
          py::arg("head_dim"), py::arg("qk_rope_head_dim"));

    // V4 CSA FP8 Decode (sparse compressed + SWA combine)
    m.def("v4_csa_fp8_decode", &v4_csa_fp8_decode,
          "V4 CSA FP8 sparse decode (split-KV). topk MUST be a multiple of 64 "
          "for optimal performance (matches FlashMLA/vLLM convention). Non-multiple "
          "values are padded internally but trigger a warning.",
          py::arg("q_nope_bf16"), py::arg("q_rope_bf16"),
          py::arg("compressed_kv"), py::arg("sparse_indices"),
          py::arg("swa_kv"), py::arg("swa_block_table"), py::arg("swa_seqlens"),
          py::arg("sm_scale"), py::arg("topk"),
          py::arg("compressed_page_block_size"), py::arg("swa_page_block_size"),
          py::arg("num_sm_parts"));

    // V4 SWA-Only Decode (pure sliding window, reuses CSA kernel with topk=0)
    m.def("v4_swa_decode", &v4_swa_decode,
          py::arg("q_nope_bf16"), py::arg("q_rope_bf16"),
          py::arg("swa_kv"), py::arg("swa_block_table"), py::arg("swa_seqlens"),
          py::arg("sm_scale"),
          py::arg("swa_page_block_size"),
          py::arg("num_sm_parts"));

    // V4 HCA FP8 Decode (dense over all compressed + SWA, reuses CSA kernel)
    m.def("v4_hca_fp8_decode", &v4_hca_fp8_decode,
          "V4 HCA FP8 dense decode. Delegates to CSA decode with topk=num_compressed. "
          "num_compressed should ideally be a multiple of 64 for optimal performance.",
          py::arg("q_nope_bf16"), py::arg("q_rope_bf16"),
          py::arg("compressed_kv"), py::arg("num_compressed"),
          py::arg("swa_kv"), py::arg("swa_block_table"), py::arg("swa_seqlens"),
          py::arg("sm_scale"),
          py::arg("compressed_page_block_size"), py::arg("swa_page_block_size"),
          py::arg("num_sm_parts"));

    // V4 CSA FP8 Prefill (non-absorbed dense attention with causal mask)
    m.def("v4_csa_fp8_prefill", &v4_csa_fp8_prefill,
          py::arg("q_bf16"), py::arg("k_bf16"), py::arg("v_bf16"),
          py::arg("sm_scale"), py::arg("causal_seqlens") = py::none());

    // TurboQuant prep kernels
    m.def("tq_fused_k_append", &tq_fused_k_append, py::arg("c_kv"), py::arg("k_rope"),
          py::arg("kv_cache"), py::arg("slot_mapping"), py::arg("Pi"), py::arg("centroids"),
          py::arg("boundaries"), py::arg("d_c"), py::arg("d_rope"), py::arg("page_size"));
    m.def("tq_dequant_ckv_indexed", &tq_dequant_ckv_indexed, py::arg("kv_cache"), py::arg("indices"),
          py::arg("Pi"), py::arg("centroids"), py::arg("d_c"), py::arg("d_rope"), py::arg("page_size"));
    m.def("tq_q_rotate", &tq_q_rotate, py::arg("q_nope"), py::arg("Pi"));
    m.def("tq_v_rotate_back", &tq_v_rotate_back, py::arg("out_rotated"), py::arg("Pi"));

    // TurboQuant decode kernels
    m.def("tq_dense_decode_v32", &tq_dense_decode_v32, py::arg("q_rot"), py::arg("q_rope"),
          py::arg("kv_cache"), py::arg("block_table"), py::arg("seqlens_k"), py::arg("centroids"),
          py::arg("sm_scale"), py::arg("page_block_size"), py::arg("num_sm_parts"));
    m.def("tq_sparse_decode_v32", &tq_sparse_decode_v32, py::arg("q_rot"), py::arg("q_rope"),
          py::arg("kv_cache"), py::arg("indices"), py::arg("centroids"),
          py::arg("sm_scale"), py::arg("page_block_size"));

    // V4 CSA TQ decode (sparse TQ scoring in rotated space)
    m.def("v4_csa_tq_decode", &v4_csa_tq_decode, py::arg("q_rot"), py::arg("q_rope"),
          py::arg("kv_cache"), py::arg("indices"), py::arg("centroids"),
          py::arg("sm_scale"), py::arg("num_sm_parts") = 1);

    // V4 TQ prefill: compose dequant + CSA FP8 prefill (Python-side torch.cat)
    // No C++ binding needed — call v4_tq_dequant_indexed then v4_csa_fp8_prefill

    // TurboQuant prefill (dequant + existing BF16 prefill)
    m.def("tq_dense_prefill_v32", &tq_dense_prefill_v32, py::arg("q_bf16"), py::arg("kv_cache"),
          py::arg("Pi"), py::arg("centroids"), py::arg("s_kv"), py::arg("sm_scale"),
          py::arg("d_c"), py::arg("d_rope"), py::arg("page_size"));
    m.def("tq_sparse_prefill_v32", &tq_sparse_prefill_v32, py::arg("q_bf16"), py::arg("kv_cache"),
          py::arg("sparse_indices"), py::arg("Pi"), py::arg("centroids"),
          py::arg("sm_scale"), py::arg("topk"),
          py::arg("d_c"), py::arg("d_rope"), py::arg("page_size"));

    m.def("get_mla_metadata", &get_mla_metadata, py::arg("seqlens_k"), py::arg("num_sm_parts"),
          py::arg("block_size_n"), py::arg("topk"));
    m.def("mla_combine", &mla_combine, py::arg("o_accum"), py::arg("lse_accum"), py::arg("num_splits"),
          py::arg("out"), py::arg("lse"), py::arg("num_sm_parts"),
          py::arg("batch_size"), py::arg("s_q"), py::arg("h_q"), py::arg("d_v"));

    m.def("dense_decode_v32", &dense_decode_v32, py::arg("q_nope_fp8"), py::arg("q_rope_bf16"),
          py::arg("q_scales"), py::arg("kv_cache"), py::arg("block_table"), py::arg("seqlens_k"),
          py::arg("sm_scale"), py::arg("page_block_size"), py::arg("num_sm_parts"));
    m.def("sparse_decode_v32", &sparse_decode_v32, py::arg("q_nope_fp8"), py::arg("q_rope_bf16"),
          py::arg("q_scales"), py::arg("kv_cache"), py::arg("indices"), py::arg("sm_scale"),
          py::arg("page_block_size"), py::arg("topk"), py::arg("num_sm_parts"));

    m.def("sparse_prefill_v32", &sparse_prefill_v32, py::arg("q_bf16"), py::arg("kv_bf16"),
          py::arg("indices"), py::arg("sm_scale"), py::arg("topk"));
    m.def("dense_prefill_v32", &dense_prefill_v32, py::arg("q_bf16"), py::arg("kv_bf16"), py::arg("sm_scale"));

    // NVFP4 attention weight GEMM
    m.def("nvfp4_gemm_preprocess", &nvfp4_gemm_preprocess, py::arg("weight_scales"),
          py::arg("N"), py::arg("K"),
          "Preprocess weight scales: [N, K/16] float32 → Sm1xx interleaved UE4M3 (1D uint8)");
    m.def("nvfp4_gemm", &nvfp4_gemm, py::arg("x_bf16"), py::arg("weight_packed_uint8"),
          py::arg("weight_scales"), py::arg("weight_scale_global"));

    // Q4_K dequant-GEMM
    m.def("q4k_dequant_gemm", &q4k_dequant_gemm,
          py::arg("x_bf16"), py::arg("weight_q4k_packed"),
          "Q4_K dequant-GEMM: x_bf16 @ dequant(W_q4k).T → BF16 output");

    // Q4_K tensor-core GEMM
    m.def("q4k_cutlass_gemm", &q4k_cutlass_gemm,
          py::arg("x_bf16"), py::arg("weight_q4k_packed"),
          "Q4_K tensor-core GEMM: x_bf16 @ dequant(W_q4k).T → BF16 output (wmma TC path)");

    // GGUF dequant-GEMM (strategy: dequant) — q2_k/q3_k/q4_k/q5_k/q6_k/q8_0
    m.def("gguf_dequant_gemm", &gguf_dequant_gemm,
          py::arg("x_bf16"), py::arg("weight_packed"), py::arg("quant_type"),
          "GGUF dequant-to-float GEMM: x_bf16 @ dequant(W).T → BF16 output. "
          "quant_type in {q2_k,q3_k,q4_k,q5_k,q6_k,q8_0}.");

    // GGUF integer mat-vec (decode) — Q8_1-activation dp4a path
    m.def("gguf_mmvq", &gguf_mmvq,
          py::arg("x_bf16"), py::arg("weight_packed"), py::arg("quant_type"),
          "GGUF integer mat-vec (decode): Q8_1-activation dp4a against packed "
          "weights → BF16. quant_type in {q2_k,q3_k,q4_k,q5_k,q6_k,q8_0}.");

    // GGUF integer mat-mat (prefill) — Q8_1-activation dp4a tiled GEMM
    m.def("gguf_mmq", &gguf_mmq,
          py::arg("x_bf16"), py::arg("weight_packed"), py::arg("quant_type"),
          "GGUF integer mat-mat (prefill): tiled dp4a GEMM, Q8_1 activation → "
          "BF16. quant_type in {q2_k,q3_k,q4_k,q5_k,q6_k,q8_0}.");

    // GGUF integer mat-mat (prefill) — fast per-type int8 tensor-core (v3)
    m.def("gguf_mmq_mma", &gguf_mmq_mma,
          py::arg("x_bf16"), py::arg("weight_packed"), py::arg("quant_type"),
          "GGUF integer mat-mat (prefill): fast int8 tensor-core, per-type "
          "k=32/k=16 (SM80 MMA). quant_type in {q2_k,...,q8_0}.");

    // GGUF integer mat-mat (prefill) — v2 int8 tensor-core (k=16, single kernel)
    m.def("gguf_mmq_cute", &gguf_mmq_cute,
          py::arg("x_bf16"), py::arg("weight_packed"), py::arg("quant_type"),
          "GGUF integer mat-mat (prefill): v2 int8 tensor-core (k=16). "
          "quant_type in {q2_k,...,q8_0}.");

    // Unified GGUF GEMM with strategy selection (default 'int')
    m.def("gguf_mul_mat", &gguf_mul_mat,
          py::arg("x_bf16"), py::arg("weight_packed"), py::arg("quant_type"),
          py::arg("strategy") = "int",
          "Unified GGUF GEMM. strategy='int' (mmvq) or 'dequant' (dequant-GEMM).");

    // DCP LSE correction
    m.def("dcp_lse_correct", &dcp_lse_correct,
          "DCP LSE correction: reweight partial attention output across DCP ranks",
          py::arg("output"), py::arg("lses"), py::arg("rank"));

    py::class_<PyDecodeGraphRunner>(m, "DecodeGraphRunner",
        "CUDA graph runner: captures fused_q_quant + decode + mla_combine")
        .def(py::init<>())
        .def("init", &PyDecodeGraphRunner::init, py::arg("kv_cache"),
             py::arg("batch_size"), py::arg("s_q"), py::arg("h_q"), py::arg("h_kv"),
             py::arg("d_qk"), py::arg("d_v"), py::arg("d_nope"),
             py::arg("page_block_size"), py::arg("max_num_blocks_per_seq"),
             py::arg("sm_scale"), py::arg("num_sm_parts"),
             py::arg("sparse")=false, py::arg("topk")=0, py::arg("extra_topk")=0)
        .def("update_metadata", &PyDecodeGraphRunner::update_metadata, py::arg("seqlens_k"), py::arg("num_sm_parts"))
        .def("update", &PyDecodeGraphRunner::update, py::arg("q_bf16"), py::arg("seqlens_k"), py::arg("block_table"))
        .def("update_with_indices", &PyDecodeGraphRunner::update_with_indices,
             py::arg("q_bf16"), py::arg("seqlens_k"), py::arg("block_table"), py::arg("indices"))
        .def("replay", &PyDecodeGraphRunner::replay)
        .def("get_output", &PyDecodeGraphRunner::get_output, py::arg("ref_tensor"))
        .def("destroy", &PyDecodeGraphRunner::destroy);

    py::class_<PyTqDecodeGraphRunner>(m, "TqDecodeGraphRunner",
        "TQ CUDA graph runner: captures tq_q_rotate + tq_decode + mla_combine + tq_v_rotate_back")
        .def(py::init<>())
        .def("init", &PyTqDecodeGraphRunner::init, py::arg("kv_cache"), py::arg("Pi"), py::arg("centroids"),
             py::arg("batch_size"), py::arg("s_q"), py::arg("h_q"),
             py::arg("d_c"), py::arg("d_rope"), py::arg("page_block_size"),
             py::arg("max_num_blocks_per_seq"), py::arg("sm_scale"), py::arg("num_sm_parts"))
        .def("update_metadata", &PyTqDecodeGraphRunner::update_metadata, py::arg("seqlens_k"), py::arg("num_sm_parts"))
        .def("update", &PyTqDecodeGraphRunner::update, py::arg("q_nope_bf16"), py::arg("q_rope_bf16"),
             py::arg("seqlens_k"), py::arg("block_table"))
        .def("replay", &PyTqDecodeGraphRunner::replay)
        .def("get_output", &PyTqDecodeGraphRunner::get_output, py::arg("ref_tensor"))
        .def("destroy", &PyTqDecodeGraphRunner::destroy);

    py::class_<PyCsaFp8DecodeGraphRunner>(m, "CsaFp8DecodeGraphRunner",
        "V4 CSA FP8 decode graph runner (attend-only: CSA decode + combine)")
        .def(py::init<>())
        .def("init", &PyCsaFp8DecodeGraphRunner::init,
             py::arg("compressed_kv"), py::arg("swa_kv"),
             py::arg("batch_size"), py::arg("s_q"), py::arg("h_q"), py::arg("topk"),
             py::arg("sm_scale"), py::arg("num_sm_parts"),
             py::arg("compressed_page_block_size"), py::arg("swa_page_block_size"),
             py::arg("max_swa_blocks"))
        .def("update_metadata", &PyCsaFp8DecodeGraphRunner::update_metadata,
             py::arg("topk_seqlens"), py::arg("num_sm_parts"))
        .def("update", &PyCsaFp8DecodeGraphRunner::update,
             py::arg("q_nope"), py::arg("q_rope"), py::arg("sparse_indices"),
             py::arg("swa_block_table"), py::arg("swa_seqlens"))
        .def("replay", &PyCsaFp8DecodeGraphRunner::replay)
        .def("get_output", &PyCsaFp8DecodeGraphRunner::get_output, py::arg("ref_tensor"))
        .def("destroy", &PyCsaFp8DecodeGraphRunner::destroy);

    py::class_<PyCsaTqDecodeGraphRunner>(m, "CsaTqDecodeGraphRunner",
        "V4 CSA TQ decode graph runner (q_rotate + TQ decode + v_rotate_back)")
        .def(py::init<>())
        .def("init", &PyCsaTqDecodeGraphRunner::init,
             py::arg("kv_cache"), py::arg("Pi"), py::arg("centroids"),
             py::arg("batch_size"), py::arg("s_q"), py::arg("h_q"),
             py::arg("topk"), py::arg("sm_scale"))
        .def("update", &PyCsaTqDecodeGraphRunner::update,
             py::arg("q_nope_bf16"), py::arg("q_rope_bf16"), py::arg("sparse_indices"))
        .def("replay", &PyCsaTqDecodeGraphRunner::replay)
        .def("get_output", &PyCsaTqDecodeGraphRunner::get_output, py::arg("ref_tensor"))
        .def("destroy", &PyCsaTqDecodeGraphRunner::destroy);
}
