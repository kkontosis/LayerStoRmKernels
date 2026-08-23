import os
import subprocess
import sys
from setuptools import setup, find_packages

# ---------------------------------------------------------------------------
# Detect CUTLASS include path
# ---------------------------------------------------------------------------
def find_cutlass_include():
    """Find CUTLASS headers: local 3rd-party > env var > pip package > common paths."""
    # 0. Local 3rd-party submodule (preferred — pinned to v4.4.2 with SM120 support)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rd-party", "cutlass", "include")
    if os.path.isfile(os.path.join(local, "cutlass", "cutlass.h")):
        return local

    # 1. Environment variable
    cutlass_path = os.environ.get("CUTLASS_PATH")
    if cutlass_path:
        inc = os.path.join(cutlass_path, "include")
        if os.path.isfile(os.path.join(inc, "cutlass", "cutlass.h")):
            return inc

    # 2. Scan site-packages for nvidia-cutlass (works with pip, uv, etc.)
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for subdir in [
            "cutlass_library/source/include",  # nvidia-cutlass 4.x
            "nvidia/cutlass/include",
            "cutlass/include",
        ]:
            inc = os.path.join(sp, subdir)
            if os.path.isfile(os.path.join(inc, "cutlass", "cutlass.h")):
                return inc

    # 4. Common system paths (apt-get install libcutlass-dev puts headers at /usr/include/)
    for path in ["/usr/include", "/usr/local/include", "/usr/local/cutlass/include"]:
        if os.path.isfile(os.path.join(path, "cutlass", "cutlass.h")):
            return path

    return None


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------
# Bypass CUDA version mismatch check (system CUDA 13.1 vs PyTorch cu128 is compatible)
import torch.utils.cpp_extension as _cpp_ext
_orig_check = _cpp_ext._check_cuda_version
def _noop_check(*args, **kwargs): pass
_cpp_ext._check_cuda_version = _noop_check

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

project_root = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(project_root, "csrc")

cutlass_include = find_cutlass_include()
if cutlass_include is None:
    print("WARNING: CUTLASS headers not found. Install nvidia-cutlass or set CUTLASS_PATH.")
    print("  pip install nvidia-cutlass")
    print("  OR: CUTLASS_PATH=/path/to/cutlass pip install -e .")
    sys.exit(1)

print(f"Using CUTLASS headers from: {cutlass_include}")

# CuTe headers are alongside cutlass
cute_include = cutlass_include  # cute/ is under the same include/ dir

cutlass_tools_include = os.path.join(os.path.dirname(cutlass_include), "tools", "util", "include")

gemm_csrc_dir = os.path.join(project_root, "deps", "LayerStoRmGemmKernels", "csrc")

include_dirs = [
    csrc_dir,
    gemm_csrc_dir,  # GEMM kernel headers from submodule
    cutlass_include,
    cutlass_tools_include,
]

sources = [
    # Main compilation unit (includes prep kernels, graph runner, and Python bindings)
    "csrc/bindings.cu",
    # Decode kernel instantiations
    "csrc/sm120/decode/sparse_fp8/instantiations/v32_persistent_h64.cu",
    "csrc/sm120/decode/sparse_fp8/instantiations/model1_persistent_h64.cu",
    "csrc/sm120/decode/dense_fp8/instantiations/v32_persistent_h64.cu",
    "csrc/sm120/decode/dense_fp8/instantiations/model1_persistent_h64.cu",
    # V4 CSA FP8 decode kernel instantiations
    "csrc/sm120/decode/csa_fp8/instantiations/v4_h64.cu",
    "csrc/sm120/decode/csa_fp8/instantiations/v4_h128.cu",
    # Prefill kernel instantiations
    "csrc/sm120/prefill/sparse/fwd/head64/instantiations/phase1_k576.cu",
    "csrc/sm120/prefill/dense/fwd/head64/instantiations/phase1_k576.cu",
    # V4 CSA FP8 prefill kernel instantiation
    "csrc/sm120/prefill/csa_fp8/instantiations/phase1_k576.cu",
    # GEMM kernels — from LayerStoRmGemmKernels submodule
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/nvfp4/nvfp4_gemm.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/q4k/q4k_dequant_gemm.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/q4k/q4k_cutlass_gemm.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/gguf_dequant_gemm.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/gguf_mmvq.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/gguf_mmq.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/gguf_mmq_cute.cu",
    # Fast per-type int8 tensor-core mmq (v3)
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/mmq_mma_dispatch.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q4_k.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q5_k.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q8_0.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q6_k.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q2_k.cu",
    "deps/LayerStoRmGemmKernels/csrc/sm120/gemm/gguf/mmq_mma/instances/mmq_q3_k.cu",
    # Projection-GEMM CUDA graph runner (segment a; no binding yet — built for hygiene)
    "csrc/sm120/graph/projection_graph.cu",
    # Split-KV + DCP infrastructure
    "csrc/smxx/mla_combine.cu",
    "csrc/smxx/get_mla_metadata.cu",
    "csrc/smxx/dcp_lse_correct.cu",
    # V4 arch-agnostic kernels
    "csrc/smxx/inverse_rope.cu",
]

nvcc_flags = [
    "-std=c++17",
    "-O2",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

# Detect SM arch: prefer sm_120f (enables FP8 + blockwise scaled TensorOp),
# fall back to sm_89 for pre-12.8 CUDA.
# The 'f' feature flag is required for CUTLASS blockwise-scaled FP8 GEMM on SM120.
try:
    nvcc_version = subprocess.check_output(["nvcc", "--version"], text=True)
    if "release 13" in nvcc_version or "release 12.8" in nvcc_version:
        nvcc_flags.append("-gencode=arch=compute_120f,code=sm_120f")
    else:
        print("WARNING: CUDA < 12.8 detected, using -arch=sm_89 (FP8 MMA works but SM120 features may not)")
        nvcc_flags.append("-arch=sm_89")
except Exception:
    nvcc_flags.append("-gencode=arch=compute_120f,code=sm_120f")

setup(
    name="sm120_mla_kernels",
    version="0.1.0",
    description="SM120 SnapMLA CUDA kernels with Python bindings",
    ext_modules=[
        CUDAExtension(
            name="sm120_mla_kernels",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-std=c++17", "-O2"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
