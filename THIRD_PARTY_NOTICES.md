# Third-Party Notices — LayerStoRmKernels

LayerStoRmKernels is licensed under the Apache License 2.0 (see `LICENSE.md`).
Portions of this repository are derived from, adapted from, or reference the
third-party projects listed below. Where a section says "see MIT License text
below", the full license text in Appendix A applies together with that
section's copyright line(s).

---

## FlashMLA

- Upstream: https://github.com/deepseek-ai/FlashMLA (and the SM120 fork
  https://github.com/IISuperluminaLII/FlashMLA_Windows_Linux_sm120)
- License: MIT — Copyright (c) 2025 DeepSeek (see MIT License text below)
- What was derived: the split-KV MLA decode/combine/metadata kernels and their
  parameter/scheduling conventions — `csrc/smxx/{mla_combine.cu,
  get_mla_metadata.cu, params.h, utils.h}`, the SM120 decode kernels under
  `csrc/sm120/decode/{sparse_fp8,dense_fp8,tq_sparse,tq_dense,csa_fp8,csa_tq}/`,
  the prefill kernels under `csrc/sm120/prefill/`, and the shared MMA helpers
  in `csrc/sm120/components/`. See the per-file headers for the specific
  upstream file each was adapted from.

## SGLang

- Upstream: https://github.com/sgl-project/sglang
- License: Apache-2.0 — Copyright 2023-2024 SGLang Team
- What was derived: the DeepSeek sparse-attention (DSA) indexer kernels
  (`csrc/sm120/indexer/lightning_*`), the FP8 KV-cache per-token quantization
  conventions of the SnapMLA path (following the "SGLang-FluentLLM" reference
  implementation), and top-k selection structure.

## vLLM

- Upstream: https://github.com/vllm-project/vllm
- License: Apache-2.0 — Copyright contributors to the vLLM project
- What was derived: the DCP LSE-correction kernel
  (`csrc/smxx/dcp_lse_correct.{cu,h}`, ported from vLLM's
  `_correct_attn_cp_out_kernel`), mHC kernel math (`csrc/smxx/mhc.{cu,h}`),
  and indexer top-k structure (`csrc/sm120/indexer/`).

## NVIDIA TensorRT-LLM

- Upstream: https://github.com/NVIDIA/TensorRT-LLM
- License: Apache-2.0 — Copyright (c) 2011-2025 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved.
- What was derived: indexer top-k kernel structure
  (`kernels/indexerTopK.cu` lineage; see `csrc/sm120/indexer/lightning_topk.h`)
  and FP8 GEMM swizzle patterns referenced in `csrc/sm120/components/`.
- TensorRT-LLM ships no Apache-2.0 NOTICE file at its repository root, so
  there are no NOTICE contents to reproduce under Apache-2.0 §4(d).

## llama.cpp / ggml

- Upstream: https://github.com/ggerganov/llama.cpp
- License: MIT — Copyright (c) 2023-2026 The ggml authors (see MIT License
  text below)
- What was derived: DeepSeek-V4 graph semantics referenced by the mHC kernels
  (`csrc/smxx/mhc.{cu,h}`, `build_hc_*` math).

## NVIDIA CUTLASS

- Upstream: https://github.com/NVIDIA/cutlass (consumed as the
  `3rd-party/cutlass` git submodule; not vendored in this tree)
- License: BSD-3-Clause — Copyright (c) 2017 - 2026 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved. (full text in Appendix B)
- What is used: CUTLASS/CuTe headers are a build dependency of every CUDA
  kernel in `csrc/` (MMA atoms, tensor layouts). Binaries built from this
  repository incorporate CUTLASS header code; the BSD-3-Clause notice applies
  to such binaries. The `python/CuTeDSL` directory of upstream CUTLASS is
  under a separate NVIDIA EULA; this project does not use it.

## TurboQuant (paper)

- Paper: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni,
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
  arXiv:2504.19874.
- The TurboQuant (TQ) 4-bit KV-cache codec kernels in `csrc/sm120/` are an
  independent implementation of the paper's method. The Lloyd-Max codebooks
  in `data/codebooks/*.json` are numeric quantizer constants (centroids and
  boundaries of the optimal scalar quantizer for the rotated unit-sphere
  coordinate distribution). No code from the GPL-3.0 third-party reference
  implementation (https://github.com/0xSero/turboquant) is included in this
  repository.

## Model configuration test data (`test-data/`)

Small model-metadata files (config.json, generation_config.json,
tokenizer_config.json, chat templates) are included for tests and were
obtained from the following model repositories:

- `test-data/DeepSeek-V3.2/` — https://huggingface.co/deepseek-ai
  (MIT, Copyright (c) 2025 DeepSeek)
- `test-data/GLM-5/` — https://huggingface.co/zai-org (MIT, Zhipu AI)
- `test-data/Kimi-K2.5/` — https://huggingface.co/moonshotai (Moonshot AI —
  K2-family models are published under Moonshot's Modified MIT License; see
  the upstream model repository for its terms)

No model weights are distributed in this repository.

## Sample text data (`sample-data/texts/`)

Per-file dataset provenance and licenses are recorded in
`sample-data/texts/SOURCES.md`. Note in particular that `prose_short.txt` and
`prose_long.txt` are English Wikipedia article text and remain under
CC BY-SA 3.0 (attribution in `SOURCES.md`), not this repository's MIT license
license.

---

## Appendix A — MIT License text

The following license text applies to the MIT-licensed material identified
above, together with the copyright lines given in each section:

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Appendix B — BSD-3-Clause (NVIDIA CUTLASS)

```
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Appendix C — Apache License 2.0

This repository is licensed under the MIT License — see `LICENSE.md`.
It applies both to LayerStoRmKernels itself (Copyright 2026 Kimon Kontosis)
and to the Apache-2.0-licensed upstream material identified above (SGLang,
vLLM, NVIDIA TensorRT-LLM).
