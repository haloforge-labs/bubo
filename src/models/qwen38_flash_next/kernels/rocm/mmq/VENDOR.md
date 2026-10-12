# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see `LICENSE`).
The upstream source directory is `ggml/src/ggml-cuda`; Gufo builds these
sources with HIP for AMD Strix Halo (`gfx1151`).

`qfn_mmq.h` is the public adapter. It supplies Q8_0 dense projections and
Q4_K, Q5_K, Q5_1 and Q8_0 routed expert projections, including the paired
Q4_K gate/up path. `qfn_ggml_stubs.*` supplies the minimal context and scratch
pool without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix. Gufo owns scheduling and HIP graphs.

When importing an upstream fix, preserve the Gufo namespace, adapter and
HIP-only dispatch, update the pin, and run the Flash-Next operator tests
under `tests/models/qwen38_flash_next/` plus the AR/MTP session replay.
The routed GEMM test compares supported formats against a floating-point
reference and checks the paired adapter against two separate projections.
Build and validate through Nix as described in `AGENTS.md`.

## Strata grouped decode experts

Adapted from https://github.com/Niko1221/Strata (MIT).
Introduction: `04e1e11afb45c040960e90c834e0fe85a475a4ab`.
Measured base (PR #1360): `82f46a8`.
Read-only reference HEAD: `a0116b2782a5b82622358c04a06496485d12b2d6`.

Imported structure: `src/kernels/cuda/iq_kernels.cu` (`S27`,
`s27_gu_kernel`, `s26_swiglu_q8_1_kernel`, `s27_down_kernel`),
`src/kernels/cuda/verify_kernels.cu` (`resident_plan_par_kernel`), and
`src/kernels/cuda/s26_tsum.cuh` (transposed butterfly helpers).
Local files: `expert_v2k.hip.cpp`, `s26_tsum.hpp`.

Modifications: HIP-only includes/namespace, separate gate/up/down tensor
bases, device expert IDs instead of blob pointers, gufo's Q4MoeFragment
and Q5_1/Q8_0 vec_dot arithmetic, gufo's SwiGLU/quantizer rules,
negative-ID zeroing, caller-owned persistent scratch, and hq-only bridge.
Only Q4_K gate/up is instantiated; Q5_K retains the existing path.
The measured four-entry chunks, eight wave32 waves, four rows per wave,
LDS staging and TSUM layout are retained.

Full upstream license follows verbatim:
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

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
