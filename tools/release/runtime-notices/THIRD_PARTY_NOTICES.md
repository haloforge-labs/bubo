# Third-party notices — Halowin Bubo for Windows

Halowin Bubo ships the files listed under "Files in the Bubo archive". It downloads the AMD ROCm 10.0.0 runtime for gfx1151 directly from AMD. AMD DLLs and kernel data are not part of the Bubo archive.

Keep `licenses/`, `THIRD_PARTY_NOTICES.md`, `NOTICE-windows.md`, `LICENSE`, `NOTICE` and `gufo-upstream/` beside the installed runtime. Read the notices and the full component texts before you install the runtime.

The file `gufo-upstream/THIRD_PARTY_NOTICES.md` is the unchanged inventory of upstream gufo. It also lists Nix, Linux, development-only and external-model entries, and ROCm, curl and OpenSSL versions that differ from the Windows versions. Use the tables below for the Windows files. All original gufo notices and licence files are kept byte for byte under `gufo-upstream/`.

The Microsoft Visual C++ runtime is a prerequisite. It is not part of the Bubo archive or of this bundle. Install it from Microsoft.

## Files in the Bubo archive

The versions are the library versions that the Bubo build records in `BUILD-MANIFEST.json`.

| File | Component | Version | Licence | Licence file |
|---|---|---|---|---|
| bubo.exe, bubo-engine.dll | Halowin Bubo (gufo) | see `bubo --version` | MIT, with the adapted-code terms below | [gufo-MIT.txt](licenses/gufo-MIT.txt) |
| libcrypto-3-x64.dll | OpenSSL | 3.6.4 | Apache-2.0 | [openssl-Apache-2.0.txt](licenses/openssl-Apache-2.0.txt) |
| libcurl.dll | curl | 8.22.0 | curl | [curl-curl.txt](licenses/curl-curl.txt) |
| icudt78.dll, icuin78.dll, icuuc78.dll | ICU | 78.3 | Unicode License V3 | [icu-Unicode-3.0.txt](licenses/icu-Unicode-3.0.txt) |
| libpng16.dll | libpng | 1.6.58 | libpng-2.0 | [libpng-libpng-2.0.txt](licenses/libpng-libpng-2.0.txt) |
| jpeg62.dll | libjpeg-turbo | 3.2.0 | IJG AND BSD-3-Clause AND Zlib | [libjpeg-turbo-IJG-BSD-3-Clause-Zlib.txt](licenses/libjpeg-turbo-IJG-BSD-3-Clause-Zlib.txt)<br>[libjpeg-turbo-IJG.txt](licenses/libjpeg-turbo-IJG.txt)<br>[libjpeg-turbo-Zlib.txt](licenses/libjpeg-turbo-Zlib.txt) |
| libwebp.dll, libsharpyuv.dll | libwebp / SharpYUV | 1.6.0 | BSD-3-Clause; patent grant included | [libwebp-BSD-3-Clause.txt](licenses/libwebp-BSD-3-Clause.txt)<br>[libwebp-PATENTS.txt](licenses/libwebp-PATENTS.txt) |
| z.dll | zlib | 1.3.2 | Zlib | [zlib-Zlib.txt](licenses/zlib-Zlib.txt) |

The audio libraries in the archive (libsndfile, libsoxr, FLAC, libogg, libvorbis, Opus, mpg123 and LAME), LLVM OpenMP and the other additions that come from upstream gufo are listed with their licences in `licenses/UPSTREAM-ADDITIONS.md` of the Bubo archive. The libspng 0.7.4 licence text is included as [libspng-BSD-2-Clause.txt](licenses/libspng-BSD-2-Clause.txt).

The libjpeg-turbo licence text is the text of release 3.2.0. This software is based in part on the work of the Independent JPEG Group.

## Files that Bubo downloads from AMD

Bubo downloads these files from AMD when you prepare the runtime. They are not in the Bubo archive.

| File / data folder | Component | Version | Licence | Licence file |
|---|---|---|---|---|
| amd_comgr.dll | COMGR | 3.3.0.0 | Apache-2.0 WITH LLVM-exception + NCSA | [comgr-Apache-2.0-LLVM-exception-NCSA.txt](licenses/comgr-Apache-2.0-LLVM-exception-NCSA.txt) |
| amdhip64_7.dll | CLR / HIP | HIP 7.15.0; file 10.0.3581.0 | MIT | [clr-MIT.txt](licenses/clr-MIT.txt) |
| hipblas.dll | hipBLAS | 3.6.0.0 | MIT, with embedded BLAS notices | [hipblas-MIT.txt](licenses/hipblas-MIT.txt) |
| hiprtc-builtins0715.dll, hiprtc0715.dll | HIPRTC | HIP 7.15.0; file 10.0.3581.0 | MIT | [hip-MIT.txt](licenses/hip-MIT.txt)<br>[clr-MIT.txt](licenses/clr-MIT.txt) |
| libhipblaslt.dll | hipBLASLt | 1.4.1.0 | MIT | [hipblaslt-MIT.txt](licenses/hipblaslt-MIT.txt) |
| origami.dll | Origami | ROCm 10.0.0 | MIT | [origami-MIT.txt](licenses/origami-MIT.txt) |
| rocblas.dll | rocBLAS | 5.6.0.0 | MIT, with embedded third-party notices | [rocblas-MIT.txt](licenses/rocblas-MIT.txt) |
| rocm_kpack.dll | rocm-kpack | ROCm 10.0.0 | MIT | [rocm-kpack-MIT.txt](licenses/rocm-kpack-MIT.txt) |
| rocm-openblas.dll, rocm-openblas64.dll | OpenBLAS | source 18638c70eff1f0d08e2833b2724deaa128d6a334 | BSD-3-Clause | [openblas-BSD-3-Clause.txt](licenses/openblas-BSD-3-Clause.txt) |
| rocsolver.dll | rocSOLVER | ROCm 10.0.0 | BSD-2-Clause, with LAPACK and MAGMA notices | [rocsolver-BSD-2-Clause.txt](licenses/rocsolver-BSD-2-Clause.txt) |
| rocsparse.dll | rocSPARSE | 5.0.0.0 | MIT | [rocsparse-MIT.txt](licenses/rocsparse-MIT.txt) |
| rocblas\library\ (150 files) | rocBLAS gfx1151 kernel data | ROCm 10.0.0 | MIT | [tensile-MIT.txt](licenses/tensile-MIT.txt)<br>[rocblas-MIT.txt](licenses/rocblas-MIT.txt) |
| hipblaslt\library\ (100 files) | hipBLASLt gfx1151 kernel data | ROCm 10.0.0 | MIT | [hipblaslt-MIT.txt](licenses/hipblaslt-MIT.txt)<br>[tensile-MIT.txt](licenses/tensile-MIT.txt)<br>[rocroller-MIT.txt](licenses/rocroller-MIT.txt) |

AMD's own terms apply to the files that you download from AMD. The core wheel also carries the hipcc licence, MIT: [hipcc-MIT.txt](licenses/hipcc-MIT.txt).

The download is the [official AMD tarball](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz), 1,684,865,477 bytes, SHA-256 `1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`. The AMD-hosted wheels below hold the same components and total 1,014,229,359 bytes.

- [rocm_sdk_core-10.0.0-py3-none-win_amd64.whl](https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/rocm_sdk_core-10.0.0-py3-none-win_amd64.whl)
- [rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl](https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-libraries/rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl)
- [rocm_sdk_device_gfx1151-10.0.0-py3-none-win_amd64.whl](https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-device-gfx1151/rocm_sdk_device_gfx1151-10.0.0-py3-none-win_amd64.whl)

## Code and credits in Bubo

These credits apply to code that Bubo adapts or builds on, and to notices that come with it.

| Component / relationship | Revision / version | Licence | Licence file | Source |
|---|---|---|---|---|
| llama.cpp / ggml; quantization, attention and HIP kernels | 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a; attention e9fa0781f1c25fc4fe8c86be1edc6970661ad6f0 | MIT | [llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt) | [upstream](https://github.com/ggml-org/llama.cpp) |
| GSQHalo.cpp / ggml; adapted routed IQ/Q2 slice decoders and WMMA staging | 16f0929d1da1001fd856c2443cdd31de8d470b73 | MIT, Copyright (c) 2023-2026 The ggml authors | [llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt) | [upstream](https://github.com/Aristo94/GSQHalo.cpp/tree/16f0929d1da1001fd856c2443cdd31de8d470b73) |
| DS4; loader, tokenizer, sessions and HIP kernels | 84cc882352757baf628a1776badf7cc54d584e28 | MIT | [ds4-MIT.txt](licenses/ds4-MIT.txt) | [upstream](https://github.com/antirez/ds4) |
| DS4 GB10/GX10 changes; Marco Palaferri | 910501e | MIT | [ds4-gb10-MIT.txt](licenses/ds4-gb10-MIT.txt) | [upstream](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA) |
| h3.c; Salvatore Sanfilippo | 8974cc055ea9c02fcd14cc27dfda3e1027c05153 | MIT | [h3.c-MIT.txt](licenses/h3.c-MIT.txt) | [upstream](https://github.com/antirez/h3.c) |
| ccv ancestry through h3.c; Liu Liu | h3.c pin above | BSD-3-Clause | [ccv-BSD-3-Clause.txt](licenses/ccv-BSD-3-Clause.txt) | [upstream](https://github.com/liuliu/ccv) |
| Qwen reference chat templates and adapted renderer behaviour | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; de4b8e4d43b917e7706784d8bb445c9af86a3540 | Apache-2.0 | [qwen-templates-Apache-2.0.txt](licenses/qwen-templates-Apache-2.0.txt) | [upstream](https://huggingface.co/Qwen/Qwen3.8-27B) |
| hipCUB; headers compiled into kernels | ROCm 7.2.3 reference of upstream gufo | BSD-3-Clause | [hipcub-BSD-3-Clause.txt](licenses/hipcub-BSD-3-Clause.txt) | [upstream](https://github.com/ROCm/hipCUB) |
| rocPRIM; headers compiled into kernels | ROCm 7.2.3 reference of upstream gufo | MIT | [rocprim-MIT.txt](licenses/rocprim-MIT.txt) | [upstream](https://github.com/ROCm/rocm-libraries) |
| rocWMMA; headers compiled into kernels | ROCm 7.2.3 reference of upstream gufo | MIT | [rocwmma-MIT.txt](licenses/rocwmma-MIT.txt) | [upstream](https://github.com/ROCm/rocWMMA) |
| Composable Kernel; adapted short-attention arithmetic | ROCm 7.2.3 reference of upstream gufo | MIT | [composable-kernel-MIT.txt](licenses/composable-kernel-MIT.txt) | [upstream](https://github.com/ROCm/composable_kernel) |
| mikealanni/gufo; Flash-Next GDN reuse and SSM stride changes | f450218743bcaf77e3e1f8fab78f4cd4e4aafae3 | MIT | [mikealanni-gufo-MIT.txt](licenses/mikealanni-gufo-MIT.txt) | [upstream](https://github.com/mikealanni/gufo) |
| Nathan Wilson; optimization inspiration | ce15ecca66e5e5a9aefa1ebc82357cb901330c86 | MIT | [strix-halo-llamacpp-MIT.txt](licenses/strix-halo-llamacpp-MIT.txt) | [upstream](https://github.com/Nathanw1014/strix-halo-llamacpp) |
| pixmaate/gufo Windows layer; compat/win32 and launchers | 2eb0778ac15d9e1c18c5b2f9e5dbbecc19e20ec2 | MIT | [pixmaate-windows-layer-MIT.txt](licenses/pixmaate-windows-layer-MIT.txt) | [upstream](https://github.com/pixmaate/gufo/tree/2eb0778ac15d9e1c18c5b2f9e5dbbecc19e20ec2) |

The Windows compatibility layer and the launcher work are derived from pixmaate/gufo, branch windows-port, commit `2eb0778ac15d9e1c18c5b2f9e5dbbecc19e20ec2`. Its full MIT licence and the notice "Copyright (c) 2026 gufo contributors" are kept in `licenses/pixmaate-windows-layer-MIT.txt`. The credit to pixmaate is also in `NOTICE-windows.md`.

Additional optimization inspiration: LaurentZuijdwijk/llama.cpp at 11bfe8a633fa02bac251db6cf21bd5ddab282a64, and gaetan-puleo/llama-cpp-strix-halo at 860c828363988d3e4b3d5c2b701dcba3d7b9f26c. These forks keep the ggml authors' MIT notice, included in `licenses/llama.cpp-MIT.txt`. The original attribution wording is in `NOTICE`.

The Bubo archive does not include FFmpeg or ffprobe. Model weights are not part of the archive or of this bundle.

## Full text sources

The full texts below are copied without rewriting their licence or copyright wording. The date is the day on which the text was copied.

| Licence file | Source | Date copied | Release or revision |
|---|---|---|---|
| [clr-MIT.txt](licenses/clr-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-systems/therock-10.0/projects/clr/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [hip-MIT.txt](licenses/hip-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-systems/therock-10.0/projects/hip/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [hipblas-MIT.txt](licenses/hipblas-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/projects/hipblas/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [hipblaslt-MIT.txt](licenses/hipblaslt-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/projects/hipblaslt/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [rocblas-MIT.txt](licenses/rocblas-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/projects/rocblas/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [rocsolver-BSD-2-Clause.txt](licenses/rocsolver-BSD-2-Clause.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/projects/rocsolver/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [rocsparse-MIT.txt](licenses/rocsparse-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/projects/rocsparse/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [origami-MIT.txt](licenses/origami-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/shared/origami/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [tensile-MIT.txt](licenses/tensile-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/shared/tensile/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [rocroller-MIT.txt](licenses/rocroller-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-libraries/therock-10.0/shared/rocroller/LICENSE.md) | 2026-10-10 | therock-10.0 |
| [rocm-kpack-MIT.txt](licenses/rocm-kpack-MIT.txt) | [source](https://raw.githubusercontent.com/ROCm/rocm-kpack/4ee2886d4b9000bd07af8a2cc73819d2c95077a7/LICENSE) | 2026-10-10 | 4ee2886d4b9000bd07af8a2cc73819d2c95077a7 |
| [comgr-Apache-2.0-LLVM-exception-NCSA.txt](licenses/comgr-Apache-2.0-LLVM-exception-NCSA.txt) | [source](https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/rocm_sdk_core-10.0.0-py3-none-win_amd64.whl) | 2026-10-10 | licence file of COMGR 3.3.0.0 in rocm_sdk_core 10.0.0 |
| [openblas-BSD-3-Clause.txt](licenses/openblas-BSD-3-Clause.txt) | [source](https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/18638c70eff1f0d08e2833b2724deaa128d6a334/LICENSE) | 2026-10-10 | 18638c70eff1f0d08e2833b2724deaa128d6a334 |
| [openssl-Apache-2.0.txt](licenses/openssl-Apache-2.0.txt) | [source](https://raw.githubusercontent.com/openssl/openssl/openssl-3.6.4/LICENSE.txt) | 2026-10-10 | openssl-3.6.4 |
| [curl-curl.txt](licenses/curl-curl.txt) | [source](https://raw.githubusercontent.com/curl/curl/curl-8_22_0/COPYING) | 2026-10-10 | curl-8_22_0 |
| [libwebp-BSD-3-Clause.txt](licenses/libwebp-BSD-3-Clause.txt) | [source](https://raw.githubusercontent.com/webmproject/libwebp/v1.6.0/COPYING) | 2026-10-10 | v1.6.0 |
| [libwebp-PATENTS.txt](licenses/libwebp-PATENTS.txt) | [source](https://raw.githubusercontent.com/webmproject/libwebp/v1.6.0/PATENTS) | 2026-10-10 | v1.6.0 |
| [zlib-Zlib.txt](licenses/zlib-Zlib.txt) | [source](https://raw.githubusercontent.com/madler/zlib/v1.3.2/LICENSE) | 2026-10-10 | v1.3.2 |
| [libpng-libpng-2.0.txt](licenses/libpng-libpng-2.0.txt) | [source](https://raw.githubusercontent.com/pnggroup/libpng/v1.6.58/LICENSE) | 2026-10-10 | v1.6.58 |
| [libjpeg-turbo-IJG-BSD-3-Clause-Zlib.txt](licenses/libjpeg-turbo-IJG-BSD-3-Clause-Zlib.txt) | [source](https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/3.2.0/LICENSE.md) | 2026-11-02 | 3.2.0 |
| [libjpeg-turbo-IJG.txt](licenses/libjpeg-turbo-IJG.txt) | [source](https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/3.2.0/README.ijg) | 2026-11-02 | 3.2.0 (the text of README.ijg is the same in 3.1.4.1 and 3.2.0) |
| [libjpeg-turbo-Zlib.txt](licenses/libjpeg-turbo-Zlib.txt) | [source](https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/3.2.0/simd/nasm/jsimdext.inc) | 2026-11-02 | 3.2.0 (complete leading comment of jsimdext.inc; it is the same in 3.1.4.1 and 3.2.0) |
| [libspng-BSD-2-Clause.txt](licenses/libspng-BSD-2-Clause.txt) | [source](https://raw.githubusercontent.com/randy408/libspng/v0.7.4/LICENSE) | 2026-10-10 | v0.7.4 |
| [pixmaate-windows-layer-MIT.txt](licenses/pixmaate-windows-layer-MIT.txt) | [source](https://raw.githubusercontent.com/pixmaate/gufo/2eb0778ac15d9e1c18c5b2f9e5dbbecc19e20ec2/LICENSE) | 2026-10-10 | 2eb0778ac15d9e1c18c5b2f9e5dbbecc19e20ec2 |
| [gufo-MIT.txt](licenses/gufo-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/LICENSE) | 2026-10-10 | gufo notice tree 4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e |
| [llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt) | [source](https://github.com/ggml-org/llama.cpp/blob/5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a/LICENSE) | 2026-10-10 | 5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a |
| [ds4-MIT.txt](licenses/ds4-MIT.txt) | [source](https://github.com/antirez/ds4/blob/84cc882352757baf628a1776badf7cc54d584e28/LICENSE) | 2026-10-10 | 84cc882352757baf628a1776badf7cc54d584e28 |
| [icu-Unicode-3.0.txt](licenses/icu-Unicode-3.0.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/icu.txt) | 2026-10-10 | copied from the notices of upstream gufo (ICU 78.3) |
| [hipcub-BSD-3-Clause.txt](licenses/hipcub-BSD-3-Clause.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/hipcub.txt) | 2026-10-10 | copied from the notices of upstream gufo (ROCm 7.2.3) |
| [rocprim-MIT.txt](licenses/rocprim-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/rocprim.txt) | 2026-10-10 | copied from the notices of upstream gufo (ROCm 7.2.3) |
| [rocwmma-MIT.txt](licenses/rocwmma-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/rocwmma.txt) | 2026-10-10 | copied from the notices of upstream gufo (ROCm 7.2.3) |
| [composable-kernel-MIT.txt](licenses/composable-kernel-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/composable_kernel.txt) | 2026-10-10 | copied from the notices of upstream gufo (ROCm 7.2.3) |
| [ds4-gb10-MIT.txt](licenses/ds4-gb10-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/DS4-GB10.txt) | 2026-10-10 | 910501e |
| [h3.c-MIT.txt](licenses/h3.c-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/h3.c.txt) | 2026-10-10 | 8974cc055ea9c02fcd14cc27dfda3e1027c05153 |
| [ccv-BSD-3-Clause.txt](licenses/ccv-BSD-3-Clause.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/ccv.txt) | 2026-10-10 | ancestry through h3.c |
| [qwen-templates-Apache-2.0.txt](licenses/qwen-templates-Apache-2.0.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/qwen.txt) | 2026-10-10 | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 / de4b8e4d43b917e7706784d8bb445c9af86a3540 |
| [strix-halo-llamacpp-MIT.txt](licenses/strix-halo-llamacpp-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/licenses/strix-halo-llamacpp.txt) | 2026-10-10 | ce15ecca66e5e5a9aefa1ebc82357cb901330c86 |
| [mikealanni-gufo-MIT.txt](licenses/mikealanni-gufo-MIT.txt) | [source](https://github.com/gufo-org/gufo/blob/4e89b9f7844dcfbd1dafddf9d09e877b22f3cc0e/LICENSE) | 2026-10-10 | f450218743bcaf77e3e1f8fab78f4cd4e4aafae3 |
