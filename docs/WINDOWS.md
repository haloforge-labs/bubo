# Windows (native, gfx1151)

Bubo builds and runs natively on Windows 11 x64 against AMD's TheRock ROCm
distribution for gfx1151 (Ryzen AI Max+ 395 / Radeon 8060S). No WSL. Linux
builds are unaffected: the port lives in `compat/win32`, Windows-only files and
`_WIN32` branches.

## Prebuilt

The archive is `bubo-v<product-version>-windows-x64-gfx1151.zip`: `bubo.exe`,
the matching lazy engine and non-AMD vcpkg dependencies in `bin\`, plus
`start.cmd`. Prerequisite: VC++ v14 x64 Redistributable (not shipped).
Install the VC++ v14 x64 Redistributable
(https://aka.ms/vs/17/release/vc_redist.x64.exe) before the PREPARE steps.
`start.cmd` (through `tools\windows\vc-runtime.ps1`) and
`tools\windows\start.ps1` check the x64 runtime registration or both
`vcruntime140.dll` and `msvcp140.dll` in native System32, and print that link
if missing. There is no checking script entry point for prepare. `bubo.exe`
links the CRT dynamically: direct `bubo.exe prepare` with the CRT missing
shows a Windows loader error before `main()`. Its in-process prerequisite
check remains a second line of defence when only some CRT parts are missing.
Install the prerequisite before other direct executable/script entry points,
including `run-flash-next.ps1`. Bubo never downloads the redistributable.
Inference also needs a Strix Halo (gfx1151) machine and a current AMD driver.
AMD runtime files and models are acquired separately by `bubo prepare`;
`README.txt` lists the steps. `tools\windows\package.ps1` makes the zip from
a release build and checks both the staged folder and final ZIP contents.
Its product version comes from `tools/release/product_version.txt` and must
match the configured build metadata.

The Bubo brand defaults to `GUFO_ENABLE_OPENMP=OFF`. The CPU/Bubo build and
its package inventory/import closure show no libomp in our Bubo archive;
no libomp license is needed for those absent files. The package gate rejects
libomp files/imports in our archive. Production HIP closure remains OPEN
and must be re-checked. The gate cannot inspect the ROCm wheels that
`bubo prepare runtime` downloads separately from AMD. Those are AMD's files;
the supplied runtime inventory lists `libomp140.x86_64.dll` in
`tools/release/runtime-notices/OPEN-ITEMS.md`, with source/license provenance
still OPEN. That DLL is not in the selected `runtime-members.tsv` extraction
list; its presence in the exact wheels is not established by this CPU check.
Release CRT and UCRT system imports are permitted; CRT files, including
`concrt*.dll`, `vccorlib`, `vcomp*.dll`, `ucrtbase.dll` and
`api-ms-win-crt-*.dll`, are forbidden inside the archive. Debug CRT imports
remain denied.

## Prerequisites

| Piece | Default location | Notes |
| --- | --- | --- |
| TheRock ROCm 10.0.0 (Windows, gfx1151) | `C:\TheRock\build` | [`therock-dist-windows-gfx1151-10.0.0.tar.gz`](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz), SHA-256 `1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`; extract so that `C:\TheRock\build\bin` exists. Its clang compiles C, C++ and HIP |
| vcpkg | `C:\vcpkg` | `vcpkg install icu curl openssl libpng libjpeg-turbo libwebp pkgconf --triplet x64-windows` |
| Visual Studio Build Tools | any | "Desktop development with C++": MSVC STL + Windows SDK only; tested with VS 18 (MSVC 14.51) |
| CMake 3.21+ and Ninja | `PATH`, or Ninja at `C:\tools\ninja` | |
| Dedicated GPU memory | AMD Software > Performance > Tuning > Variable Graphics Memory | 96 GB for Qwen3.8-Flash-Next at 256K context; see [Memory](#memory) |

The port is validated on TheRock 10.0.0. Other releases usually build, but a
newer clang can round fused kernels differently (see [Test status](#test-status)).

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # what is missing, what fits, how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # release -> build\release\bubo.exe
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test   # + tests and tools
```

`build.ps1` imports the MSVC environment, configures with TheRock clang and the
vcpkg toolchain, builds, and copies the ROCm and vcpkg runtime DLLs plus the
hipBLASLt/rocBLAS kernel libraries next to `bubo.exe`, so `build\release` runs
as it is.

`check.ps1` covers the whole setup, not only the build: the toolchain; the
build (runtime files, whether HIP sees the GPU, whether it is older than the
source); the dedicated GPU memory; every model it finds, with a memory
estimate and any missing draft or vision file (with the download command);
and runtime traps such as a busy port, a server that is already running, or
`GUFO_*` switches left in the shell.

## Starting a server

In the release folder, `start.cmd` checks the VC++ runtime and then opens the
app (`bin\Bubo.Control.Web.exe`). In a folder without the app it runs
`bin\bubo.exe serve` with the same arguments, for example
`start.cmd --port 8090`.

In a source checkout, the full launcher asks question by question:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1          # pick a model; Enter = recommended
powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1 -Last    # the previous choice again
powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1 -List    # what is here, and what fits
```

`start.ps1` looks through the Hugging Face cache, a `models` folder in or
beside the checkout, LM Studio's folders and any `-ModelDir` (remembered). It
reads each GGUF header to tell targets, MTP and DFlash2 drafts and vision
sidecars apart, lists what `bubo serve llm` runs (Qwen3.8-Flash-Next,
Qwen3.8-27B, Qwen3.6-35B-A3B and its fine-tunes) at the context that fits the
dedicated GPU memory, pairs each with its draft and `mmproj`, and offers the
speed modes those files allow, fastest measured first. Enter starts the plan
it shows; `c` changes context, sessions, thinking, draft file, port, local
network access (with an API key), a log file and extra options. It prints the
full `bubo` command, runs it in the same window, offers to ignore `GUFO_*` /
`A3B_*` switches left in the shell, and restores every variable it touched.
Choices are kept in `%LOCALAPPDATA%\gufo\start.json` (never the API key);
`-DryRun` prints the command only.

## Running Qwen3.8-Flash-Next

Download the qualified files with the Hugging Face CLI
(`pip install -U huggingface_hub`):

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"
powershell -ExecutionPolicy Bypass -File tools\windows\run-flash-next.ps1
```

`start.ps1` does the same interactively; `run-flash-next.ps1` is the
scriptable form. It finds the files in the Hugging Face cache and serves an
OpenAI-compatible API on `http://127.0.0.1:8080/v1` (thinking on, adaptive MTP,
the model's own sampler, images through the BF16 `mmproj`). `-Think off`, `-Draft mtp3|off`, `-Context N` and
`-Mode bench` (pp2048/tg128 at depths 0..128K) are the common variations. The
server is ready when the log shows `event=load_completed`.

The script picks the newest Hugging Face snapshot that holds each file, so
files downloaded at different revisions are found. `GUFO_*` environment
variables (`GUFO_PLATFORM_TUNING`, diagnostics) reach the server; the script
lists any set in the calling shell.

## Other text models

`start.ps1` offers these as well. `bubo serve llm --model PATH` also serves
Qwen3.8-27B (with a DFlash2 draft,
see [its guide](models/qwen3.8-27b/README.md)) and Qwen3.6-35B-A3B
(`qwen35moe` GGUFs, see [its guide](models/qwen3.6-35b-a3b/README.md)). Both support Windows.

## Memory

96 GB machines: use the lower-memory model. Set Variable Graphics Memory (AMD Software: Adrenalin Edition > Performance > Tuning) to the largest value offered (72 GB if available, at least 64 GB), then restart Windows. At 72 GB, Bubo uses the default 200,000-token context for one session. At 64 GB it does too when about 16 GB of system memory is free, and picks a smaller context when less is free. These values are calculated: Bubo has not yet run on a 96 GB machine.

The lower-memory model uses ISTA-DASLab GSQ-RCO experts and needs about 66 GiB instead of about 91 GiB. The decode twin head makes writing faster.

## Test status

- All Flash-Next operator tests pass, plus 86/92 CPU-labeled tests.
- Not ported: Linux-only Python dev tools (`h3_profile*`, `qwen27b.tools`,
  TTS reference verification needs numpy), and `video_jobs_test` needs
  symlink privileges (Developer Mode).
- `qwen_ssm_ops_test` and `minimax_h3_dit_analytic` assert bit-identical
  fused vs. reference kernels; TheRock's newer clang rounds differently.
