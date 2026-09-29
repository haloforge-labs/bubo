# Windows (native, gfx1151)

Gufo builds and runs natively on Windows 11 x64 against AMD's TheRock ROCm
distribution for gfx1151 (Ryzen AI Max+ 395 / Radeon 8060S). No WSL. Linux
builds are unaffected: the port lives in `compat/win32`, Windows-only files and
`_WIN32` branches.

## Prerequisites

| Piece | Default location | Notes |
| --- | --- | --- |
| TheRock ROCm 10.0.0 (Windows, gfx1151) | `C:\TheRock\build` | [`therock-dist-windows-gfx1151-10.0.0.tar.gz`](https://stable.repo.amd.com/rocm/core/tarball/therock-dist-windows-gfx1151-10.0.0.tar.gz), SHA-256 `1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`; extract so that `C:\TheRock\build\bin` exists. Its clang compiles C, C++ and HIP |
| vcpkg | `C:\vcpkg` | `vcpkg install icu curl openssl libpng libjpeg-turbo --triplet x64-windows` |
| Visual Studio Build Tools | any | "Desktop development with C++": MSVC STL + Windows SDK only; tested with VS 18 (MSVC 14.51) |
| CMake 3.21+ and Ninja | `PATH`, or Ninja at `C:\tools\ninja` | |
| Dedicated GPU memory | AMD Software > Performance > Tuning > Variable Graphics Memory | 96 GB for Qwen3.8-Flash-Next at 256K context; see [Memory](#memory) |

The port is validated on TheRock 10.0.0. Other releases usually build, but a
newer clang can round fused kernels differently (see [Test status](#test-status)).

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1   # what is missing, what fits, how to fix it
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1   # release -> build\release\gufo.exe
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test   # + tests and tools
```

`build.ps1` imports the MSVC environment, configures with TheRock clang and the
vcpkg toolchain, builds, and copies the ROCm and vcpkg runtime DLLs plus the
hipBLASLt/rocBLAS kernel libraries next to `gufo.exe`, so `build\release` runs
as it is.

`check.ps1` covers the whole setup, not only the build: the toolchain; the
build (runtime files, whether HIP sees the GPU, whether it is older than the
source); the dedicated GPU memory; every model it finds, with a memory
estimate and any missing draft or vision file (with the download command);
and runtime traps such as a busy port, a server that is already running, or
`GUFO_*` switches left in the shell.

## Starting a server

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1          # pick a model; Enter = recommended
powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1 -Last    # the previous choice again
powershell -ExecutionPolicy Bypass -File tools\windows\start.ps1 -List    # what is here, and what fits
```

`start.ps1` looks through the Hugging Face cache, a `models` folder in or
beside the checkout, LM Studio's folders and any `-ModelDir` (remembered). It
reads each GGUF header to tell targets, MTP and DFlash2 drafts and vision
sidecars apart, lists what `gufo serve llm` runs (Qwen3.8-Flash-Next,
Qwen3.8-27B, Qwen3.6-35B-A3B and its fine-tunes) at the context that fits the
dedicated GPU memory, pairs each with its draft and `mmproj`, and offers the
speed modes those files allow, fastest measured first. Enter starts the plan
it shows; `c` changes context, sessions, thinking, draft file, port, local
network access (with an API key), a log file and extra options. It prints the
full `gufo` command, runs it in the same window, offers to ignore `GUFO_*` /
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

`start.ps1` offers these as well. `gufo serve llm --model PATH` also serves
Qwen3.8-27B (with a DFlash2 draft,
see [its guide](models/qwen3.8-27b/README.md)) and Qwen3.6-35B-A3B
(`qwen35moe` GGUFs, see [its guide](models/qwen3.6-35b-a3b/README.md)). Both
carry Windows-specific decode work: VRAM-resident 27B weights and verify
kernels tuned for the Windows compiler, and A3B's own runtime.

## How the port works

- `compat/win32/include` shadows the POSIX headers Gufo uses (`unistd.h`,
  `sys/mman.h`, `fcntl.h` additions, sockets, `spawn.h`, ...) and
  `compat/win32/posix.cpp` implements them on Win32. The directory is on the
  include path only on Windows. `_CRT_DECLARE_NONSTDC_NAMES=0` keeps the
  UCRT's own POSIX aliases out of the way.
- Semantics that matter:
  - `O_DIRECT` is `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, so the
    weight uploader keeps its parallel aligned direct reads.
  - NTFS serializes direct reads of a file that is also mapped, and Gufo maps
    every model shard: 32 threads of 4 KiB direct reads fall from ~70K to ~5K
    reads/s. Large sequential reads barely notice, but the Flash-Next n-gram
    table reader does ~16 random row reads per prompt token. On Windows it
    opens the table with `O_CONCURRENT_RANDOM` (a cached overlapped handle,
    68K reads/s at 32 readers, 128K at 128) and runs 128 readers. That took
    real-text prefill from ~300 to ~680-840 tokens/s.
  - `open("/proc/self/fd/N")` is `ReOpenFile`: an independent handle to the
    same file, immune to path replacement, exactly as on Linux.
  - `flock` locks one byte far past EOF (Windows locks are mandatory) and
    `close` unlocks explicitly (Windows releases locks asynchronously).
  - `renameat` uses POSIX-semantics rename; `st_ctime` is NTFS ChangeTime.
  - `posix_spawnp` passes descriptors through the CRT `lpReserved2` block, so
    FFmpeg's `pipe:N` inputs work.
  - Uncaught C++ exceptions and crashes print a Linux-style message (with
    `what()`) or `module+offset` for `llvm-symbolizer`.
- `src/core/platform/socket.hpp` covers the socket differences call sites
  cannot hide: `closesocket`, Winsock errors to `errno`, millisecond
  `SO_RCVTIMEO`, lingering close (Windows discards data queued before a reset),
  and polled websocket reads (a Winsock `SD_RECEIVE` resets the connection).
- `compat/win32/hip_include/cmath` works around MSVC 14.5x declaring
  `isgreater` & co. `constexpr` (hence `__host__ __device__` under HIP), which
  collides with clang's HIP math declarations.
- Executables reserve an 8 MiB stack, Linux's default (Windows: 1 MiB).

## Windows-validated switches

Some behaviour changes were developed and measured on Windows only. They live
in `src/core/platform/tuning.hpp`: each is on by default on Windows and off
elsewhere, so Linux runs exactly the code it ran before. The decode switches
change timing and memory placement only. `GUFO_PLATFORM_TUNING` overrides the
defaults on any platform, for
example `GUFO_PLATFORM_TUNING=+fast_sampling` to try one on Linux, or
`GUFO_PLATFORM_TUNING=none` to run the Linux path on Windows.

| Switch | What it does | Measured on Windows |
| --- | --- | --- |
| `copy_kernels` | Qwen3.8-Flash-Next decode copies (uploads, downloads, rollback, hidden carry) as small kernels instead of `hipMemcpyAsync` | A copy-engine hand-off costs ~30 us per graph node and ~150 us per compute/copy switch; rollback 8.5 -> ~2 ms, probe +7-8% with `recorded_rollback` |
| `recorded_rollback` | The speculative state rollback replays as one recorded graph per kept length | (with `copy_kernels`, above) |
| `keep_rollback_rows` | Rollback rows survive session resets and snapshot restores | Verify and rollback graphs are captured once per session instead of on every request |
| `flush_before_wait` | `hipStreamQuery` before blocking on n-gram rows: HIP on Windows only submits queued launches at a flush | Draft-to-verify GPU idle 1.9 -> 1.15 ms per cycle |
| `flag_waits` | Decode-sized passes wait on a completion flag in coherent pinned memory instead of `hipStreamSynchronize` | `hipStreamSynchronize` returns ~0.4 ms late after a large graph; ~0.6 ms per cycle with `verify_graph_candidates` |
| `fast_sampling` | Sampled decode reads GPU-selected top-64 candidate lists (`SamplerState::DistributionFromTop`) instead of full vocabulary rows, whenever the list provably holds the whole top-k | 3-6% less time per decode cycle |
| `verify_graph_candidates` | With `fast_sampling`, the verify graph selects the candidate lists itself | (with `flag_waits`, above) |
| `fused_hc_down` | The HC mixer down projection fuses its SiLU scale into the GEMV write and prefetches deeper for 2-8 tokens | One launch fewer per mixer; part of the 3-6% above |
| `hot_first_upload` | Weights read on every token are uploaded before the routed experts | GEMVs on memory allocated late run 6-22% slower on Windows; placement only |

Checked on Windows, all decode switches on against all off (the Linux path):
`gufo bench --logit-eval` dumps are bit-identical at all 4418 positions, and a
fixed-seed sampled decode produces byte-identical text. All off decodes
7.6-9.5% slower on Windows (sampled probe, prose / code / reasoning 32.0 /
34.3 / 44.1 against 34.5 / 37.2 / 48.3 tok/s).

## Memory

On Windows, HIP allocations are not limited to the dedicated carve-out that
the BIOS or AMD Software reserves; `hipMalloc` succeeds up to ~110 GiB on a
128 GB machine, with the rest backed by shared system memory. Measured on a
128 GB Strix Halo:

- `hipMemGetInfo` double-counts memory beyond the carve-out (free drops 2 GiB
  per GiB allocated), which made Gufo refuse to size its sessions.
  `src/core/platform/device_memory.cpp` reports the process's WDDM budget
  (DXGI `QueryVideoMemoryInfo`, adapter matched by the HIP device LUID)
  instead.
- Backing new GPU memory costs ~28 GiB/s inside the carve-out but only
  ~0.63 GiB/s beyond it (Windows commits and maps system RAM page by page),
  which also halves concurrent disk reads during the load.
- Weights beyond the carve-out are also slower to read: the decode
  matrix-vector kernels run 6-22% slower on them.

Qwen3.8-Flash-Next at 256K context needs ~89 GiB of device memory (weights,
KV cache, recurrent state, scratch). With a 96 GB carve-out all of it is
dedicated: the load takes ~30 s and decode runs at full speed. A 64 GB
carve-out works, but the load takes ~2 minutes and decode is slower.

## Test status

- All Flash-Next operator tests pass, plus 86/92 CPU-labelled tests.
- Not ported: Linux-only Python dev tools (`h3_profile*`, `qwen27b.tools`,
  TTS reference verification needs numpy), and `video_jobs_test` needs
  symlink privileges (Developer Mode).
- `qwen_ssm_ops_test` and `minimax_h3_dit_analytic` assert bit-identical
  fused vs. reference kernels; TheRock's newer clang rounds differently.
