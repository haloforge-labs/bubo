# NPU prefill

Bubo can send part of the prompt processing (prefill) to the Ryzen AI NPU while the GPU does the rest. NPU prefill is on by default when the NPU, its driver and the kernel files are present. It is always off in the `exact` and `stock` profiles. If the NPU cannot start, Bubo uses the GPU path.

## What to expect

Measured on a 70 W Strix Halo laptop (128 GB, Windows 11):

| Prompt | Prefill speed change |
| --- | --- |
| 8K tokens | About 5 to 7 % faster |
| 16K tokens | About 3.5 to 5.5 % faster |

- Longer prompts were not measured.
- The NPU takes only full blocks of 4,096 prompt tokens. Prompts shorter than 4,096 tokens do not use it.
- Generation speed (decode) does not change. The NPU is used for prefill only.
- Agent work changes very little. In an agent session, each turn adds only a few hundred new tokens to a cached conversation, so most turns are below 4,096 new tokens. The NPU helps the first turn of a session and large pasted files or documents.
- The NPU path uses 8-bit block floating point for its part of the work. Output is not byte-exact.
- The NPU keeps about 1.2 GB of weights in memory.

## Requirements

- The AMD NPU driver and runtime (XRT). Bubo does not install them.
- Tested with Windows 11 build 26200, NPU driver 32.0.20102.3930 (2026-05-06) and XRT 32.00.20102.3931. Bubo checks the XRT interface version. If it differs, Bubo does not use the NPU.
- The kernel files `final_4096x2560x1024_128x64x128.xclbin` and `insts_4096x2560x1024_128x64x128.txt` in `tools\npu` in the package. The sources `mm_bfp_mixed.cc` and `n32_core_placed.py` are in the same folder.

## Turn it on or off

- Off: add `--npu-prefill off` to the server command.
- On: `--npu-prefill on`. In the `exact` and `stock` profiles this option stops the server with a startup error.
- The attention q/gate lane is off by default. Turn it on with the environment variable `GUFO_NPU_GATE_SLICES` (1 to 12). See [options](options.md).

## NPU power mode

Bubo reads the power mode and writes it to the log once. To see it, run `C:\Windows\System32\AMD\xrt-smi.exe examine --report platform` and read the line `Power Mode`. The measured gain was the same in Default and Performance mode. To change the mode, run `xrt-smi configure --pmode performance` (or `default`) in an administrator shell. The mode resets when the driver reloads.

## Licences and known issues

The NPU kernel and design come from mikealanni/gufo (`tools/npu`, MIT) and are derived from AMD mlir-aie programming examples (Apache-2.0 WITH LLVM-exception, Copyright (C) 2026 Advanced Micro Devices, Inc.). The licence texts are in the `licenses` folder.

The kernel file ships as the fork built it. The compiler toolchain that built it is not recorded. We plan a rebuild with the open-source Peano compiler in a later release.
