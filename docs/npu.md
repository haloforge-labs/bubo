# NPU prefill

Bubo can send part of the prompt processing (prefill) to the Ryzen AI NPU while the GPU does the rest. NPU prefill is off by default. To turn it on, add `--npu-prefill on` to the server command (see [Turn it on or off](#turn-it-on-or-off)). It needs the NPU, its driver and the kernel files. It is always off in the `exact` and `stock` profiles. If the NPU is on and cannot start, Bubo uses the GPU path.

## What to expect

After the prompt-reading speed-ups in v0.1.1, NPU prefill adds less than 1 % on a 70 W Strix Halo laptop. For this reason it is off by default.

- The NPU takes only full blocks of 4,096 prompt tokens. Prompts shorter than 4,096 tokens do not use it.
- Generation speed (decode) does not change. The NPU is used for prefill only.
- Agent work changes very little. In an agent session, each turn adds only a few hundred new tokens to a cached conversation, so most turns are below 4,096 new tokens. The NPU helps the first turn of a session and large pasted files or documents.
- The NPU path uses 8-bit block floating point for its part of the work. Output is not byte-exact.
- The NPU keeps about 1.2 GB of weights in memory.

## Requirements

- The AMD NPU driver and runtime (XRT). Bubo does not install them.
- Tested with Windows 11 build 26200, NPU driver 32.0.20102.3930 (2026-05-06) and XRT 32.00.20102.3931. Bubo uses the NPU only with AMD XRT runtime 32.0.20102.3931. With any other runtime, Bubo writes a warning, reports the NPU as faulted, and uses the GPU. The output does not change.
- The kernel files `final_4096x2560x1024_128x64x128.xclbin` and `insts_4096x2560x1024_128x64x128.txt` in `tools\npu` in the package. The sources `mm_bfp_mixed.cc` and `n32_core_placed.py` are in the same folder.

## Turn it on or off

- Off: this is the default. `--npu-prefill off` also sets it.
- On: add `--npu-prefill on` to the server command. In the `exact` and `stock` profiles this option stops the server with a startup error.
- The attention q/gate lane is off by default. Turn it on with the environment variable `GUFO_NPU_GATE_SLICES` (1 to 12). See [options](options.md).

## NPU power mode

Bubo reads the power mode and writes it to the log once. To see it, run `C:\Windows\System32\AMD\xrt-smi.exe examine --report platform` and read the line `Power Mode`. The measured gain was the same in Default and Performance mode. To change the mode, run `xrt-smi configure --pmode performance` (or `default`) in an administrator shell. The mode resets when the driver reloads.

## Licenses

The NPU kernel and design come from mikealanni/gufo (`tools/npu`, MIT) and are derived from AMD mlir-aie programming examples (Apache-2.0 WITH LLVM-exception, Copyright (C) 2026 Advanced Micro Devices, Inc.). The license texts are in the `licenses` folder.
