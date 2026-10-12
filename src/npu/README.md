# Windows XRT layer

P1 adds build and lazy-loading infrastructure only. The production path has no
caller of `EnsureXrtRuntime` yet. P4 must authorize immutable effective ON before
P2 calls it, then creates devices with the same pinned XRT headers/API as the
CPU tooling host. Never enable this boundary from an environment variable.

`GUFO_ENABLE_NPU=OFF` does not define NPU targets, includes, symbols, imports or
compile definitions. ON embeds the loader in the inference module and uses only
delay imports for XRT. The delay hook refuses an API call before explicit
verified loading; it never delegates XRT path search to the delay helper.

The single layer reuses the completed `npu-tooling-sol` job's Apache-2.0 XRT
headers at commit `15e6319be8de1e76a6150111a3861729e988fdb5`, generated version
header and locally generated import library by explicit CMake paths. No vendor
runtime is copied. `xrt_runtime_policy.cpp` adapts `host.cpp`'s `runtimeCheck`;
the Windows implementation retains its absolute System32 path, SHA256, PE
version and five unsigned version checks. A read handle prevents writes and
replacement through verification/load. The module is retained for process
lifetime after successful verification. No static initializer opens the DLL.

The tooling Engine/private-BO API remains the sole adapter input for P2, which
will selectively extract it; P1 does not add a second GEMM/XRT device wrapper.
The fake loader tests have no XRT headers/imports/device APIs.

Tooling provenance is in its `REPORT-20261011-2104.md` and `THIRD_PARTY.md`.
Fork-derived code remains MIT (repository LICENSE). Future copied AIE code and
artifacts must preserve Apache-2.0 WITH LLVM-exception notices; none is copied
in P1. XRT itself retains its Apache-2.0 LICENSE in the referenced checkout.
Compile/link success is not device ABI, artifact or firmware qualification.
