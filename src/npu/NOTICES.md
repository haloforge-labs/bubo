# NPU attribution and notices

The adapter, persistent session, BFP encoding and projection geometry are
adapted from the XRT host flow and kernels of the MIT fork mikealanni/gufo at
8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7, and the Apache-2.0 XRT examples.
Copyright (c) 2026 gufo contributors. The repository LICENSE preserves the
complete MIT terms.

The associated mlir-aie sources and artifacts carry these original notices:

> Copyright (C) 2026, Advanced Micro Devices, Inc.
> SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Preserve the full Apache-2.0 WITH LLVM-exception license and attribution when
redistributing those sources or artifacts. The source notice alone does not
establish redistribution rights for a precompiled artifact.

XRT public headers are referenced unmodified at commit
15e6319be8de1e76a6150111a3861729e988fdb5. XRT retains its Apache-2.0 LICENSE
and NOTICE. The build snapshots the generated import library; it does not copy
a vendor DLL, driver or SDK tree.

The fixed artifact pair is external input verified by size and SHA-256.
The adapter uses private buffers, resident model weights and at most four
in-flight runs. Completion precedes result readback. Uncertain DMA retains
buffers and runs for process lifetime. Power mode is queried read-only.
