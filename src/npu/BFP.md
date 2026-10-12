# BFP encoding

The group math and stream coordinates adapt kernels.hip.cpp from the MIT
mikealanni/gufo fork. See NOTICES.md for MIT and external mlir-aie
Apache-2.0 WITH LLVM-exception attribution.

The fixed geometry is M=4096, K=2560, N=1024. Each group has a shared
exponent and eight signed mantissas: q * 2^(e-127) / 64.
Encoding rounds to nearest even, clamps to [-128,127] and retains the
exponent-below-seven flush. Nonfinite inputs and decoded results are rejected.
This arithmetic is approximate.

EncodeA accepts pitched FP16 rows; EncodeB accepts padded CPU Q8_0 rows.
The A, B and C streams occupy 11,796,480, 2,949,120 and 4,718,592 bytes.
DecodeC checks stream groups before writing pitched FP32 output.
Input and output storage must be disjoint.

HIP storage and the invalid flag remain alive through stream completion.
The invalid flag is reset before launch and checked before results are consumed.
CPU and fake-device tests cover layout and validation; they do not establish
hardware numerical correctness or performance.
