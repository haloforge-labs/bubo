# P6 BFP contract

Selective source adaptation from `mikealanni/gufo` commit
`8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`, MIT, specifically
`kernels.hip.cpp` BFP group math and A/B/C stream coordinates. No unrelated
GPU optimization, executor, XRT wrapper, AIE source or binary was copied.
Existing MIT and AIE Apache-2.0 WITH LLVM-exception notices remain intact.

Only the artifact's full M=4096, K=2560, N=1024 geometry is supported.
Each group contains a raw shared FP32 exponent and eight signed mantissas:
`q * 2^(e-127) / 64`. Encoding uses nearest-even then clamps to [-128,127].
The fork's exponent-below-7 flush is retained. NaN/Inf and groups which
decode outside finite FP32 are refused, including exponent 254 / -128.
Zero groups are nine zero bytes. This is approximate arithmetic.

CPU EncodeA reads FP16 rows with checked pitch. EncodeB reads exactly the
owned, padded Q8_0 bytes returned by P5 StageQ8, producing an independent
2,949,120-byte B stream. A is 11,796,480 bytes and C is 4,718,592 bytes.
DecodeC validates all stream groups before writing a pitched FP32 output;
valid_cols is 1..1024, with partial groups masked per element. Input/output
storage must be disjoint. Accumulation retains the caller's FP32 addition;
callers must check final finite results before using them.

HIP counterparts compile in ON + ENGINE_ENABLE_HIP only, with no fast-math
addition. They are linked only into the inference engine, not CPU test hosts.
No GPU correctness is established by compilation or CPU tests. First device
validation is G4. Scalar loads/stores avoid inherited vector-alignment and
tail assumptions; their performance is unmeasured.

P7/P8 must keep input, output and the device invalid flag alive through stream
completion and the first-consumer fence. Zero the invalid flag before launch,
check the returned HIP status, then read/check the flag after completion. A
nonzero flag invalidates the entire operation, including any partial output.
HIP success alone is not a result-valid state. Check finite accumulated
outputs as well. Registration, dispatch, fallback, consumer fences, trace
checksums and lane integration belong to P7/P8, not this packet. Use the
single existing private-BO adapter; no HIP/XRT userptr alias is authorized.
