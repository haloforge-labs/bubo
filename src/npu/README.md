# Windows NPU prefill

The NPU path offloads eligible Q8_0 projections in complete 4096-token
prefill groups to XDNA2 through a persistent XRT context. Decode, partial
groups and incompatible model shapes use the GPU.

Default and subagents profiles request NPU prefill automatically. Exact and
stock keep it OFF and reject an explicit ON. --npu-prefill off disables it.
If startup cannot verify artifacts or XRT, create the device or complete the
startup heartbeat, one warning names the reason and the process uses the GPU.
A build without NPU support reports that reason once and uses the GPU.

GUFO_NPU_DIR overrides the artifact directory. Without an override, the
directory is the executable folder/../tools/npu, matching the packaged layout.
Lane defaults are GUFO_NPU_SLICES=6, GUFO_NPU_GATE_SLICES=0 and GUFO_NPU_SHEXP=1.
Timing defaults are GUFO_NPU_TIMEOUT_MS=8000, GUFO_NPU_KEEPALIVE=1 and
GUFO_NPU_KEEPALIVE_MS=1000. Power mode is queried once, read-only, after startup;
an unavailable query is ignored.

Windows builds with GUFO_ENABLE_NPU include the runtime through delay imports.
The runtime DLL identity and version tuple are verified before device creation.
The adapter verifies the pinned instruction and xclbin files before loading XRT.
No vendor runtime is copied into the executable.

The adapter, BFP codec and projection geometry adapt the MIT fork
mikealanni/gufo. See NOTICES.md for attribution and external artifact notices.
