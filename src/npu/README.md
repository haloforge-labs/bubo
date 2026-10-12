# Windows NPU prefill

The NPU path offloads eligible Q8_0 projections in complete 4096-token
prefill groups to XDNA2 through a persistent XRT context. Decode, partial
groups and incompatible model shapes use the GPU.

Windows builds with GUFO_ENABLE_NPU include the runtime through delay imports.
The runtime DLL identity and version tuple are verified before device creation.
The adapter verifies the pinned instruction and xclbin files before loading XRT.
No vendor runtime is copied into the executable.

The adapter, BFP codec and projection geometry adapt the MIT fork
mikealanni/gufo. See NOTICES.md for attribution and external artifact notices.
