# Linear z prefill

A ready ServerPrefill selects the trailing linear projection slices for
complete 4096-token groups. The supported trunk has hidden dimension 2560,
10240 QKV rows and 6144 z rows. Slices 1..6 retain fused GPU convolution;
7..12 also offload trailing QKV channels and use the unfused GPU prefix.

Weights are staged into independent CPU BFP streams before GPU upload and
registered once with the persistent private-buffer adapter. The HIP bridge
encodes activations in stream order. The serialized worker dispatches XRT
runs and decodes completed results before the GPU consumer.

An invalid result rejects the current prefill and disables future NPU work.
Uncertain DMA retains resources for process lifetime. Incompatible shapes,
decode and partial groups use the GPU before any NPU submission.

CPU and fake-device tests cover selection, layout, lifetime and consumer
ordering. The geometry and codec adapt the MIT mikealanni/gufo fork;
see NOTICES.md for attribution.
