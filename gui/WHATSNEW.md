# What's new in 0.1.2

1. **Run it in a lot less memory (ISTA quant support).** Bubo loads the ISTA-DASLab GSQ-RCO files of Qwen3.8 Flash Next (IQ3_S, IQ3_XXS, IQ2_XS and the Coder) as published, with no conversion.
   - IQ3_S at 55 GB matches the full-precision model on ISTA's own benchmarks.
   - IQ2_XS goes down to 39 GB, only 4 points (4 %) below full precision.
   - New download: Qwen3.8 Flash Next Coder, 61.2 GB, about 32 GB of GPU memory for its weights with the draft file. It writes 69 tokens/s (76–78 on code).
   - The lower-memory build writes 19–28 % faster than in 0.1.1.
2. **Fast mode for prompts of 4,096 or more new tokens.** Server page, Prefill mode: Fast. Such prompts read up to 16 % faster (1,718 tokens/s on a 22K-token prompt). Shorter prompts read as normal. Writing speed is unchanged. For Qwen3.8 Flash Next (not the Coder).
3. **Smaller KV cache (Q8).** Server page, KV cache. The KV cache takes about half the memory, so more or longer conversations fit.
4. **In-engine abliteration (Qwen3.8 Flash Next).** Server page. Off by default.

If you use the disk cache, the first start of 0.1.2 deletes the disk-cache files of 0.1.1.
