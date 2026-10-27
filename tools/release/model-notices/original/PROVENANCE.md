# Provenance of the model files (original model)

The original-model option of `bubo prepare` downloads eight files. Each file is pinned to a repository, a revision and a SHA-256 value. The program checks the size and the SHA-256 value of each file.

| File | Repository | Revision | Size (bytes) | SHA-256 |
| --- | --- | --- | ---: | --- |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 10,946,624 | 4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082 |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 49,859,583,136 | 3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9 |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 49,376,141,504 | 56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3 |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 12,087,983,520 | 753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a |
| MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 2,786,568,256 | 5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6 |
| mmproj-BF16.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 907,542,944 | 2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355 |
| twin-base/iq4-nofullattn-nohc.gguf | HaloForgeLabs/bubo-qwen3.8-flash-next | 0a3c343bec2095f43f2906371c83e4faa8a47da1 | 1,230,788,768 | d6aa3c0425247aca2ae37680a96a401ba09431addc552a1ee60af3a51927e13e |
| draft_head/draft-out-q4k.gguf | HaloForgeLabs/bubo-qwen3.8-flash-next | 0a3c343bec2095f43f2906371c83e4faa8a47da1 | 357,581,248 | 6867948dc9f5c467818daa4281c4878dc64b53cd1bba9d16970fd77cd1656777 |

The mmproj file is saved as mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf in the model folder. The sizes and SHA-256 values above come from the Hugging Face file listings at these revisions.

## How the two HaloForge Labs files were made

- Dense twin (`twin-base/iq4-nofullattn-nohc.gguf`): 252 selected tensors that are Q8_0 in the Unsloth source are quantised to IQ4_XS (204) and IQ4_NL (48) with the importance matrix `imatrix_unsloth.gguf_file` from unsloth/Qwen3.8-Flash-Next-GGUF. Full-attention projections and HC projections stay 8-bit in the main model path.
- Draft head (`draft_head/draft-out-q4k.gguf`): the `output.weight` tensor of the Unsloth UD-Q4_K_XL shards quantised to Q4_K with static quantisation (the importance matrix is not applied to this tensor). This is the same file that the default package uses, because `output.weight` has the same bytes in both models.
- Both files match only the Unsloth UD-Q4_K_XL shards listed above.

## Licence

The Qwen Community License 1.0 (file LICENSE) applies to all eight files. See NOTICE.
