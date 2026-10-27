# Provenance of the model files

`bubo prepare model` downloads eight files. Each file is pinned to a repository, a revision and a SHA-256 value. The program checks the size and the SHA-256 value of each file.

| File | Repository | Revision | Size (bytes) | SHA-256 |
| --- | --- | --- | ---: | --- |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf | huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF | 73e9eb7c69fdbf667def63e233e70845e7d5ac0b | 10,946,752 | 290a31ca20b74ed2c21b257d378a5678d821ad9fdd8feb1fc52c719761668903 |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf | huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF | 73e9eb7c69fdbf667def63e233e70845e7d5ac0b | 49,859,583,296 | 0ebf70d076e0e98f74dc414a90b92c67d7267876ae524bc7799cf463e27b754b |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf | huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF | 73e9eb7c69fdbf667def63e233e70845e7d5ac0b | 49,376,141,664 | e9b12f44e13e60d6ef9396449f7b4356b2e2911222b8183a596830c2418db19a |
| UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf | huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF | 73e9eb7c69fdbf667def63e233e70845e7d5ac0b | 12,087,983,680 | 983901b3e47686f56f5fcec87ba8c51e7e2c6996277dd77bc3fc804a1d563da3 |
| MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 2,786,568,256 | 5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6 |
| mmproj-BF16.gguf | unsloth/Qwen3.8-Flash-Next-GGUF | 766911a6b7369840a91dbcd95f9f997acaab6cd6 | 907,542,944 | 2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355 |
| twin/iq4-nofullattn-nohc.gguf | HaloForgeLabs/bubo-qwen3.8-flash-next | bb72204d4f9bdd1568c225cce3bf878673009780 | 1,230,788,768 | 18e0831234551a4d14a0407c24598fb02fdb1fe83a20a4fcff23f410399c66c2 |
| draft_head/draft-out-q4k.gguf | HaloForgeLabs/bubo-qwen3.8-flash-next | bb72204d4f9bdd1568c225cce3bf878673009780 | 357,581,248 | 6867948dc9f5c467818daa4281c4878dc64b53cd1bba9d16970fd77cd1656777 |

The mmproj file is saved as mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf in the model folder. The sizes and SHA-256 values above come from the Hugging Face file listings at these revisions.

## How the two HaloForge Labs files were made

- Dense twin (`iq4-nofullattn-nohc.gguf`): 252 selected tensors that are Q8_0 in the Huihui source are quantised to IQ4_XS (204) and IQ4_NL (48) with the importance matrix `imatrix_unsloth.gguf_file` from unsloth/Qwen3.8-Flash-Next-GGUF. Full-attention projections and HC projections stay 8-bit in the main model path.
- Draft head (`draft-out-q4k.gguf`): the Huihui `output.weight` quantised to Q4_K with static quantisation (the importance matrix is not applied to this tensor).
- Both files match only the Huihui UD-Q4_K_XL shards listed above.

## Licence

The Qwen Community License 1.0 (file LICENSE) applies to all eight files. See NOTICE.
