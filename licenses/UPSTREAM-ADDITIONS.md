# Additional upstream notices

These notices supplement the byte-exact published Bubo third-party notices.
The native platform, audio and embedding changes from upstream gufo retain
their additional dependencies and credits here.

| Dependency | Use | License | Upstream version/source |
|---|---|---|---|
| libsndfile | Linked audio decoding | LGPL-2.1-or-later | 1.2.2; [libsndfile](https://github.com/libsndfile/libsndfile) |
| libsoxr | Linked audio resampling | LGPL-2.1-or-later | 0.1.3; [SoX Resampler](https://sourceforge.net/projects/soxr/) |
| FLAC | Audio decoding through libsndfile | BSD-3-Clause (Xiph.Org) | 1.5.0; [license](libflac.txt) |
| libogg | Audio decoding through libsndfile | BSD-3-Clause (Xiph.Org) | 1.3.6; [license](libogg.txt) |
| libvorbis | Audio decoding through libsndfile | BSD-3-Clause (Xiph.Org) | 1.3.7; [license](libvorbis.txt) |
| Opus | Audio decoding through libsndfile | BSD-3-Clause | 1.6.1; [license](opus.txt) |
| mpg123 | MP3 decoding through libsndfile | LGPL-2.1 | 1.33.7; [license](mpg123.txt) |
| LAME | MP3 encoding through libsndfile | LGPL-2.0 (GNU Library General Public License, version 2) | 3.100; [license](mp3lame.txt) |
| LLVM OpenMP | Windows runtime built from upstream source; Microsoft-compatible DLL naming | Apache-2.0 WITH LLVM-exception | 18.1.8 (`3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff`); [license](llvm-openmp.txt) |

The [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp)
fork also informed upstream optimization work; its MIT notice is pinned at
`5081bc747b2d433964d66c37b80f350b248fb2fe` and the shared ggml notice is in
[llama.cpp.txt](llama.cpp.txt).

EmbeddingGemma 2 reads the text backbone from `google/embeddinggemma-2`
revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, whose model card
declares Apache-2.0. Weights are neither shipped nor automatically downloaded.
Its development references, Transformers and sentence-transformers, are
Apache-2.0.
