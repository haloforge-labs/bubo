# Gemma 4 vision: reference check against llama.cpp

`llama_vision.cpp` runs llama.cpp b11316's own image pipeline (decode, mtmd preprocessing, CLIP graph) on one image and
writes the final embeddings (`G4VE`, read by `gemma4_vision_probe --compare`) and, with `--dump-dir`, the layer-0 stages and
every layer output under Bubo's stage names. It loads llama.cpp's DLLs at run time and takes only the struct layouts from
the llama.cpp headers. `--rgb FILE W H` feeds raw RGB8 pixels at the encoder size, so Bubo and llama.cpp see the same
pixels; `--gpu` and `--no-fa` change the backend.

Build (ROCm clang, headers from the b11316 tree):

    clang++ -std=c++17 -O1 -I<llama>/tools/mtmd -I<llama>/include -I<llama>/ggml/include llama_vision.cpp -o llama_vision.exe

Compare: `gemma4_vision_probe --mmproj M --file IMG --gpu --no-reference --dump-dir A` then
`python compare_embeddings.py --gpu A --llama B` (relative RMS per stage). `llama_vision_golden.json` pins the build, the
mmproj, the images, the preprocessing and the numbers of the 2026-10-17 run.

