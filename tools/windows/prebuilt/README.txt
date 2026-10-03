Gufo for Windows - prebuilt
===========================

A native Windows build of gufo (https://github.com/gufo-org/gufo), the ROCm
inference engine for AMD Strix Halo, with an OpenAI-compatible server.
Source and docs: https://github.com/pixmaate/gufo

NEEDS
  - AMD Ryzen AI Max / Max+ (Strix Halo, Radeon 8060S / 8050S = gfx1151).
    No other GPU is supported.
  - Windows 11 and a current AMD Software: Adrenalin Edition driver.
  - GPU memory set in AMD Software > Performance > Tuning > Variable Graphics
    Memory: 96 GB for Qwen3.8-Flash-Next, 48-64 GB is plenty for the others.
  - Models in GGUF form (below). Nothing else: the ROCm runtime and the
    Visual C++ runtime are in bin\.

MODELS (pip install -U huggingface_hub, then pick one or more)
  Qwen3.8-Flash-Next (96 GB machines):
    hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf"
  Qwen3.8-27B (+ its DFlash2 draft):
    hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf
    hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf
  Qwen3.6-35B-A3B (MTP is inside the GGUF; the mmproj comes from the base repo):
    hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf
    hf download unsloth/Qwen3.6-35B-A3B-GGUF mmproj-BF16.gguf
  The launcher finds them in the Hugging Face cache, LM Studio's folder, or a
  "models" folder next to this one.

START
  Unzip to a short folder such as C:\gufo: the GPU kernel files have long
  names, and Windows' 260-character path limit breaks them in deep folders
  (start.cmd checks and tells you).
  Double-click start.cmd (or run it in a terminal):
    - press the number of a model,
    - toggle features with number keys, Enter starts the server.
  Everything exact and measured faster is on by default: speculative decoding
  (MTP / DFlash2), prompt lookup, survival stopping, images, thinking.
  "Latin draft vocabulary" is off by default: turn it on (key 4) if you work in
  English - about 5% faster, identical output.
  start.cmd -Last starts the previous choice again.

  The server speaks the OpenAI API at http://127.0.0.1:8080/v1 (chat
  completions with streaming, tools and images). Point any OpenAI-compatible
  client at it; the model name is shown when the server starts.

TROUBLE
  - "no Radeon GPU found" / HIP errors: update the AMD driver.
  - It loads but is slow, or runs out of memory: raise Variable Graphics
    Memory, or pick a smaller context (key 7).
  - A missing DLL: install the latest Microsoft Visual C++ Redistributable (x64).
