// llama_vision: runs llama.cpp b11316's own image pipeline (stb decode -> mtmd preprocessing -> CLIP graph on the CPU)
// on one image file and writes the final embeddings as G4VE (the file gemma4_vision_probe --compare reads) and, with
// --dump-dir, the layer-0 stages and every layer output in the probe's stage layout (u32 dims[4] + f32 data), under the probe's stage names.
// Loads llama.cpp's DLLs at run time (no import libraries); the headers only give the struct layouts.
//   llama_vision --dll-dir DIR --model GGUF --mmproj GGUF --image FILE --out OUT.g4ve [--dump-dir DIR] [--threads N]
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {

HMODULE g_mtmd = nullptr, g_llama = nullptr, g_ggml = nullptr, g_base = nullptr;

template <class Fn>
Fn Resolve(HMODULE module, const char* name) {
  auto* p = reinterpret_cast<Fn>(GetProcAddress(module, name));
  if (!p) {
    std::fprintf(stderr, "missing export %s\n", name);
    std::exit(3);
  }
  return p;
}
#define LOAD(module, fn) auto fn##_p = Resolve<decltype(&fn)>(module, #fn)

struct Capture {
  std::map<std::string, std::vector<float>> tensors;  // name -> data (ne0 x ne1)
  std::map<std::string, std::pair<int64_t, int64_t>> shapes;
  bool (*buffer_is_host)(ggml_backend_buffer_t);
  void (*tensor_get)(const ggml_tensor*, void*, size_t, size_t);
  const char* (*get_name)(const ggml_tensor*);
  size_t (*nbytes)(const ggml_tensor*);
};
Capture g_cap;

bool Wanted(const std::string& name) {
  static const char* exact[] = {"layer_inp_normed-0", "kqv_out-0", "attn_out-0", "ffn_inp-0", "ffn_gate-0", "ffn_up-0",
                                "ffn_geglu_quick-0", "ffn_out-0", "pooled", "projected", "inp_raw_scaled"};
  for (const char* e : exact)
    if (name == e) return true;
  return name.rfind("layer_out-", 0) == 0 && name.find(' ') == std::string::npos;
}

bool EvalCallback(ggml_tensor* t, bool ask, void*) {
  const std::string name = g_cap.get_name(t);
  if (!Wanted(name) || t->type != GGML_TYPE_F32) return ask ? false : true;
  if (ask) return true;
  std::vector<float> data(g_cap.nbytes(t) / 4);
  g_cap.tensor_get(t, data.data(), 0, g_cap.nbytes(t));
  g_cap.tensors[name] = std::move(data);
  g_cap.shapes[name] = {t->ne[0], t->ne[1] * t->ne[2] * t->ne[3]};
  return true;
}

void WriteStage(const std::string& path, const float* data, uint32_t width, uint32_t rows) {
  std::ofstream out(path, std::ios::binary);
  const uint32_t dims[4] = {width, rows, 1, 1};
  out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
  out.write(reinterpret_cast<const char*>(data), std::streamsize(width) * rows * 4);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dll_dir, model, mmproj, image, out, dump, rgb;
  uint32_t rgb_w = 0, rgb_h = 0;
  int threads = 4;
  bool no_fa = false, use_gpu = false;
  uint32_t width = 2816;  // the 26B-A4B projection (5376 for the 31B)
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--dll-dir") dll_dir = next();
    else if (a == "--model") model = next();
    else if (a == "--mmproj") mmproj = next();
    else if (a == "--image") image = next();
    else if (a == "--rgb") { rgb = next(); rgb_w = static_cast<uint32_t>(std::stoul(next())); rgb_h = static_cast<uint32_t>(std::stoul(next())); }
    else if (a == "--out") out = next();
    else if (a == "--dump-dir") dump = next();
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--no-fa") no_fa = true;
    else if (a == "--gpu") use_gpu = true;
    else if (a == "--width") width = static_cast<uint32_t>(std::stoul(next()));
  }
  SetDllDirectoryA(dll_dir.c_str());
  g_base = LoadLibraryA((dll_dir + "\\ggml-base.dll").c_str());
  g_ggml = LoadLibraryA((dll_dir + "\\ggml.dll").c_str());
  g_llama = LoadLibraryA((dll_dir + "\\llama.dll").c_str());
  g_mtmd = LoadLibraryA((dll_dir + "\\mtmd.dll").c_str());
  if (!g_base || !g_ggml || !g_llama || !g_mtmd) {
    std::fprintf(stderr, "cannot load llama.cpp DLLs from %s\n", dll_dir.c_str());
    return 3;
  }
  LOAD(g_ggml, ggml_backend_load_all_from_path);
  LOAD(g_llama, llama_backend_init);
  LOAD(g_llama, llama_model_default_params);
  LOAD(g_llama, llama_model_load_from_file);
  LOAD(g_mtmd, mtmd_context_params_default);
  LOAD(g_mtmd, mtmd_init_from_file);
  LOAD(g_mtmd, mtmd_default_marker);
  LOAD(g_mtmd, mtmd_input_chunks_init);
  LOAD(g_mtmd, mtmd_input_chunks_size);
  LOAD(g_mtmd, mtmd_input_chunks_get);
  LOAD(g_mtmd, mtmd_input_chunk_get_type);
  LOAD(g_mtmd, mtmd_input_chunk_get_n_tokens);
  LOAD(g_mtmd, mtmd_tokenize);
  LOAD(g_mtmd, mtmd_encode_chunk);
  LOAD(g_mtmd, mtmd_get_output_embd);
  LOAD(g_mtmd, mtmd_helper_bitmap_init_from_file);
  LOAD(g_mtmd, mtmd_bitmap_init);
  LOAD(g_mtmd, mtmd_helper_init_opt_default);
  g_cap.buffer_is_host = Resolve<decltype(&ggml_backend_buffer_is_host)>(g_base, "ggml_backend_buffer_is_host");
  g_cap.tensor_get = Resolve<decltype(&ggml_backend_tensor_get)>(g_base, "ggml_backend_tensor_get");
  g_cap.get_name = Resolve<decltype(&ggml_get_name)>(g_base, "ggml_get_name");
  g_cap.nbytes = Resolve<decltype(&ggml_nbytes)>(g_base, "ggml_nbytes");

  ggml_backend_load_all_from_path_p(dll_dir.c_str());
  llama_backend_init_p();
  llama_model_params mp = llama_model_default_params_p();
  mp.vocab_only = true;
  mp.n_gpu_layers = 0;
  llama_model* lm = llama_model_load_from_file_p(model.c_str(), mp);
  if (!lm) {
    std::fprintf(stderr, "cannot load %s\n", model.c_str());
    return 4;
  }
  mtmd_context_params cp = mtmd_context_params_default_p();
  cp.use_gpu = use_gpu;
  cp.n_threads = threads;
  cp.warmup = false;
  if (no_fa) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  cp.print_timings = true;
  if (!dump.empty()) {
    cp.cb_eval = EvalCallback;
    cp.cb_eval_user_data = nullptr;
  }
  mtmd_context* ctx = mtmd_init_from_file_p(mmproj.c_str(), lm, cp);
  if (!ctx) {
    std::fprintf(stderr, "cannot init mtmd from %s\n", mmproj.c_str());
    return 5;
  }
  // --rgb: raw RGB8 pixels already at the encoder size (llama's resize then copies them), so only the encoder differs
  std::vector<unsigned char> raw_rgb;
  mtmd_helper_bitmap_wrapper wrapper{nullptr, nullptr};
  if (!rgb.empty()) {
    std::ifstream f(rgb, std::ios::binary);
    raw_rgb.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (raw_rgb.size() != size_t(rgb_w) * rgb_h * 3) {
      std::fprintf(stderr, "bad raw size\n");
      return 6;
    }
    wrapper.bitmap = mtmd_bitmap_init_p(rgb_w, rgb_h, raw_rgb.data());
  } else {
    wrapper = mtmd_helper_bitmap_init_from_file_p(ctx, image.c_str(), false, mtmd_helper_init_opt_default_p());
  }
  if (!wrapper.bitmap) {
    std::fprintf(stderr, "cannot decode %s\n", image.c_str());
    return 6;
  }
  std::string prompt = mtmd_default_marker_p();
  mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
  mtmd_input_chunks* chunks = mtmd_input_chunks_init_p();
  const mtmd_bitmap* bitmaps[1] = {wrapper.bitmap};
  if (mtmd_tokenize_p(ctx, chunks, &text, bitmaps, 1) != 0) {
    std::fprintf(stderr, "tokenize failed\n");
    return 7;
  }
  const mtmd_input_chunk* image_chunk = nullptr;
  for (size_t i = 0; i < mtmd_input_chunks_size_p(chunks); ++i) {
    const auto* c = mtmd_input_chunks_get_p(chunks, i);
    if (mtmd_input_chunk_get_type_p(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) image_chunk = c;
  }
  if (!image_chunk) {
    std::fprintf(stderr, "no image chunk\n");
    return 8;
  }
  const size_t rows = mtmd_input_chunk_get_n_tokens_p(image_chunk);
  if (mtmd_encode_chunk_p(ctx, image_chunk) != 0) {
    std::fprintf(stderr, "encode failed\n");
    return 9;
  }
  const float* embd = mtmd_get_output_embd_p(ctx);
  {
    std::ofstream o(out, std::ios::binary);
    const uint32_t header[4] = {0x45563447U, 1U, static_cast<uint32_t>(rows), width};
    o.write(reinterpret_cast<const char*>(header), sizeof(header));
    o.write(reinterpret_cast<const char*>(embd), std::streamsize(rows) * width * 4);
  }
  std::printf("llama embeddings: %zu rows x %u\n", rows, width);
  if (!dump.empty()) {
    WriteStage(dump + "/embedding.f32", embd, width, static_cast<uint32_t>(rows));
    for (auto& [name, data] : g_cap.tensors) {
      const auto [w, r] = g_cap.shapes[name];
      std::string stage;
      if (name == "layer_inp_normed-0") stage = "l0.ln1";
      else if (name == "kqv_out-0") stage = "l0.attn";
      else if (name == "attn_out-0") stage = "l0.o";
      else if (name == "ffn_inp-0") stage = "l0.h1";
      else if (name == "ffn_geglu_quick-0") stage = "l0.act";
      else if (name == "ffn_out-0") stage = "l0.down";
      else if (name == "pooled") stage = "pooled";
      else if (name.rfind("layer_out-", 0) == 0) stage = "layer" + name.substr(10);
      else if (name == "inp_raw_scaled") stage = "input";
      if (!stage.empty()) WriteStage(dump + "/" + stage + ".f32", data.data(), (uint32_t)w, (uint32_t)r);
    }
    // gate|up per token, as the reference and the encoder lay it out
    if (g_cap.tensors.count("ffn_gate-0") && g_cap.tensors.count("ffn_up-0")) {
      const auto& gate = g_cap.tensors["ffn_gate-0"];
      const auto& up = g_cap.tensors["ffn_up-0"];
      const auto [w, r] = g_cap.shapes["ffn_gate-0"];
      std::vector<float> both(gate.size() * 2);
      for (int64_t t = 0; t < r; ++t) {
        std::memcpy(&both[size_t(t) * 2 * w], &gate[size_t(t) * w], size_t(w) * 4);
        std::memcpy(&both[size_t(t) * 2 * w + w], &up[size_t(t) * w], size_t(w) * 4);
      }
      WriteStage(dump + "/l0.gate_up.f32", both.data(), (uint32_t)(2 * w), (uint32_t)r);
    }
    std::printf("dumped %zu tensors\n", g_cap.tensors.size());
  }
  return 0;
}
