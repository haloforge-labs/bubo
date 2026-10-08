// CPU test of TwinWeights::Bind (decode twin): a synthetic two-layer trunk (one linear layer, one attention layer, Q8_0 dense tensors, one non-Q8_0 tensor)
// against tiny twin GGUF files written by this test. Accept case + every rejection case of the contract in weights.hpp (missing tensor, Q8_0 in the twin, wrong shape,
// half a qkv pair / half of q-k-v, twin tensor for a non-Q8_0 trunk tensor, empty twin). No model file, no GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct Spec {
  std::string name;
  GgmlType type;
  std::uint64_t cols, rows;
};

std::size_t BytesOf(GgmlType type, std::uint64_t cols, std::uint64_t rows) {
  std::size_t block = 1, bytes = 4;
  switch (type) {
    case GgmlType::kQ8_0: block = 32; bytes = 34; break;
    case GgmlType::kQ5_1: block = 32; bytes = 24; break;
    case GgmlType::kQ4_K: block = 256; bytes = 144; break;
    case GgmlType::kQ5_K: block = 256; bytes = 176; break;
    case GgmlType::kF32: break;
    default: throw std::runtime_error("test type");
  }
  Require(cols % block == 0, "test shape");
  return static_cast<std::size_t>(cols / block) * bytes * rows;
}

template <typename T>
void Put(std::ofstream& f, T v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof v);
}
void PutStr(std::ofstream& f, const std::string& s) {
  Put<std::uint64_t>(f, s.size());
  f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// minimal GGUF v3: one string KV, the tensor table, 32-byte aligned zero payloads
void WriteGguf(const std::filesystem::path& path, const std::vector<Spec>& tensors, const std::string& keep_q8 = "") {
  std::ofstream f(path, std::ios::binary);
  f.write("GGUF", 4);
  Put<std::uint32_t>(f, 3);
  Put<std::uint64_t>(f, tensors.size());
  Put<std::uint64_t>(f, keep_q8.empty() ? 1 : 2);
  PutStr(f, "general.architecture");
  Put<std::uint32_t>(f, 8);
  PutStr(f, "qwen4exp");
  if (!keep_q8.empty()) {
    PutStr(f, "gufo.dense_twin.keep_q8");
    Put<std::uint32_t>(f, 8);
    PutStr(f, keep_q8);
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    PutStr(f, t.name);
    Put<std::uint32_t>(f, 2);
    Put<std::uint64_t>(f, t.cols);
    Put<std::uint64_t>(f, t.rows);
    Put<std::uint32_t>(f, static_cast<std::uint32_t>(t.type));
    Put<std::uint64_t>(f, offset);
    offset += (BytesOf(t.type, t.cols, t.rows) + 31) / 32 * 32;
  }
  auto pos = static_cast<std::uint64_t>(f.tellp());
  for (; pos % 32 != 0; ++pos) f.put('\0');
  std::vector<char> zeros(4096, 0);
  for (const auto& t : tensors) {
    const std::size_t size = (BytesOf(t.type, t.cols, t.rows) + 31) / 32 * 32;
    for (std::size_t done = 0; done < size; done += zeros.size())
      f.write(zeros.data(), static_cast<std::streamsize>(std::min(zeros.size(), size - done)));
  }
}

qfn::TensorRef Trunk(GgmlType type, std::uint64_t cols, std::uint64_t rows) {
  static std::uint8_t dummy[8];
  qfn::TensorRef t;
  t.data = dummy;
  t.type = type;
  t.cols = cols;
  t.rows = rows;
  return t;
}

qfn::ModelWeights SyntheticTrunk(bool attn_k_is_q8) {
  qfn::ModelWeights w;
  for (int il = 0; il < 2; ++il) {
    qfn::LayerWeights l;
    l.linear = il == 0;
    if (l.linear) {
      l.ssm_qkv = Trunk(GgmlType::kQ8_0, 256, 8);
      l.ssm_gate = Trunk(GgmlType::kQ8_0, 256, 8);
      l.ssm_out = Trunk(GgmlType::kQ8_0, 256, 8);
    } else {
      l.attn_q = Trunk(GgmlType::kQ8_0, 256, 8);
      l.attn_k = Trunk(attn_k_is_q8 ? GgmlType::kQ8_0 : GgmlType::kF32, 256, 4);
      l.attn_v = Trunk(GgmlType::kQ8_0, 256, 4);
      l.attn_out = Trunk(GgmlType::kQ8_0, 256, 8);
    }
    for (auto* m : {&l.hc_attn, &l.hc_ffn}) {
      m->down = Trunk(GgmlType::kQ8_0, 256, 4);
      m->up = Trunk(GgmlType::kQ8_0, 64, 8);
    }
    l.shexp_gate = Trunk(GgmlType::kQ8_0, 256, 4);
    l.shexp_up = Trunk(GgmlType::kQ8_0, 256, 4);
    l.shexp_down = Trunk(GgmlType::kQ8_0, 64, 8);
    w.layers.push_back(l);
  }
  return w;
}

std::vector<Spec> GoodTwin(bool with_attn_k) {
  std::vector<Spec> t;
  const auto add = [&](const std::string& n, GgmlType ty, std::uint64_t c, std::uint64_t r) { t.push_back({n, ty, c, r}); };
  add("blk.0.attn_qkv.weight", GgmlType::kQ5_K, 256, 8);
  add("blk.0.attn_gate.weight", GgmlType::kQ4_K, 256, 8);
  add("blk.0.ssm_out.weight", GgmlType::kQ4_K, 256, 8);
  add("blk.1.attn_q.weight", GgmlType::kQ4_K, 256, 8);
  if (with_attn_k) add("blk.1.attn_k.weight", GgmlType::kQ4_K, 256, 4);
  add("blk.1.attn_v.weight", GgmlType::kQ5_K, 256, 4);
  add("blk.1.attn_output.weight", GgmlType::kQ4_K, 256, 8);
  for (int il = 0; il < 2; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    add(p + "hc_attn_down.weight", GgmlType::kQ4_K, 256, 4);
    add(p + "hc_attn_up.weight", GgmlType::kQ5_1, 64, 8);
    add(p + "hc_ffn_down.weight", GgmlType::kQ4_K, 256, 4);
    add(p + "hc_ffn_up.weight", GgmlType::kQ5_1, 64, 8);
    add(p + "ffn_gate_shexp.weight", GgmlType::kQ4_K, 256, 4);
    add(p + "ffn_up_shexp.weight", GgmlType::kQ4_K, 256, 4);
    add(p + "ffn_down_shexp.weight", GgmlType::kQ5_1, 64, 8);
  }
  return t;
}

void Drop(std::vector<Spec>& t, const std::string& name) {
  const auto n = t.size();
  std::erase_if(t, [&](const Spec& s) { return s.name == name; });
  Require(t.size() + 1 == n, "fixture: nothing to drop " + name);
}
Spec& Find(std::vector<Spec>& t, const std::string& name) {
  for (auto& s : t)
    if (s.name == name) return s;
  throw std::runtime_error("fixture: no " + name);
}

// returns the error message ("" = Bind succeeded)
std::string Bind(const std::filesystem::path& dir, const std::string& id, const std::vector<Spec>& twin, const qfn::ModelWeights& trunk, qfn::TwinWeights* out = nullptr, const std::string& keep_q8 = "") {
  const auto path = dir / (id + ".gguf");
  WriteGguf(path, twin, keep_q8);
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(path, &err);
  Require(reader != nullptr, "fixture " + id + " did not open: " + err);
  err.clear();
  auto bound = qfn::TwinWeights::Bind(*reader, trunk, &err);
  if (bound.has_value()) {
    if (out != nullptr) *out = std::move(*bound);
    return "";
  }
  Require(!err.empty(), "Bind failed without an error message (" + id + ")");
  return err;
}

void ExpectReject(const std::string& id, const std::string& err, const std::string& needle) {
  Require(!err.empty(), id + ": Bind accepted a twin it must reject");
  Require(err.find(needle) != std::string::npos, id + ": error '" + err + "' does not mention '" + needle + "'");
  std::cout << "  reject " << id << ": " << err << "\n";
}

}  // namespace

// optional real-file check: twin_weights_test TRUNK_SHARD1.gguf TWIN.gguf  (ModelWeights::Bind on the trunk, TwinWeights::Bind on the twin; headers only)
int RealCheck(const char* trunk_path, const char* twin_path) {
  std::string err;
  auto trunk_reader = gufo::core::GgufReader::OpenFile(trunk_path, &err);
  Require(trunk_reader != nullptr, "trunk open: " + err);
  auto trunk = qfn::ModelWeights::Bind(*trunk_reader, &err);
  Require(trunk.has_value(), "trunk bind: " + err);
  auto twin_reader = gufo::core::GgufReader::OpenFile(twin_path, &err);
  Require(twin_reader != nullptr, "twin open: " + err);
  auto twin = qfn::TwinWeights::Bind(*twin_reader, *trunk, &err);
  Require(twin.has_value(), "twin bind: " + err);
  std::size_t linear = 0, attention = 0;
  for (const auto& l : trunk->layers) (l.linear ? linear : attention)++;
  std::cout << "real twin bound: " << twin->tensors << " tensors over " << trunk->layers.size() << " layers (" << linear << " linear, " << attention << " attention), " << twin->bytes_4bit << " bytes (4-bit) mirroring "
            << twin->bytes_8bit << " bytes (8-bit), saved " << (twin->bytes_8bit - twin->bytes_4bit) << " bytes per full pass\n";
  return twin->tensors > 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  try {
    if (argc == 3) return RealCheck(argv[1], argv[2]);
    const auto dir = std::filesystem::temp_directory_path() / "gufo-twin-weights-test";
    std::filesystem::create_directories(dir);
    const auto trunk = SyntheticTrunk(true);

    qfn::TwinWeights ok;
    const auto err = Bind(dir, "ok", GoodTwin(true), trunk, &ok);
    Require(err.empty(), "good twin rejected: " + err);
    Require(ok.layers.size() == 2, "layer count");
    Require(ok.tensors == 3 + 4 + 14, "twinned tensor count");
    Require(ok.bytes_4bit < ok.bytes_8bit, "4-bit bytes must be smaller than the 8-bit bytes they mirror");
    Require(ok.layers[0].ssm_qkv.type == GgmlType::kQ5_K && ok.layers[1].attn_q.type == GgmlType::kQ4_K && ok.layers[1].shexp_down.type == GgmlType::kQ5_1, "types bound");
    Require(!ok.layers[0].ssm_qkv.empty() && ok.layers[0].attn_q.empty() && ok.layers[1].ssm_qkv.empty(), "only the layer's own tensors are bound");
    std::cout << "  accept ok: " << ok.tensors << " tensors, " << ok.bytes_4bit << " bytes (4-bit) mirroring " << ok.bytes_8bit << " (8-bit)\n";

    {
      auto t = GoodTwin(true);
      Drop(t, "blk.0.ssm_out.weight");
      ExpectReject("missing-tensor", Bind(dir, "missing", t, trunk), "missing blk.0.ssm_out.weight");
    }
    {
      auto t = GoodTwin(true);
      Find(t, "blk.0.attn_gate.weight").type = GgmlType::kQ8_0;
      ExpectReject("q8-in-twin", Bind(dir, "q8", t, trunk), "unsupported format");
    }
    {
      auto t = GoodTwin(true);
      Find(t, "blk.1.attn_output.weight").rows = 16;
      ExpectReject("wrong-shape", Bind(dir, "shape", t, trunk), "shape");
    }
    {
      auto t = GoodTwin(true);
      Drop(t, "blk.0.attn_gate.weight");
      ExpectReject("half-qkv-pair", Bind(dir, "pair", t, trunk), "attn_gate");
    }
    {
      auto t = GoodTwin(true);
      Drop(t, "blk.1.attn_v.weight");
      ExpectReject("half-qkv-triple", Bind(dir, "triple", t, trunk), "attn_v");
    }
    {
      // a trunk tensor that is not Q8_0 (attn_k F32) must be left alone: absent from the twin = accepted, present = rejected
      const auto f32_trunk = SyntheticTrunk(false);
      const auto accepted = Bind(dir, "k-absent", GoodTwin(false), f32_trunk);
      Require(accepted.empty(), "twin without the non-Q8_0 trunk tensor rejected: " + accepted);
      ExpectReject("twin-for-non-q8", Bind(dir, "k-present", GoodTwin(true), f32_trunk), "not Q8_0");
    }
    {
      // keep_q8: a listed Q8_0 tensor is absent from the twin, counted, and accepted; every misuse is rejected
      auto t = GoodTwin(true);
      Drop(t, "blk.0.hc_attn_up.weight");
      Drop(t, "blk.1.ffn_down_shexp.weight");
      qfn::TwinWeights kept;
      const auto err = Bind(dir, "keep-ok", t, trunk, &kept, "blk.0.hc_attn_up.weight,blk.1.ffn_down_shexp.weight");
      Require(err.empty(), "keep_q8 twin rejected: " + err);
      Require(kept.kept_tensors == 2 && kept.kept_bytes > 0 && kept.layers[0].hc_attn.up.empty() && kept.layers[1].shexp_down.empty(), "kept tensors counted and left unbound");
      Require(kept.tensors == 3 + 4 + 14 - 2, "twinned count excludes the kept tensors");
      auto missing_unlisted = GoodTwin(true);
      Drop(missing_unlisted, "blk.0.hc_attn_up.weight");
      ExpectReject("keep-unlisted-missing", Bind(dir, "keep-unlisted", missing_unlisted, trunk, nullptr, "blk.1.ffn_down_shexp.weight"), "missing blk.0.hc_attn_up");
      ExpectReject("keep-and-present", Bind(dir, "keep-present", GoodTwin(true), trunk, nullptr, "blk.0.hc_attn_up.weight"), "both in the twin file and in its keep_q8");
      ExpectReject("keep-unknown", Bind(dir, "keep-unknown", GoodTwin(true), trunk, nullptr, "blk.0.no_such.weight"), "not a Q8_0 tensor of the twin set");
      const auto f32_trunk = SyntheticTrunk(false);
      ExpectReject("keep-non-q8", Bind(dir, "keep-nonq8", GoodTwin(false), f32_trunk, nullptr, "blk.1.attn_k.weight"), "not Q8_0");
    }
    {
      std::vector<Spec> none;
      none.push_back({"unrelated.weight", GgmlType::kF32, 32, 1});
      qfn::ModelWeights nothing_to_twin;
      nothing_to_twin.layers.resize(1);  // no Q8_0 trunk tensors at all
      ExpectReject("empty-twin", Bind(dir, "empty", none, nothing_to_twin), "no twinned tensor");
    }
    std::filesystem::remove_all(dir);
    std::cout << "twin_weights_test: all cases pass\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "twin_weights_test FAILED: " << e.what() << "\n";
    return 1;
  }
}
