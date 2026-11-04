#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/stageb.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace gpu = qfn::rocm;
namespace {
void Hip(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
void Near(double a, double b, double tolerance = 1e-9) {
  if (!std::isfinite(a) ||
      std::abs(a - b) > tolerance * std::max(1., std::abs(b)))
    throw std::runtime_error("stage B GPU/independent CPU oracle mismatch");
}
template<typename T>
struct Device {
  T* p{};
  explicit Device(std::size_t n) { Hip(hipMalloc(&p, n * sizeof(T))); }
  ~Device() { (void)hipFree(p); }
  void Upload(const std::vector<T>& v) {
    Hip(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  }
  std::vector<T> Read(std::size_t n) {
    std::vector<T> v(n);
    Hip(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
  }
};
const gufo::json::Value& Token(const gufo::json::Value& fixture, unsigned t) {
  const auto* tokens = fixture.find("tokens");
  return tokens ? tokens->items().at(t) : fixture;
}
template<typename T>
void Case(const gufo::json::Value& fixture, unsigned n, unsigned stride) {
  constexpr unsigned start = 101, layer = 3, every = 4;
  const auto& first = Token(fixture, 0);
  const auto& input = *first.find("experts");
  const unsigned hidden = static_cast<unsigned>(input.items()[0].size());
  const auto parts = first.find("parts")->as_size();
  std::vector<T> rows(n * 10 * hidden);
  std::vector<float> w(n * 10), in(n * 4 * parts);
  std::vector<std::int32_t> id(n * 10);
  for (unsigned t = 0; t < n; ++t) {
    const auto& token = Token(fixture, t);
    const auto& experts = *token.find("experts");
    const auto& weights = *token.find("weights");
    const auto& ids = *token.find("ids");
    const auto& inject = *token.find("inject");
    for (unsigned s = 0; s < 10; ++s) {
      w[t * 10 + s] = static_cast<float>(weights.items()[s].as_double());
      id[t * 10 + s] = static_cast<std::int32_t>(ids.items()[s].as_size());
      for (unsigned h = 0; h < hidden; ++h)
        rows[(t * 10 + s) * hidden + h] =
            static_cast<T>(experts.items()[s].items()[h].as_double());
    }
    for (unsigned p = 0; p < 4 * parts; ++p)
      in[t * 4 * parts + p] = static_cast<float>(inject.items()[p].as_double());
  }
  Device<T> d_rows(rows.size());
  Device<float> d_weights(w.size()), d_inject(in.size());
  Device<std::int32_t> d_ids(id.size());
  Device<qfn::StageBTotal> d_totals(48);
  const unsigned capacity = (n + every - 1) / every + 1;
  Device<qfn::StageBRow> d_records(48 * capacity);
  d_rows.Upload(rows);
  d_weights.Upload(w);
  d_inject.Upload(in);
  d_ids.Upload(id);
  Hip(hipMemset(d_totals.p, 0, 48 * sizeof(qfn::StageBTotal)));
  gpu::PrefillStageB(d_rows.p, sizeof(T) == 2, d_weights.p, d_ids.p, d_inject.p,
                     static_cast<unsigned>(parts), n, hidden, start, layer,
                     stride, every, capacity, d_totals.p, d_records.p, nullptr);
  Hip(hipGetLastError());
  Hip(hipDeviceSynchronize());
  const auto actual = d_totals.Read(48)[layer];
  // Pool independent CPU expectations at the selected absolute positions.
  qfn::StageBTotal pooled{};
  for (unsigned t = 0; t < n; ++t) {
    if ((start + t) % stride != 0)
      continue;
    const auto& totals = *Token(fixture, t).find("expected")->find("totals");
    ++pooled.rows;
    pooled.top1_mass += totals.find("top1_mass")->as_double();
    pooled.top4_mass += totals.find("top4_mass")->as_double();
    pooled.gate_l2 += totals.find("gate_l2")->as_double();
    pooled.ff += totals.find("k")->find("10")->find("ff")->as_double();
    for (unsigned k = 0; k < 5; ++k) {
      const auto& e =
          *totals.find("k")->find(std::to_string(qfn::kStageBKeep[k]));
      pooled.rf[k] += e.find("rf")->as_double();
      pooled.rr[k] += e.find("rr")->as_double();
      pooled.error2[k] += e.find("error2")->as_double();
      pooled.omitted[k] += e.find("omitted_mass")->as_double();
    }
  }
  Near(static_cast<double>(actual.rows), static_cast<double>(pooled.rows));
  Near(static_cast<double>(actual.invalid), 0);
  for (unsigned k = 0; k < 5; ++k) {
    Near(actual.rf[k], pooled.rf[k]);
    Near(actual.rr[k], pooled.rr[k]);
    Near(actual.error2[k], pooled.error2[k]);
    Near(actual.omitted[k], pooled.omitted[k]);
  }
  Near(actual.ff, pooled.ff);
  Near(actual.top1_mass, pooled.top1_mass);
  Near(actual.top4_mass, pooled.top4_mass);
  Near(actual.gate_l2, pooled.gate_l2, 1e-6);
  const auto records = d_records.Read(48 * capacity);
  unsigned index = 0;
  for (unsigned t = 0; t < n; ++t)
    if ((start + t) % every == 0) {
      const auto& expected = *Token(fixture, t).find("expected");
      const auto& totals = *expected.find("totals");
      const auto& r = records[layer * capacity + index++];
      Near(r.token, start + t);
      Near(r.layer, layer);
      Near(r.top1_mass, totals.find("top1_mass")->as_double(), 1e-6);
      Near(r.top4_mass, totals.find("top4_mass")->as_double(), 1e-6);
      Near(r.f_norm, expected.find("f_norm")->as_double(), 1e-6);
      Near(r.gate_l2, expected.find("gate_l2")->as_double(), 1e-6);
      Near(r.gate_max, expected.find("gate_max")->as_double(), 1e-6);
      for (unsigned s = 0; s < 4; ++s) {
        Near(r.gates[s], expected.find("gates")->items()[s].as_double(), 1e-6);
        const auto& e =
            *totals.find("k")->find(std::to_string(qfn::kStageBKeep[s]));
        Near(r.error_norm[s], std::sqrt(e.find("error2")->as_double()), 1e-6);
        Near(r.omitted[s], e.find("omitted_mass")->as_double(), 1e-6);
      }
    }
  // The diagnostic must not modify any input. Compare every byte, including
  // FP16 payloads (operator== on half would hide signed-zero changes).
  const auto after = d_rows.Read(rows.size());
  if (std::memcmp(after.data(), rows.data(), rows.size() * sizeof(T)) != 0 ||
      d_weights.Read(w.size()) != w || d_inject.Read(in.size()) != in ||
      d_ids.Read(id.size()) != id)
    throw std::runtime_error("stage B modified a model input");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("supply stageb-synthetic.json oracle fixture");
    std::ifstream in(argv[1]);
    const auto fixture =
        gufo::json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
    for (const auto& c : fixture.items())
      for (unsigned n : {1U, 8U, 9U, 33U})
        for (unsigned stride : {1U, 3U, 128U}) {
          Case<float>(c, n, stride);
          Case<__half>(c, n, stride);
        }
    std::cout << "PASS stage B FP16/FP32 sorted/unsorted/ranked-tie oracle, "
                 "distinct token data, ragged rows, "
                 "strides, positions and immutable inputs\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
