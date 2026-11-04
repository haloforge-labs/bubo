#include "src/models/qwen38_flash_next/refusal_direction.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/refusal_math.hpp"
#include "tests/support/temporary_directory.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class T>
void Put(std::ofstream& f, T value) {
  f.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void Str(std::ofstream& f, const std::string& s) {
  Put<std::uint64_t>(f, s.size());
  f.write(s.data(), s.size());
}
void Fixture(const std::filesystem::path& path, unsigned length,
             std::string name, const std::set<std::string, std::less<>>& sites,
             std::optional<std::string> fitted = std::nullopt,
             bool numeric_fitted = false) {
  std::ofstream f(path, std::ios::binary);
  f.write("GGUF", 4);
  Put<std::uint32_t>(f, 3);
  Put<std::uint64_t>(f, 1);
  Put<std::uint64_t>(f, fitted ? 4 : 3);
  std::string list;
  for (const auto& site : sites) {
    if (!list.empty())
      list += '\n';
    list += site;
  }
  std::vector<std::pair<std::string, std::string>> metadata{
      {"general.architecture", "refusal-direction"},
      {"refusal.version", "1"},
      {"refusal.sites", list}};
  if (fitted)
    metadata.emplace_back("refusal.fitted_strength", *fitted);
  for (const auto& [key, value] : metadata) {
    Str(f, key);
    if (numeric_fitted && key == "refusal.fitted_strength") {
      Put<std::uint32_t>(f, 4);
      Put<std::uint32_t>(f, 1);
    } else {
      Put<std::uint32_t>(f, 8);
      Str(f, value);
    }
  }
  Str(f, name);
  Put<std::uint32_t>(f, 1);
  Put<std::uint64_t>(f, length);
  Put<std::uint32_t>(f, 0);
  Put<std::uint64_t>(f, 0);
  while (std::uint64_t(f.tellp()) % 32)
    f.put(0);
  for (unsigned i = 0; i < length; ++i)
    Put<float>(f, i == 0 ? 1.0F : 0.0F);
}

// Emulate the actual kernel's 256 lanes and descending shared reduction.
void Project(float* y, const float* r, unsigned n, float strength = 1.0F) {
  std::array<float, q::kRefusalThreads> sums{};
  for (unsigned lane = 0; lane < sums.size(); ++lane)
    sums[lane] = q::RefusalLaneDot(y, r, n, lane, sums.size());
  for (unsigned stride = sums.size() / 2; stride; stride /= 2)
    for (unsigned lane = 0; lane < stride; ++lane)
      sums[lane] += sums[lane + stride];
  for (unsigned i = 0; i < n; ++i)
    y[i] = q::RefusalSubtract(y[i], r[i], sums[0], strength);
}

void Arithmetic() {
  std::mt19937 rng(26030);
  std::normal_distribution<float> normal;
  double max_reference = 0, max_idempotent = 0, max_dot = 0;
  for (unsigned dim : {33U, 255U, 256U, 257U, 2560U}) {
    std::vector<float> r(dim);
    double norm = 0;
    for (auto& v : r) {
      v = normal(rng);
      norm += double(v) * v;
    }
    for (auto& v : r)
      v /= std::sqrt(norm);
    for (unsigned row = 0; row < 17; ++row) {
      std::vector<float> y(dim);
      double dot = 0;
      for (unsigned i = 0; i < dim; ++i) {
        y[i] = normal(rng);
        dot += double(y[i]) * r[i];
      }
      const auto original = y;
      Project(y.data(), r.data(), dim);
      const auto once = y;
      double remaining = 0;
      for (unsigned i = 0; i < dim; ++i) {
        max_reference = std::max(
            max_reference, std::abs(y[i] - (double(original[i]) - r[i] * dot)));
        remaining += double(y[i]) * r[i];
      }
      max_dot = std::max(max_dot, std::abs(remaining));
      Project(y.data(), r.data(), dim);
      for (unsigned i = 0; i < dim; ++i)
        max_idempotent =
            std::max(max_idempotent, double(std::abs(y[i] - once[i])));
    }
  }
  std::printf(
      "CPU kernel arithmetic max_reference=%.9g max_idempotent=%.9g "
      "max_remaining_dot=%.9g\n",
      max_reference, max_idempotent, max_dot);
  Require(max_reference < 2e-6 && max_idempotent < 2e-6 && max_dot < 2e-6,
          "projection arithmetic");
  Require(q::kRefusalDirectionKey == (std::uint64_t{1} << 52), "graph key bit");
  Require((q::kRefusalDirectionKey &
           ((std::uint64_t{1} << 39) | (std::uint64_t{1} << 42) |
            (std::uint64_t{7} << 49))) == 0,
          "graph key collision");

  // Four independent residual lanes and several strengths, including zero,
  // compared against an FP64 formula. Each lane has a different dot product.
  std::vector<float> direction(2560), lanes(4 * 2560);
  double norm = 0;
  for (auto& value : direction) {
    value = normal(rng);
    norm += double(value) * value;
  }
  for (auto& value : direction)
    value /= std::sqrt(norm);
  for (auto& value : lanes)
    value = normal(rng);
  for (float strength : {0.0F, 1.0F, q::kRefusalFittedStrength, 4.0F}) {
    auto actual = lanes;
    for (unsigned lane = 0; lane < 4; ++lane) {
      const auto offset = lane * direction.size();
      double dot = 0;
      for (unsigned i = 0; i < direction.size(); ++i)
        dot += double(lanes[offset + i]) * direction[i];
      Project(actual.data() + offset, direction.data(), direction.size(),
              strength);
      for (unsigned i = 0; i < direction.size(); ++i) {
        const double expected =
            lanes[offset + i] - strength * direction[i] * dot;
        Require(std::abs(actual[offset + i] - expected) < 3e-6,
                "strength or residual lane projection");
      }
    }
  }
}
void Loader() {
  TemporaryDirectory temp("refusal-direction");
  std::filesystem::path packaged;
  std::string default_error;
  Require(!q::ResolvePackagedRefusalDirection(temp.path(), &packaged,
                                              &default_error),
          "missing packaged default must fail");
  Require(default_error.find("packaged default missing:") != std::string::npos,
          "clear missing default diagnostic");
  std::printf("missing default: %s\n", default_error.c_str());
  std::filesystem::create_directories(packaged.parent_path());
  Fixture(packaged, 2560, "direction.r", q::RefusalSites(), "1.955837064");
  Require(q::ResolvePackagedRefusalDirection(temp.path(), &packaged,
                                             &default_error),
          "packaged default resolves");
  const auto direction = q::LoadRefusalDirection(packaged, &default_error);
  Require(direction && direction->fitted_strength == q::kRefusalFittedStrength,
          "packaged default uses file strength");
  std::printf("packaged default: %s; sites; strength=%.9g\n",
              packaged.string().c_str(), direction->fitted_strength);
  const auto path = temp.path() / "r.gguf";
  auto sites = q::RefusalSites();
  std::string error;
  Fixture(path, 2560, "direction.r", sites);
  auto good = q::LoadRefusalDirection(path, &error);
  Require(good && good->sites.size() == 96 && !good->identity.empty(),
          "valid loader: " + error);
  Require(good->fitted_strength == q::kRefusalFittedStrength,
          "legacy file fitted strength");
  Fixture(path, 2560, "direction.r", sites, "1.9");
  good = q::LoadRefusalDirection(path, &error);
  Require(good && good->fitted_strength == 1.9F, "file fitted strength");
  Fixture(path, 2560, "direction.r", sites, "1", true);
  error.clear();
  Require(!q::LoadRefusalDirection(path, &error) &&
              error.find("must be a string") != std::string::npos,
          "incorrect metadata type must fail");
  for (const auto* invalid : {"nan", "inf", "-1", "4.1", "1tail"}) {
    Fixture(path, 2560, "direction.r", sites, invalid);
    error.clear();
    Require(!q::LoadRefusalDirection(path, &error) &&
                error.find("refusal.fitted_strength") != std::string::npos,
            "invalid fitted strength must fail");
  }
  const auto failure = [&](unsigned length, std::string name, const auto& list,
                           std::string_view expected) {
    Fixture(path, length, name, list);
    error.clear();
    Require(!q::LoadRefusalDirection(path, &error),
            "expected fatal loader failure");
    Require(error.find(expected) != std::string::npos,
            "wrong diagnostic: " + error);
    std::printf("fatal test: %s\n", error.c_str());
  };
  failure(2559, "direction.r", sites, "length 2560");
  failure(2560, "wrong.r", sites, "unknown tensor name");
  sites.erase("blk.0.ssm_out.weight");
  failure(2560, "direction.r", sites, "missing site: blk.0.ssm_out.weight");
  sites.insert("blk.48.ssm_out.weight");
  failure(2560, "direction.r", sites, "unknown site: blk.48.ssm_out.weight");
}

void Controls() {
  q::RefusalControls controls;
  std::string error;
  const auto parse = [&](const char* enabled, const char* strength,
                         const char* mode, const char* first, const char* last,
                         bool legacy_path) {
    error.clear();
    return q::ParseRefusalControls(enabled, strength, mode, first, last,
                                   legacy_path, &controls, &error);
  };
  Require(parse(nullptr, nullptr, nullptr, nullptr, nullptr, false) &&
              !controls.enabled && controls.mode == q::RefusalMode::kSites &&
              controls.first_layer == 4 && controls.last_layer == 44 &&
              !controls.strength_override,
          "default controls");
  Require(parse(nullptr, nullptr, nullptr, nullptr, nullptr, true) &&
              controls.enabled,
          "legacy path enables projection");
  Require(parse("off", nullptr, nullptr, nullptr, nullptr, true) &&
              !controls.enabled,
          "explicit off overrides legacy path");
  Require(parse("on", "1.0", "stream", "1", "47", false) && controls.enabled &&
              controls.strength_override == 1.0F &&
              controls.mode == q::RefusalMode::kStream &&
              controls.first_layer == 1 && controls.last_layer == 47,
          "stream controls");
  for (const auto* invalid : {"nan", "inf", "-0.1", "4.1", "1tail"})
    Require(!parse("on", invalid, "sites", "4", "44", false),
            "invalid override accepted");
  Require(!parse("yes", "1", "sites", "4", "44", false), "bad toggle");
  Require(!parse("on", "1", "other", "4", "44", false), "bad mode");
  Require(!parse("on", "1", "stream", "0", "44", false), "layer zero");
  Require(!parse("on", "1", "stream", "5", "4", false), "empty range");
  Require(!parse("on", "1", "stream", "4", "48", false), "last layer");
  Require(!q::ParseRefusalControls("on", "1", "stream", "x", "44", false,
                                   &controls, nullptr),
          "invalid layer with null error destination");
  q::RefusalDirection direction;
  direction.identity = "vector\n";
  const auto baseline = q::RefusalConfigurationIdentity(
      direction, q::RefusalMode::kSites, 1.0F, 4, 44);
  Require(baseline != q::RefusalConfigurationIdentity(
                          direction, q::RefusalMode::kStream, 1.0F, 4, 44) &&
              baseline != q::RefusalConfigurationIdentity(
                              direction, q::RefusalMode::kSites, 2.0F, 4, 44) &&
              baseline != q::RefusalConfigurationIdentity(
                              direction, q::RefusalMode::kSites, 1.0F, 5, 44),
          "cache identities must distinguish configuration");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Arithmetic();
    Loader();
    Controls();
    if (argc == 2) {
      std::string error;
      Require(bool(q::LoadRefusalDirection(argv[1], &error)), error);
    }
    std::puts(
        "PASS refusal direction CPU arithmetic and strict loader (no HIP "
        "loaded)");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
