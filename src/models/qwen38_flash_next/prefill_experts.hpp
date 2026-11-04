#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_EXPERTS_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_EXPERTS_HPP_

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen38_flash_next/router_policy.hpp"

namespace gufo::models::qwen38_flash_next {
struct PrefillExpertsSettings {
  bool active{false};
  std::uint32_t keep{10};
  std::uint32_t tail{0};
  PrefillExpertsMode mode{PrefillExpertsMode::kRenorm};
  PrefillExpertsAudit audit{PrefillExpertsAudit::kLean};
  // Mass keeps the smallest original top-used prefix reaching tau, at least one
  // and at most keep, then renormalizes. Only tau=1 uses a 4*FLT_EPSILON
  // tolerance for FP32 accumulation. Saturated [1,0,...] therefore keeps one;
  // renorm:10 is the full-k control. Audit changes evidence, never numerics.
  float mass{1};
  std::vector<float> fold;
  std::string fold_sha256;
  std::vector<std::uint32_t> layer_keep;
  std::string kfile_sha256;
  std::uint32_t Keep(std::size_t layer) const {
    return layer_keep.empty() ? keep : layer_keep.at(layer);
  }
  const char* AuditName() const {
    switch (audit) {
      case PrefillExpertsAudit::kLean:
        return "lean";
      case PrefillExpertsAudit::kFull:
        return "full";
    }
    throw std::logic_error("invalid prefill expert audit");
  }
  const char* ModeName() const {
    switch (mode) {
      case PrefillExpertsMode::kRenorm:
        return "renorm";
      case PrefillExpertsMode::kRef10:
        return "ref10";
      case PrefillExpertsMode::kFold:
        return "fold";
      case PrefillExpertsMode::kMass:
        return "mass";
    }
    throw std::logic_error("invalid prefill expert mode");
  }
  std::string IdentityLine() const {
    if (!active)
      return {};
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "prefill_experts=" << ModeName() << ':' << keep;
    if (mode == PrefillExpertsMode::kMass)
      out << ":tau_bits=" << std::hex << std::bit_cast<std::uint32_t>(mass);
    if (mode == PrefillExpertsMode::kFold)
      out << ":scalars_sha256=" << fold_sha256;
    if (!layer_keep.empty())
      out << ":kfile_sha256=" << kfile_sha256;
    if (tail)
      out << ":tail=" << std::dec << tail;
    return out.str() + '\n';
  }
};

inline float PrefillExpertFloat(std::string_view text) {
  float value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !std::isfinite(value))
    throw std::invalid_argument("prefill experts: expected a finite float");
  return value;
}

inline PrefillExpertsSettings ParsePrefillExperts(
    const char* count, const char* mode, const char* mass,
    const char* fold_path, const char* audit = nullptr, std::uint32_t used = 10,
    std::uint32_t layers = 48, const char* tail = nullptr,
    const char* kfile_path = nullptr) {
  if (used == 0 || used > 32 || layers == 0)
    throw std::invalid_argument("invalid prefill expert geometry");
  PrefillExpertsSettings settings;
  settings.keep = used;
  if (kfile_path && count)
    throw std::invalid_argument(
        "GUFO_PREFILL_EXPERTS_KFILE replaces GUFO_PREFILL_EXPERTS");
  if (kfile_path && (used != 10 || layers != 48))
    throw std::invalid_argument(
        "GUFO_PREFILL_EXPERTS_KFILE requires Qwen3.8-Flash-Next");
  if (!count && !mode && !kfile_path) {
    // Auxiliary calibration inputs do not activate the knob.
    return settings;
  }
  settings.active = true;
  if (tail) {
    const std::string_view text(tail);
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), settings.tail);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        settings.tail > 65536)
      throw std::invalid_argument(
          "GUFO_PREFILL_EXPERTS_TAIL must be an integer in 0..65536");
    // This pass planner belongs to Flash-Next. Do not advertise a suffix
    // policy on A3B, which shares this parser but has a separate executor.
    if (settings.tail && (used != 10 || layers != 48))
      throw std::invalid_argument(
          "GUFO_PREFILL_EXPERTS_TAIL requires Qwen3.8-Flash-Next");
  }
  if (audit) {
    const std::string_view text(audit);
    if (text == "lean")
      settings.audit = PrefillExpertsAudit::kLean;
    else if (text == "full")
      settings.audit = PrefillExpertsAudit::kFull;
    else
      throw std::invalid_argument("invalid GUFO_PREFILL_EXPERTS_AUDIT");
  }
  if (count) {
    const std::string_view text(count);
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), settings.keep);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        settings.keep < 1 || settings.keep > used)
      throw std::invalid_argument(
          "GUFO_PREFILL_EXPERTS must be an integer in 1.." +
          std::to_string(used));
  }
  if (mode) {
    const std::string_view text(mode);
    if (text == "renorm")
      settings.mode = PrefillExpertsMode::kRenorm;
    else if (text == "ref10")
      settings.mode = PrefillExpertsMode::kRef10;
    else if (text == "fold")
      settings.mode = PrefillExpertsMode::kFold;
    else if (text == "mass")
      settings.mode = PrefillExpertsMode::kMass;
    else
      throw std::invalid_argument("invalid GUFO_PREFILL_EXPERTS_MODE");
  }
  if (mass && settings.mode != PrefillExpertsMode::kMass)
    throw std::invalid_argument("GUFO_PREFILL_EXPERTS_MASS requires mass mode");
  if (fold_path && settings.mode != PrefillExpertsMode::kFold)
    throw std::invalid_argument("GUFO_PREFILL_EXPERTS_FOLD requires fold mode");
  if (kfile_path && settings.mode == PrefillExpertsMode::kMass)
    throw std::invalid_argument(
        "GUFO_PREFILL_EXPERTS_KFILE does not support mass mode");
  if (settings.mode == PrefillExpertsMode::kMass) {
    if (!mass)
      throw std::invalid_argument(
          "mass mode requires GUFO_PREFILL_EXPERTS_MASS");
    settings.mass = PrefillExpertFloat(mass);
    if (!(settings.mass > 0 && settings.mass <= 1))
      throw std::invalid_argument("prefill expert mass must be in (0,1]");
  }
  if (settings.mode == PrefillExpertsMode::kFold) {
    if (!fold_path || !*fold_path)
      throw std::invalid_argument(
          "fold mode requires GUFO_PREFILL_EXPERTS_FOLD");
    std::ifstream in(fold_path);
    if (!in)
      throw std::invalid_argument("cannot open prefill expert fold file");
    settings.fold.resize(layers);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(layers) * 4);
    const auto shape_error = "prefill expert fold file needs exactly " +
                             std::to_string(layers) + " lines";
    std::string line;
    for (std::size_t layer = 0; layer < settings.fold.size(); ++layer) {
      if (!std::getline(in, line))
        throw std::invalid_argument(shape_error);
      const auto first = line.find_first_not_of(" \t\r");
      const auto last = line.find_last_not_of(" \t\r");
      const auto text =
          first == std::string::npos
              ? std::string_view{}
              : std::string_view(line).substr(first, last - first + 1);
      settings.fold[layer] = PrefillExpertFloat(text);
      const auto bits = std::bit_cast<std::uint32_t>(settings.fold[layer]);
      for (unsigned b = 0; b < 4; ++b)
        bytes[layer * 4 + b] = static_cast<std::uint8_t>(bits >> (b * 8));
    }
    if (std::getline(in, line) || !in.eof())
      throw std::invalid_argument(shape_error);
    settings.fold_sha256 = crypto::Sha256Hex(bytes);
  }
  if (kfile_path) {
    if (!*kfile_path)
      throw std::invalid_argument(
          "GUFO_PREFILL_EXPERTS_KFILE requires a file path");
    // Binary mode preserves CR so CRLF cannot silently become valid LF input.
    std::ifstream in(kfile_path, std::ios::binary);
    if (!in)
      throw std::invalid_argument("cannot open prefill expert k file");
    settings.layer_keep.resize(layers);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(layers) * 4);
    const auto shape_error = "prefill expert k file needs exactly " +
                             std::to_string(layers) + " lines";
    std::string line;
    for (std::size_t layer = 0; layer < settings.layer_keep.size(); ++layer) {
      if (!std::getline(in, line))
        throw std::invalid_argument(shape_error);
      auto& keep = settings.layer_keep[layer];
      const auto result =
          std::from_chars(line.data(), line.data() + line.size(), keep);
      if (result.ec != std::errc{} || result.ptr != line.data() + line.size() ||
          keep < 1 || keep > used)
        throw std::invalid_argument(
            "prefill expert k file values must be integers in 1..10");
      for (unsigned b = 0; b < 4; ++b)
        bytes[layer * 4 + b] = static_cast<std::uint8_t>(keep >> (b * 8));
    }
    if (std::getline(in, line) || !in.eof())
      throw std::invalid_argument(shape_error);
    settings.kfile_sha256 = crypto::Sha256Hex(bytes);
  }
  return settings;
}

template<std::uint32_t Used = 10, std::uint32_t Layers = 48>
inline const PrefillExpertsSettings& GetPrefillExperts() {
  static const auto settings =
      ParsePrefillExperts(std::getenv("GUFO_PREFILL_EXPERTS"),
                          std::getenv("GUFO_PREFILL_EXPERTS_MODE"),
                          std::getenv("GUFO_PREFILL_EXPERTS_MASS"),
                          std::getenv("GUFO_PREFILL_EXPERTS_FOLD"),
                          std::getenv("GUFO_PREFILL_EXPERTS_AUDIT"), Used,
                          Layers, std::getenv("GUFO_PREFILL_EXPERTS_TAIL"),
                          std::getenv("GUFO_PREFILL_EXPERTS_KFILE"));
  return settings;
}
}  // namespace gufo::models::qwen38_flash_next
#endif
