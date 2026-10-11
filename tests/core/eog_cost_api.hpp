#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct EogCostConfig {
  unsigned flags = 31, state = 0;
  std::size_t vocab = 248320;
  std::vector<std::uint32_t> eog = {248044, 248046};
  int top_k = 20;
  float temperature = 1, top_p = .95F;
  bool extras = true;
  bool permissive_tools = false;
};
struct EogCostSampler {
  virtual ~EogCostSampler() = default;
  virtual void Reset(unsigned state, std::uint64_t seed) = 0;
  virtual std::uint32_t Row(std::span<const float> row,
                            std::span<const float> values,
                            std::span<const std::uint32_t> ids,
                            bool candidates) = 0;
  virtual void Commit(std::uint32_t token) = 0;
  virtual std::string Snapshot() const = 0;
  virtual std::string Distribution(std::span<const float> row) const = 0;
  virtual std::vector<std::string> Counters() const = 0;
};
std::unique_ptr<EogCostSampler> EogCostOld(const EogCostConfig&);
std::unique_ptr<EogCostSampler> EogCostNew(const EogCostConfig&);
