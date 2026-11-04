#include "src/models/qwen38_flash_next/draft_chain.hpp"

#include <cstdio>
#include <set>
#include <stdexcept>

#include "src/models/qwen38_flash_next/device_accept.hpp"

namespace q = gufo::models::qwen38_flash_next;
using R = q::DraftChainReason;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  try {
    Require(q::DraftChainEligibility(false, q::DraftFalsify::kNone, false,
                                     false) == R::kEligible,
            "greedy chain refused");
    Require(q::DraftChainEligibility(true, q::DraftFalsify::kNone, false,
                                     false) == R::kSampled,
            "sampled admitted");
    Require(q::DraftChainEligibility(false, q::DraftFalsify::kWrong, false,
                                     false) == R::kFalsifier,
            "falsifier admitted");
    Require(q::DraftChainEligibility(false, q::DraftFalsify::kNone, true,
                                     false) == R::kLookup,
            "lookup admitted");
    Require(q::DraftChainEligibility(false, q::DraftFalsify::kNone, false,
                                     true) == R::kBatch,
            "batch admitted");
    q::DecodeCycleOptions options;
    std::string error;
    const auto parse = [&](const char* value) {
      return q::ParseDecodeCycleOptions(nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &options, &error, nullptr,
                                        value);
    };
    Require(parse(nullptr) && options.draft_chain, "chain default not on");
    Require(parse("1") && options.draft_chain, "chain on refused");
    Require(parse("0") && !options.draft_chain, "chain off refused");
    Require(!parse("") && !parse("true"), "invalid chain switch accepted");
    std::set<std::uint64_t> keys;
    for (unsigned rows = 1; rows <= 8; ++rows)
      for (unsigned depth = 1; depth <= 7; ++depth)
        for (unsigned mask = 0; mask < (1U << depth); ++mask)
          for (bool head : {false, true}) {
            const auto key = q::DraftChainKey(rows, depth, mask, head);
            Require(keys.insert(key).second, "chain key collision");
            Require((key & q::kDraftChainKey) && !(key & q::kDeviceAcceptKey),
                    "chain bit layout");
            Require((key & (std::uint64_t{1} << 40)) &&
                        !(key & (std::uint64_t{1} << 48)),
                    "chain tag misread by trim");
            Require((key & 0xFFFFU) == rows, "chain row field");
          }
    // Independent serial oracle: block shadows advance after each step,
    // and the first sparse step may require an unbounded pooling backlog.
    for (unsigned ratio : {2U, 4U, 128U})
      for (unsigned budget : {8U, 16U, 16384U})
        for (unsigned start = budget - 8; start <= budget + 8; ++start)
          for (unsigned rows = 1; rows <= 8; ++rows)
            for (unsigned depth = 1; depth <= 7; ++depth) {
              auto position = start;
              auto blocks = start > budget ? start / ratio : 0;
              const auto plan =
                  q::PlanDraftChain(start, blocks, rows, depth, ratio, budget);
              bool bounded = true;
              unsigned mask = 0;
              for (unsigned step = 0; step < depth; ++step) {
                const unsigned count = step ? 1 : rows;
                position += count;
                if (position <= budget)
                  continue;
                mask |= 1U << step;
                unsigned jobs = 0;
                while (blocks < position / ratio) {
                  ++jobs;
                  ++blocks;
                }
                if (jobs > count / ratio + 1)
                  bounded = false;
              }
              Require(plan.end == position && plan.blocks == blocks &&
                          plan.sparse_mask == mask && plan.bounded == bounded,
                      "chain differs from serial pool oracle");
            }
    Require(!q::PlanDraftChain(0, 0, 1, 1, 0, 8).bounded,
            "zero ratio admitted");
    std::puts(
        "PASS chain exclusions, parser, graph layouts and per-step pool "
        "oracle");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
