#include "src/models/qwen38_flash_next/device_accept.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>

#include "src/models/qwen38_flash_next/decode_cycle.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"

namespace q = gufo::models::qwen38_flash_next;
namespace s = gufo::sampling;
using R = q::DeviceAcceptReason;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Eligibility() {
  const std::array<std::int32_t, 4> chain{10, 11, 12, 13};
  const auto reason = [&](s::SamplingConfig c, bool logprobs = false,
                          bool batch = false) {
    s::SamplerState sampler(c);
    if (!c.constraint)
      sampler.Accept(chain[0]);
    const auto before = sampler.history().size();
    const auto result =
        q::DeviceAcceptEligibility(sampler, chain, logprobs, batch);
    Require(sampler.history().size() == before,
            "certificate changed live history");
    return result;
  };
  Require(reason({}) == R::kEligible, "plain greedy not eligible");
  s::SamplingConfig c;
  c.temperature = 0.7F;
  Require(reason(c) == R::kSampled, "sampled request admitted");
  c = {};
  auto constraint = std::make_shared<s::TokenConstraint>();
  constraint->grammar = s::JsonConstraint::Object();
  constraint->vocabulary = std::make_shared<s::ConstraintVocabulary>(
      128,
      [](std::uint32_t) { return s::ConstraintVocabulary::Piece{"x", false}; });
  c.constraint = constraint;
  Require(reason(c) == R::kConstraint, "grammar admitted");
  Require(reason({}, true) == R::kLogprobs, "logprobs admitted");
  c = {};
  c.frequency_penalty = 0.2F;
  Require(reason(c) == R::kPenalties, "penalties admitted");
  c = {};
  auto b = std::make_shared<s::BehaviourConfig>();
  b->think_nudge = true;
  c.behaviour = b;
  Require(reason(c) == R::kThinkNudge, "think nudge admitted");
  b->think_nudge = false;
  b->logit_bias_enabled = true;
  Require(reason(c) == R::kBehaviour, "token-specific bias admitted");
  b->logit_bias_enabled = false;
  b->model_suppress_tokens = true;
  Require(reason(c) == R::kBehaviour, "token-specific suppression admitted");
  b->model_suppress_tokens = false;
  b->marker_bias = -1;
  Require(reason(c) == R::kBehaviour, "token-specific marker bias admitted");
  b->marker_bias = 0;
  b->spec_eog_stop = true;
  b->eog_ids = {99};
  b->eog_boundary = true;
  b->eog_min_prob = 0.2;
  b->eog_defer_p = 0.9;
  b->loop_breaker = true;
  b->token_pieces =
      std::make_shared<const std::vector<std::string>>(128, "word");
  Require(reason(c) == R::kEligible,
          "ordinary default answer policy not eligible");
  Require(reason({}, false, true) == R::kBatch, "batch admitted");

  s::SamplerState forced(c);
  forced.Accept(chain[0]);
  auto draw = forced.SaveDrawState();
  draw.behaviour->Force({11}, "answer_boundary");
  forced.RestoreDrawState(std::move(draw));
  Require(!forced.ForcedToken() && forced.CanSelectArgmax(11, true) &&
              !forced.CanSelectArgmax(12, true),
          "generic force regression fixture is not selective");
  Require(
      q::DeviceAcceptEligibility(forced, chain, false, false) == R::kConstraint,
      "matching draft hid a generic forced choice");
  const std::array<std::int32_t, 4> stop_chain{10, 99, 12, 13};
  s::SamplerState stopped(c);
  stopped.Accept(stop_chain[0]);
  Require(
      q::DeviceAcceptEligibility(stopped, stop_chain, false, false) == R::kStop,
      "EOG draft certified ordinary winners");

  // A certificate promises all non-stop winners work at EVERY accepted
  // prefix, including a loop intervention caused by a preceding draft.
  for (unsigned count = 0; count < 600; ++count) {
    s::SamplerState sampler(c);
    for (unsigned i = 0; i < count; ++i)
      sampler.Accept(10);
    const std::array<std::int32_t, 4> repeated{10, 10, 10, 10};
    if (q::DeviceAcceptEligibility(sampler, repeated, false, false) !=
        R::kEligible)
      continue;
    for (unsigned row = 1; row < repeated.size(); ++row) {
      for (s::TokenId id = 0; id < 128; ++id)
        if (id != 99)
          Require(sampler.CanSelectArgmax(id, true),
                  "certificate admitted a forbidden non-stop winner");
      sampler.Accept(repeated[row]);
    }
  }

  // Exercise the actual CanSelectArgmax refusal, rather than only config
  // exclusions: a production counterfactual needs a full probability row.
  b->production_sampling = true;
  b->native_temperature = 1;
  Require(reason(c) == R::kConstraint, "CanSelectArgmax refusal ignored");
}

void KeysAndParsing() {
  q::DecodeCycleOptions options;
  std::string error;
  const auto parse = [&](const char* value) {
    return q::ParseDecodeCycleOptions(nullptr, nullptr, nullptr, nullptr,
                                      nullptr, &options, &error, value);
  };
  Require(parse(nullptr) && !options.device_accept,
          "source default is not off");
  Require(parse("1") && options.device_accept, "on switch refused");
  Require(parse("0") && !options.device_accept, "off switch refused");
  Require(!parse("") && !parse("true") && !parse("2"),
          "invalid switch accepted");
  for (std::uint32_t width = 2; width <= 8; ++width) {
    const auto key = q::DeviceRollbackKey(width);
    Require((key & 0xFFFFU) == width, "width field changed");
    Require((key & q::kDeviceAcceptKey) != 0, "missing device key bit");
    Require((key & ((std::uint64_t{1} << 32) | (std::uint64_t{1} << 40))) == 0,
            "device key misread as verify or MTP");
    for (unsigned flags = 0; flags < 64; ++flags) {
      const auto mtp = q::MtpGraphKey(width, flags & 1, flags & 2, flags & 4,
                                      flags & 8, flags & 16, flags & 32);
      Require((mtp & q::kDeviceAcceptKey) == 0 && mtp != key,
              "MTP/device key collision");
    }
  }
}

int main() {
  try {
    Eligibility();
    KeysAndParsing();
    std::puts(
        "PASS device accept eligibility, strict default-off switch and key "
        "layouts");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
