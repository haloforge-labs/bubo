#include "src/cache/identity.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace gufo::cache {
namespace {

std::span<const std::uint8_t> PrefixInputIdentity(
    std::span<const std::uint8_t> complete,
    std::span<const InputPrefix> prefixes, Rows count) {
  for (const auto& prefix : prefixes) {
    if (count <= prefix.token_count)
      return prefix.identity;
  }
  return complete;
}

void AppendWord(Identity& identity, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    identity.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
}

Identity PolicyIdentity(std::span<const std::uint8_t> input,
                        std::span<const RowPolicyFragment> policies,
                        Rows count) {
  if (count == 0)
    return {input.begin(), input.end()};
  Identity identity;
  constexpr std::uint8_t tag[] = {'r', 'o', 'w', '-', 'p', 'o', 'l', 'i',
                                 'c', 'y', '-', 'v', '1', 0};
  identity.insert(identity.end(), std::begin(tag), std::end(tag));
  AppendWord(identity, input.size());
  identity.insert(identity.end(), input.begin(), input.end());
  // Each fragment follows its complete parent chain. Leave the last consumed
  // fragment open: the lookup's token boundary already binds its consumed
  // length. In particular, an all-cut prefix contains no future L or B.
  for (const auto& policy : policies) {
    if (policy.first >= count)
      break;
    const bool closed = policy.end < count;
    identity.push_back(closed ? 1 : 0);
    AppendWord(identity, policy.first);
    AppendWord(identity, policy.descriptor.size());
    identity.insert(identity.end(), policy.descriptor.begin(),
                    policy.descriptor.end());
    if (!closed)
      break;
    AppendWord(identity, policy.end);
  }
  return identity;
}
}  // namespace

InputIdentity::InputIdentity(Rows token_count, Identity complete,
                             std::vector<InputPrefix> prefixes,
                             std::vector<RowPolicyFragment> policies)
    : token_count_(token_count),
      complete_(std::move(complete)),
      prefixes_(std::move(prefixes)) {
  Rows previous = 0;
  for (const auto& prefix : prefixes_) {
    if (prefix.token_count < previous || prefix.token_count > token_count_)
      throw std::invalid_argument("input identity boundaries are invalid");
    previous = prefix.token_count;
  }
  previous = 0;
  for (auto& policy : policies) {
    if (policy.first != previous || policy.end <= policy.first ||
        policy.end > token_count_ || policy.descriptor.empty())
      throw std::invalid_argument("input row policies are invalid");
    previous = policy.end;
    if (!policies_.empty() &&
        policies_.back().descriptor == policy.descriptor)
      policies_.back().end = policy.end;
    else
      policies_.push_back(std::move(policy));
  }
  if (!policies_.empty() && previous != token_count_)
    throw std::invalid_argument("input row policies do not cover prompt");
  if (policies_.empty())
    return;
  policy_complete_ = PolicyIdentity(complete_, policies_, token_count_);
  std::vector<Rows> boundaries{0};
  for (const auto& prefix : prefixes_)
    boundaries.push_back(prefix.token_count);
  for (const auto& policy : policies_)
    if (policy.end < token_count_)
      boundaries.push_back(policy.end);
  std::ranges::sort(boundaries);
  boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                   boundaries.end());
  policy_prefixes_.reserve(boundaries.size());
  for (const auto boundary : boundaries)
    policy_prefixes_.push_back(
        {boundary, PolicyIdentity(
                       PrefixInputIdentity(complete_, prefixes_, boundary),
                       policies_, boundary)});
}

std::span<const std::uint8_t> InputIdentity::At(Rows count) const {
  if (count > token_count_)
    throw std::invalid_argument("input identity query exceeds prompt length");
  return policies_.empty()
             ? PrefixInputIdentity(complete_, prefixes_, count)
             : PrefixInputIdentity(policy_complete_, policy_prefixes_, count);
}

InputIdentity InputIdentity::Through(Rows count) const {
  const auto complete =
      PrefixInputIdentity(complete_, prefixes_, std::min(count, token_count_));
  std::vector<InputPrefix> prefixes;
  for (const auto& prefix : prefixes_) {
    if (prefix.token_count <= count)
      prefixes.push_back(prefix);
  }
  return InputIdentity(count, {complete.begin(), complete.end()},
                       std::move(prefixes), PoliciesThrough(count));
}

std::vector<RowPolicyFragment> InputIdentity::PoliciesThrough(Rows count) const {
  std::size_t fragments = 0;
  for (const auto& policy : policies_)
    fragments += policy.first < count;
  std::vector<RowPolicyFragment> result;
  result.reserve(fragments);
  for (const auto& policy : policies_) {
    if (policy.first >= count)
      break;
    result.push_back({policy.first, std::min(policy.end, count),
                      policy.descriptor});
  }
  if (!result.empty() && count > token_count_)
    result.back().end = count;
  return result;
}

bool InputIdentity::Matches(const InputIdentity& other, Rows count) const {
  return std::ranges::equal(At(count), other.At(count));
}

}  // namespace gufo::cache
