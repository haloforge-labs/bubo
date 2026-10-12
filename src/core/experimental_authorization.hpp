// SPDX-License-Identifier: MIT
#ifndef GUFO_CORE_EXPERIMENTAL_AUTHORIZATION_HPP_
#define GUFO_CORE_EXPERIMENTAL_AUTHORIZATION_HPP_

#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::server {
struct ReleaseInputs;
struct ReleaseResolution;
ReleaseResolution ResolveReleaseSettings(const ReleaseInputs&);
}  // namespace gufo::server

namespace gufo::core {
enum class ExperimentalFeature { Prefill };

// A value capability: only the validated release resolver can authorize it.
// Default construction is denied, including all non-CLI model-create callers.
// No environment lookup, mutable global mode, or public enable constructor.
class ExperimentalAuthorization {
public:
  ExperimentalAuthorization() = default;
  bool Allows(ExperimentalFeature feature) const noexcept {
    return allowed_ && feature_ == feature;
  }
  const std::string& reason() const noexcept { return reason_; }
  void Require(ExperimentalFeature feature) const {
    if (!Allows(feature))
      throw std::invalid_argument("experimental mode denied: " + reason_);
  }
#ifdef GUFO_NPU_TESTING
  // Fake-XRT targets only. Never compiled into the product/counter host.
  static ExperimentalAuthorization FakePrefill() {
    return {ExperimentalFeature::Prefill, true, "fake-XRT test"};
  }
#endif

private:
  friend server::ReleaseResolution server::ResolveReleaseSettings(
      const server::ReleaseInputs&);
  ExperimentalAuthorization(ExperimentalFeature feature, bool allowed,
                            std::string reason)
      : feature_(feature), allowed_(allowed), reason_(std::move(reason)) {}
  ExperimentalFeature feature_ = ExperimentalFeature::Prefill;
  bool allowed_ = false;
  std::string reason_ = "no explicit CLI authorization";
};

// The production startup and CPU counter use this identical creation boundary.
// The callback is never evaluated for a denied capability.
template<class Factory>
bool CreateAuthorizedPrefill(const ExperimentalAuthorization& authorization,
                             Factory&& factory) {
  if (!authorization.Allows(ExperimentalFeature::Prefill))
    return false;
  std::forward<Factory>(factory)(authorization);
  return true;
}
}  // namespace gufo::core
#endif
