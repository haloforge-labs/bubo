#pragma once
#include <array>
#include <string_view>

namespace gufo::server {
enum class BindingExpectation { bound, disabled, rejected };
struct ArtefactBindingCase {
  std::string_view profile, artefact, source;
  BindingExpectation expected;
};
// The one T10 contract table consumed by its documentation renderer and tests.
inline constexpr std::array<ArtefactBindingCase, 48> kArtefactBindingCases = {{
#define BINDING_ROWS(profile, artefact, flag, inherited)           \
  {profile, artefact, "flag", BindingExpectation::flag},           \
      {profile, artefact, "env", BindingExpectation::inherited}, { \
    profile, artefact, "manifest", BindingExpectation::inherited   \
  }
    BINDING_ROWS("default", "twin", bound, bound),
    BINDING_ROWS("default", "draft_head", bound, bound),
    BINDING_ROWS("default", "mtp", bound, bound),
    BINDING_ROWS("default", "mmproj", bound, bound),
    BINDING_ROWS("subagents", "twin", bound, bound),
    BINDING_ROWS("subagents", "draft_head", bound, bound),
    BINDING_ROWS("subagents", "mtp", bound, bound),
    BINDING_ROWS("subagents", "mmproj", bound, bound),
    BINDING_ROWS("stock", "twin", bound, disabled),
    BINDING_ROWS("stock", "draft_head", bound, disabled),
    BINDING_ROWS("stock", "mtp", bound, disabled),
    BINDING_ROWS("stock", "mmproj", bound, bound),
    BINDING_ROWS("exact", "twin", rejected, disabled),
    BINDING_ROWS("exact", "draft_head", rejected, disabled),
    BINDING_ROWS("exact", "mtp", rejected, disabled),
    BINDING_ROWS("exact", "mmproj", bound, bound),
#undef BINDING_ROWS
}};
}  // namespace gufo::server
