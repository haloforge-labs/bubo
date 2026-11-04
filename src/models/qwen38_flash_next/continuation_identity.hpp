#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_IDENTITY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_CONTINUATION_IDENTITY_HPP_

#include <sstream>
#include <string>
#include <string_view>

#include "src/core/gguf_identity.hpp"
#include "src/models/qwen38_flash_next/continuation_layout.hpp"
#include "src/models/qwen38_flash_next/prefill_experts.hpp"

namespace gufo::models::qwen38_flash_next {
// The complete c4 canonical body, shared with the CPU regression test.
inline std::string FlashNextCompatibilityIdentity(
    std::string_view artifact_fingerprint, std::string_view mtp_fingerprint,
    bool has_mtp, std::uint32_t max_context, std::uint32_t max_draft_tokens,
    std::uint32_t decode_concurrency, std::uint32_t snapshot_version,
    std::string_view template_sha256, const PrefillExpertsSettings& experts,
    KvCacheType kv_cache_type = KvCacheType::kF16) {
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=qwen38-flash-next\n"
           << "artifact_id=" << core::kGgufIdentityScheme << ':'
           << artifact_fingerprint << '\n'
           << "tokenizer=embedded-in-artifact\n"
           << "chat_template=qwen38-reasoning-compiled-v3\n"
           << "chat_template_reference_sha256=" << template_sha256 << '\n'
           << "state_abi=qwen38-flash-next-rocm-session-v1\n"
           << "payload_layout=qfn-rocm-session-snapshot-v" << snapshot_version
           << '\n'
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "adapters=none\n"
           << KvCacheCompatibilityIdentity(kv_cache_type);
  if (has_mtp) {
    identity << "draft_backend=qfn-mtp-v1\n"
             << "draft_artifact_id=" << core::kGgufIdentityScheme << ':'
             << mtp_fingerprint << '\n'
             << "draft_max_tokens=" << max_draft_tokens << '\n'
             << "draft_cost_concurrency=" << decode_concurrency << '\n';
  }
  identity << experts.IdentityLine();
  return identity.str();
}
}  // namespace gufo::models::qwen38_flash_next
#endif
