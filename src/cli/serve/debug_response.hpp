#ifndef GUFO_SERVER_DEBUG_RESPONSE_HPP_
#define GUFO_SERVER_DEBUG_RESPONSE_HPP_

#include "src/cli/serve/text_generation_backend.hpp"

namespace gufo::server {

inline void AddDebugTimeline(json::Value& response,
                             const TextGenerationBackend::Result& result) {
  if (result.debug_timeline) {
    response["gufo"]["timeline"] = result.debug_timeline->Serialize();
    response["gufo"]["timeline_truncated"] = result.debug_timeline->truncated();
  }
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_DEBUG_RESPONSE_HPP_
