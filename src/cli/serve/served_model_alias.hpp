#ifndef GUFO_SERVER_SERVED_MODEL_ALIAS_HPP_
#define GUFO_SERVER_SERVED_MODEL_ALIAS_HPP_

#include <string_view>

namespace gufo::server {

/// Every served model also answers to this name, whatever its served id is.
inline constexpr std::string_view ServedModelAlias() {
  return "bubo";
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_SERVED_MODEL_ALIAS_HPP_
