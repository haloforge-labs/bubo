#pragma once

#include <charconv>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace bubo::prepare::metadata {
using gufo::json::Value;
inline constexpr std::size_t kMaxJsonBytes = 1024 * 1024;

inline Value Parse(std::string_view text) {
  if (text.size() > kMaxJsonBytes)
    throw std::invalid_argument("metadata exceeds 1 MiB");
  return gufo::json::parse(text);
}

inline std::string Read(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("cannot open metadata file");
  // Do not allocate from an untrusted filesystem length or read past the cap.
  std::string text(kMaxJsonBytes + 1, '\0');
  file.read(text.data(), static_cast<std::streamsize>(text.size()));
  const auto count = file.gcount();
  if (file.bad() || count > static_cast<std::streamsize>(kMaxJsonBytes))
    throw std::invalid_argument("unreadable metadata or exceeds 1 MiB");
  text.resize(static_cast<std::size_t>(count));
  return text;
}

inline void Keys(const Value& value,
                 std::initializer_list<std::string_view> allowed) {
  if (!value.is_object())
    throw std::invalid_argument("expected JSON object");
  for (const auto& [key, ignored] : value.members()) {
    bool found = false;
    for (auto name : allowed)
      found |= key == name;
    if (!found)
      throw std::invalid_argument("unknown metadata key: " + key);
  }
}

inline const Value& Required(const Value& object, const std::string& key) {
  const auto* value = object.find(key);
  if (!value)
    throw std::invalid_argument("missing metadata key: " + key);
  return *value;
}

inline const std::string& String(const Value& value) {
  if (!value.is_string())
    throw std::invalid_argument("expected JSON string");
  return value.str();
}

inline std::uint64_t Decimal(std::string_view text) {
  std::uint64_t number = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), number);
  if (text.empty() || (text.size() > 1 && text.front() == '0') ||
      error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument("expected canonical unsigned decimal integer");
  return number;
}

inline std::uint64_t Integer(const Value& value) {
  if (!value.is_number())
    throw std::invalid_argument("expected JSON integer");
  const auto result = Decimal(value.dump());
  if (result > 9007199254740991ULL)
    throw std::invalid_argument("JSON integer exceeds exact 53-bit limit");
  return result;
}
}  // namespace bubo::prepare::metadata
