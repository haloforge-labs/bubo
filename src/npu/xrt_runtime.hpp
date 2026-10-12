#ifndef GUFO_NPU_XRT_RUNTIME_HPP_
#define GUFO_NPU_XRT_RUNTIME_HPP_

#include <array>
#include <filesystem>
#include <string>

namespace gufo::npu {

// The single Windows XRT layer shares the tooling host's pinned headers,
// import library and runtime identity. Construction has no runtime side
// effects.
struct RuntimeIdentity {
  std::filesystem::path path;
  std::string sha256;
  std::array<unsigned, 4> file_version;
};

namespace detail {
// CPU tests replace only OS loading/identity operations, never XRT class
// layouts.
class RuntimeOperations {
public:
  virtual ~RuntimeOperations() = default;
  virtual RuntimeIdentity Inspect() = 0;
  virtual void Load(const std::filesystem::path& path) = 0;
  virtual std::array<unsigned, 5> Versions() = 0;
};

void ValidateAndLoad(RuntimeOperations& operations);
}  // namespace detail

// Call only after immutable effective mode authorizes NPU prefill.
// No static initializer, environment lookup or implicit device creation.
void EnsureXrtRuntime();

}  // namespace gufo::npu

#endif  // GUFO_NPU_XRT_RUNTIME_HPP_
