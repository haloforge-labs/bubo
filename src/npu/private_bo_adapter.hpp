// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_PRIVATE_BO_ADAPTER_HPP_
#define GUFO_NPU_PRIVATE_BO_ADAPTER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/experimental_authorization.hpp"

namespace gufo::npu {

inline constexpr std::size_t kABytes = 11796480;
inline constexpr std::size_t kBBytes = 2949120;
inline constexpr std::size_t kCBytes = 4718592;

enum class ErrorKind { Validation, Xrt, Timeout, Completion, State };

class AdapterError : public std::runtime_error {
public:
  AdapterError(ErrorKind kind, std::string text, bool dma_possible = false,
               std::optional<int> ert_state = {},
               std::optional<int> native_code = {});
  const ErrorKind kind;
  const bool dma_possible;
  const std::optional<int> ert_state;
  const std::optional<int> native_code;
};

std::uint32_t InstructionWordCount(std::uint64_t bytes);

class PrivateBoAdapter {
public:
  static std::unique_ptr<PrivateBoAdapter> Create(
      const core::ExperimentalAuthorization& authorization,
      const std::filesystem::path& artifact_directory,
      std::span<const std::byte> a, std::span<const std::byte> b,
      std::chrono::milliseconds timeout = std::chrono::milliseconds(8000));
  ~PrivateBoAdapter();
  PrivateBoAdapter(const PrivateBoAdapter&) = delete;
  PrivateBoAdapter& operator=(const PrivateBoAdapter&) = delete;

  void SetInputs(std::span<const std::byte> a, std::span<const std::byte> b);
  void ExecuteBatch(
      std::span<const std::byte> a,
      const std::vector<std::vector<std::uint8_t>>& weights,
      const std::function<void(std::size_t, std::span<const std::byte>)>&
          consume);
  void Heartbeat();
  void PrepareWeight(std::span<const std::byte> b);
  void SetResidentInputs(std::span<const std::byte> a,
                         std::span<const std::byte> b, bool copy_a);
  void Submit();
  void Finish();
  void CopyOutput(std::span<std::byte> destination);
  bool DmaPossible() const noexcept;
  bool Failed() const noexcept;
  std::string PowerMode();

private:
  struct Impl;
  PrivateBoAdapter();
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::npu
#endif  // GUFO_NPU_PRIVATE_BO_ADAPTER_HPP_
