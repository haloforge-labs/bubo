#include "src/cli/machine_warnings.hpp"

#include <charconv>
#include <cstdio>
#include <cwchar>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace bubo::cli {
namespace {

std::string Gib(std::uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.1f", bytes / 1073741824.0);
  return text;
}

bool ParseUnsigned(std::string_view text, std::uint64_t* value) {
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

}  // namespace

std::optional<PageFileSetting> ParsePagingFilesEntry(std::string_view entry) {
  while (!entry.empty() && (entry.back() == ' ' || entry.back() == '\r' ||
                            entry.back() == '\n'))
    entry.remove_suffix(1);
  while (!entry.empty() && entry.front() == ' ')
    entry.remove_prefix(1);
  if (entry.empty())
    return std::nullopt;
  // The path may contain spaces; the sizes are the last two words.
  std::vector<std::string_view> words;
  std::size_t start = 0;
  while (start < entry.size()) {
    const auto end = entry.find(' ', start);
    if (end == std::string_view::npos) {
      words.push_back(entry.substr(start));
      break;
    }
    if (end > start)
      words.push_back(entry.substr(start, end - start));
    start = end + 1;
  }
  PageFileSetting setting;
  std::uint64_t initial = 0, maximum = 0;
  if (words.size() >= 3 && ParseUnsigned(words[words.size() - 2], &initial) &&
      ParseUnsigned(words.back(), &maximum)) {
    setting.initial_mib = initial;
    setting.maximum_mib = maximum;
    const auto sizes_begin =
        static_cast<std::size_t>(words[words.size() - 2].data() - entry.data());
    setting.path = std::string(entry.substr(0, sizes_begin));
    while (!setting.path.empty() && setting.path.back() == ' ')
      setting.path.pop_back();
  } else {
    setting.path = std::string(entry);
  }
  // Windows writes "?:\pagefile.sys" without sizes, or with 0 0, when the
  // system chooses the size.
  setting.system_managed =
      setting.path.starts_with("?:") ||
      (setting.initial_mib == 0 && setting.maximum_mib == 0);
  return setting;
}

std::optional<std::vector<PageFileSetting>> ReadPageFileSettings() {
#ifdef _WIN32
  constexpr const wchar_t* kKey =
      L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory "
      L"Management";
  DWORD bytes = 0;
  const auto probe = RegGetValueW(HKEY_LOCAL_MACHINE, kKey, L"PagingFiles",
                                  RRF_RT_REG_MULTI_SZ, nullptr, nullptr, &bytes);
  if (probe != ERROR_SUCCESS)
    return std::nullopt;
  std::wstring buffer(bytes / sizeof(wchar_t) + 1, L'\0');
  if (RegGetValueW(HKEY_LOCAL_MACHINE, kKey, L"PagingFiles",
                   RRF_RT_REG_MULTI_SZ, nullptr, buffer.data(),
                   &bytes) != ERROR_SUCCESS)
    return std::nullopt;
  std::vector<PageFileSetting> result;
  const wchar_t* line = buffer.c_str();
  while (*line) {
    const auto length = static_cast<int>(wcslen(line));
    std::string utf8(static_cast<std::size_t>(length) * 4 + 1, '\0');
    const auto written = WideCharToMultiByte(CP_UTF8, 0, line, length,
                                             utf8.data(),
                                             static_cast<int>(utf8.size()),
                                             nullptr, nullptr);
    utf8.resize(static_cast<std::size_t>(written));
    if (const auto setting = ParsePagingFilesEntry(utf8))
      result.push_back(*setting);
    line += length + 1;
  }
  return result;
#else
  return std::nullopt;
#endif
}

std::vector<MachineWarning> EvaluateMachineWarnings(
    const MachineWarningInputs& inputs) {
  std::vector<MachineWarning> warnings;
  if (inputs.gpu_reservation_bytes &&
      *inputs.gpu_reservation_bytes < (inputs.is_96gb_strix_halo
                                           ? k96GbGpuReservationBytes
                                           : kExpectedGpuReservationBytes)) {
    if (inputs.is_96gb_strix_halo)
      warnings.push_back({"gpu_reservation",
                          "The GPU memory reservation is " +
                              Gib(*inputs.gpu_reservation_bytes) + " GiB.",
                          k96GbMemoryAdvice});
    else
      warnings.push_back(
          {"gpu_reservation",
           "The GPU memory reservation is " +
               Gib(*inputs.gpu_reservation_bytes) +
               " GiB. The tested setting is 96 GB. Large models and long "
               "contexts can run out of GPU memory.",
           "Set the GPU memory reservation (Variable Graphics Memory) to 96 "
           "GB, "
           "then restart Windows."});
  }
  if (inputs.page_files) {
    const auto& files = *inputs.page_files;
    bool fixed = !files.empty();
    for (const auto& file : files)
      fixed = fixed && !file.system_managed &&
              file.initial_mib == file.maximum_mib;
    if (files.empty()) {
      warnings.push_back(
          {"page_file", "No page file is configured. The tested setting is a "
                        "fixed-size page file.",
           "Set a fixed-size page file (initial size equal to maximum size) in "
           "System Properties > Advanced > Performance > Virtual memory."});
    } else if (!fixed) {
      warnings.push_back(
          {"page_file",
           "The page file is not a fixed size. The tested setting is a "
           "fixed-size page file. A page file that grows can stall a large "
           "model load or a long prefill.",
           "Set a fixed-size page file (initial size equal to maximum size) in "
           "System Properties > Advanced > Performance > Virtual memory."});
    }
  }
  return warnings;
}

MachineWarningInputs ReadMachineWarningInputs(MachineProbe& probe) {
  MachineWarningInputs inputs;
  const auto budgets = ReadPlanningMemoryBudgets(probe);
  inputs.gpu_reservation_bytes = budgets.gpu_dedicated;
  inputs.is_96gb_strix_halo = Is96GiBStrixHalo(budgets);
  inputs.page_files = ReadPageFileSettings();
  const auto mode = probe.EffectivePowerMode();
  if (mode.available && mode.value.is_number())
    inputs.effective_power_mode =
        static_cast<std::int64_t>(mode.value.as_double());
  return inputs;
}

std::string MachineWarningLogLine(const std::vector<MachineWarning>& warnings) {
  if (warnings.empty())
    return {};
  std::string line = "event=machine_settings differs=";
  for (std::size_t i = 0; i < warnings.size(); ++i) {
    if (i)
      line += ',';
    line += warnings[i].check;
  }
  for (const auto& warning : warnings)
    line += " " + warning.check + "=\"" + warning.message + "\"";
  line += " action=run_bubo_diagnose_for_remedies";
  return line;
}

}  // namespace bubo::cli
