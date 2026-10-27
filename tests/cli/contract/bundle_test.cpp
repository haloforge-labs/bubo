#include <zlib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/support_bundle.hpp"
#include "src/core/crypto/sha256.hpp"

namespace {
namespace fs = std::filesystem;
using gufo::json::Value;

constexpr std::array<std::string_view, 10> kSecrets = {
    "C:\\Users\\SensitiveUser\\Documents",
    "user-home-needle-7391",
    "203.0.113.77",
    "2001:db8::badd",
    "Bearer token-secret-7391",
    "http://proxy-user:proxy-pass@proxy.internal:3128",
    "prompt-private-7391",
    "response-private-7391",
    "serial-private-7391",
    "bubo-api-key-planted-7391"};

unsigned checks = 0;
void Require(bool condition, std::string_view reason) {
  if (!condition)
    throw std::runtime_error(std::string(reason));
  ++checks;
}

std::uint16_t U16(std::string_view bytes, std::size_t offset) {
  Require(offset + 2 <= bytes.size(), "short ZIP field");
  return static_cast<std::uint8_t>(bytes[offset]) |
         (static_cast<std::uint16_t>(
              static_cast<std::uint8_t>(bytes[offset + 1]))
          << 8);
}

std::uint32_t U32(std::string_view bytes, std::size_t offset) {
  return U16(bytes, offset) |
         (static_cast<std::uint32_t>(U16(bytes, offset + 2)) << 16);
}

std::string Hash(std::string_view bytes) {
  return gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

void CheckNoSecrets(const Value& value) {
  if (value.is_string()) {
    for (const auto secret : kSecrets)
      Require(value.str().find(secret) == std::string::npos,
              "secret leaked through a JSON value");
  } else if (value.is_array()) {
    for (const auto& item : value.items())
      CheckNoSecrets(item);
  } else if (value.is_object()) {
    for (const auto& [key, item] : value.members()) {
      for (const auto secret : kSecrets)
        Require(key.find(secret) == std::string::npos,
                "secret leaked through a JSON key");
      CheckNoSecrets(item);
    }
  }
}

Value FixtureStatus() {
  auto status = Value::object();
  status["prepared"] = true;
  status["phase"] = "ready";
  status["paths"] = Value::object();
  status["paths"]["runtime"] = std::string(kSecrets[0]);
  status["endpoint"] = std::string(kSecrets[2]);
  status["versions"] = Value::object();
  status["versions"]["revision"] = std::string(kSecrets[1]);
  status["effective_settings"] = Value::object();
  status["effective_settings"]["reason"] = std::string(kSecrets[5]);
  status["readiness"] = Value::object();
  status["readiness"]["runtime"] = true;
  status["readiness"]["model"] = true;
  status["last_error"] = Value::object();
  status["last_error"]["code"] = "device_lost";
  status["last_error"]["remedy"] = std::string(kSecrets[6]);
  status["gui_settings"] = Value::object();
  status["gui_settings"]["version"] = 1;
  status["gui_settings"]["profile"] = "default";
  status["gui_settings"]["npu_prefill"] = "on " + std::string(kSecrets[4]);
  status["gui_settings"]["port"] = 8080;
  status["gui_settings"]["context"] = 262144;
  status["gui_settings"]["sessions"] = std::numeric_limits<double>::infinity();
  status["gui_settings"]["model_path"] = std::string(kSecrets[0]);
  status["gui_settings"]["runtime_path"] = std::string(kSecrets[1]);
  status["gui_settings"]["data_folder"] = std::string(kSecrets[3]);
  status["gui_settings"]["auth"] = std::string(kSecrets[4]);
  status["gui_settings"]["proxy"] = std::string(kSecrets[5]);
  status["gui_settings"]["prompt"] = std::string(kSecrets[6]);
  status["gui_settings"]["response"] = std::string(kSecrets[7]);
  status["gui_settings"]["serial"] = std::string(kSecrets[8]);
  status["diagnose_state"] = Value::object();
  status["diagnose_state"]["status"] = "WARN";
  status["diagnose_state"]["message"] = std::string(kSecrets[7]);
  status["diagnose_state"]["checks"] = Value::array();
  auto check = Value::object();
  check["name"] = "gpu";
  check["status"] = "WARN";
  check["details"] = Value::object();
  check["details"]["serial"] = std::string(kSecrets[8]);
  status["diagnose_state"]["checks"].push_back(std::move(check));
  return status;
}

std::vector<Value> FixtureHistory() {
  std::vector<Value> history;
  auto event = Value::object();
  event["type"] = "diagnose";
  event["payload"] = Value::object();
  event["payload"]["status"] = "PASS";
  event["payload"]["prompt"] = std::string(kSecrets[6]);
  event["payload"]["response"] = std::string(kSecrets[7]);
  event["payload"]["serial"] = std::string(kSecrets[8]);
  history.push_back(event);
  for (int index = 0; index < 64; ++index)
    history.push_back(event);
  return history;
}

void CheckZip(const bubo::cli::SupportBundle& bundle) {
  Require(bundle.files.size() == 7, "unexpected ZIP member count");
  Require(bundle.zip.size() <= 160 * 1024, "ZIP size cap exceeded");
  Require(Hash(bundle.zip) == bundle.zip_sha256, "ZIP hash mismatch");
  std::size_t offset = 0, text_bytes = 0;
  for (const auto& expected : bundle.files) {
    Require(U32(bundle.zip, offset) == 0x04034b50, "missing local header");
    Require(U16(bundle.zip, offset + 8) == 0, "ZIP member is compressed");
    const auto crc = U32(bundle.zip, offset + 14);
    const auto packed = U32(bundle.zip, offset + 18);
    const auto unpacked = U32(bundle.zip, offset + 22);
    const auto name_size = U16(bundle.zip, offset + 26);
    const auto extra_size = U16(bundle.zip, offset + 28);
    Require(packed == unpacked && extra_size == 0, "unexpected ZIP data");
    offset += 30;
    Require(offset + name_size + packed <= bundle.zip.size(),
            "truncated ZIP member");
    const auto name = std::string_view(bundle.zip).substr(offset, name_size);
    offset += name_size;
    const auto text = std::string_view(bundle.zip).substr(offset, packed);
    offset += packed;
    Require(name == expected.name && text == expected.text,
            "ZIP and preview differ");
    Require(text.size() <= 32 * 1024, "member size cap exceeded");
    text_bytes += text.size();
    Require(Hash(text) == expected.sha256, "member hash mismatch");
    Require(crc == crc32(crc32(0L, Z_NULL, 0),
                         reinterpret_cast<const Bytef*>(text.data()),
                         static_cast<uInt>(text.size())),
            "member CRC mismatch");
    for (const auto secret : kSecrets)
      Require(text.find(secret) == std::string_view::npos,
              "secret leaked into a ZIP member");
    CheckNoSecrets(gufo::json::parse(text));
  }
  Require(text_bytes <= 128 * 1024, "total text size cap exceeded");
  Require(U32(bundle.zip, offset) == 0x02014b50,
          "central directory is missing");
  Require(U32(bundle.zip, bundle.zip.size() - 22) == 0x06054b50,
          "ZIP footer is missing");
}

void CheckEvents(const bubo::cli::SupportBundle& bundle) {
  const auto preview = bubo::cli::SupportBundleEvent(bundle, false);
  const auto saved = bubo::cli::SupportBundleEvent(bundle, true);
  Require(!preview.find("saved")->as_bool() && saved.find("saved")->as_bool(),
          "event mode mismatch");
  for (const auto* key :
       {"files", "text_preview", "sha256", "zip_bytes", "bundle_sha256"}) {
    Require(preview.find(key) && saved.find(key) &&
                preview.find(key)->dump() == saved.find(key)->dump(),
            "preview and saved proof differ");
  }
  const auto* files = preview.find("files");
  const auto* text = preview.find("text_preview");
  const auto* hashes = preview.find("sha256");
  Require(files && files->is_array() && files->size() == 7 && text && hashes,
          "event is missing the file proof");
  for (const auto& file : bundle.files) {
    Require(text->member_str(file.name) == file.text &&
                hashes->member_str(file.name) == file.sha256,
            "event member proof mismatch");
  }
  for (const auto secret : kSecrets)
    Require(preview.dump().find(secret) == std::string::npos,
            "secret leaked into preview event");
}

void CheckProjection(const bubo::cli::SupportBundle& bundle) {
  const auto settings = gufo::json::parse(bundle.files[1].text);
  Require(settings.member_str("profile") == "default" &&
              settings.member_size("context") == 262144 &&
              !settings.contains("sessions") &&
              !settings.contains("npu_prefill") &&
              !settings.contains("model_path") && !settings.contains("proxy") &&
              !settings.contains("auth"),
          "settings projection was not typed and allow-listed");
  const auto status = gufo::json::parse(bundle.files[2].text);
  Require(status.member_str("phase") == "ready" && !status.contains("paths") &&
              !status.contains("endpoint") &&
              !status.contains("effective_settings") &&
              !status.contains("versions"),
          "status copied an arbitrary field");
  const auto diagnose = gufo::json::parse(bundle.files[3].text);
  Require(diagnose.member_str("status") == "WARN" && diagnose.find("checks") &&
              diagnose.find("checks")->items().size() == 1 &&
              !diagnose.find("checks")->items()[0].contains("details"),
          "diagnosis copied arbitrary details");
  const auto history = gufo::json::parse(bundle.files[4].text);
  Require(history.is_array() && history.size() == 32,
          "history was not bounded");
  const auto logs = gufo::json::parse(bundle.files[5].text);
  Require(logs.member_size("dropped_lines") == 2 && logs.find("entries") &&
              logs.find("entries")->size() == 1 &&
              logs.find("entries")->items()[0].member_str("event") == "ready",
          "unknown raw logs were retained");
}

void CheckSave(const bubo::cli::SupportBundle& bundle) {
  const auto suffix =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto path = fs::current_path() /
                    ("bubo-bundle-contract-" + std::to_string(suffix) + ".zip");
  auto link = path;
  link += ".link.zip";
  auto absent = path;
  absent += ".absent.zip";
  try {
    bubo::cli::SaveSupportBundle(bundle, path);
    std::ifstream input(path, std::ios::binary);
    const std::string saved{std::istreambuf_iterator<char>(input),
                            std::istreambuf_iterator<char>()};
    Require(saved == bundle.zip, "saved ZIP differs from preview ZIP");
    bool rejected = false;
    try {
      bubo::cli::SaveSupportBundle(bundle, path);
    } catch (const std::exception&) {
      rejected = true;
    }
    Require(rejected, "existing output was overwritten");
    rejected = false;
    try {
      bubo::cli::SaveSupportBundle(bundle,
                                   path.parent_path() / ".." / path.filename());
    } catch (const std::exception&) {
      rejected = true;
    }
    Require(rejected, "traversal target was accepted");
    std::error_code symlink_error;
    fs::create_symlink(absent, link, symlink_error);
    if (!symlink_error) {
      rejected = false;
      try {
        bubo::cli::SaveSupportBundle(bundle, link);
      } catch (const std::exception&) {
        rejected = true;
      }
      Require(rejected && !fs::exists(absent),
              "dangling symlink target was followed");
    }
  } catch (...) {
    std::error_code ignored;
    fs::remove(path, ignored);
    fs::remove(link, ignored);
    throw;
  }
  std::error_code ignored;
  fs::remove(path, ignored);
  fs::remove(link, ignored);
}
void CheckLiveSettings() {
  auto status = FixtureStatus();
  auto settings = Value::object();
  settings["profile"] = "default";
  auto rows = Value::array();
  for (const auto* name :
       {"--context", "--mmproj", "--api-key", "BUBO_API_KEY"}) {
    auto row = Value::object();
    row["name"] = name;
    row["effective"] = std::string_view(name) == "--context"
                           ? Value("262144")
                           : std::string_view(name) == "BUBO_API_KEY"
                                 ? Value(std::string(kSecrets[9]))
                                 : Value(std::string(kSecrets[0]));
    row["source"] = "env";
    row["requested"] = std::string(kSecrets[4]);
    row["reason"] = std::string(kSecrets[6]);
    rows.push_back(std::move(row));
  }
  settings["effective_switches"] = std::move(rows);
  status["effective_settings"] = std::move(settings);
  const auto bundle = bubo::cli::BuildSupportBundle(status, {}, {});
  const auto safe = gufo::json::parse(bundle.files[1].text);
  Require(safe.member_str("profile") == "default", "live profile missing");
  Require(safe.find("effective_switches")->size() == 2,
          "authentication config retained");
  Require(
      safe.find("effective_switches")->items()[0].member_double("effective") ==
          262144,
      "typed live numeric setting missing");
  for (const auto& file : bundle.files)
    for (const auto secret : kSecrets)
      Require(file.text.find(secret) == std::string::npos,
              "live settings secret leaked");
}
}  // namespace

int main() {
  try {
    const std::vector<std::string> logs = {
        std::string(kSecrets[4]) + " " + std::string(kSecrets[6]),
        "{\"event\":\"ready\",\"level\":\"info\",\"prompt\":\"" +
            std::string(kSecrets[6]) + "\",\"proxy\":\"" +
            std::string(kSecrets[5]) + "\",\"count\":2}",
        "{\"event\":\"unknown\",\"message\":\"" + std::string(kSecrets[8]) +
            "\"}"};
    const auto bundle =
        bubo::cli::BuildSupportBundle(FixtureStatus(), FixtureHistory(), logs);
    Require(bundle.zip == bubo::cli::BuildSupportBundle(FixtureStatus(),
                                                        FixtureHistory(), logs)
                              .zip,
            "bundle image was not deterministic");
    CheckZip(bundle);
    CheckEvents(bundle);
    CheckProjection(bundle);
    CheckSave(bundle);
    CheckLiveSettings();
    std::cout << "Bundle privacy, ZIP, preview, and path checks: " << checks
              << " passed, 0 failed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Bundle contract failed: " << error.what() << '\n';
    return 1;
  }
}
