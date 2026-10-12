#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"
#include "src/core/prepare/release_catalogue.hpp"

// Release pin generator. Reads inputs only. Production provenance must be
// established by whole SHA first; --already-verified requires documented
// evidence for the exact four recipe hashes (done.txt or a previous tool run).
int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::runtime_error(
          "usage: prepare_fingerprints INPUT_DIR OUTPUT_DIR "
          "[--already-verified]");
    const bool verified =
        argc == 4 && std::string_view(argv[3]) == "--already-verified";
    if (argc == 4 && !verified)
      throw std::runtime_error("unknown argument");
    const auto input = std::filesystem::u8path(argv[1]);
    const auto output = std::filesystem::u8path(argv[2]);
    const auto& catalogue = bubo::prepare::ModelCatalogue();
    if (!verified) {
      const auto all_start = std::chrono::steady_clock::now();
      for (const auto& entry : catalogue) {
        if (entry.kind != bubo::prepare::Kind::MainShard)
          continue;
        const auto start = std::chrono::steady_clock::now();
        const auto path = input / entry.file_name;
        if (std::filesystem::file_size(path) != entry.size)
          throw std::runtime_error("source shard size mismatch");
        const auto hash = gufo::crypto::Sha256FileHex(path);
        std::cout << "full_sha " << entry.logical_name << " " << hash
                  << " elapsed_seconds="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << std::endl;
        if (hash != entry.sha256)
          throw std::runtime_error(
              "source whole SHA mismatch; no pins generated");
      }
      std::cout << "full_sha_total_seconds="
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - all_start)
                       .count()
                << std::endl;
    }
    std::filesystem::create_directories(output);
    // Keep the previously valid compiled pin header intact on parse failure.
    std::ostringstream embedded;
    embedded << "#pragma once\n#include <array>\n#include "
                "<string_view>\n#include <utility>\nnamespace bubo::prepare "
                "{\ninline constexpr std::array<std::pair<std::string_view, "
                "std::string_view>, 4>\n    kEmbeddedFingerprintTables = {{\n";
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t table_bytes = 0, prefix_bytes = 0, tensor_count = 0;
    double check_seconds = 0;
    for (const auto& entry : catalogue) {
      if (entry.kind != bubo::prepare::Kind::MainShard)
        continue;
      const auto shard_start = std::chrono::steady_clock::now();
      auto generated = bubo::prepare::GenerateGgufFingerprint(
          input / entry.file_name, entry);
      if (!generated)
        throw std::runtime_error(generated.error);
      const auto json =
          bubo::prepare::SerializeFingerprintTable(generated.table);
      const auto loaded = bubo::prepare::LoadFingerprintTable(json, entry);
      if (!loaded)
        throw std::runtime_error(loaded.error);
      std::ofstream file(output / (std::string(entry.logical_name) + ".json"),
                         std::ios::binary | std::ios::trunc);
      file << json << '\n';
      if (!file)
        throw std::runtime_error("cannot write fingerprint table");
      embedded << "    {\"" << entry.logical_name << "\", R\"BUBOFP(" << json
               << ")BUBOFP\"},\n";
      table_bytes += json.size() + 1;
      tensor_count += generated.table.tensors.size();
      for (const auto& t : generated.table.tensors)
        prefix_bytes += t.checked_length;
      std::cout << "fingerprint " << entry.logical_name
                << " tensors=" << generated.table.tensors.size()
                << " header_bytes=" << generated.table.header_bytes
                << " table_bytes=" << json.size() + 1 << " elapsed_seconds="
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - shard_start)
                       .count()
                << std::endl;
      const auto check_start = std::chrono::steady_clock::now();
      const auto check_error = bubo::prepare::CheckGgufFingerprint(
          input / entry.file_name, entry, loaded.table);
      const auto elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - check_start)
                               .count();
      check_seconds += elapsed;
      if (!check_error.empty())
        throw std::runtime_error(check_error);
      std::cout << "default_check " << entry.logical_name
                << " PASS elapsed_seconds=" << elapsed << std::endl;
    }
    embedded << "}};\n}\n";
    std::ofstream embedded_file(output / "embedded_tables.hpp",
                                std::ios::binary | std::ios::trunc);
    embedded_file << embedded.str();
    if (!embedded_file)
      throw std::runtime_error("cannot write embedded tables");
    std::cout << "generator_version="
              << bubo::prepare::kFingerprintGeneratorVersion
              << " tensor_count=" << tensor_count
              << " prefix_bytes=" << prefix_bytes
              << " table_bytes=" << table_bytes << " elapsed_seconds="
              << (std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count() -
                  check_seconds)
              << std::endl;
    std::cout << "default_check_total_seconds=" << check_seconds << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
