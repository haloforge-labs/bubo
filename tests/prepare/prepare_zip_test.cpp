#include <iostream>
#include <stdexcept>

#include "src/core/prepare/runtime_table.hpp"
#include "src/core/prepare/zip_reader.hpp"
#include "tests/prepare/zip_fixture.hpp"

namespace {
void Check(bool ok, const char* why) {
  if (!ok)
    throw std::runtime_error(why);
}
template<class F>
void Reject(F f, const char* why) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  Check(rejected, why);
}
}  // namespace
int main(int argc, char** argv) {
  using namespace bubo::prepare;
  using namespace bubo::prepare::test;
  try {
    const auto root = std::filesystem::absolute(
        argc > 1 ? std::filesystem::u8path(argv[1])
                 : std::filesystem::path("j3-zip-fixture"));
    Check(!std::filesystem::exists(root), "fixture root must be fresh");
    std::filesystem::create_directories(root);
    const auto p = root / "fixture.whl";
    for (bool wide : {false, true})
      for (bool descriptor : {false, true}) {
        WriteFixture(
            p, FixtureZip({{"_rocm_sdk_core/bin/amdhip64_7.dll", "abc", true},
                           {"README"}},
                          wide, false, descriptor));
        ZipReader z(p);
        std::string output;
        Check(z.members().size() == 2, "central enumeration");
        Check(z.Extract(
                  z.members()[0], 3,
                  [&](auto b) {
                    output.append(reinterpret_cast<const char*>(b.data()),
                                  b.size());
                  },
                  kAbcHash) == kAbcHash &&
                  output == "abc",
              "raw DEFLATE known vector/ZIP64/descriptor");
        Check(z.Extract(z.members()[1], 3, {}, kAbcHash) == kAbcHash,
              "stored vector");
        Reject([&] { z.Extract(z.members()[0], 2, {}); }, "zip bomb bound");
        Reject([&] { z.Extract(z.members()[0], 3, {}, std::string(64, '0')); },
               "wrong SHA-256 must fail");
      }
    for (const char* bad :
         {"../escape", "/absolute", "C:/drive", "x/../../escape", "x\\escape",
          "x:ads", "CON.txt", "x./file"}) {
      WriteFixture(p, FixtureZip({{"README"}, {bad}}));
      Reject([&] { ZipReader z(p); }, "unsafe unselected member must fail");
    }
    WriteFixture(p, FixtureZip({{"README"}, {"readme"}}));
    Reject([&] { ZipReader z(p); }, "duplicate case collision");
    WriteFixture(p, FixtureZip({{"data", "abc", true}}, false, true));
    Reject(
        [&] {
          ZipReader z(p);
          z.Extract(z.members()[0], 3, {});
        },
        "CRC mismatch");
    auto truncated = FixtureZip({{"data"}});
    truncated.resize(truncated.size() - 7);
    WriteFixture(p, truncated);
    Reject([&] { ZipReader z(p); }, "truncated archive");
    auto malformed = FixtureZip({{"data", "abc", true}});
    malformed[34] = static_cast<char>(0xff);
    WriteFixture(p, malformed);
    Reject(
        [&] {
          ZipReader z(p);
          z.Extract(z.members()[0], 3, {});
        },
        "malformed DEFLATE");
    const std::array<std::filesystem::path, 3> paths{
        root / "core.whl", root / "libs.whl", root / "device.whl"};
    WriteFixture(paths[0],
                 FixtureZip({{"_rocm_sdk_core/bin/amdhip64_7.dll", "abc", true},
                             {"ignored"}}));
    WriteFixture(paths[1],
                 FixtureZip({{"_rocm_sdk_libraries/bin/rocblas.dll"}}));
    WriteFixture(
        paths[2],
        FixtureZip(
            {{"_rocm_sdk_libraries/bin/rocblas/library/gfx1151/kernel.co"},
             {"_rocm_sdk_libraries/bin/hipblaslt/library/gfx1151/kernel.dat"}},
            true));
    const auto rows = GenerateRuntimeTable(paths, false);
    Check(rows.size() == 4, "fixture generator selective count");
    for (const auto& r : rows)
      Check(r.size == 3 && r.sha256 == kAbcHash,
            "fixture generator known size/hash");
    auto table = SerializeRuntimeTable(rows);
    Check(LoadRuntimeTable(table, false).size() == 4, "fixture loader");
    Reject([&] { LoadRuntimeTable(table); },
           "production cannot accept fixture");
    Reject(
        [&] {
          LoadRuntimeTable(table + table.substr(table.find('\n') + 1), false);
        },
        "duplicate table row");
    Reject([&] { GenerateRuntimeTable(paths); },
           "real generator must refuse fixture archives");
    Check(CompiledRuntimeTable().size() == 263,
          "complete compiled production table");
    std::filesystem::remove_all(root);
    std::cout << "PASS: ZIP stored/raw DEFLATE, ZIP64 sizes/offsets/end "
                 "header, descriptors, CRC, SHA, "
                 "traversal/ADS/drive/absolute/duplicate, truncation, "
                 "malformed stream, bomb limits; fixture generator/loader\n";
  } catch (const std::exception& e) {
    std::cerr << "prepare_zip_test: FAIL " << e.what() << '\n';
    return 1;
  }
}
