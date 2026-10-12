#include "src/core/prepare/runtime_table.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/release_catalogue.hpp"
#include "src/core/prepare/runtime_members.inc"
#include "src/core/prepare/zip_reader.hpp"

namespace bubo::prepare {
namespace {
// REPORT-20261010-1958.md Appendices A/B and the binding consent block.
const std::array<WheelPin, 3> wheels{
    {{"rocm_sdk_core", "rocm_sdk_core-10.0.0-py3-none-win_amd64.whl",
      "https://stable.repo.amd.com/rocm/core/whl-next/rocm-sdk-core/"
      "rocm_sdk_core-10.0.0-py3-none-win_amd64.whl",
      758050981,
      "066195cb0f7df02e6009facb17652dd992d13fae44d87652e5ab928d5b4f0851"},
     {"rocm_sdk_libraries", "rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl",
      "https://stable.repo.amd.com/rocm/core/whl-next/rocm-sdk-libraries/"
      "rocm_sdk_libraries-10.0.0-py3-none-win_amd64.whl",
      116831755,
      "bfc7a1bd3b6050fb4d36d8ac2eea80d3afd70c5604a29142463d6cb890fc73bf"},
     {"rocm_sdk_device_gfx1151",
      "rocm_sdk_device_gfx1151-10.0.0-py3-none-win_amd64.whl",
      "https://stable.repo.amd.com/rocm/core/whl-next/rocm-sdk-device-gfx1151/"
      "rocm_sdk_device_gfx1151-10.0.0-py3-none-win_amd64.whl",
      139346623,
      "1ab27ebfe61ea3c3fd2c5c1a0e780ae22ed9b662106ce0aefda0a30e3b11577f"}}};
const std::map<std::string, std::pair<unsigned, std::string>> dlls{
    {"amd_comgr.dll",
     {0, "213db0bedaa07027ce136ba1797299a2b0cfa2c6bb33584ce3861be8be067937"}},
    {"amdhip64_7.dll",
     {0, "546fb3d6e2d2194a9526fb94ec2fd3aa5b92a48a7595f04efece80162047ef69"}},
    {"hiprtc0715.dll",
     {0, "19b67aa86ba371c49b2def5e5c863346e81c1b6d67caab125cf10a8c405e0043"}},
    {"hiprtc-builtins0715.dll",
     {0, "97a4707260342e6563423a0d586e18744f7ce0de06b8634dc6dde29eceadd196"}},
    {"rocm_kpack.dll",
     {0, "97b59ca4e7b5edfcda72d66c5b6643f557168b3c60091cf5c6fe9bb226597bf1"}},
    {"rocm-openblas.dll",
     {0, "88f7631d9f7c49cfbf710551a7fa69bef2fd54052007532a518d7ba3699534c0"}},
    {"rocm-openblas64.dll",
     {0, "9020e44dc2afd30576dbe38fccc473011efadfbe86e3b6f954b8b4776c8d66fb"}},
    {"hipblas.dll",
     {1, "343bd57671dc96d035ff0e7591ecce926a102df1d52b50025f4a41a45cdf9581"}},
    {"libhipblaslt.dll",
     {1, "746fa2f7e037b3d03a0b904659bfc21bfbf0fe1a86f127564ac2dee38b85442e"}},
    {"rocblas.dll",
     {1, "8ecbe340d8052e5292a47b96f3c93c20767ced0032db47851b792da7039d1e0b"}},
    {"rocsolver.dll",
     {1, "9eedf3904e548f6b9b049abb41728ce6dc25a3b26266a5e1e94af55be1e779bd"}},
    {"rocsparse.dll",
     {1, "904a43601b71cf9b0a2bf9bb5007ed06031a17c9bb5cc5a28425f6e0d7a02931"}},
    {"origami.dll",
     {1, "f10a142efc441e90f4ba116f2f901a9433899d0f74860c20b58bf4f1381e2b56"}}};
std::string Destination(unsigned wheel, std::string_view member) {
  const std::string_view bin =
      wheel == 0 ? "_rocm_sdk_core/bin/" : "_rocm_sdk_libraries/bin/";
  if (!member.starts_with(bin))
    return {};
  const auto name = member.substr(bin.size());
  const auto d = dlls.find(std::string(name));
  if (d != dlls.end() && d->second.first == wheel)
    return std::string(name);
  if (wheel == 2 && (name.starts_with("rocblas/library/gfx1151/") ||
                     name.starts_with("hipblaslt/library/gfx1151/")))
    return std::string(name);
  return {};
}
}  // namespace
const std::array<WheelPin, 3>& RuntimeWheels() {
  return wheels;
}
std::vector<RuntimeMember> LoadRuntimeTable(std::string_view text,
                                            bool production) {
  if (text.size() > 1024 * 1024)
    throw std::runtime_error("runtime table exceeds 1 MiB");
  std::istringstream input{std::string(text)};
  std::string line;
  std::getline(input, line);
  if (line != "bubo-runtime-members-v1")
    throw std::runtime_error("runtime table schema");
  std::vector<RuntimeMember> rows;
  std::set<std::string> destinations, members;
  unsigned dcount = 0, rocblas = 0, hipblaslt = 0;
  std::uint64_t total = 0;
  while (std::getline(input, line)) {
    std::array<std::string, 5> fields;
    std::size_t p = 0;
    for (unsigned i = 0; i < fields.size(); ++i) {
      auto end = line.find('\t', p);
      if ((i == 4) != (end == line.npos))
        throw std::runtime_error("runtime table columns");
      fields[i] = line.substr(p, end - p);
      p = end + 1;
    }
    const auto wi = metadata::Decimal(fields[0]);
    RuntimeMember r{static_cast<unsigned>(wi), fields[1], fields[2],
                    metadata::Decimal(fields[3]), fields[4]};
    if (wi > 2 || !SafeZipPath(r.member) || !SafeZipPath(r.destination) ||
        Destination(r.wheel, r.member) != r.destination || !r.size ||
        r.size > 512 * 1024 * 1024 || !IsLowerHex(r.sha256, 64))
      throw std::runtime_error("invalid runtime row");
    auto folded = r.destination;
    std::transform(
        folded.begin(), folded.end(), folded.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (!destinations.insert(folded).second ||
        !members.insert(fields[0] + ":" + r.member).second)
      throw std::runtime_error("duplicate runtime row");
    const auto d = dlls.find(r.destination);
    if (d != dlls.end()) {
      ++dcount;
      if (production && r.sha256 != d->second.second)
        throw std::runtime_error("DLL pin differs from Appendix A");
    } else if (r.destination.starts_with("rocblas/"))
      ++rocblas;
    else
      ++hipblaslt;
    total += r.size;
    if (total > 1024 * 1024 * 1024ULL || rows.size() >= 263)
      throw std::runtime_error("runtime extraction limit");
    rows.push_back(std::move(r));
  }
  if (rows.empty() || (production && (rows.size() != 263 || dcount != 13 ||
                                      rocblas != 150 || hipblaslt != 100)))
    throw std::runtime_error("OPEN O1: incomplete 263-file runtime table");
  return rows;
}
std::string SerializeRuntimeTable(const std::vector<RuntimeMember>& rows) {
  std::ostringstream out;
  out << "bubo-runtime-members-v1\n";
  for (const auto& r : rows)
    out << r.wheel << '\t' << r.member << '\t' << r.destination << '\t'
        << r.size << '\t' << r.sha256 << '\n';
  return out.str();
}
std::vector<RuntimeMember> CompiledRuntimeTable() {
  return LoadRuntimeTable(kRuntimeMembers);
}
std::vector<RuntimeMember> GenerateRuntimeTable(
    const std::array<std::filesystem::path, 3>& paths, bool production) {
  std::vector<RuntimeMember> rows;
  for (unsigned i = 0; i < paths.size(); ++i) {
    ReadLease lease(paths[i]);
    if (production && (lease.Snapshot().size != wheels[i].size ||
                       HashFile(paths[i]) != wheels[i].sha256))
      throw std::runtime_error("wheel differs from Appendix B: " +
                               std::string(wheels[i].file));
    ZipReader zip(paths[i]);
    for (const auto& m : zip.members()) {
      const auto dest = Destination(i, m.name);
      if (!dest.empty())
        rows.push_back(
            {i, m.name, dest, m.size, zip.Extract(m, 512 * 1024 * 1024, {})});
    }
  }
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.destination < b.destination;
  });
  return LoadRuntimeTable(SerializeRuntimeTable(rows), production);
}
}  // namespace bubo::prepare
