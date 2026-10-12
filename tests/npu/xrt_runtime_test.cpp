#include "src/npu/xrt_runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
class FakeRuntime final : public gufo::npu::detail::RuntimeOperations {
public:
  gufo::npu::RuntimeIdentity identity{
      "C:/Windows/System32/xrt_coreutil.dll",
      "04a26d37c6e0c713491ad0bfae74ce74ea94c74136d2aa056333616dac6c3a44",
      {32, 0, 20102, 3931}};
  std::array<unsigned, 5> versions{2, 21, 0, 8679, 131093};
  unsigned inspections = 0;
  unsigned loads = 0;
  unsigned calls = 0;
  bool unavailable = false;
  gufo::npu::RuntimeIdentity Inspect() override {
    ++inspections;
    if (unavailable) {
      throw std::runtime_error("missing DLL");
    }
    return identity;
  }
  void Load(const std::filesystem::path& path) override {
    if (path != identity.path || inspections != 1) {
      throw std::runtime_error("wrong path/order");
    }
    ++loads;
  }
  std::array<unsigned, 5> Versions() override {
    if (loads != 1) {
      throw std::runtime_error("versions before load");
    }
    ++calls;
    return versions;
  }
};

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void Denied(FakeRuntime& fake, const char* label, unsigned loads) {
  bool rejected = false;
  try {
    gufo::npu::detail::ValidateAndLoad(fake);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Expect(rejected && fake.loads == loads && fake.calls == loads, label);
  std::cout << "PASS: " << label << " rejected; fake loads=" << fake.loads
            << " versions=" << fake.calls << " device calls=0\n";
}
}  // namespace

int main() {
  try {
    FakeRuntime idle;
    Expect(idle.inspections == 0 && idle.loads == 0 && idle.calls == 0,
           "construction is lazy");
    std::cout
        << "PASS: default idle performs no inspection/load/version call\n";
    FakeRuntime missing;
    missing.unavailable = true;
    Denied(missing, "missing DLL", 0);
    FakeRuntime relative;
    relative.identity.path = "xrt_coreutil.dll";
    Denied(relative, "relative path", 0);
    FakeRuntime name;
    name.identity.path = "C:/Windows/System32/other.dll";
    Denied(name, "wrong filename", 0);
    FakeRuntime hash;
    hash.identity.sha256[0] = 'f';
    Denied(hash, "hash mismatch", 0);
    FakeRuntime file_version;
    file_version.identity.file_version[3] = 3930;
    Denied(file_version, "file version mismatch", 0);
    for (size_t i = 0; i < idle.versions.size(); ++i) {
      FakeRuntime version;
      ++version.versions[i];
      Denied(version, "XRT version mismatch", 1);
    }
    gufo::npu::detail::ValidateAndLoad(idle);
    Expect(idle.inspections == 1 && idle.loads == 1 && idle.calls == 1,
           "positive control did not load once");
    std::cout << "PASS: matching identity loads absolute path once; five fake "
                 "versions match; device calls=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
