#pragma once
#include <zlib.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace bubo::prepare::test {
inline constexpr std::string_view kAbcHash =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
struct FixtureMember {
  std::string name, text = "abc";
  bool deflate = false;
};
inline void Put(std::string& s, std::uint64_t n, unsigned width) {
  for (unsigned i = 0; i < width; ++i)
    s += static_cast<char>(n >> (8 * i));
}
inline std::string FixtureZip(const std::vector<FixtureMember>& files,
                              bool zip64 = false, bool bad_crc = false,
                              bool descriptor = false) {
  std::string out, cd;
  for (const auto& f : files) {
    const auto offset = out.size();
    std::string data = f.text;
    if (f.deflate) {
      z_stream z{};
      deflateInit2(&z, 6, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
      std::string packed(compressBound(static_cast<uLong>(data.size())), '\0');
      z.next_in = reinterpret_cast<Bytef*>(data.data());
      z.avail_in = static_cast<uInt>(data.size());
      z.next_out = reinterpret_cast<Bytef*>(packed.data());
      z.avail_out = static_cast<uInt>(packed.size());
      deflate(&z, Z_FINISH);
      packed.resize(z.total_out);
      deflateEnd(&z);
      data = std::move(packed);
    }
    const auto crc = crc32(0, reinterpret_cast<const Bytef*>(f.text.data()),
                           static_cast<uInt>(f.text.size())) ^
                     (bad_crc ? 1 : 0);
    std::string extra;
    if (zip64) {
      Put(extra, 1, 2);
      Put(extra, 16, 2);
      Put(extra, f.text.size(), 8);
      Put(extra, data.size(), 8);
    }
    Put(out, 0x04034b50, 4);
    Put(out, zip64 ? 45 : 20, 2);
    Put(out, descriptor ? 8 : 0, 2);
    Put(out, f.deflate ? 8 : 0, 2);
    Put(out, 0, 4);
    Put(out, descriptor ? 0 : crc, 4);
    Put(out, zip64 ? 0xffffffff : (descriptor ? 0 : data.size()), 4);
    Put(out, zip64 ? 0xffffffff : (descriptor ? 0 : f.text.size()), 4);
    Put(out, f.name.size(), 2);
    Put(out, extra.size(), 2);
    out += f.name + extra + data;
    if (descriptor) {
      Put(out, 0x08074b50, 4);
      Put(out, crc, 4);
      Put(out, data.size(), zip64 ? 8 : 4);
      Put(out, f.text.size(), zip64 ? 8 : 4);
    }
    if (zip64) {
      Put(extra, offset, 8);
      extra[2] = 24;
    }
    Put(cd, 0x02014b50, 4);
    Put(cd, 20, 2);
    Put(cd, zip64 ? 45 : 20, 2);
    Put(cd, descriptor ? 8 : 0, 2);
    Put(cd, f.deflate ? 8 : 0, 2);
    Put(cd, 0, 4);
    Put(cd, crc, 4);
    Put(cd, zip64 ? 0xffffffff : data.size(), 4);
    Put(cd, zip64 ? 0xffffffff : f.text.size(), 4);
    Put(cd, f.name.size(), 2);
    Put(cd, extra.size(), 2);
    Put(cd, 0, 2);
    Put(cd, 0, 2);
    Put(cd, 0, 2);
    Put(cd, 0, 4);
    Put(cd, zip64 ? 0xffffffff : offset, 4);
    cd += f.name + extra;
  }
  const auto cd_offset = out.size();
  out += cd;
  if (zip64) {
    const auto zoffset = out.size();
    Put(out, 0x06064b50, 4);
    Put(out, 44, 8);
    Put(out, 45, 2);
    Put(out, 45, 2);
    Put(out, 0, 4);
    Put(out, 0, 4);
    Put(out, files.size(), 8);
    Put(out, files.size(), 8);
    Put(out, cd.size(), 8);
    Put(out, cd_offset, 8);
    Put(out, 0x07064b50, 4);
    Put(out, 0, 4);
    Put(out, zoffset, 8);
    Put(out, 1, 4);
  }
  Put(out, 0x06054b50, 4);
  Put(out, 0, 2);
  Put(out, 0, 2);
  Put(out, zip64 ? 0xffff : files.size(), 2);
  Put(out, zip64 ? 0xffff : files.size(), 2);
  Put(out, zip64 ? 0xffffffff : cd.size(), 4);
  Put(out, zip64 ? 0xffffffff : cd_offset, 4);
  Put(out, 0, 2);
  return out;
}
inline void WriteFixture(const std::filesystem::path& path,
                         std::string_view text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream f(path, std::ios::binary);
  f << text;
  if (!f)
    throw std::runtime_error("fixture write failed");
}
}  // namespace bubo::prepare::test
