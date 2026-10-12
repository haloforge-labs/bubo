#include "src/core/prepare/zip_reader.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>

#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/file_ops.hpp"
#ifdef _WIN32
#include <windows.h>
#endif

namespace bubo::prepare {
namespace {
using Bytes = std::vector<std::uint8_t>;
[[noreturn]] void Bad(std::string_view why) {
  throw std::runtime_error("ZIP: " + std::string(why));
}
class Reader {
public:
  explicit Reader(const std::filesystem::path& path) {
#ifdef _WIN32
    // Verified J2 writers retain WRITE/DELETE access. Sharing these rights
    // permits reading; their own FILE_SHARE_READ still denies mutation.
    handle_ =
        CreateFileW(path.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    LARGE_INTEGER size{};
    if (handle_ == INVALID_HANDLE_VALUE)
      Bad("cannot open archive");
    if (!GetFileSizeEx(handle_, &size) || size.QuadPart < 0) {
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
      Bad("archive size");
    }
    length = static_cast<std::uint64_t>(size.QuadPart);
#else
    file_.open(path, std::ios::binary | std::ios::ate);
    if (!file_ || file_.tellg() < 0)
      Bad("cannot open archive");
    length = static_cast<std::uint64_t>(file_.tellg());
#endif
  }
  ~Reader() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE)
      CloseHandle(handle_);
#endif
  }
  void ReadAt(std::uint64_t offset, std::span<std::uint8_t> bytes) {
#ifdef _WIN32
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(offset);
    DWORD count = 0;
    if (!SetFilePointerEx(handle_, position, nullptr, FILE_BEGIN) ||
        !ReadFile(handle_, bytes.data(), static_cast<DWORD>(bytes.size()),
                  &count, nullptr) ||
        count != bytes.size())
      Bad("truncated archive");
#else
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(offset));
    file_.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (static_cast<std::size_t>(file_.gcount()) != bytes.size())
      Bad("truncated archive");
#endif
  }
  std::uint64_t length = 0;

private:
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  std::ifstream file_;
#endif
};
std::uint64_t Number(const Bytes& b, std::size_t p, unsigned n) {
  if (p > b.size() || n > b.size() - p)
    Bad("truncated field");
  std::uint64_t v = 0;
  for (unsigned i = 0; i < n; ++i)
    v |= std::uint64_t(b[p + i]) << (8 * i);
  return v;
}
Bytes Read(Reader& f, std::uint64_t p, std::size_t n, std::uint64_t bound) {
  if (p > bound || n > bound - p || n > 16 * 1024 * 1024)
    Bad("read bounds");
  Bytes b(n);
  f.ReadAt(p, b);
  return b;
}
std::string Fold(std::string s) {
  for (auto& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
void Extra(const Bytes& b, std::size_t start, std::size_t length,
           std::uint64_t& size, std::uint64_t& compressed,
           std::uint64_t& offset, std::uint64_t& disk) {
  const bool need = size == 0xffffffff || compressed == 0xffffffff ||
                    offset == 0xffffffff || disk == 0xffff;
  bool seen = false;
  const auto end = start + length;
  while (start < end) {
    if (end - start < 4)
      Bad("truncated extra field");
    auto id = Number(b, start, 2), len = Number(b, start + 2, 2);
    start += 4;
    if (len > end - start)
      Bad("extra field bounds");
    if (id == 1) {
      if (seen)
        Bad("duplicate ZIP64 field");
      seen = true;
      auto p = start;
      auto take = [&](std::uint64_t& v, unsigned n, std::uint64_t sentinel) {
        if (v != sentinel)
          return;
        if (n > start + len - p)
          Bad("short ZIP64 field");
        v = Number(b, p, n);
        p += n;
      };
      take(size, 8, 0xffffffff);
      take(compressed, 8, 0xffffffff);
      take(offset, 8, 0xffffffff);
      take(disk, 4, 0xffff);
    }
    start += static_cast<std::size_t>(len);
  }
  if (need && !seen)
    Bad("missing ZIP64 field");
}
}  // namespace

bool SafeZipPath(std::string_view name, bool directory) {
  if (directory && name.ends_with('/'))
    name.remove_suffix(1);
  if (name.empty() || name.size() > 2048 || name.front() == '/' ||
      name.find_first_of("\\:") != name.npos)
    return false;
  for (unsigned char c : name)
    if (c < 32 || c >= 127)
      return false;
  while (!name.empty()) {
    auto end = name.find('/');
    auto part = name.substr(0, end);
    if (part.empty() || part == "." || part == ".." || part.back() == '.' ||
        part.back() == ' ' || part.find_first_of("<>\"|?*") != part.npos)
      return false;
    const auto stem = Fold(std::string(part.substr(0, part.find('.'))));
    if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" ||
        (stem.size() == 4 &&
         (stem.starts_with("com") || stem.starts_with("lpt")) &&
         stem[3] >= '1' && stem[3] <= '9'))
      return false;
    if (end == name.npos)
      break;
    name.remove_prefix(end + 1);
    if (name.empty())
      return false;
  }
  return true;
}

ZipReader::ZipReader(const std::filesystem::path& path)
    : path_(NativePath(path)) {
  const auto local = CheckLocalPath(path_);
  if (!local.empty())
    Bad(local);
  Reader f(path_);
  if (f.length < 22)
    Bad("truncated archive");
  const auto length = f.length;
  const auto tail_size =
      static_cast<std::size_t>(std::min<std::uint64_t>(length, 65557));
  const auto tail_start = length - tail_size;
  const auto tail = Read(f, tail_start, tail_size, length);
  std::size_t e = tail.size();
  for (std::size_t i = tail.size() - 22;; --i) {
    if (Number(tail, i, 4) == 0x06054b50 &&
        i + 22 + Number(tail, i + 20, 2) == tail.size()) {
      e = i;
      break;
    }
    if (!i)
      break;
  }
  if (e == tail.size())
    Bad("missing end directory");
  if (Number(tail, e + 4, 2) || Number(tail, e + 6, 2))
    Bad("multidisk archive");
  std::uint64_t count = Number(tail, e + 10, 2);
  if (Number(tail, e + 8, 2) != count)
    Bad("multidisk count");
  std::uint64_t cd_size = Number(tail, e + 12, 4), cd = Number(tail, e + 16, 4);
  std::uint64_t directory_end = tail_start + e;
  if (count == 0xffff || cd_size == 0xffffffff || cd == 0xffffffff) {
    if (directory_end < 20)
      Bad("missing ZIP64 locator");
    const auto loc = Read(f, directory_end - 20, 20, length);
    if (Number(loc, 0, 4) != 0x07064b50 || Number(loc, 4, 4) ||
        Number(loc, 16, 4) != 1)
      Bad("invalid ZIP64 locator");
    const auto pos = Number(loc, 8, 8);
    const auto end = Read(f, pos, 56, directory_end - 20);
    const auto record_size = Number(end, 4, 8);
    if (Number(end, 0, 4) != 0x06064b50 || record_size < 44 ||
        record_size > 1024 || pos > directory_end - 20 ||
        record_size != directory_end - 20 - pos - 12 || Number(end, 16, 4) ||
        Number(end, 20, 4))
      Bad("invalid ZIP64 directory");
    count = Number(end, 32, 8);
    if (Number(end, 24, 8) != count)
      Bad("ZIP64 multidisk count");
    cd_size = Number(end, 40, 8);
    cd = Number(end, 48, 8);
    directory_end = pos;
  }
  if (count > 100000 || cd_size > 16 * 1024 * 1024 || cd > directory_end ||
      cd_size != directory_end - cd)
    Bad("central directory bounds");
  std::set<std::string> names;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  std::uint64_t p = cd;
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto h = Read(f, p, 46, directory_end);
    if (Number(h, 0, 4) != 0x02014b50)
      Bad("central directory signature");
    ZipMember m;
    m.flags = static_cast<std::uint16_t>(Number(h, 8, 2));
    m.method = static_cast<std::uint16_t>(Number(h, 10, 2));
    m.crc = static_cast<std::uint32_t>(Number(h, 16, 4));
    m.compressed = Number(h, 20, 4);
    m.size = Number(h, 24, 4);
    m.offset = Number(h, 42, 4);
    const auto n = static_cast<std::size_t>(Number(h, 28, 2));
    const auto x = static_cast<std::size_t>(Number(h, 30, 2));
    const auto c = static_cast<std::size_t>(Number(h, 32, 2));
    auto disk = Number(h, 34, 2);
    const auto b = Read(f, p + 46, n + x + c, directory_end);
    m.name.assign(reinterpret_cast<const char*>(b.data()), n);
    const bool dir = m.name.ends_with('/');
    if (!SafeZipPath(m.name, dir))
      Bad("unsafe member path");
    auto folded = Fold(m.name);
    if (dir)
      folded.pop_back();
    if (!names.insert(folded).second)
      Bad("duplicate member");
    Extra(b, n, x, m.size, m.compressed, m.offset, disk);
    const auto mode = (Number(h, 38, 4) >> 16) & 0170000;
    if (disk || (m.flags & ~std::uint16_t(0x080e)) ||
        (m.method != 0 && m.method != 8) ||
        (mode && mode != 0100000 && mode != 0040000))
      Bad("unsupported/encrypted/symlink member");
    // Pinned wheels are 1 GiB combined; no runtime member may exceed 4 GiB.
    if (m.size > 0xffffffffULL || m.compressed > 0xffffffffULL)
      Bad("member size limit");
    const auto local_h = Read(f, m.offset, 30, cd);
    if (Number(local_h, 0, 4) != 0x04034b50 ||
        Number(local_h, 6, 2) != m.flags || Number(local_h, 8, 2) != m.method)
      Bad("local header mismatch");
    const auto ln = static_cast<std::size_t>(Number(local_h, 26, 2));
    const auto lx = static_cast<std::size_t>(Number(local_h, 28, 2));
    const auto local_b = Read(f, m.offset + 30, ln + lx, cd);
    if (ln != n || !std::equal(b.begin(), b.begin() + n, local_b.begin()))
      Bad("local name mismatch");
    if (!(m.flags & 8)) {
      auto ls = Number(local_h, 22, 4), lc = Number(local_h, 18, 4);
      std::uint64_t lo = 0, ld = 0;
      Extra(local_b, ln, lx, ls, lc, lo, ld);
      if (ls != m.size || lc != m.compressed || Number(local_h, 14, 4) != m.crc)
        Bad("local size/CRC mismatch");
    }
    m.data_offset = m.offset + 30 + ln + lx;
    if (m.data_offset > cd || m.compressed > cd - m.data_offset)
      Bad("payload bounds");
    auto payload_end = m.data_offset + m.compressed;
    if (m.flags & 8) {
      const auto d = Read(f, payload_end, 4, cd);
      const auto skip = Number(d, 0, 4) == 0x08074b50 ? 4 : 0;
      const bool wide =
          Number(h, 20, 4) == 0xffffffff || Number(h, 24, 4) == 0xffffffff;
      const auto descriptor = Read(f, payload_end + skip, wide ? 20 : 12, cd);
      if (Number(descriptor, 0, 4) != m.crc ||
          Number(descriptor, 4, wide ? 8 : 4) != m.compressed ||
          Number(descriptor, wide ? 12 : 8, wide ? 8 : 4) != m.size)
        Bad("data descriptor mismatch");
      payload_end += skip + descriptor.size();
    }
    ranges.emplace_back(m.offset, payload_end);
    if (dir && (m.size || m.compressed))
      Bad("nonempty directory");
    if (!dir)
      members_.push_back(m);
    p += 46 + n + x + c;
  }
  if (p != directory_end)
    Bad("directory size mismatch");
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second)
      Bad("overlapping member ranges");
}

std::string ZipReader::Extract(
    const ZipMember& member, std::uint64_t limit,
    const std::function<void(std::span<const std::uint8_t>)>& sink,
    std::string_view expected_sha256,
    const std::function<bool()>& cancelled) const {
  auto found =
      std::find_if(members_.begin(), members_.end(),
                   [&](const auto& m) { return m.name == member.name; });
  if (found == members_.end())
    Bad("member not in this archive");
  const auto& m = *found;  // Never trust a caller-supplied offset or size.
  if (m.size > limit)
    Bad("selected member exceeds trusted length");
  Reader f(path_);
  std::array<std::uint8_t, 65536> in{}, out{};
  gufo::crypto::Sha256Hasher hash;
  auto crc = crc32(0, Z_NULL, 0);
  std::uint64_t read = 0, written = 0;
  auto emit = [&](std::span<const std::uint8_t> b) {
    if (b.size() > limit - written || b.size() > m.size - written)
      Bad("decompression size limit");
    written += b.size();
    crc = crc32(crc, b.data(), static_cast<uInt>(b.size()));
    hash.Update(b);
    if (sink)
      sink(b);
  };
  z_stream z{};
  const bool deflated = m.method == 8;
  if (deflated && inflateInit2(&z, -15) != Z_OK)
    Bad("inflate initialization");
  struct End {
    z_stream* z;
    bool active;
    ~End() {
      if (active)
        inflateEnd(z);
    }
  } end{&z, deflated};
  bool ended = false;
  while (read < m.compressed) {
    if (cancelled && cancelled())
      Bad("cancelled");
    const auto n = static_cast<std::size_t>(
        std::min<std::uint64_t>(in.size(), m.compressed - read));
    f.ReadAt(m.data_offset + read, std::span(in).first(n));
    read += n;
    if (!deflated) {
      emit(std::span(in).first(n));
      continue;
    }
    z.next_in = in.data();
    z.avail_in = static_cast<uInt>(n);
    do {
      z.next_out = out.data();
      z.avail_out = static_cast<uInt>(out.size());
      const auto before = z.avail_in;
      const auto rc = inflate(&z, Z_NO_FLUSH);
      const auto produced = out.size() - z.avail_out;
      emit(std::span(out).first(produced));
      if (rc == Z_STREAM_END) {
        ended = true;
        if (z.avail_in || read != m.compressed)
          Bad("trailing compressed bytes");
        break;
      }
      if (rc != Z_OK || (before == z.avail_in && !produced))
        Bad("invalid DEFLATE stream");
    } while (z.avail_in || !z.avail_out);
    if (ended)
      break;
  }
  if ((deflated && !ended) || written != m.size || crc != m.crc)
    Bad("length/CRC mismatch");
  const auto digest = hash.FinishHex();
  if (!expected_sha256.empty() && digest != expected_sha256)
    throw ZipIntegrityError("ZIP: SHA-256 mismatch");
  return digest;
}
}  // namespace bubo::prepare
