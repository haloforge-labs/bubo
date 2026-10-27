#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>

#include "src/core/prepare/transport.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/metadata_json.hpp"

namespace bubo::prepare {
namespace {
std::string Header(const Response& response, const std::string& name) {
  const auto it = response.headers.find(name);
  return it == response.headers.end() ? std::string{} : it->second;
}
std::string Unquote(std::string value) {
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
    value = value.substr(1, value.size() - 2);
  for (auto& c : value)
    if (c >= 'A' && c <= 'F')
      c += 'a' - 'A';
  return value;
}
bool Strong(std::string_view value) {
  return value.size() >= 2 && value.front() == '"' && value.back() == '"' &&
         value.find_first_of("\r\n") == value.npos;
}
std::string LiveHex(const gufo::crypto::Sha256Hasher& hasher) {
  constexpr char hex[] = "0123456789abcdef";
  std::string text;
  for (const auto byte : hasher.Digest()) {
    text += hex[byte >> 4];
    text += hex[byte & 15];
  }
  return text;
}
struct Partial {
  std::uint64_t prefix = 0;
  std::string origin_etag, linked_etag, payload_etag, payload_host;
};
std::filesystem::path Sidecar(const std::filesystem::path& partial) {
  auto result = partial;
  result += ".json";
  return result;
}
void RequireLocalResumeInputs(const std::filesystem::path& path) {
  for (const auto& input : {path, Sidecar(path)}) {
    const auto error = CheckLocalPath(input);
    if (!error.empty())
      throw std::runtime_error(error);
  }
}
Partial Load(const Entry& entry, const std::filesystem::path& path) {
  // Locality failures are storage errors, not invalid resumable metadata.
  // Keep this outside the metadata catch so a caller cannot silently restart
  // and open/hydrate an offline input after losing its retained-prefix marker.
  RequireLocalResumeInputs(path);
  Partial result;
  try {
    if (!std::filesystem::exists(path) ||
        !std::filesystem::exists(Sidecar(path)) ||
        std::filesystem::file_size(Sidecar(path)) > 64 * 1024)
      return {};
    using namespace metadata;
    const auto value = Parse(Read(Sidecar(path)));
    Keys(value, {"schema_version", "repo", "commit", "path", "expected_size",
                 "expected_sha256", "prefix", "origin_etag", "x_linked_etag",
                 "payload_etag", "payload_host"});
    if (Integer(Required(value, "schema_version")) != 1 ||
        String(Required(value, "repo")) != entry.source_repo ||
        String(Required(value, "commit")) != entry.commit ||
        String(Required(value, "path")) != entry.repo_path ||
        Decimal(String(Required(value, "expected_size"))) != entry.size ||
        String(Required(value, "expected_sha256")) != entry.sha256)
      return {};
    result.prefix = Decimal(String(Required(value, "prefix")));
    if (result.prefix > entry.size ||
        result.prefix > std::filesystem::file_size(path))
      return {};
    result.origin_etag = String(Required(value, "origin_etag"));
    result.linked_etag = String(Required(value, "x_linked_etag"));
    result.payload_etag = String(Required(value, "payload_etag"));
    result.payload_host = String(Required(value, "payload_host"));
    for (const auto* item : {&result.origin_etag, &result.linked_etag,
                             &result.payload_etag, &result.payload_host})
      if (item->size() > 1024 || item->find_first_of("\r\n") != item->npos)
        return {};
    if (result.payload_host.find_first_of("/@?\\") != result.payload_host.npos)
      return {};
    return result;
  } catch (...) {
    return {};
  }
}
void Save(const Entry& entry, const std::filesystem::path& path,
          const Partial& partial) {
  using metadata::Value;
  auto value = Value::object();
  value["schema_version"] = 1;
  value["repo"] = std::string(entry.source_repo);
  value["commit"] = std::string(entry.commit);
  value["path"] = std::string(entry.repo_path);
  value["expected_size"] = std::to_string(entry.size);
  value["expected_sha256"] = std::string(entry.sha256);
  value["prefix"] = std::to_string(partial.prefix);
  value["origin_etag"] = partial.origin_etag;
  value["x_linked_etag"] = partial.linked_etag;
  value["payload_etag"] = partial.payload_etag;
  value["payload_host"] = partial.payload_host;
  std::string error;
  const auto text = value.dump() + "\n";
  if (text.size() > 64 * 1024)
    throw std::runtime_error("partial sidecar exceeds 64 KiB");
  if (!AtomicWrite(Sidecar(path), text, error))
    throw std::runtime_error(error);
}
// FlushFileBuffers before sidecar publication: bytes beyond this checkpoint
// are discarded on reopen. The hasher sees exactly the successfully written
// prefix; it is never serialized or reconstructed during in-process retries.
class Writer {
public:
  explicit Writer(const std::filesystem::path& path) {
#ifdef _WIN32
    handle_ = CreateFileW(
        NativePath(path).c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
        FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE)
      throw std::runtime_error(
          "cannot open partial for writing: Windows error " +
          std::to_string(GetLastError()));
#else
    stream_.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!stream_) {
      std::ofstream create(path, std::ios::binary);
      create.close();
      stream_.open(path, std::ios::binary | std::ios::in | std::ios::out);
    }
    if (!stream_)
      throw std::runtime_error("cannot open partial for writing");
#endif
  }
  ~Writer() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE)
      CloseHandle(handle_);
#endif
  }
  void Seek(std::uint64_t position) {
#ifdef _WIN32
    LARGE_INTEGER offset{};
    offset.QuadPart = static_cast<LONGLONG>(position);
    if (!SetFilePointerEx(handle_, offset, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(handle_))
      throw std::runtime_error("cannot truncate/checkpoint partial");
#else
    stream_.seekp(position);
#endif
  }
  void Write(std::span<const std::uint8_t> bytes) {
#ifdef _WIN32
    DWORD written = 0;
    if (bytes.size() > MAXDWORD ||
        !WriteFile(handle_, bytes.data(), static_cast<DWORD>(bytes.size()),
                   &written, nullptr) ||
        written != bytes.size())
      throw std::runtime_error("partial write failed (disk space/access)");
#else
    stream_.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    if (!stream_)
      throw std::runtime_error("partial write failed (disk space/access)");
#endif
  }
  void Flush() {
#ifdef _WIN32
    if (!FlushFileBuffers(handle_))
      throw std::runtime_error("partial flush failed");
#else
    stream_.flush();
    if (!stream_)
      throw std::runtime_error("partial flush failed");
#endif
  }
  std::intptr_t NativeHandle() const {
#ifdef _WIN32
    return reinterpret_cast<std::intptr_t>(handle_);
#else
    return -1;
#endif
  }
  std::shared_ptr<ReadLease> SealVerified(const std::filesystem::path& path) {
    Flush();
#ifdef _WIN32
    FILETIME observed{}, frozen{0xffffffff, 0xffffffff};
    GetSystemTimeAsFileTime(&observed);
    // The writer still denies other writers and deletion. Freeze its deferred
    // close-time update so the observed stamp remains valid after publication.
    if (!SetFileTime(handle_, nullptr, nullptr, &observed) ||
        !SetFileTime(handle_, nullptr, nullptr, &frozen))
      throw std::runtime_error("cannot seal verified file timestamp");
    auto lease = std::make_shared<ReadLease>(
        ReadLease::AdoptVerifiedHandle(NativeHandle()));
    handle_ = INVALID_HANDLE_VALUE;
    return lease;
#else
    stream_.close();
    return std::make_shared<ReadLease>(path);
#endif
  }

private:
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  std::fstream stream_;
#endif
};
}  // namespace
std::uint64_t RetainedPrefix(const Entry& entry,
                             const std::filesystem::path& partial) {
  return Load(entry, NativePath(partial)).prefix;
}
DownloadResult Download(const Entry& entry,
                        const std::filesystem::path& destination_partial,
                        Transport& transport, const DownloadOptions& options) {
  DownloadResult result;
  const auto path = NativePath(destination_partial);
  const auto fail = [&](int code, std::string text) {
    result.code = code;
    result.error = std::string(entry.logical_name) + ": " + std::move(text);
    return result;
  };
  std::string origin;
  try {
#ifdef BUBO_PREPARE_TESTING
    origin = options.origin_override.empty() ? OriginUrl(entry)
                                             : options.origin_override;
#else
    origin = OriginUrl(entry);
#endif
  } catch (const std::exception& error) {
    return fail(4, error.what());
  }
  const auto origin_checked = CheckUrl(origin, options.policy);
  if (!origin_checked)
    return fail(5, origin_checked.error);
  const auto begun = std::chrono::steady_clock::now();
  const auto cancelled = [&] {
    return options.cancelled && options.cancelled();
  };
  const auto wait = [&](std::chrono::seconds delay) {
    if (options.wait) {
      options.wait(delay);
      return;
    }
    for (auto elapsed = std::chrono::milliseconds(0);
         elapsed < delay && !cancelled() &&
         std::chrono::steady_clock::now() - begun < options.max_time;
         elapsed += std::chrono::milliseconds(100))
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  };
  try {
    // Also applies to --no-resume, which deliberately skips Load.
    RequireLocalResumeInputs(path);
    auto partial = options.resume ? Load(entry, path) : Partial{};
    std::filesystem::create_directories(path.parent_path());
    Writer writer(path);  // Deny other writers/deletion before prefix hashing.
    auto hasher = std::make_unique<gufo::crypto::Sha256Hasher>();
    if (partial.prefix) {
      result.prefix_hash_bytes = HashInto(
          path, *hasher, partial.prefix,
          [&](std::uint64_t done, std::uint64_t) {
            if (cancelled())
              throw std::runtime_error("cancelled while hashing saved prefix");
            if (std::chrono::steady_clock::now() - begun >= options.max_time)
              throw std::runtime_error(
                  "per-file time cap during saved-prefix hash");
            // Hashing a long saved prefix is not silence: report the
            // verified bytes, up to the resume offset.
            if (options.progress)
              options.progress(std::min(done, partial.prefix), entry.size);
          },
          writer.NativeHandle());
    }
    writer.Seek(partial.prefix);
    // Bytes covered by the saved checkpoint. A stall or a crash in a long
    // response keeps everything up to the last checkpoint_bytes boundary.
    std::uint64_t checkpointed = 0;
    const auto checkpoint = [&] {
      writer.Flush();
      Save(entry, path, partial);
      result.bytes = partial.prefix;
      checkpointed = partial.prefix;
    };
    const auto restart = [&] {
      partial = {};
      writer.Seek(0);
      hasher = std::make_unique<gufo::crypto::Sha256Hasher>();
      checkpoint();
    };
    checkpoint();
    unsigned stalled = 0, expired = 0;
    bool reset416 = false;
    std::uint64_t checked_space_at = 0;
    for (unsigned attempt = 0; attempt < options.max_attempts; ++attempt) {
      if (cancelled()) {
        checkpoint();
        return fail(3, "cancelled; resumable partial retained");
      }
      if (std::chrono::steady_clock::now() - begun >= options.max_time)
        return fail(5, "per-file time cap reached; resumable partial retained");
      const auto before = partial.prefix;
      const auto old = partial;
      std::string url = origin;
      unsigned hops = 0;
      bool retry = false, finished = false;
      std::chrono::seconds delay{0};
      int error_code = 0;
      std::string error;
      while (!retry && !finished && !error_code) {
        const auto checked = CheckUrl(url, options.policy);
        if (!checked)
          return fail(5, checked.error);
        const bool at_origin = hops == 0;
        Request request{url, {{"accept-encoding", "identity"}}};
        // Rebuilt allowlisted headers on every hop; no auth/cookie source
        // exists.
        if (partial.prefix) {
          request.headers["range"] =
              "bytes=" + std::to_string(partial.prefix) + "-";
          if (partial.payload_host == checked.host &&
              Strong(partial.payload_etag))
            request.headers["if-range"] = partial.payload_etag;
        }
        Response response;
        bool saw_response = false;
        bool payload = false;
        std::uint64_t response_bytes = 0, expected_response = 0;
        const auto transfer = transport.Perform(
            request,
            [&](const Response& head) {
              response = head;
              saw_response = true;
              if (at_origin && (head.status == 200 || head.status == 206 ||
                                (head.status >= 300 && head.status < 400))) {
                const auto linked = Header(head, "x-linked-etag"),
                           size = Header(head, "x-linked-size");
                bool wrong_size = false;
                try {
                  wrong_size =
                      !size.empty() && metadata::Decimal(size) != entry.size;
                } catch (...) {
                  wrong_size = true;
                }
                if ((!linked.empty() && (!IsLowerHex(Unquote(linked), 64) ||
                                         Unquote(linked) != entry.sha256)) ||
                    wrong_size) {
                  error_code = 6;
                  error =
                      "source differs from the pin (X-Linked metadata); no "
                      "payload accepted";
                  return false;
                }
                const auto etag = Header(head, "etag");
                if (etag.size() > 1024 || linked.size() > 1024) {
                  error_code = 5;
                  error = "origin validator exceeds 1024 bytes";
                  return false;
                }
                const auto current = !linked.empty() ? linked : etag;
                const auto previous = !old.linked_etag.empty()
                                          ? old.linked_etag
                                          : old.origin_etag;
                // A comparable changed identity requires a restart. With no
                // origin validator, a matching strong payload validator can
                // still resume.
                if (partial.prefix && !previous.empty() &&
                    current != previous) {
                  restart();
                  retry = true;
                  return false;
                }
                partial.linked_etag = linked;
                partial.origin_etag = etag;
              }
              if (head.status >= 300 && head.status < 400)
                return true;
              if (head.status == 429) {
                std::uint64_t seconds = 0;
                if (!ParseRetryAfter(Header(head, "retry-after"),
                                     std::time(nullptr), seconds)) {
                  error_code = 5;
                  error = "429 has invalid Retry-After";
                } else if (seconds > 60) {
                  error_code = 5;
                  error = "429 Retry-After " + std::to_string(seconds) +
                          " seconds; rerun after that interval";
                } else {
                  retry = true;
                  delay = std::chrono::seconds(seconds);
                }
                return true;
              }
              if (head.status == 401 || head.status == 403) {
                if (!at_origin && expired < 2) {
                  ++expired;
                  retry = true;
                } else {
                  error_code = 5;
                  error = "HTTP " + std::to_string(head.status) +
                          (at_origin ? " at pinned origin; public file access "
                                       "denied (no login fallback)"
                                     : " at payload host after URL refresh; "
                                       "access denied");
                }
                return true;
              }
              if (head.status == 416) {
                ContentRange range;
                if (!ParseContentRange(Header(head, "content-range"), range) ||
                    !range.unsatisfied || range.total != entry.size) {
                  error_code = 6;
                  error = "invalid 416 Content-Range";
                } else if (partial.prefix == entry.size &&
                           LiveHex(*hasher) == entry.sha256) {
                  finished = true;
                } else if (!reset416) {
                  reset416 = true;
                  restart();
                  retry = true;
                } else {
                  error_code = 6;
                  error = "416 cannot validate the retained prefix";
                }
                return true;
              }
              if (head.status == 408 || head.status >= 500) {
                retry = true;
                return true;
              }
              if (head.status != 200 && head.status != 206) {
                error_code = 5;
                error =
                    "HTTP " + std::to_string(head.status) + " at pinned source";
                return true;
              }
              const auto encoding = Header(head, "content-encoding");
              if (!encoding.empty() && encoding != "identity") {
                error_code = 6;
                error = "encoded payload forbidden";
                return false;
              }
              const auto etag = Header(head, "etag");
              if (etag.size() > 1024) {
                error_code = 5;
                error = "payload validator exceeds 1024 bytes";
                return false;
              }
              if (head.status == 200 && partial.prefix)
                restart();
              if (head.status == 206) {
                ContentRange range;
                if (!ParseContentRange(Header(head, "content-range"), range) ||
                    range.unsatisfied || range.start != partial.prefix ||
                    range.total != entry.size) {
                  error_code = 6;
                  error =
                      "invalid 206 Content-Range (offset/total differ from "
                      "pin)";
                  return false;
                }
                expected_response = range.end - range.start + 1;
                if (partial.prefix) {
                  if (old.payload_host == checked.host &&
                      Strong(old.payload_etag) && !etag.empty() &&
                      old.payload_etag != etag) {
                    restart();
                    retry = true;
                    return false;
                  }
                  const bool comparable_origin =
                      (!partial.linked_etag.empty() &&
                       partial.linked_etag == old.linked_etag) ||
                      (!partial.origin_etag.empty() &&
                       partial.origin_etag == old.origin_etag);
                  const bool comparable_payload =
                      old.payload_host == checked.host &&
                      Strong(old.payload_etag) && etag == old.payload_etag;
                  if (!comparable_origin && !comparable_payload) {
                    restart();
                    retry = true;
                    return false;
                  }
                }
              } else {
                expected_response = entry.size;
              }
              const auto length = Header(head, "content-length");
              bool wrong_length = false;
              try {
                wrong_length = !length.empty() &&
                               metadata::Decimal(length) != expected_response;
              } catch (...) {
                wrong_length = true;
              }
              if (wrong_length) {
                error_code = 6;
                error = "Content-Length differs from pinned/range length";
                return false;
              }
              partial.payload_etag = etag;
              partial.payload_host = checked.host;
              if (options.on_host)
                options.on_host(checked.host);
              payload = true;
              return true;
            },
            [&](std::span<const std::uint8_t> bytes) {
              if (!payload || error_code || retry)
                return false;
              if (cancelled()) {
                error_code = 3;
                error = "cancelled; resumable partial retained";
                return false;
              }
              if (std::chrono::steady_clock::now() - begun >=
                  options.max_time) {
                error_code = 5;
                error = "per-file time cap reached";
                return false;
              }
              if (bytes.size() > entry.size - partial.prefix ||
                  bytes.size() > expected_response - response_bytes) {
                error_code = 6;
                error = "payload exceeds the pinned size/range";
                return false;
              }
              if (options.check_space &&
                  (checked_space_at == 0 ||
                   partial.prefix -
                           std::min(checked_space_at, partial.prefix) >=
                       kFingerprintBytes)) {
                if (!options.check_space()) {
                  error_code = 7;
                  error =
                      "insufficient space while streaming; resumable partial "
                      "retained";
                  return false;
                }
                checked_space_at = partial.prefix;
              }
              try {
                writer.Write(bytes);
              } catch (const std::exception& storage_error) {
                error_code = 7;
                error = storage_error.what();
                return false;
              }
              hasher->Update(bytes);
              partial.prefix += bytes.size();
              response_bytes += bytes.size();
              if (options.checkpoint_bytes &&
                  partial.prefix - std::min(checkpointed, partial.prefix) >=
                      options.checkpoint_bytes) {
                try {
                  checkpoint();
                } catch (const std::exception& storage_error) {
                  error_code = 7;
                  error = storage_error.what();
                  return false;
                }
              }
              if (options.progress)
                options.progress(partial.prefix, entry.size);
              return true;
            },
            [&] {
              return cancelled() || std::chrono::steady_clock::now() - begun >=
                                        options.max_time;
            });
        checkpoint();
        if (error_code)
          return fail(error_code, error);
        if (cancelled())
          return fail(3, "cancelled; resumable partial retained");
        if (!saw_response && !transfer.error.empty()) {
          error = transfer.error;
          retry = true;
        }
        if (response.status >= 300 && response.status < 400) {
          const auto next = RedirectUrl(url, Header(response, "location"), hops,
                                        options.policy);
          if (!next)
            return fail(5, next.error);
          ++hops;
          url = next.url;
          continue;
        }
        if (payload) {
          if (!transfer.error.empty() || response_bytes != expected_response ||
              partial.prefix != entry.size) {
            error = transfer.error.empty()
                        ? "short body; re-resolving pinned origin"
                        : transfer.error;
            retry = true;
          } else {
            finished = true;
          }
        }
        if (!payload && !retry && !finished)
          return fail(5, transfer.error.empty() ? "no valid payload response"
                                                : transfer.error);
      }
      if (finished) {
        if (partial.prefix != entry.size || hasher->FinishHex() != entry.sha256)
          return fail(6,
                      "SHA-256 mismatch; partial retained, no canonical file "
                      "published");
        result.verified_file = writer.SealVerified(path);
        return result;
      }
      if (partial.prefix >= before && partial.prefix - before >= 1024 * 1024) {
        stalled = 0;
        expired = 0;
      } else if (++stalled >= 3) {
        return fail(
            5, "no-progress budget exhausted (1 MiB progress floor); " + error);
      }
      if (delay.count() == 0)
        delay = std::chrono::seconds(stalled < 2 ? 1 : 3);
      wait(delay);
    }
    return fail(5, "per-file attempt cap reached; resumable partial retained");
  } catch (const std::exception& error) {
    return fail(cancelled() ? 3
                : std::chrono::steady_clock::now() - begun >= options.max_time
                    ? 5
                    : 7,
                error.what());
  }
}
}  // namespace bubo::prepare
