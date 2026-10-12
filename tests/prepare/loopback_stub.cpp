#include "tests/prepare/loopback_stub.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace bubo::prepare::test {
namespace {
#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket kInvalid = INVALID_SOCKET;
void Close(Socket socket) {
  closesocket(socket);
}
void InitializeSockets() {
  static const int initialized = [] {
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data);
  }();
  if (initialized != 0)
    throw std::runtime_error("loopback WSAStartup failed");
}
#else
using Socket = int;
constexpr Socket kInvalid = -1;
void Close(Socket socket) {
  close(socket);
}
void InitializeSockets() {}
#endif
std::string Lower(std::string value) {
  for (auto& c : value)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return value;
}
bool Number(std::string_view text, std::uint64_t& value) {
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return !text.empty() && error == std::errc{} &&
         end == text.data() + text.size();
}
bool Send(Socket socket, const char* data, std::size_t size) {
  while (size) {
#ifdef _WIN32
    const auto count = send(
        socket, data, static_cast<int>(std::min(size, std::size_t{65536})), 0);
#else
    const auto count =
        send(socket, data, std::min(size, std::size_t{65536}), MSG_NOSIGNAL);
#endif
    if (count <= 0)
      return false;
    data += count;
    size -= static_cast<std::size_t>(count);
  }
  return true;
}
StubReply RangeReply(const StubRequest& request, std::uint64_t size,
                     std::string etag) {
  StubReply reply;
  reply.length = size;
  reply.headers["etag"] = std::move(etag);
  const auto range = request.headers.find("range");
  if (range != request.headers.end()) {
    const auto text = std::string_view(range->second);
    std::uint64_t start = 0;
    if (!text.starts_with("bytes=") || !text.ends_with("-") ||
        !Number(text.substr(6, text.size() - 7), start)) {
      reply.status = 400;
      reply.length = 0;
      return reply;
    }
    const auto if_range = request.headers.find("if-range");
    if (if_range == request.headers.end() ||
        if_range->second == reply.headers["etag"]) {
      if (start >= size) {
        reply.status = 416;
        reply.length = 0;
        reply.headers["content-range"] = "bytes */" + std::to_string(size);
      } else {
        reply.status = 206;
        reply.start = start;
        reply.length = size - start;
        reply.headers["content-range"] = "bytes " + std::to_string(start) +
                                         "-" + std::to_string(size - 1) + "/" +
                                         std::to_string(size);
      }
    }
  }
  return reply;
}
std::string DecodePath(std::string_view raw) {
  std::string result;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (raw[i] != '%') {
      result += raw[i];
      continue;
    }
    if (i + 2 >= raw.size())
      throw std::runtime_error("truncated percent escape");
    unsigned value = 0;
    const auto [end, error] =
        std::from_chars(raw.data() + i + 1, raw.data() + i + 3, value, 16);
    if (error != std::errc{} || end != raw.data() + i + 3 || value < 32 ||
        value == '\\' || value == '/' || value == 127)
      throw std::runtime_error("unsafe percent escape");
    result += static_cast<char>(value);
    i += 2;
  }
  return result;
}
StubReply MirrorReply(const StubRequest& request,
                      const std::filesystem::path& mirror) {
  const auto target = std::string_view(request.target);
  const auto resolve = target.find("/resolve/");
  if (!target.starts_with('/') || resolve == target.npos)
    return {404};
  const auto commit_begin = resolve + 9;
  if (target.size() < commit_begin + 41 || target[commit_begin + 40] != '/')
    return {404};
  for (const auto c : target.substr(commit_begin, 40))
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return {404};
  const auto relative =
      DecodePath(std::string(target.substr(1, resolve - 1)) + "/" +
                 std::string(target.substr(commit_begin + 41)));
  const std::filesystem::path path(relative);
  if (path.is_absolute() || path.has_root_name() ||
      relative.find_first_of("\\?:") != relative.npos)
    return {400};
  for (const auto& component : path)
    if (component == ".." || component == ".")
      return {400};
  const auto full = mirror / path;
  return std::filesystem::is_regular_file(full) ? FileReply(request, full)
                                                : StubReply{404};
}
}  // namespace

StubReply FileReply(const StubRequest& request,
                    const std::filesystem::path& file, std::string etag) {
  auto reply =
      RangeReply(request, std::filesystem::file_size(file), std::move(etag));
  reply.file = file;
  return reply;
}
StubReply GeneratedReply(const StubRequest& request, std::uint64_t size) {
  auto reply = RangeReply(request, size, "\"generated-v1\"");
  reply.generated_size = size;
  return reply;
}
struct LoopbackStub::Impl {
  Handler handler;
  Socket listener = kInvalid;
  std::uint16_t port = 0;
  std::atomic<bool> stop{false};
  mutable std::mutex mutex;
  std::vector<StubRequest> records;
  std::string error;
  std::thread worker;

  explicit Impl(Handler callback) : handler(std::move(callback)) {
    InitializeSockets();
    for (;;) {
      listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (listener == kInvalid)
        throw std::runtime_error("loopback socket failed");
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      address.sin_port = 0;
      if (bind(listener, reinterpret_cast<sockaddr*>(&address),
               sizeof(address)) != 0) {
        Close(listener);
        throw std::runtime_error("loopback bind failed");
      }
#ifdef _WIN32
      int length = sizeof(address);
#else
      socklen_t length = sizeof(address);
#endif
      if (getsockname(listener, reinterpret_cast<sockaddr*>(&address),
                      &length) != 0) {
        Close(listener);
        throw std::runtime_error("loopback getsockname failed");
      }
      port = ntohs(address.sin_port);
      if (port < 8081 || port > 8083)
        break;
      Close(listener);
    }
    if (listen(listener, 8) != 0) {
      Close(listener);
      throw std::runtime_error("loopback listen failed");
    }
    worker = std::thread([this] { Run(); });
  }
  ~Impl() {
    stop = true;
    worker.join();
    Close(listener);
  }
  void Run() {
    while (!stop) {
      fd_set readable;
      FD_ZERO(&readable);
      FD_SET(listener, &readable);
      timeval timeout{0, 100000};
      const auto ready = select(static_cast<int>(listener + 1), &readable,
                                nullptr, nullptr, &timeout);
      if (ready == 0)
        continue;
      if (ready < 0) {
        std::lock_guard guard(mutex);
        error = "loopback select failed";
        return;
      }
      sockaddr_in peer{};
#ifdef _WIN32
      int peer_length = sizeof(peer);
#else
      socklen_t peer_length = sizeof(peer);
#endif
      const auto client =
          accept(listener, reinterpret_cast<sockaddr*>(&peer), &peer_length);
      if (client == kInvalid)
        continue;
      if (ntohl(peer.sin_addr.s_addr) != INADDR_LOOPBACK) {
        Close(client);
        continue;
      }
#ifdef _WIN32
      DWORD receive_timeout = 2000;
      setsockopt(client, SOL_SOCKET, SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&receive_timeout),
                 sizeof(receive_timeout));
      setsockopt(client, SOL_SOCKET, SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&receive_timeout),
                 sizeof(receive_timeout));
#else
      timeval receive_timeout{2, 0};
      setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                 sizeof(receive_timeout));
      setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &receive_timeout,
                 sizeof(receive_timeout));
#endif
      try {
        Serve(client);
      } catch (const std::exception& failure) {
        std::lock_guard guard(mutex);
        error = failure.what();
      }
      Close(client);
    }
  }
  void Serve(Socket client) {
    std::string wire;
    std::array<char, 4096> buffer{};
    while (wire.find("\r\n\r\n") == wire.npos && wire.size() <= 65536) {
      const auto count =
          recv(client, buffer.data(), static_cast<int>(buffer.size()), 0);
      if (count <= 0)
        return;
      wire.append(buffer.data(), static_cast<std::size_t>(count));
    }
    if (wire.size() > 65536)
      throw std::runtime_error("stub request exceeds bound");
    StubRequest request;
    const auto first = wire.find(' '), second = wire.find(' ', first + 1);
    if (first == wire.npos || second == wire.npos)
      throw std::runtime_error("invalid stub request line");
    request.method = wire.substr(0, first);
    request.target = wire.substr(first + 1, second - first - 1);
    auto begin = wire.find("\r\n") + 2;
    for (;;) {
      const auto end = wire.find("\r\n", begin);
      if (end == begin || end == wire.npos)
        break;
      const auto colon = wire.find(':', begin);
      if (colon >= end)
        throw std::runtime_error("invalid stub request header");
      auto value_begin = colon + 1;
      while (value_begin < end && wire[value_begin] == ' ')
        ++value_begin;
      request.headers.emplace(Lower(wire.substr(begin, colon - begin)),
                              wire.substr(value_begin, end - value_begin));
      begin = end + 2;
    }
    {
      std::lock_guard guard(mutex);
      request.number = static_cast<unsigned>(records.size() + 1);
      records.push_back(request);
    }
    auto reply = handler(request);
    if (!reply.body.empty() && reply.length == 0)
      reply.length = reply.body.size();
    reply.headers.try_emplace("content-length", std::to_string(reply.length));
    reply.headers["connection"] = "close";
    std::string header =
        "HTTP/1.1 " + std::to_string(reply.status) + " Fixture\r\n";
    for (const auto& [key, value] : reply.headers) {
      if (key.find_first_of("\r\n:") != key.npos ||
          value.find_first_of("\r\n") != value.npos)
        throw std::runtime_error("unsafe fixture response header");
      header += key + ": " + value + "\r\n";
    }
    header += "\r\n";
    if (!Send(client, header.data(), header.size()) || request.method == "HEAD")
      return;
    std::ifstream file;
    if (!reply.file.empty()) {
      file.open(reply.file, std::ios::binary);
      file.seekg(static_cast<std::streamoff>(reply.start));
      if (!file)
        throw std::runtime_error("fixture file open/seek failed");
    }
    std::array<char, 65536> body{};
    const auto limit = std::min(reply.length, reply.drop_after);
    for (std::uint64_t sent = 0; sent < limit && !stop;) {
      const auto count = static_cast<std::size_t>(
          std::min<std::uint64_t>(body.size(), limit - sent));
      if (file.is_open()) {
        file.read(body.data(), static_cast<std::streamsize>(count));
        if (file.gcount() != static_cast<std::streamsize>(count))
          return;
      } else if (reply.generated_size) {
        for (std::size_t i = 0; i < count; ++i)
          body[i] = static_cast<char>((reply.start + sent + i) % 251);
      } else {
        if (reply.start + sent + count > reply.body.size())
          return;
        std::copy_n(reply.body.data() + reply.start + sent, count, body.data());
      }
      if (reply.corrupt_at >= reply.start + sent &&
          reply.corrupt_at < reply.start + sent + count)
        body[reply.corrupt_at - reply.start - sent] ^= 1;
      if (!Send(client, body.data(), count))
        return;
      sent += count;
    }
  }
};
LoopbackStub::LoopbackStub(Handler handler)
    : impl_(std::make_unique<Impl>(std::move(handler))) {}
LoopbackStub::LoopbackStub(const std::filesystem::path& mirror)
    : LoopbackStub([mirror](const StubRequest& request) {
        return MirrorReply(request, mirror);
      }) {}
LoopbackStub::~LoopbackStub() = default;
std::string LoopbackStub::url(std::string_view target) const {
  return "http://127.0.0.1:" + std::to_string(impl_->port) +
         std::string(target);
}
std::uint16_t LoopbackStub::port() const {
  return impl_->port;
}
std::vector<StubRequest> LoopbackStub::records() const {
  std::lock_guard guard(impl_->mutex);
  return impl_->records;
}
std::string LoopbackStub::error() const {
  std::lock_guard guard(impl_->mutex);
  return impl_->error;
}
}  // namespace bubo::prepare::test
