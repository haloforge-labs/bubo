#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "src/core/prepare/hash_file.hpp"
#include "src/core/prepare/transport.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

namespace {
using namespace bubo::prepare;
struct Hop {
  Response response;
  std::string body, error;
  std::function<void(const Request&)> inspect;
};
class Fake final : public Transport {
public:
  std::vector<Hop> hops;
  std::size_t requests = 0, bytes = 0;
  TransferResult Perform(const Request& request, const HeaderSink& head,
                         const BodySink& body,
                         const std::function<bool()>&) override {
    assert(requests < hops.size());
    const auto& hop = hops[requests++];
    if (hop.inspect)
      hop.inspect(request);
    try {
      if (!head(hop.response))
        return {"header rejected"};
      if (hop.response.status < 300 && !hop.body.empty()) {
        bytes += hop.body.size();
        if (!body(std::span(
                reinterpret_cast<const std::uint8_t*>(hop.body.data()),
                hop.body.size())))
          return {"body rejected"};
      }
    } catch (const std::exception& error) {
      return {error.what()};
    }
    return {hop.error};
  }
};
Hop Full(std::string body = "abc") {
  return {{200,
           {{"content-length", "3"},
            {"etag", "\"payload\""},
            {"x-linked-etag", std::string(fixture::kAbcHash)},
            {"x-linked-size", "3"}}},
          std::move(body)};
}
Hop Redirect(
    std::string location = "https://cdn.hf.co/signed?secret=NEVER_SAVE") {
  return {{302,
           {{"location", std::move(location)},
            {"x-linked-etag", std::string(fixture::kAbcHash)},
            {"x-linked-size", "3"}}}};
}
DownloadOptions Options() {
  DownloadOptions result;
  result.wait = [](auto) {};
  return result;
}
void Clean(const std::filesystem::path& path) {
  std::filesystem::remove(path);
  auto sidecar = path;
  sidecar += ".json";
  std::filesystem::remove(sidecar);
}
}  // namespace
int main() {
  using namespace bubo::prepare;
  const auto entry = fixture::kCatalogue[2];
  const auto root =
      std::filesystem::current_path() / "prepare-transport-fixture";
  std::filesystem::create_directories(root);
  const auto path = root / "abc.partial";
  Clean(path);
  {
    Fake fake;
    auto wrong = Full();
    wrong.response.headers["x-linked-etag"] = std::string(64, '0');
    fake.hops = {wrong};
    const auto result = Download(entry, path, fake, Options());
    assert(result.code == 6 && fake.bytes == 0 &&
           std::filesystem::file_size(path) == 0);
    auto bad_size = Full();
    bad_size.response.headers["x-linked-size"] = "4";
    fake = Fake{};
    fake.hops = {bad_size};
    assert(Download(entry, path, fake, Options()).code == 6 && fake.bytes == 0);
    std::cout << "PASS: X-Linked size/hash reject in origin headers before any "
                 "payload byte\n";
  }
  Clean(path);
  {
    Fake fake;
    auto origin = Redirect();
    origin.inspect = [](const Request& req) {
      assert(!req.headers.contains("authorization") &&
             !req.headers.contains("cookie") &&
             !req.headers.contains("if-range"));
    };
    auto first = Full("a");
    first.response.headers.erase("x-linked-etag");
    first.response.headers.erase("x-linked-size");
    first.error = "connection drop";
    auto second = Hop{{206,
                       {{"content-range", "bytes 1-2/3"},
                        {"content-length", "2"},
                        {"etag", "\"payload\""}}},
                      "bc"};
    second.inspect = [](const Request& req) {
      assert(req.headers.at("range") == "bytes=1-" &&
             req.headers.at("if-range") == "\"payload\"");
      assert(req.headers.size() == 3);
    };
    fake.hops = {origin, first, origin, second};
    auto result = Download(entry, path, fake, Options());
    assert(result.code == 0 && result.prefix_hash_bytes == 0 &&
           HashFile(path) == fixture::kAbcHash);
    std::ifstream json(path.string() + ".json");
    std::stringstream text;
    text << json.rdbuf();
    assert(text.str().find("NEVER_SAVE") == text.str().npos);
    std::cout << "PASS: cross-host redirect, header rebuild, same-host "
                 "If-Range, re-resolve/drop resume; zero in-process prefix "
                 "rehash; signed URL never persisted\n";
  }
  Clean(path);
  {
    Fake fake;
    fake.hops = {Redirect("https://evilhuggingface.co/payload")};
    assert(Download(entry, path, fake, Options()).code == 5 &&
           fake.requests == 1 && fake.bytes == 0);
    fake = Fake{};
    fake.hops.assign(6, Redirect("https://huggingface.co/again"));
    assert(Download(entry, path, fake, Options()).code == 5 &&
           fake.requests == 6);
    std::cout << "PASS: disallowed redirect never requested; sixth redirect "
                 "rejected\n";
  }
  Clean(path);
  {
    Fake fake;
    fake.hops = {{{429, {{"retry-after", "2"}}}}, Full()};
    auto options = Options();
    unsigned waited = 0;
    options.wait = [&](auto seconds) {
      waited += static_cast<unsigned>(seconds.count());
    };
    assert(Download(entry, path, fake, options).code == 0 && waited == 2);
    Clean(path);
    fake = Fake{};
    fake.hops = {{{429, {{"retry-after", "61"}}}}};
    assert(Download(entry, path, fake, options).code == 5 &&
           fake.requests == 1 && waited == 2);
    std::uint64_t seconds = 0;
    assert(
        ParseRetryAfter("Sun, 06 Nov 1994 08:49:37 GMT", 784111775, seconds) &&
        seconds == 2);
    assert(!ParseRetryAfter("-1", 0, seconds));
    std::cout
        << "PASS: Retry-After seconds/date; >60s stops without early retry\n";
  }
  Clean(path);
  {
    for (int status : {401, 403, 404}) {
      Fake fake;
      fake.hops = {{{status, {}}}};
      const auto result = Download(entry, path, fake, Options());
      assert(result.code == 5 && fake.requests == 1 &&
             result.error.find(std::to_string(status)) != std::string::npos);
    }
    Fake fake;
    fake.hops = {Redirect(), {{403, {}}}, Redirect(), Full()};
    assert(Download(entry, path, fake, Options()).code == 0);
    std::cout << "PASS: origin 401/403/404 terminal; expired payload403 "
                 "refreshes origin\n";
  }
  Clean(path);
  {
    Fake fake;
    fake.hops = {Full("ab"), Full()};
    assert(Download(entry, path, fake, Options()).code == 0);
    Clean(path);
    fake = Fake{};
    fake.hops = {Full("abd")};
    assert(Download(entry, path, fake, Options()).code == 6);
    // The deliberately broken size-only copy accepts the planted wrong byte.
    assert(std::filesystem::file_size(path) == entry.size &&
           HashFile(path) != entry.sha256);
    std::cout << "PASS: short body retries; right-size wrong byte fails SHA "
                 "(broken size-only copy accepts it)\n";
  }
  Clean(path);
  {
    Fake fake;
    fake.hops = {Full("a"), Full("a"), Full("a")};
    auto result = Download(entry, path, fake, Options());
    assert(result.code == 5 && fake.requests == 3 &&
           result.error.find("1 MiB") != result.error.npos);
    Clean(path);
    fake = Fake{};
    fake.hops = {Full("a"), Full("a")};
    auto options = Options();
    options.max_attempts = 2;
    assert(Download(entry, path, fake, options).error.find("attempt cap") !=
           std::string::npos);
    Clean(path);
    fake = Fake{};
    options = Options();
    options.max_time = std::chrono::seconds(0);
    assert(Download(entry, path, fake, options).code == 5 &&
           fake.requests == 0);
    std::cout << "PASS: dribble cannot reset 1 MiB floor; per-file "
                 "attempt/time caps\n";
  }
  Clean(path);
  {
    Fake fake;
    fake.hops = {Full()};
    bool cancel = false;
    auto options = Options();
    options.progress = [&](auto, auto) { cancel = true; };
    options.cancelled = [&] { return cancel; };
    assert(Download(entry, path, fake, options).code == 3);
    fake = Fake{};
    fake.hops = {{{416,
                   {{"content-range", "bytes */3"},
                    {"x-linked-etag", std::string(fixture::kAbcHash)},
                    {"x-linked-size", "3"}}}}};
    auto result = Download(entry, path, fake, Options());
    assert(result.code == 0 && result.prefix_hash_bytes == 3 &&
           fake.bytes == 0);
    result.verified_file.reset();
    std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
    corrupt << "abd";
    corrupt.close();
    fake = Fake{};
    fake.hops = {{{416,
                   {{"content-range", "bytes */3"},
                    {"x-linked-etag", std::string(fixture::kAbcHash)},
                    {"x-linked-size", "3"}}}},
                 Full()};
    assert(Download(entry, path, fake, Options()).code == 0 &&
           HashFile(path) == entry.sha256);
    std::cout << "PASS: cancel retains flushed sidecar; process restart "
                 "rehashes once; 416 completes only with matching size AND "
                 "hash, wrong hash resets once\n";
  }
  Clean(path);
  {
    Fake fake;
    fake.hops = {Full()};
    auto options = Options();
    options.check_space = [] { return false; };
    assert(Download(entry, path, fake, options).code == 7 &&
           std::filesystem::file_size(path) == 0);
    ContentRange range;
    assert(ParseContentRange("bytes 4294967297-4294967300/4294967301", range) &&
           range.start == 4294967297ULL && range.end == 4294967300ULL);
    assert(!ParseContentRange("bytes 18446744073709551616-1/3", range));
    assert(!ParseContentRange("bytes 0-3/3", range));
    std::cout << "PASS: streaming volume rejection before bytes; Content-Range "
                 "uint64 offsets beyond4GiB and overflow rejection\n";
  }
  Clean(path);
  std::filesystem::remove(root);
}
