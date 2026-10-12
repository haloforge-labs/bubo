#include <cassert>
#include <iostream>

#include "src/core/prepare/runtime_table.hpp"
#include "src/core/prepare/transport.hpp"

template<class T>
concept HasLoopbackHook = requires(T value) { value.test_loopback; };
static_assert(!HasLoopbackHook<bubo::prepare::HostPolicy>);
#ifdef BUBO_PREPARE_TESTING
#error Production policy proof must never have the test injection definition
#endif
int main() {
  using namespace bubo::prepare;
  assert(CheckUrl("https://huggingface.co/a"));
  assert(CheckUrl("https://cdn-lfs.hf.co/a"));
  assert(CheckUrl("https://x.huggingface.co/a"));
  for (const auto* bad :
       {"https://evilhuggingface.co/a", "https://hf.co.evil.com/a",
        "http://huggingface.co/a", "http://127.0.0.1:5555/a",
        "https://user:pw@huggingface.co/a", "https://@huggingface.co/a",
        "https://huggingface.co./a", "https://huggingface%2eco/a",
        "https://huggingface.co\\@evil.com/a", "https://huggingface.co/a#x"})
    assert(!CheckUrl(bad));
  assert(RedirectUrl("https://huggingface.co/a/b", "../c", 4));
  assert(!RedirectUrl("https://huggingface.co/a", "https://cdn.hf.co/b", 5));
  assert(!RedirectUrl("https://huggingface.co/a", "//evil.com/a", 0));
  HostPolicy amd;
  amd.source = HostPolicy::Source::Amd;
  assert(CheckUrl("https://stable.repo.amd.com/a", amd));
  assert(!CheckUrl("https://x.stable.repo.amd.com/a", amd));
  for (const auto& wheel : RuntimeWheels()) {
    Entry entry{};
    entry.direct_url = wheel.url;
    assert(OriginUrl(entry) == wheel.url);
    assert(!ValidateFields(std::span(&entry, 1), false).empty());
  }
  const auto broken = [](std::string_view host) {
    return host.find("huggingface.co") != host.npos ||
           host.find("hf.co") != host.npos;
  };
  assert(broken("evilhuggingface.co") && broken("hf.co.evil.com"));
  std::cout << "PASS: production policy has no loopback hook (compile-time and "
               "HTTP refusal); suffix/userinfo/HTTPS/5-hop policy; broken "
               "substring policy accepts both planted near-misses\n";
}
