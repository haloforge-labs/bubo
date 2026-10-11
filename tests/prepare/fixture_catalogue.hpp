#pragma once
#include <array>

#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::prepare::fixture {
// Synthetic fixture commits/repos only. These never fill production PIN SLOTs.
// SHA-256 of ASCII abc: published SHA-256 known-answer vector.
inline constexpr std::string_view kAbcHash =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
inline constexpr std::array<Entry, 3> kCatalogue = {{
    {"fixture_main", "main.gguf", 3, kAbcHash, "fixture/huihui",
     "1111111111111111111111111111111111111111", "shards/main.gguf",
     Kind::MainShard, "qwen-community-1.0", "main.gguf",
     "fingerprints/main.json"},
    {"fixture_mtp", "mtp.gguf", 3, kAbcHash, "fixture/unsloth",
     "2222222222222222222222222222222222222222", "MTP/mtp.gguf", Kind::Mtp,
     "qwen-community-1.0", "mtp/mtp.gguf"},
    {"fixture_twin", "twin.gguf", 3, kAbcHash, "fixture/bubo",
     "3333333333333333333333333333333333333333", "twin/twin.gguf", Kind::Twin,
     "qwen-community-1.0", "twin/twin.gguf"},
}};
}  // namespace bubo::prepare::fixture
