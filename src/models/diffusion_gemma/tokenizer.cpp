#include "src/models/diffusion_gemma/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "src/core/json.hpp"

namespace gufo::models::diffusion_gemma {
namespace {

constexpr std::string_view kSpaceMark = "\xE2\x96\x81";  // U+2581

std::size_t Utf8Length(unsigned char lead) {
  if (lead < 0x80)
    return 1;
  if ((lead >> 5) == 0x6)
    return 2;
  if ((lead >> 4) == 0xE)
    return 3;
  if ((lead >> 3) == 0x1E)
    return 4;
  return 1;  // Invalid lead byte: one byte, resolved by byte fallback.
}

std::uint64_t PairKey(std::int32_t left, std::int32_t right) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32) |
         static_cast<std::uint32_t>(right);
}

}  // namespace

Tokenizer::Tokenizer(const std::filesystem::path& root) {
  const auto path = root / "tokenizer.json";
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const auto document = json::parse(buffer.str());
  const auto* model = document.find("model");
  if (model == nullptr || model->find("type") == nullptr ||
      model->find("type")->get_str() != "BPE")
    throw std::runtime_error("tokenizer.json: expected a BPE model");
  if (const auto* fallback = model->find("byte_fallback");
      fallback == nullptr || !fallback->as_bool())
    throw std::runtime_error("tokenizer.json: expected byte_fallback");

  const auto& vocab = model->find("vocab")->members();
  std::int32_t max_id = -1;
  for (const auto& [piece, id] : vocab)
    max_id = std::max(max_id, static_cast<std::int32_t>(id.as_size()));
  pieces_.resize(static_cast<std::size_t>(max_id) + 1);
  vocab_.reserve(vocab.size() * 2);
  for (const auto& [piece, id] : vocab) {
    const auto index = static_cast<std::int32_t>(id.as_size());
    pieces_[index] = piece;
    vocab_.emplace(piece, index);
  }

  byte_value_.assign(pieces_.size(), -1);
  for (int b = 0; b < 256; ++b) {
    char name[8];
    std::snprintf(name, sizeof(name), "<0x%02X>", b);
    const auto found = vocab_.find(name);
    if (found == vocab_.end())
      throw std::runtime_error("tokenizer.json: missing byte token");
    byte_ids_[b] = found->second;
    byte_value_[found->second] = static_cast<std::int16_t>(b);
  }

  const auto& merges = model->find("merges")->items();
  merges_.reserve(merges.size() * 2);
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    std::string left, right;
    if (merges[rank].is_array()) {
      left = merges[rank].items()[0].get_str();
      right = merges[rank].items()[1].get_str();
    } else {
      const auto& text = merges[rank].get_str();
      const auto space = text.find(' ');
      left = text.substr(0, space);
      right = text.substr(space + 1);
    }
    const auto l = vocab_.find(left), r = vocab_.find(right),
               m = vocab_.find(left + right);
    if (l == vocab_.end() || r == vocab_.end() || m == vocab_.end())
      continue;  // tokenizers skips merges whose result is absent too.
    merges_.emplace(PairKey(l->second, r->second),
                    (static_cast<std::uint64_t>(rank) << 32) |
                        static_cast<std::uint32_t>(m->second));
  }

  special_.assign(pieces_.size(), false);
  if (const auto* added = document.find("added_tokens")) {
    for (const auto& token : added->items()) {
      const auto content = token.find("content")->get_str();
      const auto id = static_cast<std::int32_t>(token.find("id")->as_size());
      if (static_cast<std::size_t>(id) >= pieces_.size()) {
        pieces_.resize(static_cast<std::size_t>(id) + 1);
        // special_ was sized from the vocabulary above; an added token beyond it must not set a bit past the end.
        special_.resize(pieces_.size(), false);
      }
      if (pieces_[id].empty())
        pieces_[id] = content;
      vocab_.emplace(content, id);
      added_.emplace_back(content, id);
      if (const auto* s = token.find("special"); s != nullptr && s->as_bool())
        special_[id] = true;
    }
  }
  byte_value_.resize(pieces_.size(), -1);
  special_.resize(pieces_.size(), false);
  // Longest first, so a scan takes the longest match at each position.
  std::sort(added_.begin(), added_.end(), [](const auto& a, const auto& b) {
    return a.first.size() > b.first.size();
  });
}

bool Tokenizer::IsSpecial(std::int32_t id) const {
  return id >= 0 && static_cast<std::size_t>(id) < special_.size() &&
         special_[id];
}

std::int32_t Tokenizer::TokenId(std::string_view token) const {
  const auto found = vocab_.find(std::string(token));
  return found == vocab_.end() ? -1 : found->second;
}

std::vector<std::int32_t> Tokenizer::Encode(std::string_view text) const {
  std::vector<std::int32_t> out;
  out.reserve(text.size() / 3 + 8);
  std::size_t start = 0, i = 0;
  while (i < text.size()) {
    bool matched = false;
    if (text[i] == '<') {
      for (const auto& [content, id] : added_) {
        if (text.compare(i, content.size(), content) == 0) {
          EncodeSegment(text.substr(start, i - start), &out);
          out.push_back(id);
          i += content.size();
          start = i;
          matched = true;
          break;
        }
      }
    }
    if (!matched)
      ++i;
  }
  EncodeSegment(text.substr(start), &out);
  return out;
}

void Tokenizer::EncodeSegment(std::string_view text,
                              std::vector<std::int32_t>* out) const {
  if (text.empty())
    return;
  std::string normal;
  normal.reserve(text.size() + text.size() / 4);
  for (const char c : text) {
    if (c == ' ')
      normal += kSpaceMark;
    else
      normal += c;
  }

  // Initial symbols: one per character, or its bytes when the character is
  // not in the vocabulary.
  std::vector<std::int32_t> ids;
  ids.reserve(normal.size());
  std::string scratch;
  for (std::size_t p = 0; p < normal.size();) {
    std::size_t n = Utf8Length(static_cast<unsigned char>(normal[p]));
    n = std::min(n, normal.size() - p);
    scratch.assign(normal, p, n);
    const auto found = vocab_.find(scratch);
    if (found != vocab_.end()) {
      ids.push_back(found->second);
    } else {
      for (std::size_t b = 0; b < n; ++b)
        ids.push_back(byte_ids_[static_cast<unsigned char>(normal[p + b])]);
    }
    p += n;
  }

  const std::size_t count = ids.size();
  std::vector<std::int32_t> prev(count), next(count);
  std::vector<bool> alive(count, true);
  for (std::size_t s = 0; s < count; ++s) {
    prev[s] = static_cast<std::int32_t>(s) - 1;
    next[s] = s + 1 < count ? static_cast<std::int32_t>(s + 1) : -1;
  }
  struct Merge {
    std::uint32_t rank;
    std::int32_t pos;
    std::int32_t merged;
    bool operator>(const Merge& o) const {
      return rank != o.rank ? rank > o.rank : pos > o.pos;
    }
  };
  std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>> queue;
  const auto push = [&](std::int32_t pos) {
    if (pos < 0 || next[pos] < 0)
      return;
    const auto found = merges_.find(PairKey(ids[pos], ids[next[pos]]));
    if (found != merges_.end())
      queue.push(Merge{static_cast<std::uint32_t>(found->second >> 32), pos,
                       static_cast<std::int32_t>(found->second & 0xffffffffU)});
  };
  for (std::size_t s = 0; s + 1 < count; ++s)
    push(static_cast<std::int32_t>(s));
  while (!queue.empty()) {
    const Merge top = queue.top();
    queue.pop();
    if (!alive[top.pos] || next[top.pos] < 0)
      continue;
    const auto right = next[top.pos];
    const auto found = merges_.find(PairKey(ids[top.pos], ids[right]));
    if (found == merges_.end() ||
        static_cast<std::uint32_t>(found->second >> 32) != top.rank ||
        static_cast<std::int32_t>(found->second & 0xffffffffU) != top.merged)
      continue;  // Stale entry: one side merged since it was queued.
    ids[top.pos] = top.merged;
    alive[right] = false;
    next[top.pos] = next[right];
    if (next[right] >= 0)
      prev[next[right]] = top.pos;
    push(prev[top.pos]);
    push(top.pos);
  }
  for (std::int32_t s = 0; s >= 0 && static_cast<std::size_t>(s) < count;
       s = next[s])
    out->push_back(ids[s]);
}

std::string Tokenizer::Decode(std::span<const std::int32_t> ids,
                              bool skip_special) const {
  std::string text;
  for (const auto id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size())
      continue;
    if (skip_special && special_[id])
      continue;
    if (byte_value_[id] >= 0) {
      text += static_cast<char>(byte_value_[id]);
      continue;
    }
    const auto& piece = pieces_[id];
    for (std::size_t p = 0; p < piece.size();) {
      if (piece.compare(p, kSpaceMark.size(), kSpaceMark) == 0) {
        text += ' ';
        p += kSpaceMark.size();
      } else {
        text += piece[p++];
      }
    }
  }
  return text;
}

}  // namespace gufo::models::diffusion_gemma
