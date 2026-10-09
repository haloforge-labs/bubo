#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/tokenizer.hpp"
#ifdef GUFO_BATCH_QUALITY_HIP
#include "src/models/qwen38_flash_next/engine.hpp"
#endif

namespace {
using gufo::json::Value;
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
std::string Read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Require(static_cast<bool>(in), "cannot open " + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
const Value& At(const Value& value, const std::string& key) {
  const auto* member = value.find(key);
  Require(member != nullptr, "missing manifest member: " + key);
  return *member;
}
std::vector<std::int32_t> Tokens(const Value& value) {
  std::vector<std::int32_t> out;
  for (const auto& t : value.items())
    out.push_back(static_cast<std::int32_t>(t.as_size()));
  return out;
}
const std::string Env(const char* key) {
  const char* v = std::getenv(key);
  return v ? v : "";
}
#ifdef GUFO_BATCH_QUALITY_HIP
void Dump(std::ofstream& compact, std::ofstream& full, std::size_t& offset,
          const Value& meta, std::span<const float> row, std::int32_t target) {
  Require(target >= 0 && static_cast<std::size_t>(target) < row.size(),
          "target outside vocabulary");
  for (auto v : row)
    Require(std::isfinite(v), "non-finite output logit");
  const double peak = *std::max_element(row.begin(), row.end());
  double sum = 0;
  for (auto v : row)
    sum += std::exp(static_cast<double>(v) - peak);
  const double lse = peak + std::log(sum);
  std::vector<std::size_t> ids(row.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto top = std::min<std::size_t>(512, row.size());
  std::partial_sort(ids.begin(), ids.begin() + top, ids.end(),
                    [&row](auto a, auto b) {
                      return row[a] > row[b] || (row[a] == row[b] && a < b);
                    });
  auto record = meta;
  record["lse"] = lse;
  record["target"] = target;
  record["target_logit"] = row[target];
  record["vocab"] = row.size();
  record["offset"] = offset;
  record["top"] = Value::array();
  for (std::size_t i = 0; i < top; ++i) {
    auto pair = Value::array();
    pair.push_back(ids[i]);
    pair.push_back(row[ids[i]]);
    record["top"].push_back(std::move(pair));
  }
  compact << record.dump() << '\n';
  full.write(reinterpret_cast<const char*>(row.data()), row.size_bytes());
  Require(compact.good() && full.good(), "quality dump write failed");
  offset += row.size_bytes();
}

int Run(const std::unordered_map<std::string, std::string>& args) {
  namespace qfn = gufo::models::qwen38_flash_next;
  const auto manifest = gufo::json::parse(Read(args.at("--manifest")));
  const auto c = std::stoul(args.at("--c"));
  const auto arm = args.at("--arm");
  Require(c == 1 || c == 2 || c == 4, "quality c must be 1, 2 or 4");
  Require(arm == "wmma" || arm == "narrow" || arm == "self" || arm == "q8",
          "unknown quality arm");
  const bool narrow = arm == "narrow" || arm == "self";
  Require(Env("GUFO_DENSE_TWIN_MAX_N").empty() &&
              Env("GUFO_DENSE_TWIN_FORCE").empty(),
          "quality runs forbid twin caps/forcing");
  if (arm == "q8") {
    Require(Env("GUFO_DENSE_TWIN_MODEL").empty(),
            "Q8 arm must not load a twin");
  } else {
    Require(!Env("GUFO_DENSE_TWIN_MODEL").empty(),
            "twin arm needs a twin model");
    if (!narrow && c > 1)
      Require(Env("GUFO_DENSE_TWIN_BATCH") == "1" &&
                  Env("GUFO_IQ4_WIDE_VARIANT") == "4",
              "WMMA requires batch=1 and variant=4");
    if (narrow || c == 1)
      Require(Env("GUFO_DENSE_TWIN_BATCH") != "1",
              "c1 narrow must disable batch");
  }
  qfn::ModelOptions options;
  options.max_context =
      static_cast<std::uint32_t>(At(manifest, "context").as_size());
  options.mtp_model_path = args.at("--mtp-model");
  options.max_draft_tokens = 7;
  options.decode_concurrency = static_cast<std::uint32_t>(narrow ? 1 : c);
  std::string error;
  auto model = qfn::Model::Load(args.at("--model"), options, &error);
  Require(model != nullptr, error);
  const auto& sessions = At(manifest, "sessions").items();
  std::vector<std::vector<std::int32_t>> tokens;
  for (const auto& s : sessions)
    tokens.push_back(Tokens(At(s, "tokens")));
  std::ofstream compact(args.at("--out") + ".jsonl", std::ios::binary);
  std::ofstream full(args.at("--out") + ".f32", std::ios::binary);
  Require(compact.good() && full.good(), "cannot open quality dumps");
  std::size_t offset = 0;
  const auto& groups = At(At(manifest, "schedules"), std::to_string(c)).items();
  for (const auto& group : groups) {
    std::vector<std::unique_ptr<qfn::Session>> active;
    std::vector<std::size_t> indices;
    for (const auto& index : At(group, "sessions").items()) {
      const auto i = index.as_size();
      Require(i < sessions.size(), "session outside manifest");
      indices.push_back(i);
      auto session = model->CreateSession(gufo::core::SessionMode::kSpeculative,
                                          options.max_context, &error);
      Require(session != nullptr, error);
      const auto start = At(sessions[i], "start").as_size();
      Require(start > 0 && start < tokens[i].size(), "bad prefill boundary");
      Require(session->Sync(std::span(tokens[i]).first(start), &error), error);
      active.push_back(std::move(session));
    }
    for (const auto& pass : At(group, "passes").items()) {
      std::vector<std::vector<float>> rows(active.size());
      std::vector<qfn::Session::TeacherForceRequest> requests;
      std::size_t total = 0;
      Require(pass.size() == active.size(), "pass c does not match group");
      for (std::size_t j = 0; j < active.size(); ++j) {
        const auto p = At(pass.items()[j], "position").as_size();
        const auto k = At(pass.items()[j], "width").as_size();
        const auto i = indices[j];
        Require(k >= 1 && k <= 8 && p == active[j]->Tokens().size() &&
                    p + k < tokens[i].size(),
                "bad forced position or width");
        requests.push_back(
            {active[j].get(), std::span(tokens[i]).subspan(p, k), &rows[j]});
        total += k;
      }
      if (narrow) {
        for (const auto& r : requests)
          Require(r.session->TeacherForce(r.tokens, r.rows, &error), error);
      } else {
        Require(qfn::Session::TeacherForceBatch(requests, &error), error);
      }
      for (std::size_t j = 0; j < active.size(); ++j) {
        const auto i = indices[j];
        const auto p = At(pass.items()[j], "position").as_size();
        const auto k = At(pass.items()[j], "width").as_size();
        auto meta = Value::object();
        meta["arm"] = arm;
        meta["c"] = c;
        meta["session"] = At(sessions[i], "name");
        meta["manifest_sha256"] = args.at("--manifest-sha256");
        meta["K"] = k;
        meta["N"] = total;
        for (std::size_t t = 0; t < k; ++t) {
          meta["position"] = p + t;
          Dump(compact, full, offset, meta,
               std::span(rows[j]).subspan(t * model->VocabSize(),
                                          model->VocabSize()),
               tokens[i][p + t + 1]);
        }
      }
    }
  }
  return 0;
}
#endif
}  // namespace

int main(int argc, char** argv) {
  try {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      Require(i + 1 < argc, "every argument needs a value");
      Require(args.emplace(argv[i], argv[i + 1]).second, "duplicate argument");
    }
    if (args.contains("--tokenize")) {
      std::string error;
      auto reader =
          gufo::core::GgufReader::OpenFile(args.at("--model"), &error);
      Require(reader != nullptr, error);
      auto tokenizer =
          gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
      Require(tokenizer != nullptr, error);
      auto sources = gufo::json::parse(Read(args.at("--tokenize")));
      auto tokenized = Value::array();
      for (auto s : At(sources, "sessions").items()) {
        auto ids = Value::array();
        for (auto id : tokenizer->Encode(s["text"].str()))
          ids.push_back(static_cast<std::size_t>(id));
        s["tokens"] = std::move(ids);
        s["text"] = Value{};
        tokenized.push_back(std::move(s));
      }
      sources["sessions"] = std::move(tokenized);
      std::ofstream out(args.at("--out"), std::ios::binary);
      out << sources.dump() << '\n';
      Require(out.good(), "token manifest write failed");
      return 0;
    }
#ifdef GUFO_BATCH_QUALITY_HIP
    return Run(args);
#else
    throw std::runtime_error(
        "engine quality execution requires ENGINE_ENABLE_HIP; --tokenize is "
        "CPU-only");
#endif
  } catch (const std::exception& e) {
    std::cerr << "INVALID: " << e.what() << '\n';
    return 2;
  }
}
