// Included after expert_iq_decode.hpp, inside expert_decode_bench's namespace.
// Survey opens only shard headers. Timing maps the split file and uploads one
// layer's real expert tensors at a time.

const Fmt& RealFormat(GgmlType type) {
  for (const Fmt* f :
       {&kQ4_K, &kQ5_K, &kQ5_1, &kQ8_0, &kIQ3_S, &kIQ3_XXS, &kIQ2_S, &kIQ2_XXS,
        &kIQ2_XS, &kIQ1_M, &kIQ4_XS, &kIQ4_NL, &kQ2_0})
    if (f->type == type)
      return *f;
  throw std::runtime_error("Unsupported real expert type " +
                           std::to_string(int(type)));
}

std::string RealName(int layer, const char* role) {
  return "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
}

struct RealLayer {
  const gufo::core::GgufTensorInfo* gate = nullptr;
  const gufo::core::GgufTensorInfo* up = nullptr;
  const gufo::core::GgufTensorInfo* down = nullptr;
};

void RealAssign(RealLayer& layer, const gufo::core::GgufTensorInfo& t,
                int index, const char* role) {
  const int rows = std::strcmp(role, "down") == 0 ? kHidden : kFF;
  const int k = std::strcmp(role, "down") == 0 ? kFF : kHidden;
  const auto& f = RealFormat(t.type);
  if (t.dimensions != std::vector<std::uint64_t>{std::uint64_t(k),
                                                 std::uint64_t(rows),
                                                 kExperts} ||
      k % f.block != 0 ||
      t.size_bytes != std::size_t(kExperts) * rows * RowBytes(f, k))
    throw std::runtime_error("Malformed expert tensor " +
                             RealName(index, role));
  auto*& target = std::strcmp(role, "gate") == 0 ? layer.gate
                  : std::strcmp(role, "up") == 0 ? layer.up
                                                 : layer.down;
  if (target)
    throw std::runtime_error("Duplicate expert tensor " +
                             RealName(index, role));
  target = &t;
}

std::array<RealLayer, 48> RealLayers(
    const std::vector<std::unique_ptr<gufo::core::GgufReader>>& readers) {
  std::array<RealLayer, 48> layers{};
  for (const auto& reader : readers)
    for (int layer = 0; layer < 48; ++layer)
      for (const char* role : {"gate", "up", "down"})
        if (const auto* t = reader->FindTensor(RealName(layer, role)))
          RealAssign(layers[layer], *t, layer, role);
  for (int layer = 0; layer < 48; ++layer) {
    const auto& l = layers[layer];
    if (!l.gate || !l.up || !l.down || l.gate->type != l.up->type)
      throw std::runtime_error("Missing or mismatched expert layer " +
                               std::to_string(layer));
  }
  return layers;
}

std::vector<std::unique_ptr<gufo::core::GgufReader>> RealHeaders(
    const std::filesystem::path& first) {
  std::string error;
  std::vector<std::unique_ptr<gufo::core::GgufReader>> readers;
  auto header = gufo::core::GgufReader::OpenHeaderFile(first, &error);
  if (!header)
    throw std::runtime_error("Cannot open first GGUF header: " + error);
  const unsigned count = header->GetMetadataUint32("split.count").value_or(1);
  if (count < 1 || count > 1024)
    throw std::runtime_error("Invalid split.count");
  readers.push_back(std::move(header));
  if (count == 1)
    return readers;
  const std::string name = first.filename().string();
  const auto split = name.rfind("-of-");
  const auto previous = split == std::string::npos ? std::string::npos
                                                   : name.rfind('-', split - 1);
  if (previous == std::string::npos || split == std::string::npos ||
      split <= previous + 1 || name.substr(split + 4).size() < 6 ||
      name.substr(split + 4).substr(name.substr(split + 4).size() - 5) !=
          ".gguf")
    throw std::runtime_error("Expected first split GGUF filename");
  const auto width = split - previous - 1;
  const auto prefix = name.substr(0, previous);
  const auto suffix = name.substr(split);
  if (readers.front()->GetMetadataUint32("split.no").value_or(9999) != 0 ||
      name.substr(previous + 1, width) != std::string(width - 1, '0') + "1")
    throw std::runtime_error("Survey requires the first GGUF shard");
  std::ostringstream expected_suffix;
  expected_suffix << "-of-" << std::setfill('0')
                  << std::setw(int(suffix.size() - 9)) << count << ".gguf";
  if (suffix != expected_suffix.str())
    throw std::runtime_error("GGUF shard count differs from filename");
  for (unsigned i = 2; i <= count; ++i) {
    std::ostringstream part;
    part << prefix << '-' << std::setfill('0') << std::setw(int(width)) << i
         << suffix;
    auto shard = gufo::core::GgufReader::OpenHeaderFile(
        first.parent_path() / part.str(), &error);
    if (!shard)
      throw std::runtime_error("Cannot open GGUF shard " + part.str() + ": " +
                               error);
    if (shard->GetMetadataUint32("split.count").value_or(0) != count ||
        shard->GetMetadataUint32("split.no").value_or(9999) != i - 1)
      throw std::runtime_error("Wrong shard metadata in " + part.str());
    readers.push_back(std::move(shard));
  }
  return readers;
}

int RealSurvey(const std::string& path) {
  const auto readers = RealHeaders(path);
  const auto layers = RealLayers(readers);
  std::map<std::tuple<std::string, std::string, int, int, std::size_t>, int>
      counts;
  for (int i = 0; i < 48; ++i)
    for (const auto& [role, t] : {std::pair{"gate", layers[i].gate},
                                  {"up", layers[i].up},
                                  {"down", layers[i].down}}) {
      const auto& f = RealFormat(t->type);
      std::printf(
          "SURVEY layer=%d role=%s fmt=%s K=%llu M=%llu experts=%llu "
          "bytes=%zu\n",
          i, role, f.name, static_cast<unsigned long long>(t->dimensions[0]),
          static_cast<unsigned long long>(t->dimensions[1]),
          static_cast<unsigned long long>(t->dimensions[2]), t->size_bytes);
      ++counts[{role, f.name, int(t->dimensions[0]), int(t->dimensions[1]),
                t->size_bytes}];
    }
  for (const auto& [key, count] : counts) {
    const auto& [role, fmt, k, m, bytes] = key;
    std::printf("SURVEY_COUNT role=%s fmt=%s K=%d M=%d bytes=%zu count=%d\n",
                role.c_str(), fmt.c_str(), k, m, bytes, count);
  }
  std::printf(
      "SURVEY_SUMMARY shards=%zu layers=48 tensors=144 types=%zu "
      "header_only=1\n",
      readers.size(), counts.size());
  return 0;
}

struct RealRoute {
  int layer, n;
  std::vector<std::int32_t> ids;
};

using RealRoutes = std::map<std::pair<int, int>, std::vector<RealRoute>>;

RealRoutes LoadRealRoutes(const std::string& path) {
  RealRoutes routes;
  if (path.empty())
    return routes;
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot open routing replay: " + path);
  std::string line;
  int number = 0;
  while (std::getline(file, line)) {
    ++number;
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream in(line);
    int layer, n;
    if (!(in >> layer >> n) || layer < 0 || layer >= 48 || n < 1 || n > 8)
      throw std::runtime_error("Invalid routing layer/width at line " +
                               std::to_string(number));
    RealRoute route{layer, n, {}};
    for (int token = 0; token < n; ++token) {
      for (int slot = 0; slot < kUsed; ++slot) {
        int id;
        if (!(in >> id) || id < -1 || id >= kExperts)
          throw std::runtime_error("Invalid routing id at line " +
                                   std::to_string(number));
        route.ids.push_back(id);
      }
    }
    std::string extra;
    if (in >> extra)
      throw std::runtime_error("Extra routing fields at line " +
                               std::to_string(number));
    routes[{layer, n}].push_back(std::move(route));
  }
  if (routes.empty())
    throw std::runtime_error("Empty routing replay");
  return routes;
}

void RealRouteStats(int layer, int n,
                    const std::vector<std::vector<std::int32_t>>& sets,
                    const char* source) {
  std::array<int, 9> histogram{};
  double distinct = 0;
  for (const auto& ids : sets) {
    if (ids.size() != std::size_t(n) * kUsed)
      throw std::runtime_error("Routing width differs from id count");
    std::array<int, kExperts> columns{};
    for (int t = 0; t < n; ++t) {
      std::set<int> active;
      for (int s = 0; s < kUsed; ++s)
        if (const int id = ids[t * kUsed + s]; id >= 0) {
          if (id >= kExperts)
            throw std::runtime_error("Routing expert exceeds model bounds");
          active.insert(id);
        }
      for (int id : active)
        ++columns[id];
    }
    for (int value : columns)
      if (value > 0) {
        if (value > n)
          throw std::runtime_error("Routing histogram exceeds token width");
        ++histogram[value];
        ++distinct;
      }
  }
  std::printf(
      "ROUTING layer=%d N=%d source=%s sets=%zu mean_distinct=%.3f "
      "columns_hist=",
      layer, n, source, sets.size(), distinct / sets.size());
  for (int i = 1; i <= n; ++i)
    std::printf("%s%d:%d", i == 1 ? "" : ",", i, histogram[i]);
  std::puts("");
}

int RealRoutingStats(const std::string& path) {
  const auto routes = LoadRealRoutes(path);
  std::size_t captures = 0;
  for (const auto& [key, records] : routes) {
    std::vector<std::vector<std::int32_t>> sets;
    for (const auto& record : records)
      sets.push_back(record.ids);
    RealRouteStats(key.first, key.second, sets, "real_replay");
    captures += records.size();
  }
  std::printf("ROUTING_SUMMARY source=real_replay groups=%zu captures=%zu\n",
              routes.size(), captures);
  return 0;
}

void RealTimedLayer(int layer, const RealLayer& tensors,
                    const RealRoutes& replay, int iters, int synthetic_sets) {
  const auto& gu = RealFormat(tensors.gate->type);
  const auto& dn = RealFormat(tensors.down->type);
  IqBuffer gate(tensors.gate->size_bytes), up(tensors.up->size_bytes),
      down(tensors.down->size_bytes);
  Check(hipMemcpy(gate.data, tensors.gate->data, tensors.gate->size_bytes,
                  hipMemcpyHostToDevice),
        "upload real gate");
  Check(hipMemcpy(up.data, tensors.up->data, tensors.up->size_bytes,
                  hipMemcpyHostToDevice),
        "upload real up");
  Check(hipMemcpy(down.data, tensors.down->data, tensors.down->size_bytes,
                  hipMemcpyHostToDevice),
        "upload real down");
  for (int n = 1; n <= 8; ++n) {
    std::vector<std::vector<std::int32_t>> host;
    const auto found = replay.find({layer, n});
    if (!replay.empty() && found == replay.end()) {
      std::printf(
          "ROUTING_SKIP layer=%d N=%d source=replay reason=no_capture\n", layer,
          n);
      continue;
    }
    if (found != replay.end())
      for (const auto& set : found->second)
        host.push_back(set.ids);
    else {
      auto synthetic = MakeIds(n, synthetic_sets, 0x91C41u + n + layer * 17);
      host = std::move(synthetic.host);
      (void)hipFree(synthetic.device);
    }
    const char* source =
        replay.empty() ? "independent_synthetic" : "real_replay";
    RealRouteStats(layer, n, host, source);
    const int sets = int(host.size());
    std::vector<std::int32_t> flat;
    std::vector<int> distinct;
    for (const auto& ids : host) {
      flat.insert(flat.end(), ids.begin(), ids.end());
      std::set<int> active;
      for (int id : ids)
        if (id >= 0)
          active.insert(id);
      distinct.push_back(int(active.size()));
    }
    IqBuffer device_ids(flat.size() * sizeof(std::int32_t));
    Check(hipMemcpy(device_ids.data, flat.data(),
                    flat.size() * sizeof(std::int32_t), hipMemcpyHostToDevice),
          "upload routing");
    const auto input = IqInputs(n, kHidden, 1.0F);
    IqBuffer x(input.size() * sizeof(float));
    Check(hipMemcpy(x.data, input.data(), input.size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "upload real input");
    const std::size_t gu_elements = std::size_t(n) * kUsed * kFF;
    const std::size_t dn_elements = std::size_t(n) * kUsed * kHidden;
    IqBuffer gated(gu_elements * sizeof(float)),
        out(dn_elements * sizeof(float));
    double stats_mean = 0;
    for (int count : distinct)
      stats_mean += count;
    stats_mean /= sets;
    const double mean = expert_bench::MeanDistinct(distinct, iters);
    const double gu_bytes =
        mean * (tensors.gate->size_bytes + tensors.up->size_bytes) / kExperts;
    const double dn_bytes = mean * tensors.down->size_bytes / kExperts;
    const double requested_gu_bytes =
        double(n * kUsed) *
        (tensors.gate->size_bytes + tensors.up->size_bytes) / kExperts;
    const double requested_dn_bytes =
        double(n * kUsed) * tensors.down->size_bytes / kExperts;
    std::printf(
        "REAL_BYTES layer=%d N=%d source=%s stats_mean_distinct=%.3f "
        "timed_mean_distinct=%.3f unique_gu=%.0f unique_down=%.0f "
        "requested_gu=%.0f requested_down=%.0f "
        "metric=logical_packed_bytes_not_dram_counter\n",
        layer, n, source, stats_mean, mean, gu_bytes, dn_bytes,
        requested_gu_bytes, requested_dn_bytes);
    const auto ids = [&](int i) {
      return device_ids.As<std::int32_t>() + std::size_t(i % sets) * n * kUsed;
    };
    IqBuffer up_out(gu_elements * sizeof(float));
    const bool gated_type = gu.type == GgmlType::kQ4_K ||
                            gu.type == GgmlType::kQ5_K ||
                            gu.type == GgmlType::kQ8_0;
    // The narrow HC mixer supplies ReadyQ8(x); shared expert projections
    // retain it across their own down input. Match GatedExperts' preq entry.
    IqBuffer ready(qfn_mmq_q8_1_bytes(n, kHidden));
    if (gated_type) {
      if (qfn_mmq_quantize_q8_1(x.As<float>(), ready.data, n, kHidden, nullptr))
        throw std::runtime_error("prepare gated ReadyQ8 failed");
      Check(hipDeviceSynchronize(), "prepare gated ReadyQ8 completion");
    }
    // Match Executor::ExpertV2kEnabled: unset is the grouped gated path.
    const char* v2k_setting = std::getenv("GUFO_EXPERT_V2K");
    const bool v2k = v2k_setting && std::strcmp(v2k_setting, "1") == 0 &&
                     gu.type == GgmlType::kQ4_K && n >= 2 &&
                     (dn.type == GgmlType::kQ5_1 || dn.type == GgmlType::kQ8_0);
    const auto scratch_bytes = qfn_mmq_expert_v2k_scratch_bytes(8);
    IqBuffer scratch(scratch_bytes);
    const auto gate_up = [&](int i) {
      if (gated_type)
        return qfn_mmq_moe_gated_vec_preq(
            int(gu.type), gate.data, up.data, ready.data, ids(i),
            gated.As<float>(), kFF, kHidden, n, kExperts, kUsed, nullptr);
      const int rc =
          qfn_mmq_moe_vec(int(gu.type), gate.data, x.As<float>(), ids(i),
                          gated.As<float>(), kFF, kHidden, n, kExperts, kUsed,
                          nullptr, up.data, up_out.As<float>());
      if (rc)
        return rc;
      gufo::models::qwen38_flash_next::rocm::Swiglu(
          gated.As<float>(), up_out.As<float>(), gu_elements, nullptr);
      return 0;
    };
    const auto pair_timing = TimeUs(
        [&](int i) {
          return qfn_mmq_moe_vec(int(gu.type), gate.data, x.As<float>(), ids(i),
                                 gated.As<float>(), kFF, kHidden, n, kExperts,
                                 kUsed, nullptr, up.data, up_out.As<float>());
        },
        iters);
    Report("real-pair|" + std::to_string(layer) + "|" + std::to_string(n),
           "real-pair", gu.name, source, n, mean, pair_timing, gu_bytes,
           "vec-pair");
    IqBuffer raw_gate(gu_elements * sizeof(float));
    Check(hipMemcpy(raw_gate.data, gated.data, gu_elements * sizeof(float),
                    hipMemcpyDeviceToDevice),
          "save raw gate");
    const double swiglu_bytes = 3.0 * gu_elements * sizeof(float);
    const auto swiglu_timing = TimeUs(
        [&](int) {
          gufo::models::qwen38_flash_next::rocm::Swiglu(
              gated.As<float>(), up_out.As<float>(), gu_elements, nullptr);
          return 0;
        },
        iters,
        [&] {
          Check(hipMemcpy(gated.data, raw_gate.data,
                          gu_elements * sizeof(float), hipMemcpyDeviceToDevice),
                "restore raw gate");
        });
    Report("real-swiglu|" + std::to_string(layer) + "|" + std::to_string(n),
           "real-swiglu", gu.name, source, n, mean, swiglu_timing, swiglu_bytes,
           "standalone");
    const auto gu_timing = TimeUs(gate_up, iters);
    Report("real-gu|" + std::to_string(layer) + "|" + std::to_string(n),
           "real-gu", gu.name, source, n, mean, gu_timing, gu_bytes,
           gated_type ? "gated" : "vec-pair+swiglu");
    const auto chain_timing = TimeUs(
        [&](int i) {
          if (v2k)
            return qfn_mmq_expert_v2k(
                int(gu.type), int(dn.type), gate.data, up.data, down.data,
                x.As<float>(), ids(i), gated.As<float>(), up_out.As<float>(),
                out.As<float>(), n, scratch.data, scratch_bytes, nullptr);
          const int rc = gate_up(i);
          return rc ? rc
                    : qfn_mmq_moe_vec(int(dn.type), down.data,
                                      gated.As<float>(), ids(i),
                                      out.As<float>(), kHidden, kFF, n * kUsed,
                                      kExperts, 1, nullptr);
        },
        iters);
    Report("real-chain|" + std::to_string(layer) + "|" + std::to_string(n),
           "real-chain", gu.name, source, n, mean, chain_timing,
           gu_bytes + dn_bytes,
           v2k          ? "v2k"
           : gated_type ? "gated+down"
                        : "vec-pair+swiglu+down");
    std::printf(
        "REAL_CELL layer=%d N=%d kind=chain us=%.6f valid=%d source=%s "
        "unique_bytes=%.0f requested_bytes=%.0f route=%s gate_input=%s\n",
        layer, n, chain_timing.us,
        int(expert_bench::Valid(gu_bytes + dn_bytes, chain_timing)), source,
        gu_bytes + dn_bytes, requested_gu_bytes + requested_dn_bytes,
        v2k          ? "v2k"
        : gated_type ? "gated+down"
                     : "vec-pair+swiglu+down",
        gated_type && !v2k ? "producer_ready_q8" : "raw_quantized_in_chain");
    IqBuffer down_input(gu_elements * sizeof(float));
    Check(hipMemcpy(down_input.data, IqInputs(n * kUsed, kFF, 1.0F).data(),
                    gu_elements * sizeof(float), hipMemcpyHostToDevice),
          "upload down input");
    const auto timing = TimeUs(
        [&](int i) {
          return qfn_mmq_moe_vec(
              int(dn.type), down.data, down_input.As<float>(), ids(i),
              out.As<float>(), kHidden, kFF, n * kUsed, kExperts, 1, nullptr);
        },
        iters);
    Report("real-down|" + std::to_string(layer) + "|" + std::to_string(n),
           "real-down", dn.name, source, n, mean, timing, dn_bytes);
  }
}

std::optional<int> TryRealBench(int argc, char** argv) {
  std::string mode, model, routing, unknown;
  bool all_layers = false, strict = false;
  int iters = 40, sets = 16;
  const bool real_selected = [&] {
    for (int i = 1; i < argc; ++i)
      if (std::strcmp(argv[i], "--survey") == 0 ||
          std::strcmp(argv[i], "--bench-real") == 0 ||
          std::strcmp(argv[i], "--routing-stats") == 0)
        return true;
    return false;
  }();
  if (!real_selected)
    return std::nullopt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--survey" || arg == "--bench-real" ||
        arg == "--routing-stats") {
      if (!mode.empty() || i + 1 == argc)
        throw std::runtime_error(
            "Exactly one real benchmark mode and file required");
      mode = arg;
      model = argv[++i];
    } else if (arg == "--all-layers") {
      all_layers = true;
    } else if (arg == "--strict-timing") {
      strict = true;
    } else if (i + 1 < argc && arg == "--routing") {
      routing = argv[++i];
    } else if (i + 1 < argc && arg == "--iters") {
      iters = std::stoi(argv[++i]);
    } else if (i + 1 < argc && arg == "--sets") {
      sets = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::puts(
          "Real experts: --survey FIRST-SHARD | --bench-real FIRST-SHARD "
          "[--routing FILE] [--all-layers] [--iters N] [--sets N] "
          "[--strict-timing] | --routing-stats FILE");
      return 0;
    } else {
      unknown = arg;
    }
  }
  if (mode.empty())
    return std::nullopt;
  if (!unknown.empty())
    throw std::runtime_error("Unknown real benchmark option: " + unknown);
  if (iters < 1 || sets < 1 ||
      (mode != "--bench-real" && (!routing.empty() || all_layers)))
    throw std::runtime_error("Invalid real benchmark options");
  if (mode == "--survey")
    return RealSurvey(model);
  if (mode == "--routing-stats")
    return RealRoutingStats(model);
  const auto replay = LoadRealRoutes(routing);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model, &error);
  if (!reader)
    throw std::runtime_error("Cannot map real GGUF: " + error);
  std::vector<std::unique_ptr<gufo::core::GgufReader>> readers;
  readers.push_back(std::move(reader));
  const auto layers = RealLayers(readers);
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init failed");
  std::set<std::pair<int, int>> seen;
  int selected = 0;
  for (int layer = 0; layer < 48; ++layer) {
    const auto& l = layers[layer];
    if (!replay.empty() &&
        std::none_of(replay.begin(), replay.end(), [&](const auto& item) {
          return item.first.first == layer;
        }))
      continue;
    const auto types = std::pair{int(l.gate->type), int(l.down->type)};
    if (!all_layers && !seen.insert(types).second)
      continue;
    std::printf(
        "REAL_COVERAGE layer=%d gate_up=%s down=%s policy=%s routing=%s\n",
        layer, RealFormat(l.gate->type).name, RealFormat(l.down->type).name,
        all_layers ? "all_layers" : "representative_type_pair",
        replay.empty() ? "independent_synthetic" : "real_replay");
    RealTimedLayer(layer, l, replay, iters, sets);
    ++selected;
  }
  std::printf(
      "REAL_SUMMARY selected_layers=%d total_layers=48 mode=%s routing=%s\n",
      selected, mode.c_str(),
      replay.empty() ? "independent_synthetic" : "real_replay");
  return TimingStatus(strict);
}
