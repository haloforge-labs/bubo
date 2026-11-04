// Included after the common expert_decode_bench helpers, inside its namespace.
namespace iq_bits = gufo::models::qwen38_flash_next::iq_decode;
using IqVariant = iq_bits::Variant;

const char* IqVariantName(IqVariant variant) {
  switch (variant) {
    case IqVariant::kBaseline:
      return "baseline";
    case IqVariant::kPacked2:
      return "packed2";
    case IqVariant::kPacked4:
      return "packed4";
    case IqVariant::kPacked8:
      return "packed8";
    case IqVariant::kLds4:
      return "lds4";
  }
  throw std::runtime_error("Invalid IQ variant");
}

std::vector<IqVariant> IqVariants() {
  const char* setting = std::getenv("GUFO_IQ_BENCH_VARIANTS");
  const std::string value = setting && *setting ? setting : "all";
  const std::array<IqVariant, 5> all{IqVariant::kBaseline, IqVariant::kPacked2,
                                     IqVariant::kPacked4, IqVariant::kPacked8,
                                     IqVariant::kLds4};
  if (value == "all")
    return {all.begin(), all.end()};
  std::vector<IqVariant> variants{IqVariant::kBaseline};
  std::istringstream stream(value);
  std::string name;
  while (std::getline(stream, name, ',')) {
    const auto found = std::find_if(all.begin(), all.end(), [&](IqVariant v) {
      return name == IqVariantName(v);
    });
    if (found == all.end())
      throw std::runtime_error("Unknown GUFO_IQ_BENCH_VARIANTS value: " + name);
    if (std::find(variants.begin(), variants.end(), *found) == variants.end())
      variants.push_back(*found);
  }
  return variants;
}

struct IqBuffer {
  void* data = nullptr;
  explicit IqBuffer(std::size_t bytes) {
    Check(hipMalloc(&data, bytes), "IQ exact allocation");
  }
  IqBuffer(const IqBuffer&) = delete;
  IqBuffer& operator=(const IqBuffer&) = delete;
  ~IqBuffer() { (void)hipFree(data); }
  template<typename T>
  T* As() const {
    return static_cast<T*>(data);
  }
};

std::uint32_t IqFloatBits(float value) {
  std::uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

bool IqTarget(const Fmt& format) {
  return format.type == GgmlType::kIQ2_S || format.type == GgmlType::kQ2_0 ||
         format.type == GgmlType::kIQ2_XXS ||
         format.type == GgmlType::kIQ2_XS || format.type == GgmlType::kIQ1_M ||
         format.type == GgmlType::kIQ3_XXS || format.type == GgmlType::kIQ3_S;
}

bool IqSupported(IqVariant variant, const Fmt& format) {
  if (!IqTarget(format))
    return variant == IqVariant::kBaseline;
  return iq_bits::Supported(variant, format.type == GgmlType::kIQ2_S);
}

const char* IqCallName(int variant) {
  return variant == -2 ? "matrix"
         : variant < 0 ? "production"
                       : IqVariantName(IqVariant(variant));
}

int IqMatrix(const Fmt& format, const void* w, const float* x,
             const std::int32_t* ids, float* out, int rows, int k, int tokens,
             int experts, int used) {
  switch (format.type) {
    case GgmlType::kIQ2_XXS:
      return qfn_mmq_iq2_xxs_moe_raw(w, x, ids, out, rows, k, tokens, experts,
                                     used, nullptr);
    case GgmlType::kIQ2_XS:
      return qfn_mmq_iq2_xs_moe_raw(w, x, ids, out, rows, k, tokens, experts,
                                    used, nullptr);
    case GgmlType::kIQ1_M:
      return qfn_mmq_iq1_m_moe_raw(w, x, ids, out, rows, k, tokens, experts,
                                   used, nullptr);
    default:
      throw std::runtime_error("No matrix oracle route for this format");
  }
}

int IqCall(const Fmt& format, const void* w, const void* up, const float* x,
           const std::int32_t* ids, float* out, float* out_up, int rows, int k,
           int tokens, int experts, int used, int variant) {
  if (variant == -2) {
    const int rc =
        IqMatrix(format, w, x, ids, out, rows, k, tokens, experts, used);
    return rc || !up ? rc
                     : IqMatrix(format, up, x, ids, out_up, rows, k, tokens,
                                experts, used);
  }
  if (variant < 0 || !IqTarget(format))
    return qfn_mmq_moe_vec(int(format.type), w, x, ids, out, rows, k, tokens,
                           experts, used, nullptr, up, out_up);
  return qfn_mmq_moe_vec_iq_bench(int(format.type), w, x, ids, out, rows, k,
                                  tokens, experts, used, nullptr, up, out_up,
                                  variant);
}

std::vector<std::int32_t> IqRoutes(int tokens, int used, int experts,
                                   int pattern) {
  std::vector<std::int32_t> ids(std::size_t(tokens) * used);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = pattern == 1 ? int(i % used) % experts : int(i % experts);
    if (pattern == 3 || (pattern == 2 && i % 3 == 0))
      ids[i] = -1;
  }
  if (pattern == 0 && ids.size() < std::size_t(experts)) {
    ids.front() = 0;
    if (ids.size() > 2)
      ids[ids.size() / 2] = experts / 2;
    ids.back() = experts - 1;
  }
  return ids;
}

std::vector<float> IqInputs(int tokens, int k, float scale) {
  std::vector<float> values(std::size_t(tokens) * k);
  std::uint32_t state = 0x3D13119B;
  for (auto& value : values)
    value = scale * float(int(NextRandom(&state) % 2001) - 1000) / 997.0F;
  return values;
}

std::vector<std::uint8_t> IqWeights(const Fmt& format, int experts, int rows,
                                    int k, std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(std::size_t(experts) * rows *
                                  RowBytes(format, k));
  for (auto& value : bytes)
    value = std::uint8_t(NextRandom(&seed));
  constexpr std::uint16_t scales[]{0,      1,      0x8001, 0x1C00, 0x3C00,
                                   0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0x7E01};
  for (std::size_t block = 0; block < bytes.size() / format.bytes; ++block) {
    const auto scale =
        scales[block % (rows == 1 && k == int(format.block) ? 10 : 8)];
    if (format.type == GgmlType::kIQ1_M) {
      std::uint16_t sc[4];
      std::memcpy(sc, bytes.data() + block * format.bytes + 48, sizeof(sc));
      for (int i = 0; i < 4; ++i)
        sc[i] = (sc[i] & 0x0FFF) | (((scale >> (4 * i)) & 15) << 12);
      std::memcpy(bytes.data() + block * format.bytes + 48, sc, sizeof(sc));
    } else {
      std::memcpy(bytes.data() + block * format.bytes, &scale, sizeof(scale));
    }
  }
  return bytes;
}

constexpr std::size_t kIqGuardBytes = 128;
std::uint64_t iq_comparisons = 0;
std::uint64_t iq_nonzero = 0;

void IqReset(IqBuffer& output, std::size_t bytes) {
  Check(hipMemset(output.data, 0xFF, bytes), "poison IQ output");
  Check(hipMemset(output.As<char>() + bytes, 0xA5, kIqGuardBytes),
        "IQ output guard");
}

std::vector<float> IqRead(const IqBuffer& output, std::size_t elements,
                          const std::vector<std::int32_t>& ids, int rows) {
  std::vector<float> values(elements + kIqGuardBytes / sizeof(float));
  Check(hipMemcpy(values.data(), output.data, values.size() * sizeof(float),
                  hipMemcpyDeviceToHost),
        "read IQ output");
  for (std::size_t i = 0; i < elements; ++i) {
    if (!std::isfinite(values[i]))
      throw std::runtime_error("Nonfinite or unwritten IQ output at " +
                               std::to_string(i));
    if (ids[i / rows] < 0 && IqFloatBits(values[i]) != 0)
      throw std::runtime_error("Skipped expert did not write positive zero");
    iq_nonzero += values[i] != 0.0F;
  }
  const auto* bytes =
      reinterpret_cast<const std::uint8_t*>(values.data() + elements);
  if (!std::all_of(bytes, bytes + kIqGuardBytes,
                   [](std::uint8_t b) { return b == 0xA5; }))
    throw std::runtime_error("IQ output guard changed");
  values.resize(elements);
  return values;
}

std::string IqHash(const std::vector<float>& values) {
  return gufo::crypto::Sha256Hex(
      {reinterpret_cast<const std::uint8_t*>(values.data()),
       values.size() * sizeof(float)});
}

void IqCheckCase(const char* kind, const char* tensor, const Fmt& format,
                 const void* w, const void* up, int rows, int k, int n,
                 int experts, bool down, int pattern, float scale,
                 const std::vector<IqVariant>& variants,
                 const std::vector<float>* exact_input = nullptr,
                 const std::vector<std::vector<float>>* reference = nullptr) {
  const int tokens = down ? n * kUsed : n;
  const int used = down ? 1 : kUsed;
  const auto ids = IqRoutes(tokens, used, experts, pattern);
  const auto input = exact_input ? *exact_input : IqInputs(tokens, k, scale);
  IqBuffer device_ids(ids.size() * sizeof(std::int32_t));
  IqBuffer device_input(input.size() * sizeof(float));
  Check(hipMemcpy(device_ids.data, ids.data(),
                  ids.size() * sizeof(std::int32_t), hipMemcpyHostToDevice),
        "upload IQ check ids");
  Check(hipMemcpy(device_input.data, input.data(), input.size() * sizeof(float),
                  hipMemcpyHostToDevice),
        "upload IQ check input");
  const std::size_t elements = std::size_t(tokens) * used * rows;
  const std::size_t bytes = elements * sizeof(float);
  IqBuffer output(bytes + kIqGuardBytes);
  IqBuffer output_up(bytes + kIqGuardBytes);
  const auto execute = [&](int variant) {
    IqReset(output, bytes);
    if (up)
      IqReset(output_up, bytes);
    if (IqCall(format, w, up, device_input.As<float>(),
               device_ids.As<std::int32_t>(), output.As<float>(),
               up ? output_up.As<float>() : nullptr, rows, k, tokens, experts,
               used, variant) != 0)
      throw std::runtime_error("IQ identity launch failed");
    Check(hipDeviceSynchronize(), "IQ identity synchronize");
    std::vector<std::vector<float>> result;
    result.push_back(IqRead(output, elements, ids, rows));
    if (up)
      result.push_back(IqRead(output_up, elements, ids, rows));
    return result;
  };
  const auto baseline = execute(int(IqVariant::kBaseline));
  if (reference && (reference->size() != baseline.size()))
    throw std::runtime_error("CPU oracle projection count differs");
  if (reference)
    for (std::size_t p = 0; p < baseline.size(); ++p)
      if ((*reference)[p].size() != elements ||
          std::memcmp((*reference)[p].data(), baseline[p].data(), bytes) != 0)
        throw std::runtime_error(
            "Vector baseline differs from independent exact CPU dequant/dot "
            "oracle");
  for (std::size_t p = 0; p < baseline.size(); ++p)
    std::printf(
        "CHECK kind=%s tensor=%s fmt=%s n=%d pattern=%d scale=%.9g "
        "projection=%zu variant=baseline sha256=%s\n",
        kind, tensor, format.name, n, pattern, scale, p,
        IqHash(baseline[p]).c_str());
  std::vector<int> selected{-1};
  if (reference)
    selected.push_back(-2);
  if (IqTarget(format)) {
    for (const auto variant : variants)
      if (variant != IqVariant::kBaseline && IqSupported(variant, format))
        selected.push_back(int(variant));
  }
  for (int variant : selected) {
    const auto actual = execute(variant);
    for (std::size_t p = 0; p < baseline.size(); ++p) {
      if (std::memcmp(baseline[p].data(), actual[p].data(), bytes) != 0) {
        std::size_t first = 0;
        while (IqFloatBits(baseline[p][first]) == IqFloatBits(actual[p][first]))
          ++first;
        std::fprintf(
            stderr,
            "IDENTITY_FAIL tensor=%s fmt=%s n=%d pattern=%d projection=%zu "
            "variant=%s first=%zu old=%08x new=%08x\n",
            tensor, format.name, n, pattern, p, IqCallName(variant), first,
            IqFloatBits(baseline[p][first]), IqFloatBits(actual[p][first]));
        throw std::runtime_error("IQ bit identity failed");
      }
      ++iq_comparisons;
      std::printf(
          "CHECK kind=%s tensor=%s fmt=%s n=%d pattern=%d scale=%.9g "
          "projection=%zu variant=%s elements=%zu sha256=%s changed_bits=0\n",
          kind, tensor, format.name, n, pattern, scale, p, IqCallName(variant),
          elements, IqHash(actual[p]).c_str());
    }
  }
  std::fflush(stdout);
}

void IqSynthetic(const std::vector<IqVariant>& variants,
                 const std::vector<int>& widths) {
  for (const Fmt* format :
       {&kIQ2_S, &kQ2_0, &kIQ2_XXS, &kIQ2_XS, &kIQ1_M, &kIQ3_XXS, &kIQ3_S}) {
    const auto nonzero_before = iq_nonzero;
    for (const int k : {int(format->block), int(format->block * 19)}) {
      for (int rows : {1, 3, 17}) {
        const auto weights = IqWeights(*format, 13, rows, k, 0xA18B11);
        const auto up = IqWeights(*format, 13, rows, k, 0x198B11);
        IqBuffer device_w(weights.size());
        IqBuffer device_up(up.size());
        Check(hipMemcpy(device_w.data, weights.data(), weights.size(),
                        hipMemcpyHostToDevice),
              "IQ synthetic weights");
        Check(hipMemcpy(device_up.data, up.data(), up.size(),
                        hipMemcpyHostToDevice),
              "IQ synthetic up");
        for (int n : widths) {
          for (int pattern = 0; pattern < 4; ++pattern) {
            for (bool down : {false, true}) {
              IqCheckCase("synthetic", down ? "slot-down" : "gate-up-pair",
                          *format, device_w.data,
                          down ? nullptr : device_up.data, rows, k, n, 13, down,
                          pattern, pattern == 0 ? 0.00003F : 1.0F, variants);
            }
          }
        }
      }
    }
    if (iq_nonzero == nonzero_before)
      throw std::runtime_error(
          std::string(format->name) +
          " synthetic fixtures produced no nonzero outputs");
  }
  std::printf(
      "CHECK_SUMMARY synthetic_comparisons=%llu nonzero_words=%llu PASS\n",
      static_cast<unsigned long long>(iq_comparisons),
      static_cast<unsigned long long>(iq_nonzero));
}

#include "expert_iq_cpu_oracle.hpp"

const Fmt& IqRealFormat(GgmlType type) {
  for (const Fmt* format : {&kIQ2_S, &kQ2_0, &kIQ3_XXS, &kIQ3_S, &kIQ4_XS,
                            &kIQ4_NL, &kIQ2_XXS, &kIQ2_XS, &kIQ1_M})
    if (format->type == type)
      return *format;
  throw std::runtime_error("Hybrid tensor has an unexpected expert type");
}

std::map<GgmlType, int> IqExpected(const std::string& layout) {
  if (layout == "hybrid")
    return {{GgmlType::kIQ2_S, 40},   {GgmlType::kQ2_0, 9},
            {GgmlType::kIQ3_XXS, 34}, {GgmlType::kIQ3_S, 20},
            {GgmlType::kIQ4_XS, 2},   {GgmlType::kIQ4_NL, 39}};
  if (layout == "iq3-xxs")
    return {{GgmlType::kIQ2_XS, 20},  {GgmlType::kIQ2_XXS, 18},
            {GgmlType::kIQ2_S, 20},   {GgmlType::kIQ3_S, 26},
            {GgmlType::kIQ3_XXS, 12}, {GgmlType::kQ2_0, 30},
            {GgmlType::kIQ4_NL, 18}};
  if (layout == "iq2-xs")
    return {{GgmlType::kIQ1_M, 6},
            {GgmlType::kIQ2_S, 68},
            {GgmlType::kIQ2_XXS, 22},
            {GgmlType::kQ2_0, 48}};
  throw std::runtime_error("Unknown expert layout: " + layout);
}

void IqReal(const std::string& model, const std::string& layout,
            bool expert_shard, const std::vector<IqVariant>& variants,
            const std::vector<int>& widths) {
  const auto comparisons_before = iq_comparisons;
  const auto nonzero_before = iq_nonzero;
  std::string error;
  const auto reader =
      expert_shard ? gufo::core::GgufReader::OpenHeaderFile(model, &error)
                   : gufo::core::GgufReader::OpenFile(model, &error);
  if (!reader)
    throw std::runtime_error("Cannot open expert GGUF: " + error);
  const auto expected = IqExpected(layout);
  struct RealTensor {
    std::string name;
    const Fmt* format;
    const void* data;
    std::size_t bytes;
    int rows, k;
    bool down;
  };
  std::vector<RealTensor> targets;
  std::map<GgmlType, int> counts;
  int tensors = 0;
  for (int layer = 0; layer < 48; ++layer) {
    for (const char* role : {"gate", "up", "down"}) {
      const std::string name =
          "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
      const auto* tensor = reader->FindTensor(name);
      const bool down = std::strcmp(role, "down") == 0;
      const int rows = down ? kHidden : kFF;
      const int k = down ? kFF : kHidden;
      if (!tensor || tensor->dimensions !=
                         std::vector<std::uint64_t>{
                             std::uint64_t(k), std::uint64_t(rows), kExperts})
        throw std::runtime_error("Missing or malformed hybrid tensor: " + name);
      const auto& format = IqRealFormat(tensor->type);
      if (tensor->size_bytes !=
          std::size_t(kExperts) * rows * RowBytes(format, k))
        throw std::runtime_error("Hybrid tensor byte size differs: " + name);
      ++counts[tensor->type];
      ++tensors;
      if (IqTarget(format))
        targets.push_back(
            {name, &format, tensor->data, tensor->size_bytes, rows, k, down});
    }
  }
  if (counts != expected || tensors != 144 || targets.empty())
    throw std::runtime_error(
        "Real expert coverage differs from selected layout");
  for (const auto& tensor : targets) {
    const auto tensor_nonzero_before = iq_nonzero;
    IqBuffer device_w(tensor.bytes);
    Check(hipMemcpy(device_w.data, tensor.data, tensor.bytes,
                    hipMemcpyHostToDevice),
          "upload one real expert tensor");
    for (int n : widths) {
      for (int pattern = 0; pattern < 4; ++pattern)
        IqCheckCase("real", tensor.name.c_str(), *tensor.format, device_w.data,
                    nullptr, tensor.rows, tensor.k, n, kExperts, tensor.down,
                    pattern, pattern == 0 ? 0.00003F : 1.0F, variants);
    }
    if (iq_nonzero == tensor_nonzero_before)
      throw std::runtime_error("Real expert fixture is all zero: " +
                               tensor.name);
  }
  if (iq_nonzero == nonzero_before)
    throw std::runtime_error(
        "Real expert fixtures produced no nonzero outputs");
  std::printf(
      "IDENTITY_SUMMARY layout=%s layers=48 counted_tensors=%d "
      "checked_tensors=%zu "
      "IQ2_S=%d Q2_0=%d IQ2_XXS=%d IQ2_XS=%d IQ1_M=%d IQ3_XXS=%d IQ3_S=%d "
      "IQ4_XS=%d IQ4_NL=%d comparisons=%llu "
      "nonzero_words=%llu PASS\n",
      layout.c_str(), tensors, targets.size(), counts[GgmlType::kIQ2_S],
      counts[GgmlType::kQ2_0], counts[GgmlType::kIQ2_XXS],
      counts[GgmlType::kIQ2_XS], counts[GgmlType::kIQ1_M],
      counts[GgmlType::kIQ3_XXS], counts[GgmlType::kIQ3_S],
      counts[GgmlType::kIQ4_XS], counts[GgmlType::kIQ4_NL],
      static_cast<unsigned long long>(iq_comparisons - comparisons_before),
      static_cast<unsigned long long>(iq_nonzero - nonzero_before));
}

void IqTiming(const Fmt& format, int iters, int sets,
              const std::vector<IqVariant>& variants,
              const std::vector<int>& widths) {
  const bool down =
      format.type == GgmlType::kQ2_0 || format.type == GgmlType::kIQ4_NL;
  const int rows = down ? kHidden : kFF;
  const int k = down ? kFF : kHidden;
  const auto weight_bytes = std::size_t(kExperts) * rows * RowBytes(format, k);
  void* w = RandomDevice(weight_bytes, 0x1234U);
  void* up = down ? nullptr : RandomDevice(weight_bytes, 0x4321U);
  for (int n : widths) {
    IdSets ids = MakeIds(n, sets, 0x9E3779B9U + n);
    const int tokens = down ? kUsed * n : n;
    const int used = down ? 1 : kUsed;
    float* input = DeviceFloats(std::size_t(tokens) * k, 1.0F, 77U);
    float* output = DeviceFloats(std::size_t(tokens) * used * rows, 0.0F, 1U);
    float* output_up =
        down ? nullptr
             : DeviceFloats(std::size_t(tokens) * used * rows, 0.0F, 2U);
    const double distinct = expert_bench::MeanDistinct(ids.distinct, iters);
    const double moved =
        distinct * (down ? 1.0 : 2.0) * rows * RowBytes(format, k);
    std::optional<expert_bench::Timing> reference_timing;
    std::vector<IqVariant> supported;
    for (IqVariant variant : variants)
      if (IqSupported(variant, format))
        supported.push_back(variant);
    expert_bench::Sweep(
        supported,
        [&](IqVariant variant) {
          const auto call = [&](int i) {
            if (format.type == GgmlType::kQ4_K)
              return qfn_mmq_moe_gated_vec(
                  int(format.type), w, up, input,
                  ids.device + std::size_t(i % sets) * n * kUsed, output, rows,
                  k, tokens, kExperts, used, nullptr);
            return IqCall(format, w, up, input,
                          ids.device + std::size_t(i % sets) * n * kUsed,
                          output, output_up, rows, k, tokens, kExperts, used,
                          int(variant));
          };
          return TimeUs(call, iters);
        },
        [&](IqVariant variant, const expert_bench::Timing& timing) {
          if (reference_timing)
            expert_bench::RequireComparable(*reference_timing, timing);
          else
            reference_timing = timing;
          const char* route = down ? "vec" : "vec-pair";
          if (format.type == GgmlType::kQ4_K)
            route = "gated";
          Report(std::string("iq|") + format.name + "|" +
                     IqVariantName(variant) + "|" + std::to_string(n),
                 down ? "down" : "gu", format.name, route, n, distinct, timing,
                 moved, IqVariantName(variant));
        });
    (void)hipFree(input);
    (void)hipFree(output);
    (void)hipFree(output_up);
    (void)hipFree(ids.device);
  }
  (void)hipFree(w);
  (void)hipFree(up);
}

std::vector<int> IqWidths(const std::string& text) {
  std::istringstream stream(text);
  std::string field;
  std::vector<int> widths;
  while (std::getline(stream, field, ',')) {
    std::size_t consumed = 0;
    const int n = std::stoi(field, &consumed);
    if (consumed != field.size() || n < 1 || n > 64)
      throw std::runtime_error("IQ widths must be integers in 1..64");
    if (std::find(widths.begin(), widths.end(), n) == widths.end())
      widths.push_back(n);
  }
  if (widths.empty())
    throw std::runtime_error("IQ widths cannot be empty");
  return widths;
}

std::optional<int> TryIqBench(int argc, char** argv) {
  bool selected = false;
  if (const char* value = std::getenv("GUFO_IQ_BENCH_VARIANTS"))
    selected = *value != '\0';
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    selected |= arg == "--identity" || arg == "--iq-only";
    if (arg == "--help") {
      std::puts(
          "expert_decode_bench [--iters N] [--sets K]\n"
          "IQ qualification: --iq-only | --identity --model experts.gguf\n"
          "[--layout hybrid|iq3-xxs|iq2-xs] [--expert-shard]\n"
          "[--widths 1,2,3,4,5,6,7,8] [--strict-timing] [--csv path]\n"
          "Timings evict 256 MiB before each separately timed call; GBps >250 "
          "is INVALID and --strict-timing exits nonzero.\n"
          "GUFO_IQ_BENCH_VARIANTS=all|baseline,packed2,packed4,packed8,lds4\n"
          "lds4 requires IQ2_S. Every IQ run compares the production default "
          "too.");
      return 0;
    }
  }
  if (!selected)
    return std::nullopt;
  bool identity = false;
  bool strict_timing = false;
  bool expert_shard = false;
  int iters = 40;
  int sets = 16;
  std::string model;
  std::string csv;
  std::string layout = "hybrid";
  std::vector<int> widths{1, 2, 3, 4, 5, 6, 7, 8};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--identity") {
      identity = true;
    } else if (arg == "--strict-timing") {
      strict_timing = true;
    } else if (arg == "--iq-only") {
      continue;
    } else if (arg == "--expert-shard") {
      expert_shard = true;
    } else if (i + 1 < argc &&
               (arg == "--iters" || arg == "--sets" || arg == "--widths" ||
                arg == "--model" || arg == "--layout" || arg == "--csv")) {
      const std::string value = argv[++i];
      if (arg == "--iters")
        iters = std::stoi(value);
      else if (arg == "--sets")
        sets = std::stoi(value);
      else if (arg == "--widths")
        widths = IqWidths(value);
      else if (arg == "--layout")
        layout = value;
      else if (arg == "--csv")
        csv = value;
      else
        model = value;
    } else {
      throw std::runtime_error("Unknown or incomplete IQ bench option: " + arg);
    }
  }
  if (iters < 1 || sets < 1 || (identity && model.empty()) ||
      (!identity && (!model.empty() || expert_shard || layout != "hybrid")))
    throw std::runtime_error(
        "Invalid IQ bench counts or model/identity combination");
  const auto variants = IqVariants();
  if (!csv.empty())
    OpenTimingCsv(csv);
  (void)IqExpected(layout);
  if (qfn_mmq_init(0) != 0)
    throw std::runtime_error("qfn_mmq_init failed");
  IqExactCpuOracle(variants, widths);
  if (identity) {
    IqReal(model, layout, expert_shard, variants, widths);
  } else {
    IqSynthetic(variants, widths);
    IqTiming(kIQ2_S, iters, sets, variants, widths);
    IqTiming(kQ2_0, iters, sets, variants, widths);
    IqTiming(kIQ2_XXS, iters, sets, variants, widths);
    IqTiming(kIQ2_XS, iters, sets, variants, widths);
    IqTiming(kIQ1_M, iters, sets, variants, widths);
    IqTiming(kIQ3_XXS, iters, sets, variants, widths);
    IqTiming(kIQ3_S, iters, sets, variants, widths);
    IqTiming(kIQ4_XS, iters, sets, variants, widths);
    IqTiming(kIQ4_NL, iters, sets, variants, widths);
    IqTiming(kQ4_K, iters, sets, variants, widths);
  }
  return identity ? 0 : TimingStatus(strict_timing);
}
