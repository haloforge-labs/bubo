#pragma once

// Included in dense_gemv_bench's private namespace. Reuse its buffer, GGUF,
// PRNG and launch/check helpers; this is not a second benchmark executable.
struct DenseError {
  double max_abs = 0, max_relative = 0;
  double absolute_tolerance;
  std::size_t outside_tolerance = 0, byte_differences = 0;

  explicit DenseError(double absolute = 0.002) : absolute_tolerance(absolute) {}

  void Add(double expected, float actual) {
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      ++outside_tolerance;
      max_abs = max_relative = std::numeric_limits<double>::infinity();
      return;
    }
    const double delta = std::abs(double(actual) - expected);
    max_abs = std::max(max_abs, delta);
    max_relative =
        std::max(max_relative, delta / std::max(std::abs(expected), 1e-12));
    outside_tolerance +=
        delta > absolute_tolerance + 0.0002 * std::abs(expected);
    byte_differences += std::bit_cast<std::uint32_t>(float(expected)) !=
                        std::bit_cast<std::uint32_t>(actual);
  }
};

void CompareHcBf16(const std::string& path) {
  namespace r = gufo::models::qwen38_flash_next::rocm;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(bool(reader), error.c_str());
  std::size_t checked = 0;
  for (const auto& tensor : reader->GetTensors()) {
    if (int(tensor.type) != 30 || tensor.dimensions.size() != 2 ||
        tensor.name.find("hc_") == std::string::npos)
      continue;
    const int k = int(tensor.dimensions[0]), m = int(tensor.dimensions[1]);
    if (!gufo::models::qwen38_flash_next::hc_bf16_dense::Shape(2, m, k))
      continue;
    Buffer w(tensor.size_bytes);
    Check(
        hipMemcpy(w.p, tensor.data, tensor.size_bytes, hipMemcpyHostToDevice));
    const auto* packed = static_cast<const std::uint16_t*>(tensor.data);
    for (float amplitude : {1.0F, 0.001F}) {
      std::uint32_t seed = 38127;
      std::vector<float> host_x(std::size_t(8) * k),
          width_one(std::size_t(8) * m);
      for (float& value : host_x)
        value = float(int(Next(seed) % 2001) - 1000) / 997.0F * amplitude;
      // Each reference row is a genuinely separate N1 call on the same input.
      Buffer scalar_x(std::size_t(k) * sizeof(float));
      Buffer scalar_y(std::size_t(m) * sizeof(float));
      for (int t = 0; t < 8; ++t) {
        Check(hipMemcpy(scalar_x.p, host_x.data() + std::size_t(t) * k,
                        std::size_t(k) * sizeof(float), hipMemcpyHostToDevice));
        r::SmallGemmHcBf16Variant(w.p, static_cast<const float*>(scalar_x.p),
                                  static_cast<float*>(scalar_y.p), 1, m, k,
                                  true, nullptr);
        Check(hipGetLastError());
        Check(hipMemcpy(width_one.data() + std::size_t(t) * m, scalar_y.p,
                        std::size_t(m) * sizeof(float), hipMemcpyDeviceToHost));
      }
      for (int n = 1; n <= 8; ++n) {
        // Exact-sized input/output allocations expose invalid token columns.
        Buffer x(std::size_t(n) * k * sizeof(float));
        Buffer old(std::size_t(n) * m * sizeof(float));
        Buffer candidate(std::size_t(n) * m * sizeof(float));
        Check(hipMemcpy(x.p, host_x.data(), std::size_t(n) * k * sizeof(float),
                        hipMemcpyHostToDevice));
        Check(hipMemset(old.p, 0xff, std::size_t(n) * m * sizeof(float)));
        Check(hipMemset(candidate.p, 0xff, std::size_t(n) * m * sizeof(float)));
        for (bool selected : {false, true}) {
          r::SmallGemmHcBf16Variant(
              w.p, static_cast<const float*>(x.p),
              static_cast<float*>(selected ? candidate.p : old.p), n, m, k,
              selected, nullptr);
          Check(hipGetLastError());
        }
        std::vector<float> expected(std::size_t(n) * m),
            actual(expected.size());
        Check(hipMemcpy(expected.data(), old.p, expected.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        Check(hipMemcpy(actual.data(), candidate.p,
                        actual.size() * sizeof(float), hipMemcpyDeviceToHost));
        DenseError change(0.002 * amplitude), invariant(0.002 * amplitude),
            old_invariant(0.002 * amplitude), oracle(0.002 * amplitude);
        for (std::size_t i = 0; i < actual.size(); ++i) {
          change.Add(expected[i], actual[i]);
          invariant.Add(width_one[i], actual[i]);
          old_invariant.Add(width_one[i], expected[i]);
        }
        // Independent FP64 oracle, three output rows and first/last token.
        for (int row : {0, m / 2, m - 1})
          for (int token : {0, n - 1}) {
            double sum = 0;
            for (int i = 0; i < k; ++i) {
              const float weight = std::bit_cast<float>(
                  std::uint32_t(packed[std::size_t(row) * k + i]) << 16);
              sum += double(weight) * host_x[std::size_t(token) * k + i];
            }
            oracle.Add(sum, actual[std::size_t(token) * m + row]);
          }
        std::printf(
            "HC_COMPARE name=%s K=%d M=%d N=%d amplitude=%.4g "
            "old_new_max_abs=%.9g old_new_max_rel=%.9g byte_differences=%zu "
            "n1_nN_max_abs=%.9g n1_nN_max_rel=%.9g n1_nN_byte_differences=%zu "
            "old_n1_nN_max_abs=%.9g oracle_max_abs=%.9g oracle_max_rel=%.9g "
            "abs_tolerance=%.9g rel_tolerance=0.0002 valid=%d\n",
            std::string(tensor.name).c_str(), k, m, n, amplitude,
            change.max_abs, change.max_relative, change.byte_differences,
            invariant.max_abs, invariant.max_relative,
            invariant.byte_differences, old_invariant.max_abs, oracle.max_abs,
            oracle.max_relative, 0.002 * amplitude,
            int(!change.outside_tolerance && !invariant.outside_tolerance &&
                !oracle.outside_tolerance));
        std::fflush(stdout);
        Require(!change.outside_tolerance && !invariant.outside_tolerance &&
                    !oracle.outside_tolerance,
                "BF16 HC comparison tolerance exceeded");
        Require(invariant.byte_differences == 0,
                "BF16 HC candidate differs between N1 and the wider call");
      }
    }
    ++checked;
  }
  std::printf(
      "HC_COMPARE_SUMMARY file=%s tensors=%zu widths=1..8 "
      "weights=real inputs=synthetic gpu_measured_by_operator=1\n",
      path.c_str(), checked);
}
