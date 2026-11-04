// Included inside the benchmark namespace after its launch/check helpers.
void IqExactCpuOracle(const std::vector<IqVariant>& variants,
                      const std::vector<int>& widths) {
  std::uint64_t cases = 0;
  for (const Fmt* format : {&kIQ2_XXS, &kIQ2_XS, &kIQ1_M}) {
    constexpr int experts = 13;
    constexpr int k = 256;
    for (int rows : {1, 3, 17}) {
      std::array<std::vector<std::uint8_t>, 2> weights{
          IqWeights(*format, experts, rows, k, 0xA11B37),
          IqWeights(*format, experts, rows, k, 0x163B77)};
      std::array<std::vector<float>, 2> decoded;
      for (int p = 0; p < 2; ++p) {
        decoded[p].resize(std::size_t(experts) * rows * k);
        for (int row = 0; row < experts * rows; ++row) {
          auto* block = weights[p].data() + row * format->bytes;
          if (format->type == GgmlType::kIQ1_M) {
            std::uint16_t sc[4];
            std::memcpy(sc, block + 48, sizeof(sc));
            for (int i = 0; i < 4; ++i)
              sc[i] = (sc[i] & 0x0FFF) | (((0x3C00 >> (4 * i)) & 15) << 12);
            std::memcpy(block + 48, sc, sizeof(sc));
          } else {
            constexpr std::uint16_t half = 0x4800;
            std::memcpy(block, &half, sizeof(half));
            for (int chunk = 0; chunk < k / 32; ++chunk) {
              block[2 + 8 * chunk] = 0;
              if (format->type == GgmlType::kIQ2_XS)
                block[3 + 8 * chunk] &= 0xFE;
            }
          }
          float* out = decoded[p].data() + row * k;
          if (format->type == GgmlType::kIQ1_M)
            gufo::quant::DequantizeIQ1_M(block, out, k);
          else if (format->type == GgmlType::kIQ2_XS)
            gufo::quant::DequantizeIQ2_XS(block, out, k);
          else
            gufo::quant::DequantizeIQ2_XXS(block, out, k);
        }
      }
      IqBuffer device_w(weights[0].size());
      IqBuffer device_up(weights[1].size());
      Check(hipMemcpy(device_w.data, weights[0].data(), weights[0].size(),
                      hipMemcpyHostToDevice),
            "CPU oracle weights");
      Check(hipMemcpy(device_up.data, weights[1].data(), weights[1].size(),
                      hipMemcpyHostToDevice),
            "CPU oracle up weights");
      for (int n : widths) {
        for (bool down : {false, true}) {
          const int tokens = down ? n * kUsed : n;
          const int used = down ? 1 : kUsed;
          std::vector<float> input(std::size_t(tokens) * k);
          std::uint32_t seed = 0x731319;
          for (auto& value : input)
            value = float(8 * (int(NextRandom(&seed) % 31) - 15));
          for (int token = 0; token < tokens; ++token)
            for (int chunk = 0; chunk < k / 32; ++chunk)
              input[token * k + 32 * chunk] = 127.0F;
          // Every 32-value amax is 127, giving exact Q8 scale 1. IQ2's
          // corresponding grid magnitude is 8;
          // all other activations are multiples of 8, so integer divisions
          // are exact. All CPU sums are representable, independent of order.
          for (int pattern = 0; pattern < 4; ++pattern) {
            const auto ids = IqRoutes(tokens, used, experts, pattern);
            std::vector<std::vector<float>> reference(down ? 1 : 2);
            for (int p = 0; p < int(reference.size()); ++p) {
              auto& result = reference[p];
              result.resize(std::size_t(tokens) * used * rows);
              for (int token = 0; token < tokens; ++token) {
                for (int slot = 0; slot < used; ++slot) {
                  const int expert = ids[token * used + slot];
                  if (expert < 0)
                    continue;
                  for (int row = 0; row < rows; ++row) {
                    float sum = 0.0F;
                    const auto* w =
                        decoded[p].data() + (expert * rows + row) * k;
                    for (int j = 0; j < k; ++j)
                      sum += w[j] * input[token * k + j];
                    result[(token * used + slot) * rows + row] = sum;
                  }
                }
              }
            }
            IqCheckCase("cpu-exact", "new-type-reference", *format,
                        device_w.data, down ? nullptr : device_up.data, rows, k,
                        n, experts, down, pattern, 1.0F, variants, &input,
                        &reference);
            ++cases;
          }
        }
      }
    }
  }
  std::printf(
      "CPU_ORACLE_SUMMARY cases=%llu vector_matrix_cpu_bytes=equal PASS\n",
      static_cast<unsigned long long>(cases));
}
