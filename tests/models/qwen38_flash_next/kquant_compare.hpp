#pragma once

// Shares buffers, quantized inputs and error reporting with the existing bench.
// Real payloads also supply compact, exact-sized ragged-M and K-tail cases.
// candidate kquant_wmma: N1..3 keep the current bytes and the N1 invariant is
// against the current width-one kernel. candidate dot4: every width runs the
// candidate, so its own N1 is the reference and must match N = 1..8 bit for bit.
void CompareKQuantSlice(int type, const void* payload, Shape s, bool ragged,
                        int candidate_variant = int(d::Variant::kKQuantWmma)) {
  const bool dot4 = candidate_variant == int(d::Variant::kDot4);
  const int width_one_variant =
      dot4 ? candidate_variant : int(d::ProductionVariant(type, 1, s.k));
  namespace q = gufo::quant;
  const auto weight_bytes = WeightBytes(type, s);
  Buffer w(weight_bytes), all_x(qfn_mmq_q8_1_bytes(8, s.k)),
      ids(8 * sizeof(int32_t));
  Check(hipMemcpy(w.p, payload, weight_bytes, hipMemcpyHostToDevice));
  const auto stride_bytes = qfn_mmq_q8_1_bytes(8, s.k) / 8;
  for (float amplitude : {1.0F, 0.001F}) {
    Inputs(all_x, ids, s, 8, amplitude);
    std::vector<std::uint8_t> quantized(stride_bytes * 8);
    Check(hipMemcpy(quantized.data(), all_x.p, quantized.size(),
                    hipMemcpyDeviceToHost));
    std::vector<float> width_one(std::size_t(8) * s.m);
    Buffer scalar_x(stride_bytes), scalar_y(std::size_t(s.m) * sizeof(float));
    for (int token = 0; token < 8; ++token) {
      Check(hipMemcpy(scalar_x.p, quantized.data() + token * stride_bytes,
                      stride_bytes, hipMemcpyHostToDevice));
      Require(Call(type, width_one_variant, w.p, scalar_x.p, ids.p,
                   scalar_y.p, s, 1) == 0,
              "K-quant N1 reference");
      Check(hipMemcpy(width_one.data() + std::size_t(token) * s.m, scalar_y.p,
                      std::size_t(s.m) * sizeof(float), hipMemcpyDeviceToHost));
    }
    for (int n = 1; n <= 8; ++n) {
      Buffer x(qfn_mmq_q8_1_bytes(n, s.k));
      const auto bytes = std::size_t(n) * s.m * sizeof(float);
      Buffer old(bytes), candidate(bytes);
      Check(hipMemcpy(x.p, quantized.data(), stride_bytes * n,
                      hipMemcpyHostToDevice));
      Check(hipMemset(old.p, 0xff, bytes));
      Check(hipMemset(candidate.p, 0xff, bytes));
      Require(Call(type, int(d::ProductionVariant(type, n, s.k)), w.p, x.p,
                   ids.p, old.p, s, n) == 0,
              "K-quant current launch");
      const auto launched =
          qfn_mmq_dense_gemv_launch_count(type, candidate_variant, n);
      Require(Call(type, candidate_variant, w.p, x.p, ids.p, candidate.p, s,
                   n) == 0,
              "K-quant candidate launch");
      // kquant_wmma reroutes N1..3 by design; dot4 never reroutes a K-quant.
      Require(!dot4 || qfn_mmq_dense_gemv_launch_count(
                           type, candidate_variant, n) == launched + 1,
              "the dot4 kernel ran (launch counter)");
      std::vector<float> expected(std::size_t(n) * s.m),
          actual(expected.size());
      Check(hipMemcpy(expected.data(), old.p, bytes, hipMemcpyDeviceToHost));
      Check(
          hipMemcpy(actual.data(), candidate.p, bytes, hipMemcpyDeviceToHost));
      DenseError change(0.002 * amplitude), invariant(0.002 * amplitude),
          old_invariant(0.002 * amplitude), oracle(0.002 * amplitude);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        change.Add(expected[i], actual[i]);
        invariant.Add(width_one[i], actual[i]);
        old_invariant.Add(width_one[i], expected[i]);
      }
      // Independent vendored dequantizer + FP64 dot against the ACTUAL Q8_1
      // inputs (not the unquantized source). Every real tensor gets samples.
      std::vector<float> decoded(s.k);
      for (int row : {0, s.m / 2, s.m - 1}) {
        const auto* packed = static_cast<const std::uint8_t*>(payload) +
                             std::size_t(row) * weight_bytes / s.m;
        if (type == 14)
          q::DequantizeQ6_K(packed, decoded.data(), s.k);
        else if (type == 13)
          q::DequantizeQ5_K(packed, decoded.data(), s.k);
        else
          q::DequantizeQ4_K(packed, decoded.data(), s.k);
        for (int token : {0, n - 1}) {
          double sum = 0;
          for (int i = 0; i < s.k; ++i) {
            const auto* block =
                quantized.data() + token * stride_bytes + (i / 32) * 36;
            std::uint16_t half;
            std::memcpy(&half, block, 2);
            const auto code = static_cast<std::int8_t>(block[4 + i % 32]);
            sum += double(decoded[i]) * q::Fp16ToFloat(half) * code;
          }
          oracle.Add(sum, actual[std::size_t(token) * s.m + row]);
        }
      }
      std::printf(
          "KQUANT_COMPARE candidate=%s name=%s fmt=%s K=%d M=%d N=%d "
          "amplitude=%.4g ragged=%d "
          "old_new_abs=%.9g old_new_rel=%.9g changed=%zu "
          "N1_NN_abs=%.9g N1_NN_rel=%.9g N1_NN_changed=%zu "
          "old_N1_NN_abs=%.9g oracle_abs=%.9g oracle_rel=%.9g "
          "outside=%zu tolerance_abs=%.9g tolerance_relative=0.0002 exact=%d\n",
          d::VariantName(static_cast<d::Variant>(candidate_variant)), s.name,
          Format(type), s.k, s.m, n, amplitude, int(ragged),
          change.max_abs, change.max_relative, change.byte_differences,
          invariant.max_abs, invariant.max_relative, invariant.byte_differences,
          old_invariant.max_abs, oracle.max_abs, oracle.max_relative,
          change.outside_tolerance + invariant.outside_tolerance +
              oracle.outside_tolerance,
          0.002 * amplitude, int(change.byte_differences == 0));
      std::fflush(stdout);
      Require(change.outside_tolerance == 0 &&
                  invariant.outside_tolerance == 0 &&
                  oracle.outside_tolerance == 0,
              "K-quant numerical tolerance");
      if (dot4)
        Require(invariant.byte_differences == 0,
                "dot4: a row computes to the same bits at N = 1..8");
      else
        Require(n > 3 || change.byte_differences == 0,
                "N1..3 must retain current bytes");
    }
  }
}

void CompareKQuant(const std::string& path,
                   int candidate_variant = int(d::Variant::kKQuantWmma)) {
  const bool dot4 = candidate_variant == int(d::Variant::kDot4);
  std::set<std::tuple<int, int, int>> shapes;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(bool(reader), error.c_str());
  std::set<int> tails;
  std::size_t checked = 0;
  for (const auto& t : reader->GetTensors()) {
    const int type = int(t.type);
    // dot4 has no shape list: one tensor of every distinct (format, K, M),
    // heads and small projections included.
    if (!d::IsKQuantTarget(type) || t.dimensions.size() != 2 ||
        (dot4 ? !shapes.insert({type, int(t.dimensions[0]),
                                int(t.dimensions[1])})
                     .second
              : !d::KQuantMatrixShape(int(t.dimensions[0]),
                                      int(t.dimensions[1]))))
      continue;
    const std::string name(t.name);
    const Shape s{name.c_str(), int(t.dimensions[0]), int(t.dimensions[1])};
    CompareKQuantSlice(type, t.data, s, false, candidate_variant);
    ++checked;
    if (tails.insert(type).second) {
      for (int m : {1, 15, 16, 17, 31}) {
        const int k = m == 1 ? 256 : 768;
        std::vector<std::uint8_t> compact(std::size_t(m) * (k / 256) *
                                          Bytes(type));
        const auto src_stride = std::size_t(s.k / 256) * Bytes(type);
        const auto dst_stride = std::size_t(k / 256) * Bytes(type);
        for (int row = 0; row < m; ++row)
          std::memcpy(
              compact.data() + row * dst_stride,
              static_cast<const std::uint8_t*>(t.data) + row * src_stride,
              dst_stride);
        CompareKQuantSlice(type, compact.data(), {name.c_str(), k, m}, true,
                           candidate_variant);
      }
    }
  }
  std::printf(
      "KQUANT_COMPARE_SUMMARY candidate=%s file=%s real_tensors=%zu "
      "ragged_formats=%zu status=%s\n",
      d::VariantName(static_cast<d::Variant>(candidate_variant)), path.c_str(),
      checked, tails.size(), checked ? "pass" : "not-applicable");
}
