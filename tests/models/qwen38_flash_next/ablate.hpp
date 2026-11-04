#pragma once

// Operator-only explicit-kernel cells for the few-row bandwidth ablation.
// Included inside the bench's anonymous namespace (shares Buffer, Shape,
// Weights, Inputs, TimeCall). ONE cell function (MeasureCell) serves synthetic
// shapes and real-payload crops, so the protocol is identical: `sets`
// physically distinct 256-byte-aligned copies per timed interval, 256 MiB
// eviction before every interval, nine intervals per width. Every cell diffs
// the launch counters around itself and aborts when the kernel that ran is not
// the kernel that was asked for. Flags only; the environment must not select a
// kernel.

enum class AblKind {
  kProduction,
  kVariant,
  kStage,
  kLoads,
  kQ8Dp4a,
  kQ8Wmma,
  kIq4Wmma,
  kTubo  // tubo_k4.hpp: VALU dot4 control (copied kernels, bench only)
};
struct AblKernel {
  AblKind kind = AblKind::kProduction;
  int variant = 0;
  int stage = 0;
  int cfg = -1;  // tubo config id
  std::string name;
};
struct AblTuboInfo {
  const char* name = "";
  int type = 0, rows32 = 0, act = 0;
};
AblTuboInfo AblTubo(int cfg) {
  AblTuboInfo info;
  Require(bench_tubo_config(cfg, &info.name, &info.type, &info.rows32,
                            &info.act) == 0,
          "tubo config");
  return info;
}
// Where a kernel's weights come from: tubo's *_rows32 configs read the
// re-laid-out copy (bench-only ceiling control, not shippable).
bool AblRepacked(const AblKernel& k) {
  return k.kind == AblKind::kTubo && AblTubo(k.cfg).rows32;
}
struct AblTuboBuffers {
  const void* xq = nullptr;
  const void* xds = nullptr;
  const void* xsum = nullptr;
};

int AblType(const std::string& text) {
  for (int type : {8, 12, 13, 14, 20, 21, 23, 7, 18, 17})
    if (text == Format(type))
      return type;
  throw std::runtime_error("unknown --fmt " + text);
}

AblKernel AblParseKernel(const std::string& text) {
  AblKernel k;
  k.name = text;
  if (text == "production")
    k.kind = AblKind::kProduction;
  else if (text == "existing" || text == "baseline")
    k.kind = AblKind::kVariant, k.variant = 0;
  else if (text.rfind("stage:", 0) == 0) {
    k.kind = AblKind::kStage;
    k.stage = d::ParseKQuantStage(text.c_str() + 6);
    Require(k.stage > 0, "unknown stage (A A0 B B0 C1 C2 C3 D)");
  } else if (text == "reuse1_loads")
    k.kind = AblKind::kLoads;
  else if (text == "q8_dp4a")
    k.kind = AblKind::kQ8Dp4a;
  else if (text == "q8_wmma")
    k.kind = AblKind::kQ8Wmma;
  else if (text == "iq4_wmma")
    k.kind = AblKind::kIq4Wmma;
  else if (text.rfind("tubo:", 0) == 0) {
    k.kind = AblKind::kTubo;
    for (int cfg = 0; cfg < bench_tubo_config_count(); ++cfg)
      if (text.substr(5) == AblTubo(cfg).name)
        k.cfg = cfg;
    Require(k.cfg >= 0, "unknown tubo config");
  } else {
    k.kind = AblKind::kVariant;
    k.variant = d::ParseVariant(text.c_str());
    Require(k.variant > 0, "unknown --kernel");
  }
  return k;
}

struct AblCounts {
  std::vector<std::pair<std::string, unsigned long long>> named;
  unsigned long long iq4_total = 0;
};
AblCounts AblSnapshot(int type, int n) {
  AblCounts c;
  if (d::FormatIndex(type) >= 0) {
    for (int v = 0; v < d::kVariantCount; ++v)
      c.named.push_back({d::VariantName(static_cast<d::Variant>(v)),
                         qfn_mmq_dense_gemv_launch_count(type, v, n)});
    c.named.push_back(
        {"reuse1_loads", qfn_mmq_reuse1_loads_launch_count(type)});
  }
  if (d::IsKQuantTarget(type))
    for (int s = 1; s <= d::kStageCount; ++s)
      c.named.push_back({std::string("stage:") + d::KQuantStageName(s),
                         qfn_mmq_kquant_stage_launch_count(type, s, n)});
  if (type == 8) {
    c.named.push_back({"q8_dp4a", qfn_mmq_q8_0_control_launch_count(0, n)});
    c.named.push_back({"q8_wmma", qfn_mmq_q8_0_control_launch_count(1, n)});
  }
  for (int cfg = 0; cfg < bench_tubo_config_count(); ++cfg)
    c.named.push_back({std::string("tubo:") + AblTubo(cfg).name,
                       bench_tubo_launch_count(cfg, n)});
  c.iq4_total = qfn_mmq_iq4_dense_launch_count();
  return c;
}
// The kernel the counters say ran `expected_launches` times, or a description
// of the mismatch.
std::string AblObserved(const AblCounts& before, const AblCounts& after,
                        unsigned long long expected_launches, bool* ok) {
  std::string moved, ran;
  int exact = 0, other = 0;
  for (std::size_t i = 0; i < before.named.size(); ++i) {
    const auto delta = after.named[i].second - before.named[i].second;
    if (!delta)
      continue;
    moved += " " + before.named[i].first + "+" + std::to_string(delta);
    if (delta == expected_launches) {
      ++exact;
      ran = before.named[i].first;
    } else {
      ++other;
    }
  }
  const auto iq4 = after.iq4_total - before.iq4_total;
  *ok = true;
  if (exact == 1 && other == 0)
    return ran;
  if (exact == 0 && other == 0 && iq4 == expected_launches)
    return "iq4_wmma";
  *ok = false;
  return "MISMATCH[" + moved + " iq4_total+" + std::to_string(iq4) + "]";
}

int AblCall(const AblKernel& k, int type, const void* w, const void* x,
            const void* ids, void* y, Shape s, int n, int extra_lds,
            const AblTuboBuffers& tubo = {}) {
  switch (k.kind) {
    case AblKind::kTubo: {
      const int id = bench_tubo_launch(k.cfg, w, tubo.xq, tubo.xds, tubo.xsum,
                                       static_cast<float*>(y), n, s.m, s.k);
      return id == k.cfg ? 0 : (id < 0 ? id : -4);
    }
    case AblKind::kProduction:
      return qfn_mmq_dense_vec_preq(
          type, w, x, static_cast<const int32_t*>(ids), static_cast<float*>(y),
          s.m, n, s.k, nullptr);
    case AblKind::kVariant:
      return qfn_mmq_dense_vec_preq_variant(
          type, w, x, static_cast<const int32_t*>(ids), static_cast<float*>(y),
          s.m, n, s.k, k.variant, nullptr);
    case AblKind::kStage: {
      const int id = qfn_mmq_kquant_stage_launch(type, k.stage, w, x,
                                                 static_cast<float*>(y), s.k,
                                                 s.m, n, extra_lds, nullptr);
      return id == type * 256 + k.stage ? 0 : (id < 0 ? id : -4);
    }
    case AblKind::kLoads:
      return qfn_mmq_reuse1_loads_launch(type, w, x, static_cast<float*>(y),
                                         s.m, s.k, nullptr);
    case AblKind::kQ8Dp4a:
      return qfn_mmq_q8_0_control_launch(0, w, x, static_cast<float*>(y), s.m,
                                         n, s.k, nullptr);
    case AblKind::kQ8Wmma:
      return qfn_mmq_q8_0_control_launch(1, w, x, static_cast<float*>(y), s.m,
                                         n, s.k, nullptr);
    case AblKind::kIq4Wmma:
      Require(gufo::models::qwen38_flash_next::Iq4WideVariant() == 4,
              "iq4_wmma control requires GUFO_IQ4_WIDE_VARIANT=4 or unset");
      return qfn_mmq_iq4_dense_batch_preq(type, w, x, static_cast<float*>(y),
                                          s.m, n, s.k, nullptr);
  }
  return -1;
}

// Why this kernel cannot be asked for here (empty = fine). Unsupported
// requests are reported, never rerouted.
std::string AblUnsupported(const AblKernel& k, int type, Shape s, int n) {
  if (k.kind == AblKind::kStage && (!d::IsKQuantTarget(type) || s.k % 256))
    return "stage kernels are Q4_K/Q5_K/Q6_K with K % 256 == 0";
  if (k.kind == AblKind::kLoads && (n != 1 || d::FormatIndex(type) < 0))
    return "reuse1_loads has no activations: run at N=1 on a dense format";
  if ((k.kind == AblKind::kQ8Dp4a || k.kind == AblKind::kQ8Wmma) && type != 8)
    return "Q8_0 controls need --fmt Q8_0";
  if (k.kind == AblKind::kIq4Wmma && type != 20 && type != 23)
    return "iq4_wmma is IQ4_XS / IQ4_NL";
  if (k.kind == AblKind::kTubo) {
    const auto info = AblTubo(k.cfg);
    if (info.type != type)
      return "tubo config is for another format";
    if (n != 1 && n != 2 && n != 4 && n != 8)
      return "tubo control is compiled for widths 1, 2, 4, 8";
    if (s.k % 256)
      return "K % 256 != 0";
    if (info.rows32 && s.m % 32)
      return "rows32 repack needs M % 32 == 0 (no fallback exists in tubo)";
  }
  if ((k.kind == AblKind::kVariant || k.kind == AblKind::kProduction) &&
      (d::FormatIndex(type) < 0))
    return "format has no dense variants";
  return {};
}

struct AblConfig {
  std::vector<AblKernel> kernels;
  int type = 12;
  std::vector<int> widths{1, 2, 3, 4, 5, 6, 7, 8};
  int sets = 16;
  int samples = 9;
  int occupancy = 0;     // 0 = no forced residency
  int extra_lds = -1;    // explicit dynamic LDS bytes (-1 unset)
  double max_gib = 6.0;  // weight footprint cap per cell
  bool failed = false;
};

// Kernels whose output is a checksum of what they loaded (stages A..C3 and the
// loads rung), not a product: their output is checked for finiteness only.
// Stage B0 (the decode-to-LDS rung without the per-step barriers) reads tile
// words that other waves write in the same step: the read-back races by
// construction, so its checksum is not reproducible. Its instruction stream and
// addresses are B's minus the barriers, which is what it measures.
bool AblRacy(const AblKernel& k) {
  return k.kind == AblKind::kStage && k.stage == d::kStageLdsNoBarrier;
}

bool AblSinkOutput(const AblKernel& k) {
  return k.kind == AblKind::kLoads ||
         (k.kind == AblKind::kStage && k.stage != d::kStageFull);
}

// Output check of one timed cell: finite everywhere, not all zero, and (for
// kernels that compute the product) within a fraction of the largest value of
// the reference result. Tubo quantises the activations to int8 with its own
// scales, so it gets 10 %; every kernel that shares the Q8_1 inputs gets 2 %.
std::string AblOutputProblem(const AblKernel& kernel,
                             const std::vector<float>& got,
                             const std::vector<float>* reference) {
  bool finite = true, nonzero = false;
  for (float v : got) {
    finite &= std::isfinite(v);
    nonzero |= v != 0.0F;
  }
  if (!finite)
    return "output-not-finite";
  if (AblSinkOutput(kernel))
    return {};
  if (!nonzero)
    return "output-all-zero";
  if (!reference || reference->size() != got.size())
    return {};
  double largest = 0, worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    largest = std::max(largest, std::fabs(double((*reference)[i])));
    worst =
        std::max(worst, std::fabs(double((*reference)[i]) - double(got[i])));
  }
  const double fraction = kernel.kind == AblKind::kTubo ? 0.10 : 0.02;
  if (!(worst <= fraction * largest)) {
    char text[96];
    std::snprintf(text, sizeof text,
                  "output-mismatch(max_abs_diff=%.3g,ref=%.3g)", worst,
                  largest);
    return text;
  }
  return {};
}

// One shape's cells: allocates the distinct copies once, then every kernel and
// width reuses them (same bytes, same protocol). Returns the number of
// ABL_CELL lines printed (a caller that expects cells treats zero as failure).
int MeasureCell(AblConfig& cfg, Buffer& eviction, const std::string& source,
                const std::string& name, Shape s, const void* payload) {
  const int type = cfg.type;
  const auto bytes = WeightBytes(type, s);
  const auto stride = (bytes + 255) / 256 * 256;
  // The re-laid-out copy (tubo rows32) doubles the footprint; the cap counts
  // it.
  bool want_repack = false, want_tubo = false;
  for (const auto& kernel : cfg.kernels) {
    want_tubo |= kernel.kind == AblKind::kTubo;
    want_repack |= AblRepacked(kernel) && s.m % 32 == 0 && s.k % 256 == 0;
  }
  const double copies = want_repack ? 2.0 : 1.0;
  int sets = cfg.sets;
  const int requested_sets = sets;
  while (sets > 4 &&
         copies * double(sets) * double(stride) / (1024.0 * 1024 * 1024) >
             cfg.max_gib)
    --sets;
  int cells = 0;
  Buffer weights(sets * stride), input(qfn_mmq_q8_1_bytes(8, s.k)),
      ids(8 * sizeof(int32_t)), output(std::size_t(8) * s.m * sizeof(float));
  for (int i = 0; i < sets; ++i) {
    if (payload)
      Check(hipMemcpy(static_cast<char*>(weights.p) + i * stride, payload,
                      bytes, hipMemcpyHostToDevice));
    else
      WeightsAt(static_cast<char*>(weights.p) + i * stride, bytes, type);
  }
  Inputs(input, ids, s, 8);
  // tubo control: the re-laid-out copy (only if a rows32 kernel is asked for
  // and the shape allows it) and the float activations it quantises itself.
  std::unique_ptr<Buffer> repacked;
  if (want_repack) {
    std::vector<std::uint8_t> native(bytes), laid_out(bytes);
    if (payload)
      std::memcpy(native.data(), payload, bytes);
    else
      Check(hipMemcpy(native.data(), weights.p, bytes, hipMemcpyDeviceToHost));
    Require(bench_tubo_repack(native.data(), laid_out.data(), s.m, s.k,
                              Bytes(type)) == 0,
            "tubo rows32 repack");
    repacked = std::make_unique<Buffer>(sets * stride);
    for (int i = 0; i < sets; ++i)
      Check(hipMemcpy(static_cast<char*>(repacked->p) + i * stride,
                      laid_out.data(), bytes, hipMemcpyHostToDevice));
  }
  std::unique_ptr<Buffer> raw, tubo_q, tubo_ds, tubo_sum;
  if (want_tubo) {
    const auto host = InputHost(8, s.k, 1.0F);
    raw = std::make_unique<Buffer>(host.size() * sizeof(float));
    Check(hipMemcpy(raw->p, host.data(), host.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    tubo_q = std::make_unique<Buffer>(std::size_t(8) * s.k);
    tubo_ds = std::make_unique<Buffer>(std::size_t(8) * (s.k / 16) * 8);
    tubo_sum =
        std::make_unique<Buffer>(std::size_t(s.k / 256) * 9 * 8 * 4 + 64);
  }
  std::printf(
      "ABL_SHAPE source=%s name=%s fmt=%s K=%d M=%d weight_bytes=%zu "
      "copy_stride=%zu sets=%d sets_requested=%d footprint_mib=%.1f "
      "copies=%.0f max_gib=%.1f%s\n",
      source.c_str(), name.c_str(), Format(type), s.k, s.m, bytes, stride, sets,
      requested_sets, copies * double(sets) * double(stride) / (1024.0 * 1024),
      copies, cfg.max_gib, sets < requested_sets ? " sets_reduced=1" : "");
  // Reference result per width (reuse1 on the same bytes and Q8_1 inputs; the
  // Q8_0 dp4a route for Q8_0). Computed before a cell's counter snapshot.
  std::map<int, std::vector<float>> references;
  const auto reference_for = [&](int n) -> const std::vector<float>* {
    auto found = references.find(n);
    if (found != references.end())
      return found->second.empty() ? nullptr : &found->second;
    std::vector<float> values;
    int status = -1;
    if (type == 8)
      status = qfn_mmq_q8_0_control_launch(0, weights.p, input.p,
                                           static_cast<float*>(output.p), s.m,
                                           n, s.k, nullptr);
    else if (d::Covers(type, n, s.k))
      status = qfn_mmq_dense_vec_preq_variant(
          type, weights.p, input.p, static_cast<const int32_t*>(ids.p),
          static_cast<float*>(output.p), s.m, n, s.k, int(d::Variant::kReuse1),
          nullptr);
    if (status == 0) {
      Check(hipDeviceSynchronize());
      values.resize(std::size_t(n) * s.m);
      Check(hipMemcpy(values.data(), output.p, values.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
    }
    auto& slot = references[n];
    slot = std::move(values);
    return slot.empty() ? nullptr : &slot;
  };
  for (const auto& kernel : cfg.kernels) {
    for (int n : cfg.widths) {
      const auto why = AblUnsupported(kernel, type, s, n);
      if (!why.empty()) {
        std::printf("ABL_SKIP kernel=%s fmt=%s K=%d M=%d N=%d reason=%s\n",
                    kernel.name.c_str(), Format(type), s.k, s.m, n,
                    why.c_str());
        continue;
      }
      int extra = cfg.extra_lds >= 0 ? cfg.extra_lds : 0;
      int resident = -1;
      if (kernel.kind == AblKind::kStage) {
        if (cfg.occupancy > 0) {
          extra = -1;
          int lowest = -1, last_extra = 0;
          std::string problem;
          for (int e = 0; e <= 65536; e += 512) {
            const int occ =
                qfn_mmq_kquant_stage_occupancy(type, kernel.stage, e);
            if (occ < 0) {
              problem =
                  "occupancy query failed at extra_lds=" + std::to_string(e);
              break;
            }
            lowest = occ;
            last_extra = e;
            if (occ <= cfg.occupancy) {
              extra = e;
              resident = occ;
              break;
            }
          }
          if (extra < 0) {
            // An unreachable residency is a measurement we cannot make here,
            // not an error of the run.
            std::printf(
                "ABL_SKIP kernel=%s fmt=%s K=%d M=%d N=%d reason=residency %d "
                "CTAs/WGP unreachable (lowest reported %d at "
                "extra_lds=%d)%s%s\n",
                kernel.name.c_str(), Format(type), s.k, s.m, n, cfg.occupancy,
                lowest, last_extra, problem.empty() ? "" : "; ",
                problem.c_str());
            continue;
          }
        } else {
          resident = qfn_mmq_kquant_stage_occupancy(type, kernel.stage, extra);
        }
      }
      // Expected counter owner (production: what the selector says).
      std::string expected = kernel.name;
      if (kernel.kind == AblKind::kProduction)
        expected =
            d::VariantName(d::SelectVariant(type, n, s.k, s.m, -1, true));
      else if (kernel.kind == AblKind::kVariant)
        expected = d::VariantName(static_cast<d::Variant>(kernel.variant));
      const bool repacked_layout = AblRepacked(kernel);
      AblTuboBuffers tubo;
      if (kernel.kind == AblKind::kTubo) {
        // Activation quantisation is outside the timed interval (as the
        // Q8_1 quantisation is for every other kernel).
        Require(
            bench_tubo_prepare(kernel.cfg, static_cast<const float*>(raw->p),
                               tubo_q->p, tubo_ds->p, tubo_sum->p, n, s.k) == 0,
            "tubo activation quantisation");
        Check(hipDeviceSynchronize());
        tubo = {tubo_q->p, tubo_ds->p, tubo_sum->p};
      }
      const char* base =
          static_cast<const char*>(repacked_layout ? repacked->p : weights.p);
      const std::vector<float>* reference =
          AblSinkOutput(kernel) ? nullptr : reference_for(n);
      const auto before = AblSnapshot(type, n);
      std::vector<double> times;
      bool launch_ok = true;
      std::string launch_error;
      for (int sample = 0; sample < cfg.samples && launch_ok; ++sample) {
        try {
          times.push_back(TimeCall(
              [&](int i) {
                return AblCall(kernel, type, base + i * stride, input.p, ids.p,
                               output.p, s, n, extra, tubo);
              },
              eviction, 256U << 20, sets));
        } catch (const std::exception& e) {
          launch_ok = false;
          launch_error = e.what();
        }
      }
      const auto after = AblSnapshot(type, n);
      bool ok = false;
      const auto ran = AblObserved(
          before, after, (unsigned long long)times.size() * (sets + 1), &ok);
      const bool verified = launch_ok && ok && ran == expected;
      if (!verified) {
        std::printf(
            "CELL_ABORT kernel_requested=%s kernel_expected=%s kernel_ran=%s "
            "fmt=%s K=%d M=%d N=%d launch_error=%s\n",
            kernel.name.c_str(), expected.c_str(), ran.c_str(), Format(type),
            s.k, s.m, n, launch_error.c_str());
        cfg.failed = true;
        continue;
      }
      // The output of the last launch of the last interval: counters prove
      // which kernel was launched, this proves it wrote a plausible result.
      Check(hipDeviceSynchronize());
      std::vector<float> got(std::size_t(n) * s.m);
      Check(hipMemcpy(got.data(), output.p, got.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
      std::string problem = AblOutputProblem(kernel, got, reference);
      std::vector<double> gbps;
      for (std::size_t sample = 0; sample < times.size(); ++sample) {
        const double us = times[sample];
        const double g = double(bytes) / us / 1000.0;
        const double total =
            (double(bytes) + double(n) * (s.k / 32) * 36) / us / 1000.0;
        gbps.push_back(g);
        std::printf(
            "ABL name=%s fmt=%s K=%d M=%d N=%d sample=%zu us=%.6f GBps=%.3f "
            "GBps_with_activations=%.3f valid=%d kernel_requested=%s "
            "kernel_ran=%s verified=1 source=%s protocol=distinct-matrices "
            "sets=%d evict_bytes=268435456 extra_lds=%d ctas_per_wgp=%d "
            "layout=%s shippable=%d racy=%d\n",
            name.c_str(), Format(type), s.k, s.m, n, sample, us, g, total,
            int(us > 0 && g <= 256), kernel.name.c_str(), ran.c_str(),
            source.c_str(), sets, extra, resident,
            repacked_layout ? "rows32_repacked" : "native",
            repacked_layout ? 0 : 1, int(AblRacy(kernel)));
      }
      std::sort(gbps.begin(), gbps.end());
      const double median = gbps[gbps.size() / 2];
      // The bandwidth rule: a median above the 256 GB/s ceiling (or no time)
      // is not a measurement.
      if (!(median > 0 && median <= 256))
        problem += (problem.empty() ? "" : ";") + std::string("above-256-GB/s");
      const bool valid = problem.empty();
      const double median_us = double(bytes) / median / 1000.0;
      std::printf(
          "ABL_CELL name=%s fmt=%s K=%d M=%d N=%d kernel=%s median_GBps=%.3f "
          "min_GBps=%.3f max_GBps=%.3f median_us=%.3f samples=%zu sets=%d "
          "extra_lds=%d ctas_per_wgp=%d layout=%s shippable=%d source=%s "
          "status=%s reason=%s short_kernel=%d racy=%d\n",
          name.c_str(), Format(type), s.k, s.m, n, kernel.name.c_str(), median,
          gbps.front(), gbps.back(), median_us, gbps.size(), sets, extra,
          resident, repacked_layout ? "rows32_repacked" : "native",
          repacked_layout ? 0 : 1, source.c_str(), valid ? "VALID" : "INVALID",
          valid ? "ok" : problem.c_str(), int(median_us < 25.0),
          int(AblRacy(kernel)));
      if (!valid)
        cfg.failed = true;
      ++cells;
      std::fflush(stdout);
    }
  }
  return cells;
}

// About one second of back-to-back eviction traffic so that the first timed
// cell of a process does not run at an idle GPU clock after the uploads.
void AblWarmup(Buffer& eviction) {
  for (int i = 0; i < 400; ++i)
    Check(ExpertBenchEvict(eviction.p, 256U << 20));
  Check(hipDeviceSynchronize());
}

void AblDevice() {
  hipDeviceProp_t p{};
  Check(hipGetDeviceProperties(&p, 0));
  // HIP's "multiprocessor" on gfx10/gfx11 is the WGP (the kernels are built in
  // workgroup-processor mode, .amdhsa_workgroup_processor_mode 1): two CUs of
  // two SIMD32 each share one 128 KiB LDS. multiProcessorCount is therefore the
  // number of WGPs and the occupancy API counts CTAs per WGP. The CU count is
  // DERIVED (two per WGP), not queried.
  std::printf(
      "ABL_DEVICE name=%s arch=%s wgps=%d cus_derived=%d "
      "lds_per_wgp_bytes=%zu max_lds_per_block_bytes=%zu wave_size=%d "
      "l2_bytes=%d residency_unit=WGP\n",
      p.name, p.gcnArchName, p.multiProcessorCount, 2 * p.multiProcessorCount,
      p.maxSharedMemoryPerMultiProcessor, p.sharedMemPerBlock, p.warpSize,
      p.l2CacheSize);
}

int AblRun(int argc, char** argv) {
  AblConfig cfg;
  struct Named {
    std::string name;
    int k, m;
  };
  std::vector<Named> shapes;
  std::string gguf;
  const auto value = [&](int& i) -> std::string {
    Require(i + 1 < argc, "missing flag value");
    return argv[++i];
  };
  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--fmt")
      cfg.type = AblType(value(i));
    else if (flag == "--kernel") {
      const auto text = value(i);
      std::size_t pos = 0;
      while (pos <= text.size()) {
        const auto end = text.find(',', pos);
        cfg.kernels.push_back(AblParseKernel(text.substr(pos, end - pos)));
        if (end == std::string::npos)
          break;
        pos = end + 1;
      }
    } else if (flag == "--shape") {
      const auto text = value(i);
      const auto x = text.find('x');
      Require(x != std::string::npos, "--shape KxM");
      shapes.push_back({"synthetic-" + text, std::stoi(text.substr(0, x)),
                        std::stoi(text.substr(x + 1))});
    } else if (flag == "--gguf")
      gguf = value(i);
    else if (flag == "--widths") {
      const auto text = value(i);
      cfg.widths.clear();
      if (text.find('-') != std::string::npos) {
        const int lo = std::stoi(text.substr(0, text.find('-')));
        const int hi = std::stoi(text.substr(text.find('-') + 1));
        for (int n = lo; n <= hi; ++n)
          cfg.widths.push_back(n);
      } else {
        std::size_t pos = 0;
        while (pos <= text.size()) {
          const auto end = text.find(',', pos);
          cfg.widths.push_back(std::stoi(text.substr(pos, end - pos)));
          if (end == std::string::npos)
            break;
          pos = end + 1;
        }
      }
    } else if (flag == "--sets")
      cfg.sets = std::stoi(value(i));
    else if (flag == "--samples")
      cfg.samples = std::stoi(value(i));
    else if (flag == "--occupancy")
      cfg.occupancy = std::stoi(value(i));
    else if (flag == "--extra-lds")
      cfg.extra_lds = std::stoi(value(i));
    else if (flag == "--max-gib")
      cfg.max_gib = std::stod(value(i));
    else
      throw std::runtime_error("unknown --ablate flag " + flag);
  }
  Require(!cfg.kernels.empty(), "--kernel is required");
  Require(!shapes.empty() || !gguf.empty(), "--shape or --gguf is required");
  Require(cfg.sets >= 4 && cfg.sets <= 64 && cfg.samples >= 1,
          "sets 4..64 (16 for the distinct-16 protocol)");
  for (int n : cfg.widths)
    Require(n >= 1 && n <= 8, "widths are 1..8");
  Require(cfg.occupancy == 0 || cfg.extra_lds < 0,
          "--occupancy and --extra-lds are exclusive");
  for (const char* name : {"GUFO_DENSE_IQ_VARIANT", "GUFO_DENSE_W1",
                           "GUFO_IQ4_WIDE_VARIANT", "GUFO_IQ4_WMMA_WAVES"})
    Require(!std::getenv(name),
            "kernel selection by environment variable is refused in --ablate");
  Check(hipSetDevice(0));
  AblDevice();
  Buffer eviction(256U << 20);
  Check(hipMemset(eviction.p, 0, 256U << 20));
  AblWarmup(eviction);
  // A shape that cannot be set up (device memory, a repack error) is reported
  // and the job goes on with the next shape; the exit code says it failed.
  const auto guarded = [&](const std::string& source, const std::string& name,
                           Shape s, const void* payload) {
    try {
      MeasureCell(cfg, eviction, source, name, s, payload);
    } catch (const std::exception& e) {
      std::printf("SHAPE_FAILED source=%s name=%s fmt=%s K=%d M=%d error=%s\n",
                  source.c_str(), name.c_str(), Format(cfg.type), s.k, s.m,
                  e.what());
      std::fflush(stdout);
      cfg.failed = true;
    }
  };
  for (const auto& named : shapes) {
    Require(named.k % Block(cfg.type) == 0,
            "K is not a multiple of the block size");
    guarded("synthetic", named.name,
            Shape{named.name.c_str(), named.k, named.m}, nullptr);
  }
  if (!gguf.empty()) {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(gguf, &error);
    Require(bool(reader), error.c_str());
    std::set<std::pair<int, int>> seen;
    for (const auto& t : reader->GetTensors()) {
      // The output head is a dense projection of the same formats (very
      // large M); it is measured with the trunk.
      if (t.dimensions.size() != 2 || int(t.type) != cfg.type ||
          !(t.name.starts_with("blk.") || t.name == "output.weight"))
        continue;
      const Shape s{nullptr, int(t.dimensions[0]), int(t.dimensions[1])};
      if (!seen.insert({s.k, s.m}).second)
        continue;
      Require(WeightBytes(cfg.type, s) <= t.size_bytes, "tensor too small");
      const std::string name(t.name);
      guarded("real", name, s, t.data);
    }
    Require(!seen.empty(), "no tensor of this format in --gguf");
  }
  std::printf("ABL_DONE failed=%d\n", int(cfg.failed));
  return cfg.failed ? 1 : 0;
}

// One tubo config against `reuse1` on the same small matrix. The tubo kernels
// quantise activations to int8 with their own scales, so the comparison is
// "same product within the activation-rounding error" (10 % of the largest
// reference magnitude), not bit equality; it catches a wrong layout, a wrong
// row or a wrong token, which move results by order one.
bool AblTuboAgrees(int cfg, int n, int k, int m, std::string* detail) {
  const auto info = AblTubo(cfg);
  const int type = info.type;
  const Shape s{"selfcheck", k, m};
  const auto bytes = WeightBytes(type, s);
  Buffer w(bytes), x(qfn_mmq_q8_1_bytes(8, s.k)), ids(8 * sizeof(int32_t));
  Buffer y_ref(std::size_t(8) * s.m * sizeof(float)),
      y_tubo(std::size_t(8) * s.m * sizeof(float));
  Buffer raw(std::size_t(8) * s.k * sizeof(float)), q(std::size_t(8) * s.k),
      ds(std::size_t(8) * (s.k / 16) * 8),
      sum(std::size_t(s.k / 256) * 9 * 8 * 4 + 64);
  WeightsAt(w.p, bytes, type);
  Inputs(x, ids, s, 8);
  const auto host = InputHost(8, s.k, 1.0F);
  Check(hipMemcpy(raw.p, host.data(), host.size() * sizeof(float),
                  hipMemcpyHostToDevice));
  const auto out_bytes = std::size_t(n) * s.m * sizeof(float);
  Check(hipMemset(y_ref.p, 0, out_bytes));
  Check(hipMemset(y_tubo.p, 0xFF, out_bytes));
  Require(qfn_mmq_dense_vec_preq_variant(
              type, w.p, x.p, static_cast<const int32_t*>(ids.p),
              static_cast<float*>(y_ref.p), s.m, n, s.k,
              int(d::Variant::kReuse1), nullptr) == 0,
          "reuse1 reference");
  Buffer laid_out_device(info.rows32 ? bytes : 1);
  const void* weights = w.p;
  if (info.rows32) {
    std::vector<std::uint8_t> native(bytes), laid_out(bytes);
    Check(hipMemcpy(native.data(), w.p, bytes, hipMemcpyDeviceToHost));
    Require(bench_tubo_repack(native.data(), laid_out.data(), s.m, s.k,
                              Bytes(type)) == 0,
            "tubo rows32 repack");
    Check(hipMemcpy(laid_out_device.p, laid_out.data(), bytes,
                    hipMemcpyHostToDevice));
    weights = laid_out_device.p;
  }
  Require(bench_tubo_prepare(cfg, static_cast<const float*>(raw.p), q.p, ds.p,
                             sum.p, n, s.k) == 0,
          "tubo prepare");
  Require(bench_tubo_launch(cfg, weights, q.p, ds.p, sum.p,
                            static_cast<float*>(y_tubo.p), n, s.m, s.k) == cfg,
          "tubo launch");
  Check(hipDeviceSynchronize());
  std::vector<float> a(std::size_t(n) * s.m), b(a.size());
  Check(hipMemcpy(a.data(), y_ref.p, out_bytes, hipMemcpyDeviceToHost));
  Check(hipMemcpy(b.data(), y_tubo.p, out_bytes, hipMemcpyDeviceToHost));
  double largest = 0, worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    largest = std::max(largest, std::fabs(double(a[i])));
  bool finite = true;
  for (std::size_t i = 0; i < a.size(); ++i) {
    finite &= std::isfinite(b[i]);
    worst = std::max(worst, std::fabs(double(a[i]) - double(b[i])));
  }
  char text[160];
  std::snprintf(text, sizeof text,
                "max_abs_diff=%.3g largest_ref=%.3g ratio=%.3g", worst, largest,
                largest > 0 ? worst / largest : 0.0);
  *detail = text;
  return finite && largest > 0 && worst <= 0.10 * largest;
}

// The dot4 gate (no timing; block 0 of the phase 2 operator commands). For
// every format the dot4 kernel instantiates, on the tails M 1/15/16/17/31 at
// K 256/768, an M tail at K 2560, an odd K256-block count and a non-Qwen K:
// (a) N = 1..8 against the production route on the same Q8_1 inputs: finite,
//     max |diff| <= 1e-3 of the largest reference value (the only difference
//     is float summation order; a wrong row, token or layout moves results by
//     order one), and the number of changed outputs is printed;
// (b) a row computes to the same bits at every width: each N < 8 call equals
//     the first N tokens of the N = 8 call, and token t alone (N1 on the input
//     offset by t rows) equals token t of the N = 8 call;
// (c) every dot4 call moved the dot4 launch counter by exactly one.
void AblDot4Gate(const std::function<void(bool, const std::string&)>& check) {
  struct Geometry {
    int k, m;
  };
  const Geometry geometries[] = {
      {256, 1},     {256, 15},  {256, 16},  {256, 17}, {256, 31},
      {768, 1},     {768, 15},  {768, 16},  {768, 17}, {768, 31},
      {2560, 4097}, {3840, 97}, {14336, 33}};
  const int dot4 = int(d::Variant::kDot4);
  int cases = 0;
  for (int type : {12, 13, 14, 21, 23}) {
    if (!d::Dot4Kernel(type))
      continue;
    for (const auto& g : geometries) {
      const Shape s{"dot4gate", g.k, g.m};
      const auto bytes = WeightBytes(type, s);
      const std::size_t token_bytes = qfn_mmq_q8_1_bytes(8, s.k) / 8;
      Buffer w(bytes), x(qfn_mmq_q8_1_bytes(8, s.k)), ids(8 * sizeof(int32_t));
      Buffer y(std::size_t(8) * s.m * sizeof(float));
      WeightsAt(w.p, bytes, type);
      Inputs(x, ids, s, 8);
      bool counted = true;
      const auto run = [&](int variant, int n, int first_token) {
        Check(hipMemset(y.p, 0xFF, std::size_t(8) * s.m * sizeof(float)));
        const auto before = qfn_mmq_dense_gemv_launch_count(type, dot4, n);
        Require(
            qfn_mmq_dense_vec_preq_variant(
                type, w.p,
                static_cast<const char*>(x.p) + first_token * token_bytes,
                static_cast<const int32_t*>(ids.p), static_cast<float*>(y.p),
                s.m, n, s.k, variant, nullptr) == 0,
            "dot4 gate launch");
        Check(hipDeviceSynchronize());
        if (variant == dot4)
          counted &=
              qfn_mmq_dense_gemv_launch_count(type, dot4, n) == before + 1;
        std::vector<float> out(std::size_t(n) * s.m);
        Check(hipMemcpy(out.data(), y.p, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        return out;
      };
      const auto same_bits = [](float a, float b) {
        return std::memcmp(&a, &b, sizeof a) == 0;
      };
      const auto full = run(dot4, 8, 0);
      double worst_ratio = 0;
      std::size_t changed = 0, outputs = 0;
      bool close = true, identical = true;
      for (int n = 1; n <= 8; ++n) {
        const auto reference =
            run(int(d::ProductionVariant(type, n, s.k)), n, 0);
        const auto candidate = run(dot4, n, 0);
        double largest = 0, worst = 0;
        for (float v : reference)
          largest = std::max(largest, std::fabs(double(v)));
        for (std::size_t i = 0; i < candidate.size(); ++i) {
          close &= std::isfinite(candidate[i]);
          worst =
              std::max(worst, std::fabs(double(candidate[i]) - reference[i]));
          changed += !same_bits(candidate[i], reference[i]);
          identical &= same_bits(candidate[i], full[i]);
        }
        outputs += candidate.size();
        close &= largest > 0 && worst <= 1e-3 * largest;
        worst_ratio =
            std::max(worst_ratio, largest > 0 ? worst / largest : 1.0);
      }
      for (int t = 0; t < 8; ++t) {
        const auto alone = run(dot4, 1, t);
        for (int row = 0; row < s.m; ++row)
          identical &= same_bits(alone[row], full[std::size_t(t) * s.m + row]);
      }
      const std::string label = std::string("dot4 gate ") + Format(type) +
                                " K=" + std::to_string(s.k) +
                                " M=" + std::to_string(s.m);
      char stats[160];
      std::snprintf(stats, sizeof stats,
                    " N1..8 vs production: max_ratio=%.3g changed=%zu/%zu",
                    worst_ratio, changed, outputs);
      check(close, label + stats);
      check(identical,
            label + " rows bit-identical across N = 1..8 and token offsets");
      check(counted, label + " every call counted as dot4");
      ++cases;
    }
  }
  check(cases > 0, "dot4 gate ran");
}

// No timing: one tiny cell per kernel and stage on a small synthetic matrix.
int AblSelfcheck() {
  Check(hipSetDevice(0));
  AblDevice();
  Buffer eviction(256U << 20);
  Check(hipMemset(eviction.p, 0, 256U << 20));
  AblWarmup(eviction);
  bool failed = false;
  const auto check = [&](bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "SELFCHECK_OK" : "SELFCHECK_FAIL",
                what.c_str());
    failed |= !ok;
  };
  // 1. Stage kernels: the returned id matches, D equals the production kernel,
  //    every stage is deterministic, and the occupancy query answers.
  for (int type : {12, 13, 14}) {
    const Shape s{"selfcheck", 768, 37};
    const int n = 5;
    const auto bytes = WeightBytes(type, s);
    Buffer w(bytes), x(qfn_mmq_q8_1_bytes(8, s.k)), ids(8 * sizeof(int32_t));
    Buffer y1(std::size_t(8) * s.m * sizeof(float)),
        y2(std::size_t(8) * s.m * sizeof(float));
    WeightsAt(w.p, bytes, type);
    Inputs(x, ids, s, 8);
    const auto out_bytes = std::size_t(n) * s.m * sizeof(float);
    std::vector<std::uint8_t> a(out_bytes), b(out_bytes), production(out_bytes);
    Check(hipMemset(y1.p, 0xFF, out_bytes));
    Require(qfn_mmq_dense_vec_preq_variant(
                type, w.p, x.p, static_cast<const int32_t*>(ids.p),
                static_cast<float*>(y1.p), s.m, n, s.k,
                int(d::Variant::kKQuantWmma), nullptr) == 0,
            "production kquant_wmma launch");
    Check(hipMemcpy(production.data(), y1.p, out_bytes, hipMemcpyDeviceToHost));
    for (int stage = 1; stage <= d::kStageCount; ++stage) {
      const std::string label =
          std::string(Format(type)) + " stage " + d::KQuantStageName(stage);
      Check(hipMemset(y1.p, 0xFF, out_bytes));
      Check(hipMemset(y2.p, 0xFF, out_bytes));
      const int id1 = qfn_mmq_kquant_stage_launch(type, stage, w.p, x.p,
                                                  static_cast<float*>(y1.p),
                                                  s.k, s.m, n, 0, nullptr);
      const int id2 = qfn_mmq_kquant_stage_launch(type, stage, w.p, x.p,
                                                  static_cast<float*>(y2.p),
                                                  s.k, s.m, n, 1024, nullptr);
      Check(hipDeviceSynchronize());
      check(id1 == type * 256 + stage && id2 == id1, label + " kernel id");
      Check(hipMemcpy(a.data(), y1.p, out_bytes, hipMemcpyDeviceToHost));
      Check(hipMemcpy(b.data(), y2.p, out_bytes, hipMemcpyDeviceToHost));
      if (stage == d::kStageLdsNoBarrier) {
        // B0 reads, without a barrier, tile words that other waves write in
        // the same step (wave w reads row `sub` written by wave sub/2): its
        // checksum races by construction. Determinism is not expected; the
        // rung is accepted when the outputs are finite floats.
        bool finite = true;
        for (const auto* bytes : {&a, &b})
          for (std::size_t i = 0; i + 4 <= bytes->size(); i += 4) {
            float v;
            std::memcpy(&v, bytes->data() + i, 4);
            finite &= std::isfinite(v);
          }
        check(finite, label +
                          " nondeterministic by design (LDS read-back races "
                          "with the tile writes, no barrier); outputs finite");
      } else {
        check(a == b, label + " deterministic (also with 1 KiB extra LDS)");
      }
      if (stage == d::kStageFull)
        check(a == production, label + " equals production kquant_wmma bytes");
      check(qfn_mmq_kquant_stage_occupancy(type, stage, 0) >= 1,
            label + " occupancy query");
    }
  }
  // 2. Full cells (sets 4, 1 sample) through the shared cell function, which
  //    proves the counter logic for each kernel kind.
  struct Case {
    int type;
    const char* kernel;
    int n;
    bool expect_abort;
    int m = 37;
    int k = 768;
    bool expect_skip = false;  // a deliberate skip (no cell printed)
  };
  const Case cases[] = {
      {12, "production", 6, false},
      {12, "reuse1", 3, false},
      {12, "existing", 3, false},
      {13, "stage:A", 4, false},
      {13, "stage:A0", 4, false},
      {13, "stage:B", 4, false},
      {13, "stage:B0", 4, false},
      {13, "stage:C1", 4, false},
      {13, "stage:C2", 4, false},
      {13, "stage:C3", 4, false},
      {13, "stage:D", 4, false},
      {14, "stage:D", 1, false},
      {14, "kquant_wmma", 6, false},
      {14, "kquant_wmma", 2, true},  // rerouted: must abort
      {14, "reuse1_loads", 1, false},
      {21, "reuse1", 4, false},
      {21, "reuse1_loads", 1, false},
      {23, "iq4_wmma", 5, false},
      {23, "reuse1", 5, false},
      {8, "q8_dp4a", 6, false},
      {8, "q8_wmma", 6, false},
      {12, "tubo:q4_native_k4", 4, false, 64},
      {12, "tubo:q4_native_k4h", 8, false, 64},
      {12, "tubo:q4_rows32_k4h", 8, false, 64},
      {12, "tubo:q4_rows32_r2k8hp", 4, false, 64},
      {12, "dot4", 8, false, 64},
      {12, "tubo:q4_rows32_k4h", 4, false, 37, 768, true},  // M % 32: skipped
      {13, "tubo:q5_native_k4h", 2, false, 64},
      {13, "tubo:q5_rows32_r2k8hp", 8, false, 64},
      {13, "dot4", 3, false, 37},
      {14, "tubo:q6_rows32_r2k8hp", 4, false, 64},
      {14, "dot4", 1, false, 64},
      {14, "dot4", 7, false, 37},
      {23, "tubo:iq4_rows32_r2k4hp", 8, false, 64},
      {23, "dot4", 4, false, 64},
      {21, "dot4", 8, false, 64},
      {20, "dot4", 4, true},  // IQ4_NL is not a dot4 format: must abort
      // The shapes the timed run meets first: K = 2560 / 3840 / 14336 and an
      // M tail (4097 for the kernels that take it, 4128 for rows32).
      {12, "stage:D", 8, false, 4097, 2560},
      {12, "stage:A", 8, false, 4097, 2560},
      {14, "stage:D", 8, false, 4097, 2560},
      {12, "production", 8, false, 4097, 2560},
      {12, "tubo:q4_native_k4", 8, false, 4097, 2560},
      {12, "tubo:q4_native_k4h", 8, false, 4097, 2560},
      {12, "tubo:q4_rows32_k4h", 8, false, 4128, 2560},
      {12, "tubo:q4_rows32_r2k8hp", 8, false, 4128, 2560},
      {12, "dot4", 8, false, 4097, 2560},
      {13, "tubo:q5_native_k4h", 8, false, 4097, 2560},
      {13, "tubo:q5_rows32_r2k8hp", 8, false, 4128, 2560},
      {13, "dot4", 5, false, 4097, 2560},
      {14, "tubo:q6_rows32_r2k8hp", 8, false, 4128, 2560},
      {14, "dot4", 8, false, 4097, 2560},
      {23, "tubo:iq4_rows32_r2k4hp", 8, false, 4128, 2560},
      {23, "dot4", 8, false, 4097, 2560},
      {21, "dot4", 8, false, 4097, 2560},
      {13, "dot4", 6, false, 97, 14336},
      {14, "dot4", 2, false, 4097, 3840},
      {12, "tubo:q4_native_k4h", 4, false, 4097, 3840},
      {12, "tubo:q4_rows32_r2k8hp", 4, false, 4128, 3840},
      {12, "tubo:q4_native_k4h", 8, false, 97, 14336},
      {12, "tubo:q4_rows32_k4h", 8, false, 96, 14336},
      {13, "tubo:q5_rows32_r2k8hp", 8, false, 96, 14336},
      {14, "tubo:q6_rows32_r2k8hp", 8, false, 96, 14336},
      {12, "stage:D", 8, false, 97, 14336}};
  int tubo_cells = 0;
  for (const auto& c : cases) {
    AblConfig cfg;
    cfg.type = c.type;
    cfg.kernels = {AblParseKernel(c.kernel)};
    cfg.widths = {c.n};
    cfg.sets = 4;
    cfg.samples = 1;
    const Shape s{"selfcheck", c.k, c.m};
    const int cells =
        MeasureCell(cfg, eviction, "synthetic", "selfcheck", s, nullptr);
    if (cfg.kernels[0].kind == AblKind::kTubo)
      tubo_cells += cells;
    const std::string label = std::string(Format(c.type)) + " " + c.kernel +
                              " N=" + std::to_string(c.n) +
                              " K=" + std::to_string(c.k) +
                              " M=" + std::to_string(c.m);
    check(cfg.failed == c.expect_abort,
          label + (c.expect_abort ? " aborts (reroute detected)"
                                  : " verified, output checked"));
    // A case that printed no cell proves nothing: a silent skip is a failure
    // unless the case is the deliberate skip.
    check(c.expect_abort || (c.expect_skip ? cells == 0 : cells == 1),
          label + (c.expect_skip ? " skipped on purpose" : " ran one cell"));
  }
  check(tubo_cells > 0,
        "at least one tubo case ran a cell (all-skip is a failure)");
  // 3. The tubo controls compute the product: compare each config with
  //    reuse1 on the same weights (int8-activation tolerance, not exactness)
  //    at K=768, and at the larger K and M tail the timed run meets first.
  int agreements = 0;
  for (int cfg = 0; cfg < bench_tubo_config_count(); ++cfg) {
    const bool rows32 = AblTubo(cfg).rows32 != 0;
    struct Geometry {
      int k, m;
      std::vector<int> widths;
    };
    for (const Geometry& g : {Geometry{768, 64, {1, 2, 4, 8}},
                              Geometry{2560, rows32 ? 4128 : 4097, {2, 8}},
                              Geometry{3840, rows32 ? 4128 : 4097, {8}},
                              Geometry{14336, rows32 ? 96 : 97, {8}}})
      for (int n : g.widths) {
        std::string detail;
        const bool ok = AblTuboAgrees(cfg, n, g.k, g.m, &detail);
        ++agreements;
        check(ok, std::string("tubo:") + AblTubo(cfg).name +
                      " N=" + std::to_string(n) + " K=" + std::to_string(g.k) +
                      " M=" + std::to_string(g.m) +
                      " agrees with reuse1: " + detail);
      }
  }
  check(agreements > 0, "tubo agreement checks ran");
  // 4. The dot4 gate.
  AblDot4Gate(check);
  std::printf(failed ? "SELFCHECK_FAILED\n" : "SELFCHECK_PASS\n");
  return failed ? 1 : 0;
}
