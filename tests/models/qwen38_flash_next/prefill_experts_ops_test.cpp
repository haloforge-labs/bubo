#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace gpu = qfn::rocm;
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Hip(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}
template<class T>
struct Device {
  T* p = nullptr;
  explicit Device(std::size_t n) { Hip(hipMalloc(&p, n * sizeof(T))); }
  ~Device() { (void)hipFree(p); }
};
void RouterCases() {
  constexpr unsigned tokens = 9, experts = 512, k = 10;
  std::vector<float> logits(tokens * experts, 0);
  for (unsigned t = 1; t < tokens; ++t)
    for (unsigned e = 0; e < experts; ++e)
      logits[t * experts + e] = float((e * 13 + t) % 73) * 0.125f;
  // Finite logits whose FP32 softmax saturates to [1,0,...].
  for (unsigned e = 0; e < experts; ++e)
    logits[(tokens - 1) * experts + e] = e == 0 ? 10000.0f : -10000.0f;
  Device<float> d_logits(logits.size()), weights(tokens * k);
  Device<int> ids(tokens * k);
  Device<std::uint32_t> counts(experts);
  Hip(hipMemcpy(d_logits.p, logits.data(), logits.size() * 4,
                hipMemcpyHostToDevice));
  gpu::RouterTopK(d_logits.p, experts, ids.p, weights.p, tokens, experts, k,
                  nullptr);
  std::vector<int> base_ids(tokens * k), out_ids(tokens * k);
  std::vector<float> base_weights(tokens * k), out_weights(tokens * k);
  Hip(hipMemcpy(base_ids.data(), ids.p, base_ids.size() * 4,
                hipMemcpyDeviceToHost));
  Hip(hipMemcpy(base_weights.data(), weights.p, base_weights.size() * 4,
                hipMemcpyDeviceToHost));
  using M = qfn::PrefillExpertsMode;
  for (auto mode : {M::kRenorm, M::kRef10, M::kFold, M::kMass}) {
    for (auto cap : {1u, 2u, 4u, 6u, 8u, 10u}) {
      for (auto tau : {0.25f, 0.75f, 1.0f}) {
        gpu::RouterTopK(d_logits.p, experts, ids.p, weights.p, tokens, experts,
                        k, nullptr, cap, mode, tau, 1.25f);
        Hip(hipMemcpy(out_ids.data(), ids.p, out_ids.size() * 4,
                      hipMemcpyDeviceToHost));
        Hip(hipMemcpy(out_weights.data(), weights.p, out_weights.size() * 4,
                      hipMemcpyDeviceToHost));
        unsigned active = 0;
        for (unsigned t = 0; t < tokens; ++t) {
          unsigned keep = cap;
          if (mode == M::kMass) {
            const float threshold =
                tau == 1 ? tau - 4 * std::numeric_limits<float>::epsilon()
                         : tau;
            float mass = 0;
            keep = 0;
            do {
              mass += base_weights[t * k + keep++];
            } while (keep < cap && mass < threshold);
          }
          active += keep;
          float sum = 0;
          for (unsigned s = 0; s < keep; ++s)
            sum += base_weights[t * k + s];
          for (unsigned s = 0; s < k; ++s) {
            const auto i = t * k + s;
            Require(out_ids[i] == (s < keep ? base_ids[i] : -1),
                    "masked id/ranking");
            float expected = s >= keep ? 0 : base_weights[i];
            if (mode == M::kRenorm || mode == M::kMass)
              expected /= sum;
            if (mode == M::kFold)
              expected *= 1.25f;
            Require(std::abs(out_weights[i] - expected) < 2e-6f,
                    "variant weight");
            if (cap == k && (mode == M::kRenorm || mode == M::kRef10))
              Require(std::bit_cast<unsigned>(out_weights[i]) ==
                          std::bit_cast<unsigned>(base_weights[i]),
                      "keep=k changed bits");
          }
        }
        gpu::ExpertCounts(ids.p, counts.p, tokens, experts, k, nullptr);
        std::vector<std::uint32_t> actual(experts);
        Hip(hipMemcpy(actual.data(), counts.p,
                      actual.size() * sizeof(actual[0]),
                      hipMemcpyDeviceToHost));
        unsigned counted = 0;
        for (auto n : actual)
          counted += n;
        Require(counted == active, "live counts differ from selected prefix");
      }
    }
  }
  for (unsigned s = 0; s < k; ++s)
    Require(base_ids[s] == int(s), "equal-logit ties");
}

template<class T>
void LeanClearing(bool f16) {
  Device<int> ids(3);
  Device<T> rows(3 * 16);
  // Reuse the same poisoned scratch with a different kept set per invocation.
  for (const auto routing :
       {std::vector<int>{7, -1, 9}, std::vector<int>{-1, 8, -1}}) {
    Hip(hipMemcpy(ids.p, routing.data(), 3 * sizeof(int),
                  hipMemcpyHostToDevice));
    std::vector<T> output(48, static_cast<T>(2.0f));
    for (unsigned r = 0; r < 3; ++r)
      if (routing[r] < 0)
        for (unsigned c = 0; c < 16; ++c)
          output[r * 16 + c] = static_cast<T>(NAN);
    Hip(hipMemcpy(rows.p, output.data(), 48 * sizeof(T),
                  hipMemcpyHostToDevice));
    gpu::PrepareRoutedExpertRows(rows.p, f16, ids.p, 3, 16, nullptr,
                                 qfn::PrefillExpertsAudit::kLean);
    Hip(hipMemcpy(output.data(), rows.p, 48 * sizeof(T),
                  hipMemcpyDeviceToHost));
    for (unsigned c = 0; c < 16; ++c) {
      float combined = 0;
      unsigned kept = 0;
      for (unsigned r = 0; r < 3; ++r) {
        const auto value = static_cast<float>(output[r * 16 + c]);
        Require(value == (routing[r] < 0 ? 0 : 2),
                "lean clearing/preservation");
        combined += (routing[r] < 0 ? 0.0f : 1.0f) * value;
        kept += routing[r] >= 0;
      }
      Require(std::isfinite(combined) && combined == 2 * kept,
              "lean epilogue stand-in reads stale poison");
    }
  }
}

template<class T>
void RowEvidence(bool f16) {
  // Missing output writes must fail, and dropped poison must be cleared.
  Device<int> ids(3);
  Device<T> rows(3 * 16);
  Device<qfn::RoutedExpertAudit> counts(1);
  const int routing[3] = {7, -1, 9};
  Hip(hipMemcpy(ids.p, routing, sizeof(routing), hipMemcpyHostToDevice));
  Hip(hipMemset(rows.p, 0xff, 3 * 16 * sizeof(T)));
  Hip(hipMemset(counts.p, 0, sizeof(qfn::RoutedExpertAudit)));
  gpu::PrepareRoutedExpertRows(rows.p, f16, ids.p, 3, 16, nullptr);
  const T written = static_cast<T>(2.0f);
  Hip(hipMemcpy(rows.p, &written, sizeof(written), hipMemcpyHostToDevice));
  gpu::CountRoutedExpertRows(rows.p, f16, ids.p, 3, 16, counts.p, nullptr);
  qfn::RoutedExpertAudit actual;
  Hip(hipMemcpy(&actual, counts.p, sizeof(actual), hipMemcpyDeviceToHost));
  Require(actual.ExecutedRows() == 1 && actual.InvalidRows() == 1,
          "missing row not detected");
  std::vector<T> output(48);
  Hip(hipMemcpy(output.data(), rows.p, output.size() * sizeof(T),
                hipMemcpyDeviceToHost));
  for (unsigned c = 0; c < 16; ++c)
    Require(static_cast<float>(output[16 + c]) == 0, "dropped stale row");
  Hip(hipMemset(counts.p, 0, sizeof(qfn::RoutedExpertAudit)));
  gpu::PrepareRoutedExpertRows(rows.p, f16, ids.p, 3, 16, nullptr);
  Hip(hipMemcpy(rows.p, &written, sizeof(written), hipMemcpyHostToDevice));
  Hip(hipMemcpy(rows.p + 32, &written, sizeof(written), hipMemcpyHostToDevice));
  const T zero = static_cast<T>(0.0f);
  // Even a zero write to a dropped first column proves an ignored mask.
  Hip(hipMemcpy(rows.p + 16, &zero, sizeof(zero), hipMemcpyHostToDevice));
  gpu::CountRoutedExpertRows(rows.p, f16, ids.p, 3, 16, counts.p, nullptr);
  Hip(hipMemcpy(&actual, counts.p, sizeof(actual), hipMemcpyDeviceToHost));
  Require(actual.ExecutedRows() == 3 && actual.InvalidRows() == 1,
          "write to dropped row not detected");
}

void AuditReduction() {
  constexpr unsigned slots = 513;
  Device<int> ids(slots);
  Device<float> rows(slots);
  Device<qfn::RoutedExpertAudit> audit(1);
  std::vector<int> routing(slots, 7);
  std::vector<float> output(slots, 2);
  output.back() = INFINITY;
  Hip(hipMemcpy(ids.p, routing.data(), slots * sizeof(int),
                hipMemcpyHostToDevice));
  Hip(hipMemcpy(rows.p, output.data(), slots * sizeof(float),
                hipMemcpyHostToDevice));
  Hip(hipMemset(audit.p, 0, sizeof(qfn::RoutedExpertAudit)));
  gpu::CountRoutedExpertRows(rows.p, false, ids.p, slots, 1, audit.p, nullptr);
  qfn::RoutedExpertAudit actual;
  Hip(hipMemcpy(&actual, audit.p, sizeof(actual), hipMemcpyDeviceToHost));
  Require(actual.ExecutedRows() == slots && actual.InvalidRows() == 1,
          "block audit reduction lost ragged tail/nonfinite first scalar");
}
}  // namespace
int main() {
  try {
    RouterCases();
    LeanClearing<float>(false);
    LeanClearing<__half>(true);
    RowEvidence<float>(false);
    RowEvidence<__half>(true);
    AuditReduction();
    std::cout << "PASS: router pruning variants, default bits, row evidence\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
