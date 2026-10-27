// SPDX-License-Identifier: MIT
// Lane split selectively adapted from the MIT mikealanni/gufo fork.
#include "linear_z.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "layer_worker.hpp"
#include "request_stats.hpp"

namespace gufo::npu {
std::shared_ptr<const LinearZWeights> PrepareLinearZ(const Q8Matrix& qkv,
                                                     const Q8Matrix& gate,
                                                     unsigned slices) {
  if (slices == 0 || slices > 12)
    throw std::invalid_argument("NPU linear slice count outside 1..12");
  // Values 7..12 include trailing QKV channels as well as all z rows.
  // Their first consumer is convolution, so integration fences before GDN.
  constexpr std::size_t row_bytes = kK / 32 * 34;
  if (slices > 12 || qkv.type != core::GgmlType::kQ8_0 ||
      gate.type != core::GgmlType::kQ8_0 || qkv.cols != kK || gate.cols != kK ||
      qkv.rows != 10240 || gate.rows != 6144 ||
      qkv.bytes.size() != qkv.rows * row_bytes ||
      gate.bytes.size() != gate.rows * row_bytes || !qkv.bytes.data() ||
      !gate.bytes.data())
    return {};
  std::vector<std::byte> stacked(16384 * row_bytes);
  std::memcpy(stacked.data(), qkv.bytes.data(), qkv.bytes.size());
  std::memcpy(stacked.data() + qkv.bytes.size(), gate.bytes.data(),
              gate.bytes.size());
  Q8Matrix matrix{core::GgmlType::kQ8_0, 16384, kK, stacked};
  auto result = std::make_shared<LinearZWeights>();
  const auto plan = LinearPlan(matrix, slices);
  result->gpu_rows = plan.front().row0;
  for (const auto& slice : plan)
    result->b.push_back(bfp::EncodeB(StageQ8(matrix, slice)));
  return result;
}

bool LinearZQualified(unsigned tokens, unsigned capacity, bool prefill,
                      bool speculative, bool projections_ready,
                      const LinearZWeights* weights) noexcept {
  return prefill && !speculative && !projections_ready && weights &&
         !weights->b.empty() && weights->b.size() <= 12 &&
         weights->gpu_rows == 16384 - weights->b.size() * kN &&
         QualifiedChunk(tokens, capacity);
}

LinearZWork::LinearZWork(std::shared_ptr<const LinearZWeights> weights,
                         std::vector<std::vector<std::uint8_t>> a, int layer,
                         LinearDispatch dispatch,
                         std::function<void(std::string)> reject,
                         BatchDispatch batch, PackedConsumer packed,
                         BorrowedActivations borrowed) {
  LayerInputs inputs{std::move(a), std::move(borrowed)};
  if (!weights || weights->b.empty() || weights->b.size() > 12 ||
      weights->gpu_rows != 16384 - weights->b.size() * kN ||
      inputs.size() == 0 || inputs.size() > 4 || !dispatch)
    throw std::invalid_argument("NPU linear work geometry invalid");
  for (std::size_t g = 0; g < inputs.size(); ++g)
    if (inputs[g].size() != bfp::ABytes)
      throw std::invalid_argument("NPU linear A size invalid");
  for (const auto& bytes : weights->b)
    if (bytes.size() != bfp::BBytes)
      throw std::invalid_argument("NPU linear B size invalid");
  auto promise = std::make_shared<std::promise<LinearZResult>>();
  result_ = promise->get_future().share();
  auto submit = inputs.borrowed.submit;
  auto command = [weights = std::move(weights), inputs = std::move(inputs),
                  layer, dispatch = std::move(dispatch),
                  reject = std::move(reject), promise, batch = std::move(batch),
                  packed = std::move(packed)]() {
    LinearZResult result;
    try {
      if (inputs.borrowed.ready)
        inputs.borrowed.ready();
      const auto columns = weights->b.size() * kN;
      result.packed = bool(packed);
      if (!packed)
        result.tail.resize(inputs.size() * kTokens * columns);
      for (std::size_t g = 0; g < inputs.size(); ++g) {
        if (batch) {
          const bool ok =
              batch(std::as_bytes(inputs[g]), weights->b,
                    [&](std::size_t s, std::span<const std::byte> output) {
                      if (packed) {
                        packed(g, s, output);
                        return;
                      }
                      bfp::DecodeC(
                          {reinterpret_cast<const std::uint8_t*>(output.data()),
                           output.size()},
                          std::span(result.tail).subspan(g * kTokens * columns),
                          columns, s * kN, kN, false);
                    });
          if (!ok)
            throw std::runtime_error("NPU linear batch failed");
          continue;
        }

        for (std::size_t s = 0; s < weights->b.size(); ++s) {
          DispatchInfo info;
          info.layer = layer;
          info.slice = static_cast<int>(s);
          info.lane = "linear-z";
          info.group = static_cast<int>(g);
          auto r = dispatch(std::as_bytes(inputs[g]),
                            std::as_bytes(std::span(weights->b[s])), info);
          result.dma_uncertain = r.dma_uncertain;
          if (!r.submitted || !r.executed || !r.healthy || !r.result_valid ||
              r.dma_uncertain || r.output.size() != bfp::CBytes)
            throw std::runtime_error("NPU linear result rejected: " + r.error);
          bfp::DecodeC({reinterpret_cast<const std::uint8_t*>(r.output.data()),
                        r.output.size()},
                       std::span(result.tail).subspan(g * kTokens * columns),
                       columns, s * kN, kN, false);
        }
      }
      if (!std::all_of(result.tail.begin(), result.tail.end(),
                       [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("NPU linear output is not finite");
      result.valid = true;
    } catch (const std::exception& e) {
      result.error = e.what();
      result.tail.clear();
    } catch (...) {
      result.error = "NPU linear non-standard dispatch/decode failure";
      result.tail.clear();
    }
    if (!result.valid && reject) {
      try {
        reject(result.error);
      } catch (...) {
      }
    }
    promise->set_value(std::move(result));
  };
  if (submit)
    submit(std::move(command));
  else
    EnqueueLayerWork(std::move(command));
}

LinearZResult LinearZWork::Finish(std::chrono::milliseconds deadline) const {
  auto& stats = Stats();
  stats.finish_calls.fetch_add(1, std::memory_order_relaxed);
  if (deadline.count() <= 0) {
    stats.failures.fetch_add(1, std::memory_order_relaxed);
    return {false, false, "NPU linear first-consumer deadline", {}};
  }
  if (result_.wait_for(std::chrono::milliseconds(0)) !=
      std::future_status::ready) {
    const auto t0 = NowNs();
    const bool ready = result_.wait_for(deadline) == std::future_status::ready;
    stats.finish_not_ready.fetch_add(1, std::memory_order_relaxed);
    stats.finish_wait_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    if (!ready) {
      stats.failures.fetch_add(1, std::memory_order_relaxed);
      return {false, false, "NPU linear first-consumer deadline", {}};
    }
  }
  return result_.get();
}
std::unique_ptr<LinearZWork> BeginLinearZ(
    std::shared_ptr<const LinearZWeights> weights, LinearZTransfer& transfer,
    int layer, LinearDispatch dispatch, std::function<void(std::string)> reject,
    BatchDispatch batch) {
  NoteBegin(layer, true);
  auto borrowed = transfer.BorrowInput();
  auto input = borrowed.groups.empty()
                   ? transfer.FenceInput()
                   : std::vector<std::vector<std::uint8_t>>{};
  return std::make_unique<LinearZWork>(
      std::move(weights), std::move(input), layer, std::move(dispatch),
      std::move(reject), std::move(batch), transfer.PackedOutputConsumer(),
      std::move(borrowed));
}
LinearZResult FinishLinearZ(const LinearZWork& work, LinearZTransfer& transfer,
                            unsigned columns,
                            std::chrono::milliseconds deadline) {
  WaitLane() = 0;
  if (transfer.WaitOnStream(work.Result(), deadline)) {
    const auto t0 = NowNs();
    transfer.FencePackedOutput(columns);
    Stats().fence_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    return {true, false, {}, {}, true};
  }
  auto result = work.Finish(deadline);
  if (!result.valid)
    return result;
  const auto fence_t0 = NowNs();
  try {
    if (result.packed)
      transfer.FencePackedOutput(columns);
    else
      transfer.FenceOutput(result.tail, columns);
  } catch (const std::exception& e) {
    result.valid = false;
    result.error = e.what();
    result.tail.clear();
  } catch (...) {
    result.valid = false;
    result.error = "NPU output transfer non-standard failure";
    result.tail.clear();
  }
  Stats().fence_ns.fetch_add(NowNs() - fence_t0, std::memory_order_relaxed);
  if (!result.valid)
    Stats().failures.fetch_add(1, std::memory_order_relaxed);
  return result;
}
}  // namespace gufo::npu
