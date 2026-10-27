// SPDX-License-Identifier: MIT
// Selective lane geometry from the MIT mikealanni/gufo fork; private CPU
// copies.
#include "projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "layer_worker.hpp"
#include "request_stats.hpp"
namespace gufo::npu {
namespace {
bool Matrix(const Q8Matrix& q, unsigned rows) {
  return q.type == core::GgmlType::kQ8_0 && q.cols == kK && q.rows == rows &&
         q.bytes.data() && q.bytes.size() == std::size_t{rows} * (kK / 32 * 34);
}
}  // namespace
std::shared_ptr<const ProjectionWeights> PrepareQueryGate(
    const Q8Matrix& query, const Q8Matrix& key, const Q8Matrix& value,
    unsigned slices, unsigned query_heads, unsigned kv_heads,
    unsigned head_dim) {
  if (slices > 12)
    throw std::invalid_argument("NPU query slices outside 0..12");
  if (!slices)
    return {};
  if (query_heads != 24 || kv_heads != 2 || head_dim != 256 ||
      !Matrix(query, 12288) || !Matrix(key, 512) || !Matrix(value, 512))
    return {};
  // Stage only query/gate rows; K/V stay GPU. GatePlan verifies the complete
  // stacked shape without copying the K/V payload into NPU B streams.
  std::vector<std::byte> packed(13312 * (kK / 32 * 34));
  std::memcpy(packed.data(), query.bytes.data(), query.bytes.size());
  Q8Matrix q{core::GgmlType::kQ8_0, 13312, kK, packed};
  const auto plan = GatePlan(q, slices, query_heads, kv_heads, head_dim);
  auto w = std::make_shared<ProjectionWeights>();
  w->rows = slices * kN;
  for (const auto& s : plan)
    w->b.push_back(bfp::EncodeB(StageQ8(q, s)));
  return w;
}
std::shared_ptr<const ProjectionWeights> PrepareSharedExpert(
    const Q8Matrix& gate, const Q8Matrix& up, bool enabled) {
  if (!enabled)
    return {};
  if (!gate.rows || gate.rows > kN || up.rows != gate.rows ||
      !Matrix(gate, static_cast<unsigned>(gate.rows)) ||
      !Matrix(up, static_cast<unsigned>(gate.rows)))
    return {};
  const auto plan = SharedPlan(gate, up, true);
  auto w = std::make_shared<ProjectionWeights>();
  w->lane = Lane::SharedGate;
  w->rows = static_cast<unsigned>(gate.rows);
  w->b.push_back(bfp::EncodeB(StageQ8(gate, plan[0])));
  w->b.push_back(bfp::EncodeB(StageQ8(up, plan[1])));
  return w;
}
bool ProjectionGeometry(const ProjectionWeights& w) noexcept {
  return (w.lane == Lane::QueryGate && QueryGpuRows(w.rows) != 0 &&
          w.b.size() == w.rows / kN) ||
         (w.lane == Lane::SharedGate && w.rows > 0 && w.rows <= kN &&
          w.b.size() == 2);
}
bool ProjectionQualified(unsigned tokens, unsigned capacity, bool prefill,
                         bool speculative, bool prepared, bool last_only,
                         const ProjectionWeights* w) noexcept {
  return prefill && !speculative && !prepared && !last_only && w &&
         ProjectionGeometry(*w) && QualifiedChunk(tokens, capacity);
}
ProjectionWork::ProjectionWork(std::shared_ptr<const ProjectionWeights> w,
                               std::vector<std::vector<std::uint8_t>> a,
                               int layer, LinearDispatch dispatch,
                               std::function<void(std::string)> reject,
                               BatchDispatch batch, PackedConsumer packed,
                               BorrowedActivations borrowed) {
  if (w)
    lane_ = w->lane;
  LayerInputs inputs{std::move(a), std::move(borrowed)};
  if (!w || !ProjectionGeometry(*w) || inputs.size() == 0 ||
      inputs.size() > 4 || !dispatch)
    throw std::invalid_argument("NPU projection work geometry invalid");
  for (std::size_t g = 0; g < inputs.size(); ++g)
    if (inputs[g].size() != bfp::ABytes)
      throw std::invalid_argument("NPU projection A size");
  for (const auto& v : w->b)
    if (v.size() != bfp::BBytes)
      throw std::invalid_argument("NPU projection B size");
  auto promise = std::make_shared<std::promise<LinearZResult>>();
  result_ = promise->get_future().share();
  auto submit = inputs.borrowed.submit;
  auto command = [w = std::move(w), inputs = std::move(inputs), layer,
                  dispatch = std::move(dispatch), reject = std::move(reject),
                  promise, batch = std::move(batch),
                  packed = std::move(packed)] {
    LinearZResult result;
    try {
      if (inputs.borrowed.ready)
        inputs.borrowed.ready();
      const unsigned cols = w->lane == Lane::QueryGate ? w->rows : 2 * w->rows;
      result.packed = bool(packed);
      if (!packed)
        result.tail.resize(inputs.size() * kTokens * cols);
      for (std::size_t g = 0; g < inputs.size(); ++g) {
        if (batch) {
          const unsigned valid = w->lane == Lane::QueryGate ? kN : w->rows;
          const bool ok =
              batch(std::as_bytes(inputs[g]), w->b,
                    [&](std::size_t s, std::span<const std::byte> output) {
                      if (packed) {
                        packed(g, s, output);
                        return;
                      }
                      bfp::DecodeC(
                          {reinterpret_cast<const std::uint8_t*>(output.data()),
                           output.size()},
                          std::span(result.tail).subspan(g * kTokens * cols),
                          cols, s * valid, valid, false);
                    });
          if (!ok)
            throw std::runtime_error("NPU projection batch failed");
          continue;
        }
        for (std::size_t s = 0; s < w->b.size(); ++s) {
          DispatchInfo info;
          info.layer = layer;
          info.group = static_cast<int>(g);
          info.slice = static_cast<int>(s);
          info.lane = w->lane == Lane::QueryGate
                          ? "attention-q-gate"
                          : (s == 0 ? "shared-gate" : "shared-up");
          auto r = dispatch(std::as_bytes(inputs[g]),
                            std::as_bytes(std::span(w->b[s])), info);
          result.dma_uncertain = r.dma_uncertain;
          if (!r.submitted || !r.executed || !r.healthy || !r.result_valid ||
              r.dma_uncertain || r.output.size() != bfp::CBytes)
            throw std::runtime_error("NPU projection result rejected: " +
                                     r.error);
          const unsigned valid = w->lane == Lane::QueryGate ? kN : w->rows;
          bfp::DecodeC({reinterpret_cast<const std::uint8_t*>(r.output.data()),
                        r.output.size()},
                       std::span(result.tail).subspan(g * kTokens * cols), cols,
                       s * valid, valid, false);
        }
      }
      if (!std::all_of(result.tail.begin(), result.tail.end(),
                       [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("NPU projection output is not finite");
      result.valid = true;
    } catch (const std::exception& e) {
      result.error = e.what();
      result.tail.clear();
    } catch (...) {
      result.error = "NPU projection non-standard failure";
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
LinearZResult ProjectionWork::Finish(std::chrono::milliseconds deadline) const {
  auto& stats = Stats();
  stats.finish_calls.fetch_add(1, std::memory_order_relaxed);
  if (deadline.count() <= 0) {
    stats.failures.fetch_add(1, std::memory_order_relaxed);
    return {false, false, "NPU projection first-consumer deadline", {}};
  }
  if (result_.wait_for(std::chrono::milliseconds(0)) !=
      std::future_status::ready) {
    const auto t0 = NowNs();
    const bool ready = result_.wait_for(deadline) == std::future_status::ready;
    stats.finish_not_ready.fetch_add(1, std::memory_order_relaxed);
    stats.finish_wait_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    if (!ready) {
      stats.failures.fetch_add(1, std::memory_order_relaxed);
      return {false, false, "NPU projection first-consumer deadline", {}};
    }
  }
  return result_.get();
}
std::unique_ptr<ProjectionWork> BeginProjection(
    std::shared_ptr<const ProjectionWeights> w, LinearZTransfer& t, int layer,
    LinearDispatch d, std::function<void(std::string)> reject,
    BatchDispatch batch) {
  NoteBegin(layer, false);
  (w && w->lane == Lane::SharedGate ? Stats().shared_begins
                                    : Stats().qgate_begins)
      .fetch_add(1, std::memory_order_relaxed);
  auto borrowed = t.BorrowInput();
  auto a = borrowed.groups.empty() ? t.FenceInput()
                                   : std::vector<std::vector<std::uint8_t>>{};
  return std::make_unique<ProjectionWork>(
      std::move(w), std::move(a), layer, std::move(d), std::move(reject),
      std::move(batch), t.PackedOutputConsumer(), std::move(borrowed));
}
LinearZResult FinishProjection(const ProjectionWork& w, LinearZTransfer& t,
                               unsigned cols,
                               std::chrono::milliseconds deadline) {
  WaitLane() = w.lane() == Lane::SharedGate ? 2 : 1;
  if (t.WaitOnStream(w.Result(), deadline)) {
    const auto t0 = NowNs();
    t.FencePackedOutput(cols);
    Stats().fence_ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
    return {true, false, {}, {}, true};
  }
  auto r = w.Finish(deadline);
  if (!r.valid)
    return r;
  const auto fence_t0 = NowNs();
  try {
    if (r.packed)
      t.FencePackedOutput(cols);
    else
      t.FenceOutput(r.tail, cols);
  } catch (const std::exception& e) {
    r.valid = false;
    r.error = e.what();
    r.tail.clear();
  } catch (...) {
    r.valid = false;
    r.error = "NPU output transfer non-standard failure";
    r.tail.clear();
  }
  Stats().fence_ns.fetch_add(NowNs() - fence_t0, std::memory_order_relaxed);
  if (!r.valid)
    Stats().failures.fetch_add(1, std::memory_order_relaxed);
  return r;
}
}  // namespace gufo::npu
