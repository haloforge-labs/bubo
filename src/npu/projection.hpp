// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_PROJECTION_HPP_
#define GUFO_NPU_PROJECTION_HPP_
#include "linear_z.hpp"
#include "projection_geometry.hpp"
namespace gufo::npu {
// CPU-owned streams only; query rows are paired q/gate, shared rows gate/up.
struct ProjectionWeights {
  Lane lane = Lane::QueryGate;
  unsigned rows = 0;
  std::vector<std::vector<std::uint8_t>> b;
};
std::shared_ptr<const ProjectionWeights> PrepareQueryGate(
    const Q8Matrix& query, const Q8Matrix& key, const Q8Matrix& value,
    unsigned slices, unsigned query_heads, unsigned kv_heads,
    unsigned head_dim);
std::shared_ptr<const ProjectionWeights> PrepareSharedExpert(
    const Q8Matrix& gate, const Q8Matrix& up, bool enabled);
bool ProjectionGeometry(const ProjectionWeights& weights) noexcept;
bool ProjectionQualified(unsigned tokens, unsigned capacity, bool prefill,
                         bool speculative, bool prepared, bool last_only,
                         const ProjectionWeights* weights) noexcept;
// The persistent worker owns every input and the dispatch capability.
class ProjectionWork {
public:
  ProjectionWork(std::shared_ptr<const ProjectionWeights> weights,
                 std::vector<std::vector<std::uint8_t>> a, int layer,
                 LinearDispatch dispatch,
                 std::function<void(std::string)> reject = {},
                 BatchDispatch batch = {}, PackedConsumer packed = {},
                 BorrowedActivations borrowed = {});
  LinearZResult Finish(std::chrono::milliseconds deadline =
                           std::chrono::milliseconds(30000)) const;
  const std::shared_future<LinearZResult>& Result() const { return result_; }
  Lane lane() const { return lane_; }

private:
  Lane lane_ = Lane::QueryGate;
  std::shared_future<LinearZResult> result_;
};
std::unique_ptr<ProjectionWork> BeginProjection(
    std::shared_ptr<const ProjectionWeights> weights, LinearZTransfer& transfer,
    int layer, LinearDispatch dispatch,
    std::function<void(std::string)> reject = {}, BatchDispatch batch = {});
LinearZResult FinishProjection(
    const ProjectionWork& work, LinearZTransfer& transfer, unsigned columns,
    std::chrono::milliseconds deadline = std::chrono::milliseconds(30000));
}  // namespace gufo::npu
#endif
