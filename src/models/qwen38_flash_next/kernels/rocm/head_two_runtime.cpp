#include "head_two_runtime.hpp"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <limits>
#include <string_view>

#include "kernels.hpp"
#include "mmq/qfn_mmq.h"
#include "src/cli/serve/logging.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
bool Fail(std::string* error, const std::string& message) {
  if (error)
    *error = "headtwo: " + message;
  return false;
}
const char* Env(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value && *value ? value : fallback;
}
}  // namespace

bool HeadTwoRuntime::Allocate(void** p, std::size_t bytes, std::string* error) {
  if (hipMalloc(p, bytes) != hipSuccess)
    return Fail(error, "device allocation failed");
  allocations_.push_back(*p);
  device_bytes_ += bytes;
  return true;
}

bool HeadTwoRuntime::Create(const DeviceTensor& output,
                            const std::filesystem::path& cheap_path,
                            std::unique_ptr<HeadTwoRuntime>& result,
                            std::string* error, int bench_shortlist) {
  auto h = std::unique_ptr<HeadTwoRuntime>(new HeadTwoRuntime());
  h->report_ = false;
  if (bench_shortlist == 0) {
    const std::string_view enabled = Env("GUFO_HEAD_SHORTLIST", "0");
    const std::string_view check = Env("GUFO_HEAD_CHECK", "0");
    if ((enabled != "0" && enabled != "1") || (check != "0" && check != "1"))
      return Fail(error,
                  "GUFO_HEAD_SHORTLIST and GUFO_HEAD_CHECK must be 0 or 1");
    if (check == "1" && enabled != "1")
      return Fail(error, "GUFO_HEAD_CHECK=1 requires GUFO_HEAD_SHORTLIST=1");
    h->check_ = check == "1";
    h->report_ = enabled == "1";
    if (enabled == "0") {
      h->unavailable_ = HeadTwoReason::kDisabled;
      result = std::move(h);
      return true;
    }
  }
  h->shortlist_ = bench_shortlist ? bench_shortlist : kHeadTop;
  h->rescore_ = bench_shortlist && bench_shortlist != kHeadTop ? bench_shortlist
                                                               : kHeadCapacity;
  if (h->shortlist_ != 64 && h->shortlist_ != 128 && h->shortlist_ != 256 &&
      h->shortlist_ != 512)
    return Fail(error, "bench shortlist must be 64,128,256 or 512");
  const auto unavailable = [&](const std::string& why) {
    if (h->report_)
      server::Logger::Raw("[headtwo-init] FULL HEAD: %s; Q4_K copy=%s\n",
                          why.c_str(), cheap_path.string().c_str());
    result = std::move(h);
    return true;
  };
  if (output.type != core::GgmlType::kQ8_0 || output.cols != 2560 ||
      output.rows < kHeadTop)
    return unavailable("target head type/shape is incompatible");
  std::string detail;
  auto reader = core::GgufReader::OpenFile(cheap_path, &detail);
  const auto* tensor = reader ? reader->FindTensor("output.weight") : nullptr;
  if (!tensor || tensor->type != core::GgmlType::kQ4_K ||
      tensor->dimensions !=
          std::vector<std::uint64_t>{output.cols, output.rows})
    return unavailable("missing or incompatible sidecar (" + detail + ")");
  h->rows_ = output.rows;
  h->cols_ = output.cols;
  h->exact_ = output.data;
  const auto alloc = [&](auto** p, std::size_t bytes) {
    return h->Allocate(reinterpret_cast<void**>(p), bytes, error);
  };
  if (!alloc(&h->cheap_, tensor->size_bytes) ||
      hipMemcpy(h->cheap_, tensor->data, tensor->size_bytes,
                hipMemcpyHostToDevice) != hipSuccess)
    return Fail(error, "Q4_K upload failed");
  h->weight_bytes_ = tensor->size_bytes;
  auto& b = h->buffers_;
  const auto pairs = std::size_t(kHeadMaxWidth) *
                     ((output.rows + kHeadTile - 1) / kHeadTile) *
                     h->shortlist_;
  h->q8_row_bytes_ = ((output.cols + 511) / 512) * 16 * 36;
  if (!alloc(&h->q8_, kHeadMaxWidth * h->q8_row_bytes_) ||
      !alloc(&h->gathered_, std::size_t(kHeadMaxWidth) * h->rescore_ *
                                output.cols / 32 * 34) ||
      !alloc(&h->cross_,
             kHeadMaxWidth * kHeadMaxWidth * h->rescore_ * sizeof(float)) ||
      !alloc(&b.scores,
             std::size_t(kHeadMaxWidth) * output.rows * sizeof(float)) ||
      !alloc(&b.sort[0], pairs * sizeof(HeadTwoPair)) ||
      !alloc(&b.sort[1], pairs * sizeof(HeadTwoPair)) ||
      !alloc(&b.ids, kHeadMaxWidth * h->rescore_ * sizeof(int)) ||
      !alloc(&b.extras, kHeadExtras * sizeof(int)) ||
      !alloc(&b.exact, kHeadMaxWidth * h->rescore_ * sizeof(float)) ||
      !alloc(&b.result, kHeadMaxWidth * sizeof(HeadTwoResult)) ||
      hipHostMalloc(&h->extras_host_, kHeadExtras * sizeof(int)) !=
          hipSuccess ||
      hipHostMalloc(&h->host_, kHeadMaxWidth * sizeof(HeadTwoResult)) !=
          hipSuccess)
    return Fail(error, "workspace allocation failed");
  std::fill_n(h->extras_host_, kHeadExtras, INT_MAX);
  if (hipMemcpy(b.extras, h->extras_host_, kHeadExtras * sizeof(int),
                hipMemcpyHostToDevice) != hipSuccess)
    return Fail(error, "extra rows initialization failed");
  if (h->report_)
    server::Logger::Raw(
        "[headtwo-init] A shortlist=128 check=%d Q4_K=%s bytes=%llu\n",
        h->check_, cheap_path.string().c_str(),
        static_cast<unsigned long long>(h->weight_bytes_));
  result = std::move(h);
  return true;
}

bool HeadTwoRuntime::PrepareExtras(std::span<const std::uint32_t> ids,
                                   hipStream_t stream,
                                   std::string* error) const {
  if (ids.size() > kHeadExtras)
    return Fail(error, "too many extra rows");
  std::fill_n(extras_host_, kHeadExtras, INT_MAX);
  for (std::size_t i = 0; i < ids.size(); ++i)
    extras_host_[i] = ids[i] < unsigned(rows_) ? int(ids[i]) : INT_MAX;
  return hipMemcpyAsync(buffers_.extras, extras_host_,
                        kHeadExtras * sizeof(int), hipMemcpyHostToDevice,
                        stream) == hipSuccess ||
         Fail(error, "extra rows upload failed");
}

bool HeadTwoRuntime::Enqueue(const void* q8, float* full, int width,
                             hipStream_t stream, std::string* error,
                             int row_rotation) const {
  CopyMapped(q8, q8_, width * q8_row_bytes_, stream);
  if (hipGetLastError() != hipSuccess ||
      !HeadTwoPre(cheap_, q8, width, rows_, cols_, shortlist_, buffers_, stream,
                  row_rotation, rescore_) ||
      qfn_mmq_head_rescore(exact_, q8, buffers_.ids, buffers_.exact, cols_,
                           rescore_, rows_, width, gathered_, cross_, stream) ||
      !HeadTwoExtract(full, width, rows_, rescore_, buffers_, stream))
    return Fail(error, "shortlist kernel launch failed");
  if (check_ &&
      (qfn_mmq_q8_0_dense_vec_preq(exact_, nullptr, q8, full, rows_, width,
                                   cols_, stream) ||
       !HeadTwoCheck(full, width, rows_, rescore_, buffers_, stream) ||
       !HeadTwoExtract(full, width, rows_, rescore_, buffers_, stream, false)))
    return Fail(error, "full-head debug comparison failed");
  CopyMapped(buffers_.result, host_, width * sizeof(HeadTwoResult), stream);
  return hipGetLastError() == hipSuccess ||
         Fail(error, "compact download failed");
}

bool HeadTwoRuntime::SaveFrontier(void* destination, int row,
                                  hipStream_t stream) const {
  CopyMapped(static_cast<const char*>(q8_) + row * q8_row_bytes_, destination,
             q8_row_bytes_, stream);
  return hipGetLastError() == hipSuccess;
}

void HeadTwoRuntime::Row(int row, float* logits) const {
  std::fill_n(logits, rows_, -std::numeric_limits<float>::infinity());
  for (int i = 0; i < kHeadCapacity; ++i)
    if (host_[row].ids[i] < unsigned(rows_))
      logits[host_[row].ids[i]] = host_[row].logits[i];
}

const float* HeadTwoRuntime::RebuildFrontier(const void* saved, int width,
                                             int row, hipStream_t stream,
                                             std::string* error) const {
  if (!HeadTwoWidth(width) || row < 0 || row >= width) {
    Fail(error, "invalid saved frontier geometry");
    return nullptr;
  }
  // Repeated inputs retain the original width and the token's reduction slot.
  // The gather workspace is idle here; keep q8_ intact for other saved rows.
  for (int r = 0; r < width; ++r)
    CopyMapped(saved, static_cast<char*>(gathered_) + r * q8_row_bytes_,
               q8_row_bytes_, stream);
  if (hipGetLastError() != hipSuccess ||
      qfn_mmq_q8_0_dense_vec_preq(exact_, nullptr, gathered_, buffers_.scores,
                                  rows_, width, cols_, stream)) {
    Fail(error, "full frontier projection failed");
    return nullptr;
  }
  return buffers_.scores + std::size_t(row) * rows_;
}

MtpCandidateLogits HeadTwoRuntime::Candidates(int row) const {
  MtpCandidateLogits out;
  // This optional shortcut retains its existing 64-entry capacity.
  // Larger top-k consumers read the complete sparse 128-entry row instead.
  out.size = std::min<std::size_t>(kHeadTop, kMtpCandidates);
  std::copy_n(host_[row].ids, out.size, out.ids.begin());
  std::copy_n(host_[row].logits, out.size, out.logits.begin());
  return out;
}

void HeadTwoRuntime::Record(int width, HeadTwoReason reason) const {
  ++counts_[static_cast<int>(reason)];
  if (reason == HeadTwoReason::kShortlist && check_)
    for (int r = 0; r < width; ++r) {
      ++checked_;
      mismatches_ += host_[r].mismatch;
      missed_ += host_[r].missed;
    }
}

HeadTwoRuntime::~HeadTwoRuntime() {
  if (report_) {
    std::uint64_t passes = 0;
    for (auto n : counts_)
      passes += n;
    const auto shortlist = counts_[static_cast<int>(HeadTwoReason::kShortlist)];
    server::Logger::Raw(
        "[headtwo] passes=%llu shortlist=%llu full=%llu debug_full=%llu "
        "topk0=%llu "
        "topk>128=%llu logprobs=%llu grammar=%llu fullnorm=%llu "
        "extra_rows=%llu "
        "prefill=%llu frontier=%llu disabled=%llu unavailable=%llu width=%llu "
        "other=%llu\n",
        static_cast<unsigned long long>(passes),
        static_cast<unsigned long long>(shortlist),
        static_cast<unsigned long long>(passes - shortlist),
        static_cast<unsigned long long>(check_ ? shortlist : 0),
        static_cast<unsigned long long>(counts_[1]),
        static_cast<unsigned long long>(counts_[2]),
        static_cast<unsigned long long>(counts_[3]),
        static_cast<unsigned long long>(counts_[4]),
        static_cast<unsigned long long>(counts_[5]),
        static_cast<unsigned long long>(counts_[6]),
        static_cast<unsigned long long>(counts_[7]),
        static_cast<unsigned long long>(counts_[8]),
        static_cast<unsigned long long>(counts_[9]),
        static_cast<unsigned long long>(counts_[10]),
        static_cast<unsigned long long>(counts_[11]),
        static_cast<unsigned long long>(counts_[12]));
    if (check_)
      server::Logger::Raw(
          "[headcheck] passes=%llu shortlist-missed-top20=%llu "
          "rescored-byte-mismatch=%llu\n",
          static_cast<unsigned long long>(checked_),
          static_cast<unsigned long long>(missed_),
          static_cast<unsigned long long>(mismatches_));
  }
  for (void* p : allocations_)
    (void)hipFree(p);
  if (host_)
    (void)hipHostFree(host_);
  if (extras_host_)
    (void)hipHostFree(extras_host_);
}
}  // namespace gufo::models::qwen38_flash_next::rocm
