#include "src/models/qwen38_flash_next/ngram.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "src/core/platform/fs.hpp"
#include "src/core/platform/mapped_file.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen38_flash_next {
namespace {

constexpr std::size_t kPage = 4096;
// Next-chunk prefetch hides large gathers behind the current transformer
// pass. More readers increased CPU work without improving served prefill.
constexpr std::size_t kWorkers = 128;
constexpr std::size_t kReadBatch = 8;
constexpr std::size_t kBatchJobs = 1024;
constexpr std::size_t kCacheBytes = 128 * 1024 * 1024;

}  // namespace

void HashNgramRows(const Config& c, NgramHistory& history,
                   std::span<const std::int32_t> tokens,
                   std::span<std::uint32_t> rows) {
  const std::uint32_t n = c.ple_ngram_size;
  const std::int64_t eos = c.ple_eos_token;
  std::array<std::uint64_t, Config::kMaxPleNgram> ctx{};
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    // The token's own EOS does not cut its context, only an older one does.
    ctx[0] = static_cast<std::uint64_t>(tokens[i]);
    bool cut = false;
    for (std::uint32_t s = 1; s < n; ++s) {
      const std::int32_t t = cut ? NgramHistory::kNone : history.prev[s - 1];
      cut = cut || t < 0 || t == eos;
      ctx[s] = static_cast<std::uint64_t>(cut ? eos : t);
    }
    std::uint32_t* out = rows.data() + i * c.ple_heads;
    for (std::uint32_t order = 2; order <= n; ++order) {
      std::uint64_t mixed = ctx[0] * c.ple_multipliers[0];
      for (std::uint32_t j = 1; j < order; ++j) {
        mixed ^= ctx[j] * c.ple_multipliers[j];
      }
      const std::uint32_t base = (order - 2) * c.ple_heads_per_ngram;
      for (std::uint32_t g = 0; g < c.ple_heads_per_ngram; ++g) {
        const std::uint32_t h = base + g;
        out[h] = static_cast<std::uint32_t>(mixed % c.ple_head_vocab[h]) +
                 c.ple_head_offsets[h];
      }
    }
    for (std::size_t s = history.prev.size(); s-- > 1;) {
      history.prev[s] = history.prev[s - 1];
    }
    history.prev[0] = tokens[i];
  }
}

NgramTable::~NgramTable() {
  (void)WaitRead();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [&] { return ahead_pending_ == 0; });
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& w : workers_) {
    w.join();
  }
  if (platform::FileValid(file_)) {
    platform::CloseFile(file_);
  }
}

std::unique_ptr<NgramTable> NgramTable::Open(
    platform::PlatformFile file_descriptor, std::uint64_t file_offset, std::uint64_t rows,
    std::uint32_t row_dim, core::GgmlType type, std::string* error_msg) {
  std::unique_ptr<NgramTable> t(new NgramTable());
  const std::size_t block = type == core::GgmlType::kIQ4_NL ? 32 : 1;
  const std::size_t block_bytes = type == core::GgmlType::kIQ4_NL ? 18 : 2;
  if ((type != core::GgmlType::kIQ4_NL && type != core::GgmlType::kBF16) ||
      row_dim == 0 || row_dim % block != 0) {
    if (error_msg != nullptr) {
      *error_msg = "unsupported n-gram table format";
    }
    return nullptr;
  }
  t->type_ = type;
  t->row_dim_ = row_dim;
  t->row_bytes_ = row_dim / block * block_bytes;
  t->cache_count_ = std::bit_floor(kCacheBytes / t->row_bytes_);
  t->cache_entries_ = std::make_unique<CacheEntry[]>(t->cache_count_);
  t->cache_rows_.resize(t->cache_count_ * t->row_bytes_);
  t->rows_ = rows;
  t->base_offset_ = file_offset;
  t->file_ = platform::ReopenRandomReadFile(file_descriptor, t->direct_);
  if (!platform::FileValid(t->file_)) {
    t->file_ = platform::DuplicateFile(file_descriptor);
  }
  if (!platform::FileValid(t->file_)) {
    if (error_msg != nullptr) {
      *error_msg = "cannot open bound n-gram table";
    }
    return nullptr;
  }
  for (std::size_t i = 0; i < kWorkers; ++i) {
    if (PrefillDiagEnabled()) {
      t->workers_.emplace_back([raw = t.get()] { raw->Worker<true>(); });
    } else {
      t->workers_.emplace_back([raw = t.get()] { raw->Worker<false>(); });
    }
  }
  return t;
}

void NgramTable::DecodeRow(const std::uint8_t* src, float* dst) const {
  if (type_ == core::GgmlType::kIQ4_NL) {
    gufo::quant::DequantizeIQ4_NL(src, dst, row_dim_);
  } else {
    for (std::uint32_t i = 0; i < row_dim_; ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, src + 2 * i, 2);
      const std::uint32_t f = static_cast<std::uint32_t>(bits) << 16;
      std::memcpy(dst + i, &f, sizeof(float));
    }
  }
}

bool NgramTable::ReadCached(std::uint32_t row, float* dst, std::uint8_t* raw) {
  if (cache_count_ == 0) {
    return false;
  }
  const std::size_t slot = (row * 2654435761U) & (cache_count_ - 1);
  CacheEntry& entry = cache_entries_[slot];
  if (entry.busy.test_and_set(std::memory_order_acquire)) {
    return false;
  }
  const bool hit = entry.valid && entry.row == row;
  if (hit) {
    const auto* src = cache_rows_.data() + slot * row_bytes_;
    if (raw)
      std::memcpy(raw, src, row_bytes_);
    else
      DecodeRow(src, dst);
  }
  entry.busy.clear(std::memory_order_release);
  return hit;
}

template<bool Diagnostic>
bool NgramTable::ReadOne(std::uint32_t row, float* dst,
                         std::vector<std::uint8_t>& buf,
                         TableReadCounters* counters, NgramCounters* owner,
                         std::uint8_t* raw) {
  if (row >= rows_) {
    return false;
  }
  if (ReadCached(row, dst, raw)) {
    if (owner)
      owner->cache_hits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  const std::uint64_t offset = base_offset_ + row * row_bytes_;
  const std::uint64_t begin = direct_ ? offset & ~(kPage - 1) : offset;
  const std::uint64_t end =
      direct_ ? (offset + row_bytes_ + kPage - 1) & ~(kPage - 1)
              : offset + row_bytes_;
  const std::size_t length = end - begin;
  if (buf.size() < length + kPage) {
    buf.resize(length + kPage);
  }
  // O_DIRECT needs a page-aligned buffer; align inside the vector.
  auto* base =
      reinterpret_cast<std::uintptr_t>(buf.data()) % kPage == 0
          ? buf.data()
          : buf.data() +
                (kPage - reinterpret_cast<std::uintptr_t>(buf.data()) % kPage);
  // The aligned window may run past the end of the file: only the row's own
  // bytes have to arrive.
  const std::size_t needed = (offset - begin) + row_bytes_;
  const std::size_t slot =
      cache_count_ != 0 ? (row * 2654435761U) & (cache_count_ - 1) : 0;
  CacheEntry* entry = cache_count_ != 0 ? &cache_entries_[slot] : nullptr;
  std::uint8_t* cached =
      entry != nullptr ? cache_rows_.data() + slot * row_bytes_ : nullptr;
  std::size_t got = 0;
  // Compile-time specialization keeps the flag-off read loop unchanged.
  struct NoTimer {};
  std::conditional_t<Diagnostic, DiagClock::time_point, NoTimer> started;
  if constexpr (Diagnostic) {
    if (counters)
      started = DiagClock::now();
  }
  const auto finish = [&] {
    if constexpr (Diagnostic) {
      if (counters)
        counters->table_read_us += DiagMicroseconds(DiagClock::now() - started);
    }
  };
  while (got < needed) {
    std::error_code ec;
    const std::size_t n =
        platform::ReadAt(file_, begin + got, base + got, length - got, ec);
    if constexpr (Diagnostic) {
      if (counters) {
        ++counters->table_reads;
        if (n > 0)
          counters->table_bytes += static_cast<std::uint64_t>(n);
      }
    }
    if (n == 0) {
      if (ec == std::errc::interrupted) {
        continue;
      }
      finish();
      return false;
    }
    got += static_cast<std::size_t>(n);
  }
  finish();
  if (owner)
    owner->disk_reads_.fetch_add(1, std::memory_order_relaxed);
  // Cache only a complete row, still in its original quantized format.
  // A busy slot is skipped; readers never wait for another row's I/O.
  if (entry != nullptr &&
      !entry->busy.test_and_set(std::memory_order_acquire)) {
    std::memcpy(cached, base + (offset - begin), row_bytes_);
    entry->row = row;
    entry->valid = true;
    entry->busy.clear(std::memory_order_release);
  }
  if (raw)
    std::memcpy(raw, base + (offset - begin), row_bytes_);
  else
    DecodeRow(base + (offset - begin), dst);
  return true;
}

template<bool Diagnostic>
void NgramTable::Worker() {
  std::vector<std::uint8_t> buf;
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] {
      return stop_ || (active_ && next_job_ < jobs_.size()) ||
             !ahead_jobs_.empty();
    });
    if (stop_) {
      return;
    }
    // Large gathers amortize the queue lock. Small gathers keep one read
    // per worker, so decode does not serialize its few rows into a batch.
    std::array<Job, kReadBatch> batch;
    const bool ahead = !(active_ && next_job_ < jobs_.size());
    std::size_t count = 1;
    if (ahead) {
      batch[0] = std::move(ahead_jobs_.front());
      ahead_jobs_.pop_front();
    } else {
      count = std::min(jobs_.size() >= kBatchJobs ? kReadBatch : 1,
                       jobs_.size() - next_job_);
      std::copy_n(jobs_.data() + next_job_, count, batch.data());
      next_job_ += count;
    }
    struct NoCounters {};
    std::conditional_t<Diagnostic, TableReadCounters, NoCounters>
        batch_counters;
    TableReadCounters* counters = nullptr;
    if constexpr (Diagnostic) {
      if (!ahead && diagnostic_)
        counters = &batch_counters;
    }
    lock.unlock();
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i) {
      ok = ReadOne<Diagnostic>(batch[i].row, batch[i].dst, buf, counters,
                               batch[i].owner, batch[i].raw) &&
           ok;
    }
    if (ahead) {
      auto& read = *batch[0].ahead;
      const std::lock_guard read_lock(read.mutex_);
      read.failed_ = read.failed_ || !ok;
      if (--read.pending_ == 0)
        read.done_.notify_all();
    }
    lock.lock();
    if (ahead) {
      if (--ahead_pending_ == 0)
        done_.notify_all();
      continue;
    }
    if constexpr (Diagnostic) {
      if (diagnostic_)
        diagnostic_->Add(batch_counters);
    }
    failed_ = failed_ || !ok;
    pending_ -= count;
    if (pending_ == 0) {
      if constexpr (Diagnostic) {
        if (diagnostic_)
          read_finished_ = DiagClock::now();
      }
      done_.notify_all();
    }
  }
}

bool NgramReadAhead::Wait() {
  std::unique_lock lock(mutex_);
  done_.wait(lock, [&] { return pending_ == 0; });
  return !failed_;
}

std::shared_ptr<NgramReadAhead> NgramTable::ReadAhead(
    std::span<const std::uint32_t> rows,
    const std::shared_ptr<NgramCounters>& owner) {
  if (rows.empty() ||
      rows.size() > std::numeric_limits<std::size_t>::max() / row_bytes_ ||
      std::ranges::any_of(rows, [this](auto row) { return row >= rows_; }))
    return {};
  auto read = std::make_shared<NgramReadAhead>();
  read->raw_.resize(rows.size() * row_bytes_);
  read->owner_ = owner;
  std::size_t queued = 0;
  {
    const std::lock_guard lock(mutex_);
    if (stop_)
      return {};
    // Allocate queue entries before publishing the pending count. A failed
    // allocation removes only this submission, never another gather's jobs.
    const auto previous = ahead_jobs_.size();
    std::size_t cached = 0;
    try {
      for (std::size_t i = 0; i < rows.size(); ++i) {
        auto* raw = read->raw_.data() + i * row_bytes_;
        if (ReadCached(rows[i], nullptr, raw)) {
          ++cached;
          continue;
        }
        ahead_jobs_.push_back({rows[i], nullptr, owner.get(), read, raw});
        ++queued;
      }
    } catch (...) {
      ahead_jobs_.resize(previous);
      throw;
    }
    read->pending_ = queued;
    ahead_pending_ += queued;
    if (owner) {
      owner->rows_.fetch_add(rows.size(), std::memory_order_relaxed);
      owner->cache_hits_.fetch_add(cached, std::memory_order_relaxed);
    }
  }
  for (std::size_t i = 0; i < std::min(queued, workers_.size()); ++i)
    wake_.notify_one();
  return read;
}

void PleReadAhead::Begin(const NgramHistory& before) {
  Discard();
  before_ = after_ = before;
}

bool PleReadAhead::Append(NgramTable& table, const Config& config,
                          std::int32_t token,
                          const std::shared_ptr<NgramCounters>& owner) {
  if (token < 0 || config.ple_heads == 0 ||
      table.RowDim() != config.ple_head_dim)
    return false;
  std::array<std::uint32_t, Config::kMaxPleHeads> rows{};
  auto next = after_;
  HashNgramRows(config, next, std::span(&token, 1),
                std::span(rows).first(config.ple_heads));
  // Reserve both lists before any jobs are made visible to readers.
  tokens_.reserve(tokens_.size() + 1);
  reads_.reserve(reads_.size() + 1);
  auto read = table.ReadAhead(std::span(rows).first(config.ple_heads), owner);
  if (!read)
    return false;
  tokens_.push_back(token);
  reads_.push_back(std::move(read));
  row_bytes_ = table.RowBytes();
  heads_ = config.ple_heads;
  after_ = next;
  totals_.issued += heads_;
  return true;
}

bool PleReadAhead::Matches(const NgramHistory& before,
                           std::span<const std::int32_t> tokens) const {
  return !tokens.empty() && before == before_ &&
         tokens.size() <= tokens_.size() &&
         std::equal(tokens.begin(), tokens.end(), tokens_.begin());
}

bool PleReadAhead::CopyRows(std::size_t tokens, std::span<std::uint8_t> out) {
  if (tokens == 0 || tokens > reads_.size() ||
      out.size() < tokens * heads_ * row_bytes_)
    return false;
  const auto started = DiagClock::now();
  bool ok = true;
  for (std::size_t i = 0; i < tokens; ++i) {
    if (!reads_[i]->Wait()) {
      ok = false;
      continue;
    }
    const auto raw = reads_[i]->RawRows();
    std::memcpy(out.data() + i * heads_ * row_bytes_, raw.data(), raw.size());
  }
  totals_.wait_ms_total +=
      DiagMicroseconds(DiagClock::now() - started) / 1000.0;
  if (ok) {
    totals_.used += tokens * heads_;
    totals_.wasted += (reads_.size() - tokens) * heads_;
    reads_.clear();
    tokens_.clear();
  } else {
    Discard();
  }
  return ok;
}

void PleReadAhead::Discard() {
  totals_.wasted += reads_.size() * heads_;
  reads_.clear();
  tokens_.clear();
}

bool NgramTable::Read(std::span<const std::uint32_t> rows,
                      std::span<float> out) {
  return StartRead(rows, out) && WaitRead();
}

bool NgramTable::StartRead(std::span<const std::uint32_t> rows,
                           std::span<float> out,
                           TableReadCounters* diagnostic) {
  return StartRead(rows, out, {}, diagnostic);
}

void NgramTable::ReserveGather(std::size_t rows) {
  const std::lock_guard lock(mutex_);
  if (active_)
    throw std::logic_error("cannot reserve an outstanding n-gram gather");
  if (rows > std::numeric_limits<std::size_t>::max() / 4)
    throw std::length_error("n-gram gather is too large");
  const auto capacity = std::bit_ceil(std::max<std::size_t>(2, rows * 2));
  if (first_rows_.size() < capacity)
    first_rows_.resize(capacity);
  jobs_.resize(rows);
  jobs_.clear();
  copies_.resize(rows);
  copies_.clear();
}

bool NgramTable::StartRead(std::span<const std::uint32_t> rows,
                           std::span<float> out,
                           std::span<const NgramGatherOwner> owners,
                           TableReadCounters* diagnostic) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t owned = 0;
  for (const auto& owner : owners)
    owned += owner.rows;
  if (active_ || rows.size() > out.size() / row_dim_ ||
      rows.size() > std::numeric_limits<std::size_t>::max() / 4 ||
      (!owners.empty() && owned != rows.size()) ||
      std::any_of(rows.begin(), rows.end(),
                  [this](std::uint32_t row) { return row >= rows_; })) {
    return false;
  }
  if (diagnostic) {
    *diagnostic = {};
    read_started_ = DiagClock::now();
  }
  diagnostic_ = diagnostic;
  for (const auto& owner : owners)
    if (owner.counters)
      owner.counters->rows_.fetch_add(owner.rows, std::memory_order_relaxed);
  constexpr auto empty = std::numeric_limits<std::size_t>::max();
  const auto capacity = std::bit_ceil(std::max<std::size_t>(2, rows.size() * 2));
  if (first_rows_.size() < capacity)
    first_rows_.resize(capacity);
  auto first = std::span(first_rows_).first(capacity);
  std::ranges::fill(first, empty);
  jobs_.clear();
  copies_.clear();
  jobs_.reserve(rows.size());
  copies_.reserve(rows.size());
  auto owner = owners.begin();
  std::size_t owner_end = owner != owners.end() ? owner->rows : rows.size();
  for (std::size_t i = 0; i < rows.size(); ++i) {
    while (owner != owners.end() && i == owner_end) {
      ++owner;
      if (owner != owners.end())
        owner_end += owner->rows;
    }
    auto* counters = owner != owners.end() ? owner->counters : nullptr;
    const auto hash = std::uint64_t{rows[i]} * 0x9e3779b97f4a7c15ULL;
    auto slot = static_cast<std::size_t>(hash ^ (hash >> 32)) & (capacity - 1);
    while (first[slot] != empty && rows[first[slot]] != rows[i])
      slot = (slot + 1) & (capacity - 1);
    if (first[slot] == empty) {
      first[slot] = i;
      jobs_.push_back({rows[i], out.data() + i * row_dim_, counters});
    } else {
      copies_.push_back(
          {out.data() + first[slot] * row_dim_, out.data() + i * row_dim_});
      if (counters)
        counters->dedup_rows_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (jobs_.size() >= kBatchJobs) {
    std::sort(jobs_.begin(), jobs_.end(),
              [](const Job& a, const Job& b) { return a.row < b.row; });
  } else {
    // Small cached gathers avoid waking the I/O pool for every decode
    // token. Large prefills still parallelize their row conversions.
    std::erase_if(jobs_, [this](const Job& job) {
      if (!ReadCached(job.row, job.dst))
        return false;
      if (job.owner)
        job.owner->cache_hits_.fetch_add(1, std::memory_order_relaxed);
      return true;
    });
  }
  next_job_ = 0;
  pending_ = jobs_.size();
  failed_ = false;
  active_ = true;
  if (diagnostic && pending_ == 0)
    read_finished_ = DiagClock::now();
  // A decode gather has a handful of reads: wake one reader per read, not
  // the whole pool. Readers keep taking jobs until the queue is empty.
  if (pending_ >= workers_.size()) {
    wake_.notify_all();
  } else {
    for (std::size_t i = 0; i < pending_; ++i)
      wake_.notify_one();
  }
  return true;
}

bool NgramTable::WaitRead() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!active_) {
    return false;
  }
  done_.wait(lock, [&] { return pending_ == 0; });
  if (!failed_) {
    for (const Copy& copy : copies_) {
      std::copy_n(copy.src, row_dim_, copy.dst);
    }
  }
  jobs_.clear();
  copies_.clear();
  active_ = false;
  return !failed_;
}

}  // namespace gufo::models::qwen38_flash_next
