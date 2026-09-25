#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <thread>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// Re-encodes Q6_K rows as Q8_0 (fp16 scale + 32 int8 per block).
/// Q8_0's per-32 step is about a quarter of Q6_K's per-16 step, so the added
/// rounding is small next to the Q6_K quantization it sits on.
void Q6KRowsToQ8_0(const std::uint8_t* src, std::uint8_t* dst, std::size_t rows,
                   std::size_t cols) {
  const std::size_t src_row = cols / 256 * 210;
  const std::size_t dst_row = cols / 32 * 34;
  std::vector<float> values(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    gufo::quant::DequantizeQ6_K(src + r * src_row, values.data(), cols);
    std::uint8_t* out = dst + r * dst_row;
    for (std::size_t b = 0; b < cols / 32; ++b) {
      const float* x = values.data() + b * 32;
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(x[i]));
      const auto scale = static_cast<_Float16>(amax / 127.0F);
      const float d = static_cast<float>(scale);
      const float inverse = d != 0.0F ? 1.0F / d : 0.0F;
      std::memcpy(out + b * 34, &scale, sizeof(scale));
      auto* q = reinterpret_cast<std::int8_t*>(out + b * 34 + 2);
      for (int i = 0; i < 32; ++i) {
        const float v = std::nearbyint(x[i] * inverse);
        q[i] = static_cast<std::int8_t>(std::clamp(v, -127.0F, 127.0F));
      }
    }
  }
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  /// Layer() leaves the routed experts for a later Experts() pass.
  bool defer_experts{false};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// A Q6_K matrix (the output head of unsloth UD-IQ4_XS) re-encoded as
  /// Q8_0 on the host, so it runs on the tuned Q8_0 dense tier instead of a
  /// path the HIP kernels do not have.
  DeviceTensor CopyQ6KAsQ8_0(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.cols % 256 != 0 || t.experts != 1) {
      Fail("Q6_K tensor " + std::string(t.name) + " has an unsupported shape");
      return d;
    }
    const std::size_t src_row = t.cols / 256 * 210;
    const std::size_t dst_row = t.cols / 32 * 34;
    const std::size_t size = dst_row * t.rows;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name));
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    const auto* src = static_cast<const std::uint8_t*>(t.data);
    constexpr std::size_t kChunkRows = 16384;
    std::vector<std::uint8_t> host(kChunkRows * dst_row);
    const std::size_t workers =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    for (std::size_t r0 = 0; r0 < t.rows; r0 += kChunkRows) {
      const std::size_t rows = std::min<std::size_t>(kChunkRows, t.rows - r0);
      const std::size_t per = (rows + workers - 1) / workers;
      std::vector<std::jthread> pool;
      for (std::size_t w = 0; w < workers && w * per < rows; ++w) {
        const std::size_t begin = w * per;
        const std::size_t count = std::min(per, rows - begin);
        pool.emplace_back([&, begin, count] {
          Q6KRowsToQ8_0(src + (r0 + begin) * src_row,
                        host.data() + begin * dst_row, count, t.cols);
        });
      }
      pool.clear();
      if (hipMemcpy(static_cast<std::uint8_t*>(ptr) + r0 * dst_row, host.data(),
                    rows * dst_row, hipMemcpyHostToDevice) != hipSuccess) {
        Fail("upload failed for " + std::string(t.name));
        return d;
      }
    }
    (void)hipMemset(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin);
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = 1;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    if (!defer_experts) {
      Experts(l, d);
    }
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  void Experts(const LayerWeights& l, DeviceLayer& d) {
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // The CPU reference also reads Q6_K, but the production embedding, dense
  // and routed kernels do not. Reject it before allocating device weights;
  // the embedding and output head are re-encoded as Q8_0 instead.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K || &t == &w.token_embd ||
        &t == &w.output)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, error_msg};
  const auto head = [&](const TensorRef& t) {
    return t.type == core::GgmlType::kQ6_K ? up.CopyQ6KAsQ8_0(t) : up.Copy(t);
  };
  // Tuning::hot_first_upload: allocation order decides placement. Once most
  // of the GPU memory is taken, later allocations get slower memory (up to
  // ~20% lower GEMV bandwidth at the tail on gfx1151 under Windows).
  // Everything read in full on every token goes first: the output head,
  // then every dense projection (target and MTP). The routed experts follow
  // (each byte is read by a few percent of tokens), and the token embedding,
  // a row gather, comes last.
  const bool hot_first = platform::PlatformTuning().hot_first_upload;
  const bool tied = w.output.data == w.token_embd.data;
  if (tied || !hot_first) {
    m->token_embd_ = head(w.token_embd);
  }
  m->output_ = tied ? m->token_embd_ : head(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  up.defer_experts = hot_first;
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
    up.shard_base = 0;
  }
  if (hot_first) {
    for (std::size_t i = 0; i < w.layers.size() && up.ok; ++i) {
      up.Experts(w.layers[i], m->layers_[i]);
    }
    if (mtp != nullptr && up.ok) {
      up.shard_base = shard_count;
      up.Experts(mtp->block, m->mtp_);
      up.shard_base = 0;
    }
    if (!tied) {
      m->token_embd_ = head(w.token_embd);
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
