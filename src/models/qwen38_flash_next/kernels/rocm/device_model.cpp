#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <thread>
#include <vector>

#ifdef GUFO_NPU
#include "src/cli/serve/logging.hpp"
#include "src/npu/linear_z.hpp"
#include "src/npu/projection.hpp"
#include "src/npu/server_prefill.hpp"
#endif

#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/core/quant/bf16_q8_0.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/hc_load_q8_policy.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/native_dispatch.hpp"

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

/// GUFO_HC_LOAD_Q8_POISON: ends the life of a weight stager. With the switch on
/// the pinned staging buffers are overwritten with the poison pattern and kept
/// allocated; otherwise the stager is destroyed as before (bit-identical).
void RetireStager(std::unique_ptr<hip::WeightUpload>& stager,
                  hc_load_q8::PoisonLedger* poison) {
  if (poison == nullptr || stager == nullptr) {
    return;  // switch off: the stager is destroyed where it always was
  }
  const std::size_t bytes = stager->PoisonAndRetainStaging(hc_load_q8::kPoisonByte);
  // one event per staging slot (each is 16 MiB + one page)
  poison->host_events += bytes / ((std::size_t{16} << 20) + 4096);
  poison->host_bytes += bytes;
  poison->kept_host_bytes += bytes;
  stager.reset();
}

/// The same for the stagers of the optional uploads (draft output head, decode
/// twin, shadow): prints its own line, does nothing when the switch is off.
void RetireOptionalStager(std::unique_ptr<hip::WeightUpload>& stager, const char* what) {
  bool on = false;
  try {
    on = hc_load_q8::PoisonEnabled();
  } catch (const std::exception&) {
    return;  // Upload() already reported the bad value
  }
  if (!on) {
    return;
  }
  hc_load_q8::PoisonLedger ledger;
  RetireStager(stager, &ledger);
  std::fprintf(stderr,
               "[loader-poison] %s poisoned_host=%zu bytes=%zu kept_device=0 bytes=0 "
               "kept_host_bytes=%zu\n",
               what, ledger.host_events, ledger.host_bytes, ledger.kept_host_bytes);
}

/// GUFO_HC_LOAD_Q8: the device tensors of every BF16 hyper-connection tensor,
/// made BEFORE the weight stager exists (see PreloadHc), keyed by the address
/// of the mapped BF16 payload they replace.
struct HcPreload {
  std::map<const void*, DeviceTensor> tensors;
  std::size_t down{0}, up{0}, inject{0};
  std::size_t bf16_bytes{0}, q8_bytes{0}, inject_added_bytes{0};
  double ms{0};
};

/// Converts and uploads every BF16 HC down/up (-> Q8_0) and inject (-> F32)
/// tensor of `w` and `mtp`, one synchronous host-to-device copy per tensor that
/// carries the zero tail with it, then synchronizes the device. It runs before
/// `hip::WeightUpload::Create`, so no stager thread, stream or pending queue
/// exists while these null-stream copies happen: the conversion cannot
/// interleave with the asynchronous upload of any other tensor. Nothing is kept
/// on the host after its copy returned; no BF16 copy is ever allocated.
bool PreloadHc(const ModelWeights& w, const MtpWeights* mtp,
               std::vector<void*>& allocations, std::size_t& bytes,
               std::size_t& max_q8_cols, HcPreload& out, std::string* error,
               hc_load_q8::PoisonLedger* poison) {
  const auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  std::vector<std::uint8_t> image;  // reused: every byte is rewritten per tensor
  const auto upload = [&](const TensorRef& t, bool inject) {
    if (t.empty()) {
      return true;
    }
    if (t.type != core::GgmlType::kBF16) {
      return true;  // Q8_0 / F32 tensors take the ordinary stager path
    }
    if (t.experts != 1 || (!inject && t.cols % 32 != 0)) {
      return fail("GUFO_HC_LOAD_Q8: " + std::string(t.name) +
                  " is not a single BF16 matrix with 32-aligned rows");
    }
    const auto begin = std::chrono::steady_clock::now();
    std::string message;
    if (inject ? !quant::BuildBf16F32Image(t.data, t.cols * t.rows, kTailMargin,
                                           image, &message)
               : !quant::BuildBf16Q8_0Image(t.data, t.rows, t.cols, kTailMargin,
                                            hc_load_q8::kQuantizeThreads, image,
                                            &message)) {
      return fail("GUFO_HC_LOAD_Q8: " + std::string(t.name) + ": " + message);
    }
    const std::size_t size = image.size() - kTailMargin;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, image.size()) != hipSuccess) {
      return fail("hipMalloc failed for " + std::string(t.name) + " (" +
                  std::to_string(image.size()) + " bytes)");
    }
    allocations.push_back(ptr);
    bytes += image.size();
    // Tensor and zero tail in one synchronous copy.
    if (hipMemcpy(ptr, image.data(), image.size(), hipMemcpyHostToDevice) !=
        hipSuccess) {
      return fail("GUFO_HC_LOAD_Q8: upload failed for " + std::string(t.name));
    }
    // GUFO_HC_LOAD_Q8_POISON: the host image is released right here (the next
    // tensor rewrites it); if hipMemcpy had returned before its DMA read the
    // buffer, the device would now hold NaN and the trunk would break.
    if (poison != nullptr) {
      poison->Host(image.data(), image.size());
    }
    DeviceTensor d;
    d.data = ptr;
    d.type = inject ? core::GgmlType::kF32 : core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = 1;
    d.name = t.name;
    out.tensors.emplace(t.data, std::move(d));
    if (inject) {
      ++out.inject;
      out.inject_added_bytes += size - t.SizeBytes();
    } else {
      out.bf16_bytes += t.SizeBytes();
      out.q8_bytes += size;
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    out.ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - begin)
                  .count();
    return true;
  };
  const auto mixer = [&](const HcMixer& m) {
    const std::size_t before_down = out.tensors.size();
    if (!upload(m.down, false)) {
      return false;
    }
    out.down += out.tensors.size() - before_down;
    const std::size_t before_up = out.tensors.size();
    if (!upload(m.up, false)) {
      return false;
    }
    out.up += out.tensors.size() - before_up;
    return upload(m.inject, true);
  };
  if (!mixer(w.hc_head)) {
    return false;
  }
  for (const auto& l : w.layers) {
    if (!mixer(l.hc_attn) || !mixer(l.hc_ffn)) {
      return false;
    }
  }
  if (mtp != nullptr &&
      (!mixer(mtp->block.hc_attn) || !mixer(mtp->block.hc_ffn) ||
       !mixer(mtp->block.nextn_head))) {
    return false;
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    return fail("GUFO_HC_LOAD_Q8: device synchronize failed: " +
                std::string(hipGetErrorString(status)));
  }
  return true;
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::size_t& max_native_gemm_cols;
  std::size_t& max_native_gemm_rows;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  /// Layer() leaves the routed experts for a later Experts() pass.
  bool defer_experts{false};
  // Legacy F32 packages retain their established F16 stack arithmetic.
  // Published BF16-router packages preserve every type, including sidecars.
  // The one exception is the opt-in GUFO_HC_LOAD_Q8 narrowing below: it never
  // enters `conversions`, it quantises on the host while uploading.
  bool preserve_types{false};
  /// GUFO_HC_LOAD_Q8 decided to narrow (hc_load_q8::Decide): PreloadHc already
  /// made the device tensors; Mixer() takes them from here instead of copying
  /// the BF16 payload.
  HcPreload* hc_preload{nullptr};
  /// GUFO_HC_LOAD_Q8_POISON: non-null = poison what the loader releases and
  /// do not free device buffers.
  hc_load_q8::PoisonLedger* poison{nullptr};

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
    d.name = t.name;
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if ((t.type == core::GgmlType::kQ8_0 || NativeQuantDense(t.type)) &&
        t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    // Pad sizes of the native prefill GEMM (a pass under 96 rows runs on a
    // zero-padded 96-row copy); native types are never stacked.
    if (t.experts == 1 && NativeWmmaGemm(t.type, t.rows, t.cols, true)) {
      max_native_gemm_cols = std::max<std::size_t>(max_native_gemm_cols, t.cols);
      max_native_gemm_rows = std::max<std::size_t>(max_native_gemm_rows, t.rows);
    }
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
    if (poison != nullptr) {
      // GUFO_HC_LOAD_Q8_POISON: the staging tensor is poisoned and NOT freed;
      // it stays in `allocations` (the destructor frees it) and in `bytes`,
      // because it is still resident.
      const std::size_t whole = t.SizeBytes() + kTailMargin;
      if (hipMemset(combined.data, hc_load_q8::kPoisonByte, whole) != hipSuccess) {
        Fail("MTP staging tensor poisoning failed");
        return;
      }
      poison->KeptDevice(whole);
      return;
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
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0 &&
           type != core::GgmlType::kBF16 && type != core::GgmlType::kF16)) {
        Fail("tensor " + std::string(t->name) + " type=" +
             std::string(core::ToString(t->type)) +
             " pass=decode/verify/prefill: stacked projection types differ");
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
    if (StackedProjectionType(type, preserve_types) == type) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      d.name = (*parts.begin())->name;
      if (type == core::GgmlType::kQ8_0)
        max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      if (type == core::GgmlType::kBF16 || type == core::GgmlType::kF16)
        max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
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
    d.name = (*parts.begin())->name;
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  /// A tensor PreloadHc already made (BF16 HC projection -> Q8_0, BF16 inject
  /// -> F32), or the ordinary stager copy for everything else (the MTP
  /// sidecar's Q8_0 mixers, F32 norms, a Q8_0 inject).
  DeviceTensor Hc(const TensorRef& t) {
    if (t.empty() || !ok) {
      return {};
    }
    if (hc_preload != nullptr) {
      const auto it = hc_preload->tensors.find(t.data);
      if (it != hc_preload->tensors.end()) {
        return it->second;
      }
      if (t.type == core::GgmlType::kBF16) {
        Fail("GUFO_HC_LOAD_Q8: BF16 tensor " + std::string(t.name) +
             " was not converted although the plan covered every mixer");
        return {};
      }
    }
    return Copy(t);
  }

  DeviceMixer Mixer(const HcMixer& m) {
    if (hc_preload == nullptr) {
      return {Copy(m.norm), Copy(m.down), Copy(m.up), Copy(m.inject)};
    }
    DeviceMixer out;
    out.norm = Copy(m.norm);
    out.down = Hc(m.down);
    out.up = Hc(m.up);
    out.inject = Hc(m.inject);
    return out;
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
    std::string* error_msg
#ifdef GUFO_NPU
    ,
    std::shared_ptr<npu::ServerPrefill> prefill
#endif
) {
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
#ifdef GUFO_NPU
  // Stage before GPU uploads. Only a successfully started authorized service
  // can be injected; direct model callers default to no capability.
  std::vector<std::shared_ptr<const npu::LinearZWeights>> z(w.layers.size());
  std::vector<std::shared_ptr<const npu::ProjectionWeights>> gate(
      w.layers.size()),
      shared(w.layers.size());
  if (prefill && prefill->ready() && w.config.num_layers == 48 &&
      w.config.full_attention_interval == 4 && w.config.hidden_size == 2560 &&
      w.config.SsmConvChannels() == 10240 && w.config.SsmValueDim() == 6144) {
    try {
      for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        // Validate metadata before constructing any byte span/SizeBytes.
        if (!l.linear || l.ssm_qkv.experts != 1 || l.ssm_gate.experts != 1 ||
            l.ssm_qkv.empty() || l.ssm_gate.empty() ||
            l.ssm_qkv.type != core::GgmlType::kQ8_0 ||
            l.ssm_gate.type != core::GgmlType::kQ8_0 ||
            l.ssm_qkv.cols != npu::kK || l.ssm_gate.cols != npu::kK ||
            l.ssm_qkv.rows != 10240 || l.ssm_gate.rows != 6144)
          continue;
        const auto matrix = [](const TensorRef& t) {
          return npu::Q8Matrix{
              t.type,
              t.rows,
              t.cols,
              {static_cast<const std::byte*>(t.data), t.SizeBytes()}};
        };
        z[i] = npu::PrepareLinearZ(matrix(l.ssm_qkv), matrix(l.ssm_gate),
                                   prefill->knobs().slices);
      }
      const auto count = std::count_if(z.begin(), z.end(),
                                       [](const auto& p) { return bool(p); });
      server::Logger::Info(
          "npu",
          "linear-z qualified layers=" + std::to_string(count) +
              "/36; slices=" + std::to_string(prefill->knobs().slices) +
              "; other weight shapes and non-full prefill use GPU fallback");
      const auto q8 = [](const TensorRef& t, unsigned rows) {
        return !t.empty() && t.experts == 1 &&
               t.type == core::GgmlType::kQ8_0 && t.cols == npu::kK &&
               t.rows == rows;
      };
      const auto matrix = [](const TensorRef& t) {
        return npu::Q8Matrix{
            t.type,
            t.rows,
            t.cols,
            {static_cast<const std::byte*>(t.data), t.SizeBytes()}};
      };
      for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        if (!l.linear && w.config.num_heads == 24 &&
            w.config.num_kv_heads == 2 && w.config.head_dim == 256 &&
            w.config.rotary_dim == 64 && q8(l.attn_q, 12288) &&
            q8(l.attn_k, 512) && q8(l.attn_v, 512))
          gate[i] = npu::PrepareQueryGate(
              matrix(l.attn_q), matrix(l.attn_k), matrix(l.attn_v),
              prefill->knobs().gate_slices, 24, 2, 256);
        if (w.config.shared_expert_ff > 0 &&
            w.config.shared_expert_ff <= 1024 &&
            w.config.shared_expert_ff % 32 == 0 &&
            q8(l.shexp_gate, w.config.shared_expert_ff) &&
            q8(l.shexp_up, w.config.shared_expert_ff) &&
            !l.shexp_down.empty() && l.shexp_down.experts == 1 &&
            l.shexp_down.type == core::GgmlType::kQ8_0 &&
            l.shexp_down.rows == 2560 &&
            l.shexp_down.cols == w.config.shared_expert_ff)
          shared[i] =
              npu::PrepareSharedExpert(matrix(l.shexp_gate), matrix(l.shexp_up),
                                       prefill->knobs().shared_expert);
      }
      const auto resident = [&](const auto& layers) {
        for (const auto& weights : layers)
          if (weights)
            for (const auto& b : weights->b)
              prefill->session().PrepareWeight(std::as_bytes(std::span(b)));
      };
      resident(z);
      resident(gate);
      resident(shared);
      server::Logger::Info(
          "npu", "attention-q-gate qualified layers=" +
                     std::to_string(
                         std::count_if(gate.begin(), gate.end(),
                                       [](const auto& p) { return bool(p); })) +
                     "/12; shared gate/up qualified layers=" +
                     std::to_string(
                         std::count_if(shared.begin(), shared.end(),
                                       [](const auto& p) { return bool(p); })) +
                     "/48");
      m->npu_prefill_ = std::move(prefill);
    } catch (const std::exception& e) {
      prefill->Reject(e.what());
      if (error_msg)
        *error_msg = e.what();
      return nullptr;
    }
  }
#endif
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
  const bool preserve_types = std::any_of(
      w.layers.begin(), w.layers.end(), [](const LayerWeights& l) {
        return l.router.type == core::GgmlType::kBF16;
      });
  // GUFO_HC_LOAD_Q8 (default on): one decision for every mixer, taken before
  // any HC tensor is uploaded, so the model is never half converted.
  hc_load_q8::Plan hc_plan;
  bool hc_switch = false;
  bool hc_verify = false;
  bool hc_poison = false;
  try {
    hc_switch = hc_load_q8::Enabled();
    hc_verify = hc_load_q8::VerifyEnabled();
    hc_poison = hc_load_q8::PoisonEnabled();
  } catch (const std::exception& e) {
    if (error_msg != nullptr)
      *error_msg = e.what();
    return nullptr;
  }
  if (hc_switch) {
    std::vector<hc_load_q8::Tensor> listing;
    std::uint32_t mixer_id = 0;
    const auto list = [&](hc_load_q8::Group group, const HcMixer& mixer) {
      if (mixer.down.empty() && mixer.up.empty() && mixer.inject.empty())
        return;  // a trunk layer has no draft-head mixer
      ++mixer_id;
      const std::pair<hc_load_q8::Role, const TensorRef*> parts[] = {
          {hc_load_q8::Role::kDown, &mixer.down},
          {hc_load_q8::Role::kUp, &mixer.up},
          {hc_load_q8::Role::kInject, &mixer.inject}};
      for (const auto& [role, ref] : parts)
        if (!ref->empty())
          listing.push_back(
              {mixer_id, group, role, ref->type, ref->rows, ref->cols});
    };
    list(hc_load_q8::Group::kOutputHead, w.hc_head);
    for (const auto& l : w.layers) {
      list(hc_load_q8::Group::kTrunkAttn, l.hc_attn);
      list(hc_load_q8::Group::kTrunkFfn, l.hc_ffn);
    }
    if (mtp != nullptr) {
      list(hc_load_q8::Group::kMtpAttn, mtp->block.hc_attn);
      list(hc_load_q8::Group::kMtpFfn, mtp->block.hc_ffn);
      list(hc_load_q8::Group::kMtpHead, mtp->block.nextn_head);
    }
    if (!preserve_types) {
      hc_plan.reason = "not a published-types (BF16 router) package";
    } else if (w.config.hc_count != hc_load_q8::kHcCount ||
               w.config.hidden_size % 32 != 0 ||
               w.config.hc_low_rank != hc_load_q8::kLowRank) {
      hc_plan.reason = "hc_count/hidden_size/hc_low_rank is not 4/32k/320";
    } else {
      hc_plan = hc_load_q8::Decide(listing);
    }
    if (!hc_plan.convert) {
      std::fprintf(stderr,
                   "[hcq8] switch=1 active=0 %s reason=\"%s\" tensors=0 "
                   "resident_saved=0\n",
                   hc_plan.nothing_to_do ? "noop=1" : "refused=1",
                   hc_plan.reason.c_str());
    }
  }
  // The conversion runs BEFORE the stager exists: no stager thread, stream or
  // queued copy is alive while the null-stream copies of the converted tensors
  // and the device synchronize happen (decode-hcq8 intermittent-draft review:
  // the interleaved form could not be shown safe, this one needs no argument).
  HcPreload hc_pre;
  hc_load_q8::PoisonLedger poison_ledger;
  hc_load_q8::PoisonLedger* const poison = hc_poison ? &poison_ledger : nullptr;
  if (hc_plan.convert &&
      !PreloadHc(w, mtp, m->allocations_, m->bytes_, m->max_q8_cols_, hc_pre,
                 error_msg, poison)) {
    return nullptr;
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, m->max_native_gemm_cols_,
              m->max_native_gemm_rows_, error_msg};
  up.preserve_types = preserve_types;
  up.poison = poison;
  if (hc_plan.convert) {
    up.hc_preload = &hc_pre;
  }
  const auto head = [&](const TensorRef& t) {
    return up.Copy(t);
  };







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
#ifdef GUFO_NPU
    m->layers_.back().npu_z = std::move(z[m->layers_.size() - 1]);
    m->layers_.back().npu_gate = std::move(gate[m->layers_.size() - 1]);
    m->layers_.back().npu_shared = std::move(shared[m->layers_.size() - 1]);
#endif
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
  if (up.preserve_types && !conversions.empty()) {
    if (error_msg != nullptr)
      *error_msg = "native weights pass=upload: persistent type conversion is forbidden";
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
    if (poison != nullptr) {
      // GUFO_HC_LOAD_Q8_POISON: the F32 source of a narrowed stack is poisoned
      // and kept (still in allocations_ and bytes_), not freed.
      const std::size_t whole = c.count * sizeof(float) + kTailMargin;
      if (hipMemset(c.source, hc_load_q8::kPoisonByte, whole) != hipSuccess) {
        if (error_msg != nullptr)
          *error_msg = "GUFO_HC_LOAD_Q8_POISON: poisoning a narrowed stack source failed";
        return nullptr;
      }
      poison->KeptDevice(whole);
      continue;
    }
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  // The weight stager's pinned staging buffers end their life here (the
  // executor allocates its own pinned memory later).
  RetireStager(stager, poison);
  if (hc_plan.convert) {
    // Every HC projection must now be Q8_0 on the device; the consumers branch
    // on DeviceTensor::type alone and no BF16 copy was ever allocated.
    std::size_t down = 0, up_count = 0, injects = 0, f32_injects = 0;
    std::array<std::size_t, hc_load_q8::kGroups> mixers{};
    bool all_q8 = true;
    const auto verify = [&](hc_load_q8::Group group, const DeviceMixer& mixer) {
      if (mixer.down.empty() && mixer.up.empty())
        return;
      all_q8 = all_q8 && mixer.down.type == core::GgmlType::kQ8_0 &&
               mixer.up.type == core::GgmlType::kQ8_0;
      down += mixer.down.empty() ? 0 : 1;
      up_count += mixer.up.empty() ? 0 : 1;
      if (!mixer.inject.empty()) {
        ++injects;
        f32_injects += mixer.inject.type == core::GgmlType::kF32 ? 1 : 0;
        all_q8 = all_q8 && mixer.inject.type != core::GgmlType::kBF16;
      }
      ++mixers[static_cast<std::size_t>(group)];
    };
    verify(hc_load_q8::Group::kOutputHead, m->hc_head_);
    for (const auto& l : m->layers_) {
      verify(hc_load_q8::Group::kTrunkAttn, l.hc_attn);
      verify(hc_load_q8::Group::kTrunkFfn, l.hc_ffn);
    }
    if (m->has_mtp_) {
      verify(hc_load_q8::Group::kMtpAttn, m->mtp_.hc_attn);
      verify(hc_load_q8::Group::kMtpFfn, m->mtp_.hc_ffn);
      verify(hc_load_q8::Group::kMtpHead, m->mtp_.nextn_head);
    }
    const auto& st = hc_pre;
    if (!all_q8 || st.down != hc_plan.down || st.up != hc_plan.up ||
        st.inject != hc_plan.inject || st.bf16_bytes != hc_plan.bf16_bytes ||
        st.q8_bytes != hc_plan.q8_bytes ||
        st.inject_added_bytes != hc_plan.inject_added_bytes ||
        m->max_q8_cols_ < hc_load_q8::kHcDim) {
      if (error_msg != nullptr)
        *error_msg = "GUFO_HC_LOAD_Q8: loaded state differs from the plan (a "
                     "HC tensor is not Q8_0, counts or bytes differ, or "
                     "max_q8_cols < 10240)";
      return nullptr;
    }
    m->hc_load_q8_active_ = true;
    std::fprintf(
        stderr,
        "[hcq8] tensors=%zu bf16_bytes=%zu q8_bytes=%zu resident_saved=%zu "
        "load_ms=%.1f switch=1 active=1 down=%zu up=%zu inject_f32=%zu "
        "inject_added_bytes=%zu resident_net_saved=%zu mixers(trunk_attn=%zu "
        "trunk_ffn=%zu mtp_attn=%zu mtp_ffn=%zu mtp_head=%zu output_head=%zu) "
        "verified_q8_down_up=%zu max_q8_cols=%zu max_half_cols=%zu "
        "resident_bytes=%zu bf16_dense_kernel=idle(no BF16 HC tensor)\n",
        st.down + st.up + st.inject, st.bf16_bytes, st.q8_bytes,
        st.bf16_bytes - st.q8_bytes, st.ms, st.down, st.up, st.inject,
        st.inject_added_bytes,
        st.bf16_bytes - st.q8_bytes - st.inject_added_bytes,
        mixers[static_cast<std::size_t>(hc_load_q8::Group::kTrunkAttn)],
        mixers[static_cast<std::size_t>(hc_load_q8::Group::kTrunkFfn)],
        mixers[static_cast<std::size_t>(hc_load_q8::Group::kMtpAttn)],
        mixers[static_cast<std::size_t>(hc_load_q8::Group::kMtpFfn)],
        mixers[static_cast<std::size_t>(hc_load_q8::Group::kMtpHead)],
        mixers[static_cast<std::size_t>(hc_load_q8::Group::kOutputHead)],
        down + up_count, m->max_q8_cols_, m->max_half_cols_, m->bytes_);
  }
  if (up.preserve_types) {
    std::fprintf(stderr, "[native-upload] published_types=1 converted_weights=0 resident_bytes=%zu",
                 m->bytes_);
    // converted_weights counts the F32 -> F16 `conversions` list, which the
    // guard above keeps empty; the opt-in HC narrowing is in memory at upload
    // time and reported by the [hcq8] line.
    if (hc_plan.convert)
      std::fprintf(stderr, " hc_load_q8_converted=%zu",
                   hc_pre.down + hc_pre.up + hc_pre.inject);
    std::fputc('\n', stderr);
  }
  if (poison != nullptr) {
    std::size_t free_bytes = 0, total_bytes = 0;
    (void)hipMemGetInfo(&free_bytes, &total_bytes);
    std::fprintf(stderr,
                 "[loader-poison] on poisoned_host=%zu bytes=%zu kept_device=%zu "
                 "bytes=%zu kept_host_bytes=%zu pattern=0xFF gpu_free_mib=%zu "
                 "gpu_total_mib=%zu\n",
                 poison_ledger.host_events, poison_ledger.host_bytes,
                 poison_ledger.kept_device, poison_ledger.kept_device_bytes,
                 poison_ledger.kept_host_bytes, free_bytes >> 20, total_bytes >> 20);
  }
  // GUFO_HC_LOAD_Q8_VERIFY (debug, default off): read the converted tensors and
  // the whole MTP block back from the device and compare with the file.
  if (hc_verify && !m->VerifyContent(w, mtp, "upload", error_msg)) {
    return nullptr;
  }
  return m;
}

namespace {

/// One reference range on the host: bytes the device tensor must hold, in order.
struct RefPart {
  const std::uint8_t* data;
  std::size_t size;
};

enum class VerifyKind : std::size_t { kConverted, kHc, kMtp, kHead, kCount };

struct VerifyState {
  static constexpr std::size_t kChunk = 32u << 20;
  std::vector<std::uint8_t> chunk = std::vector<std::uint8_t>(kChunk);
  std::array<std::size_t, static_cast<std::size_t>(VerifyKind::kCount)> count{};
  std::size_t bytes{0}, skipped{0};
  std::vector<std::string> bad;
  std::string read_failure;
};

/// Reads `size` device bytes at `device` in chunks and compares with
/// `reference`; `first_diff` is the offset of the first differing byte.
bool CompareRange(VerifyState& s, const void* device, const std::uint8_t* reference,
                  std::size_t size, std::size_t& first_diff) {
  for (std::size_t offset = 0; offset < size;) {
    const std::size_t n = std::min(s.chunk.size(), size - offset);
    const auto status =
        hipMemcpy(s.chunk.data(), static_cast<const std::uint8_t*>(device) + offset,
                  n, hipMemcpyDeviceToHost);
    if (status != hipSuccess) {
      s.read_failure = std::string("device read-back failed: ") + hipGetErrorString(status);
      first_diff = offset;
      return false;
    }
    if (std::memcmp(s.chunk.data(), reference + offset, n) != 0) {
      std::size_t i = 0;
      while (s.chunk[i] == reference[offset + i]) {
        ++i;
      }
      first_diff = offset + i;
      return false;
    }
    offset += n;
  }
  return true;
}

/// The device tensor at `device` must hold the concatenation of `parts`, then
/// kTailMargin zero bytes.
void CheckParts(VerifyState& s, VerifyKind kind, const std::string& name,
                const void* device, std::span<const RefPart> parts) {
  std::size_t total = 0;
  for (const RefPart& p : parts) {
    total += p.size;
  }
  ++s.count[static_cast<std::size_t>(kind)];
  s.bytes += total;
  std::size_t offset = 0;
  for (const RefPart& p : parts) {
    std::size_t diff = 0;
    if (!CompareRange(s, static_cast<const std::uint8_t*>(device) + offset, p.data,
                      p.size, diff)) {
      s.bad.push_back(name + "@" + std::to_string(offset + diff) + "/" +
                      std::to_string(total));
      return;
    }
    offset += p.size;
  }
  const std::vector<std::uint8_t> zeros(kTailMargin, 0);
  std::size_t diff = 0;
  if (!CompareRange(s, static_cast<const std::uint8_t*>(device) + total,
                    zeros.data(), kTailMargin, diff)) {
    s.bad.push_back(name + "@tail+" + std::to_string(diff));
  }
}

}  // namespace

bool DeviceModel::VerifyContent(const ModelWeights& w, const MtpWeights* mtp,
                                const char* phase, std::string* error_msg) const {
  const auto begin = std::chrono::steady_clock::now();
  VerifyState s;
  std::vector<std::uint8_t> image;
  const auto file = [](const TensorRef& t) {
    return RefPart{static_cast<const std::uint8_t*>(t.data), t.SizeBytes()};
  };
  // The device tensor `d` must equal the file bytes of `t` (same type).
  const auto plain = [&](VerifyKind kind, const TensorRef& t, const DeviceTensor& d) {
    if (t.empty()) {
      return;
    }
    if (d.empty()) {
      s.bad.push_back(std::string(t.name) + "@missing");
      return;
    }
    if (d.type != t.type) {
      ++s.skipped;  // narrowed on purpose (not for a published-types package)
      return;
    }
    const RefPart p = file(t);
    CheckParts(s, kind, std::string(t.name), d.data, {&p, 1});
  };
  // A BF16 HC tensor that was converted: compare with a fresh SERIAL conversion
  // of the file payload (independent of the threaded path the loader used).
  const auto hc_tensor = [&](const TensorRef& t, const DeviceTensor& d, bool inject) {
    if (t.empty()) {
      return;
    }
    if (!hc_load_q8_active_ || t.type != core::GgmlType::kBF16) {
      plain(VerifyKind::kHc, t, d);
      return;
    }
    if (d.empty()) {
      s.bad.push_back(std::string(t.name) + "@missing");
      return;
    }
    std::string message;
    const bool built =
        inject ? quant::BuildBf16F32Image(t.data, t.cols * t.rows, 0, image, &message)
               : quant::BuildBf16Q8_0Image(t.data, t.rows, t.cols, 0, 1, image, &message);
    if (!built) {
      s.bad.push_back(std::string(t.name) + "@reference:" + message);
      return;
    }
    const RefPart p{image.data(), image.size()};
    CheckParts(s, VerifyKind::kConverted, std::string(t.name), d.data, {&p, 1});
  };
  const auto hc_mixer = [&](const HcMixer& t, const DeviceMixer& d) {
    plain(VerifyKind::kHc, t.norm, d.norm);
    hc_tensor(t.down, d.down, false);
    hc_tensor(t.up, d.up, false);
    hc_tensor(t.inject, d.inject, true);
  };
  hc_mixer(w.hc_head, hc_head_);
  for (std::size_t i = 0; i < w.layers.size() && i < layers_.size(); ++i) {
    hc_mixer(w.layers[i].hc_attn, layers_[i].hc_attn);
    hc_mixer(w.layers[i].hc_ffn, layers_[i].hc_ffn);
  }
  // The draft path's inputs outside the block: embedding and output head.
  plain(VerifyKind::kHead, w.token_embd, token_embd_);
  if (w.output.data != w.token_embd.data) {
    plain(VerifyKind::kHead, w.output, output_);
  }
  if (mtp != nullptr && has_mtp_) {
    const LayerWeights& b = mtp->block;
    const DeviceLayer& l = mtp_;
    const auto mtp_plain = [&](const TensorRef& t, const DeviceTensor& d) {
      plain(VerifyKind::kMtp, t, d);
    };
    for (const auto& [t, d] : {std::pair{&b.hc_attn, &l.hc_attn},
                               std::pair{&b.hc_ffn, &l.hc_ffn},
                               std::pair{&b.nextn_head, &l.nextn_head}}) {
      mtp_plain(t->norm, d->norm);
      mtp_plain(t->down, d->down);
      mtp_plain(t->up, d->up);
      mtp_plain(t->inject, d->inject);
    }
    mtp_plain(b.attn_out, l.attn_out);
    mtp_plain(b.attn_q_norm, l.attn_q_norm);
    mtp_plain(b.attn_k_norm, l.attn_k_norm);
    mtp_plain(b.indexer_q, l.indexer_q);
    mtp_plain(b.indexer_k, l.indexer_k);
    mtp_plain(b.indexer_q_norm, l.indexer_q_norm);
    mtp_plain(b.indexer_k_norm, l.indexer_k_norm);
    mtp_plain(b.shexp_gate, l.shexp_gate);
    mtp_plain(b.shexp_up, l.shexp_up);
    mtp_plain(b.shexp_down, l.shexp_down);
    mtp_plain(b.nextn_enorm, l.nextn_enorm);
    mtp_plain(b.nextn_hnorm, l.nextn_hnorm);
    mtp_plain(b.ffn_gate_exps, l.ffn_gate_exps);
    mtp_plain(b.ffn_up_exps, l.ffn_up_exps);
    mtp_plain(b.ffn_down_exps, l.ffn_down_exps);
    if (!l.attn_qkv.empty()) {  // q | k | v stacked
      const RefPart parts[] = {file(b.attn_q), file(b.attn_k), file(b.attn_v)};
      CheckParts(s, VerifyKind::kMtp, std::string(b.attn_q.name) + "+k+v",
                 l.attn_qkv.data, parts);
    } else {
      mtp_plain(b.attn_q, l.attn_q);
      mtp_plain(b.attn_k, l.attn_k);
      mtp_plain(b.attn_v, l.attn_v);
    }
    if (!l.router.empty()) {  // router | shared-expert gate stacked
      if (l.router.type == b.router.type) {
        const RefPart parts[] = {file(b.router), file(b.shexp_gate_inp)};
        CheckParts(s, VerifyKind::kMtp, std::string(b.router.name) + "+shexp_gate_inp",
                   l.router.data, parts);
      } else {
        ++s.skipped;
      }
    }
    if (!b.nextn_eh_proj.empty()) {  // [embedding | hidden] split by column halves
      const TensorRef& t = b.nextn_eh_proj;
      const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
      for (int half = 0; half < 2; ++half) {
        std::vector<RefPart> rows;
        rows.reserve(t.rows);
        for (std::uint64_t r = 0; r < t.rows; ++r) {
          rows.push_back({static_cast<const std::uint8_t*>(t.data) +
                              r * 2 * row_bytes + half * row_bytes,
                          row_bytes});
        }
        CheckParts(s, VerifyKind::kMtp,
                   std::string(t.name) + (half == 0 ? ".embedding" : ".hidden"),
                   (half == 0 ? l.nextn_fc_embedding : l.nextn_fc_hidden).data, rows);
      }
    }
  }
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
  const auto n = [&](VerifyKind k) { return s.count[static_cast<std::size_t>(k)]; };
  if (s.bad.empty() && s.read_failure.empty()) {
    std::fprintf(stderr,
                 "[hcq8-verify] ok phase=%s converted=%zu hc_plain=%zu mtp=%zu head=%zu "
                 "skipped=%zu bytes=%zu ms=%.1f\n",
                 phase, n(VerifyKind::kConverted), n(VerifyKind::kHc),
                 n(VerifyKind::kMtp), n(VerifyKind::kHead), s.skipped, s.bytes, ms);
    return true;
  }
  std::string list;
  for (std::size_t i = 0; i < s.bad.size() && i < 6; ++i) {
    list += (i ? "," : "") + s.bad[i];
  }
  std::fprintf(stderr,
               "[hcq8-verify] MISMATCH phase=%s bad=%zu first=[%s] %s checked: converted=%zu "
               "hc_plain=%zu mtp=%zu head=%zu ms=%.1f\n",
               phase, s.bad.size(), list.c_str(), s.read_failure.c_str(),
               n(VerifyKind::kConverted), n(VerifyKind::kHc), n(VerifyKind::kMtp),
               n(VerifyKind::kHead), ms);
  if (error_msg != nullptr) {
    *error_msg = std::string("GUFO_HC_LOAD_Q8_VERIFY: device content differs from the file (phase ") +
                 phase + "): " + list + (s.read_failure.empty() ? "" : " " + s.read_failure);
  }
  return false;
}

bool DeviceModel::UploadDraftOut(const TensorRef& ref,
                                 const core::GgufReader& reader,
                                 std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = "draft output head: " + message;
    return false;
  };
  if (!draft_out_.empty())
    return fail("already uploaded");
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager)
    return false;
  std::vector<Conversion> conversions;
  const std::size_t before = bytes_;
  Uploader up{*stager,        conversions,  allocations_, bytes_,
              max_half_cols_, max_q8_cols_, max_native_gemm_cols_, max_native_gemm_rows_,
              error_msg};
  DeviceTensor t = up.Copy(ref);
  if (!up.ok)
    return fail(error_msg != nullptr ? *error_msg : "upload failed");
  if (!stager->Finish(error_msg))
    return false;
  if (!conversions.empty())
    return fail("converted during upload (forbidden)");
  RetireOptionalStager(stager, "draft-out");
  if (t.type != core::GgmlType::kQ4_K && t.type != core::GgmlType::kQ5_K)
    return fail("not a 4/5-bit K-quant");
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess)
    return fail(std::string("synchronize failed: ") +
                hipGetErrorString(status));
  draft_out_ = t;
  std::fprintf(stderr,
               "[draftout] loaded output.weight %s %ux%u, %zu bytes resident "
               "(trunk head stays %zu bytes Q8_0)\n",
               t.type == core::GgmlType::kQ4_K ? "Q4_K" : "Q5_K", t.cols,
               t.rows, bytes_ - before, std::size_t{t.cols} / 32 * 34 * t.rows);
  return true;
}

bool DeviceModel::UploadTwin(const TwinWeights& twin,
                             const core::GgufReader& twin_reader,
                             std::string* error_msg) {
  return UploadTwinImpl(twin, twin_reader, error_msg, twin_layers_, true);
}

bool DeviceModel::UploadShadow(const TwinWeights& shadow,
                               const core::GgufReader& shadow_reader,
                               std::string* error_msg) {
  if (twin_layers_.empty()) {
    if (error_msg != nullptr)
      *error_msg = "decode twin shadow: needs the 4-bit twin first";
    return false;
  }
  if (!UploadTwinImpl(shadow, shadow_reader, error_msg, shadow_layers_, false))
    return false;
  for (std::size_t il = 0;
       il < twin_layers_.size() && il < shadow_layers_.size(); ++il) {
    const DeviceLayer& a = twin_layers_[il];
    const DeviceLayer& b = shadow_layers_[il];
    const auto add = [&](const DeviceTensor& x, const DeviceTensor& y,
                         const char* name) {
      if (!x.empty() && !y.empty() && x.type != core::GgmlType::kQ8_0 &&
          y.type == core::GgmlType::kQ8_0 && x.data != y.data &&
          x.rows == y.rows && x.cols == y.cols)
        shadow_map_[x.data] =
            ShadowEntry{&y, "blk." + std::to_string(il) + "." + name};
    };
    add(a.ssm_qkv, b.ssm_qkv, "attn_qkv");
    add(a.ssm_gate, b.ssm_gate, "attn_gate");
    add(a.ssm_out, b.ssm_out, "ssm_out");
    add(a.attn_q, b.attn_q, "attn_q");
    add(a.attn_k, b.attn_k, "attn_k");
    add(a.attn_v, b.attn_v, "attn_v");
    add(a.attn_out, b.attn_out, "attn_output");
    add(a.hc_attn.down, b.hc_attn.down, "hc_attn_down");
    add(a.hc_attn.up, b.hc_attn.up, "hc_attn_up");
    add(a.hc_ffn.down, b.hc_ffn.down, "hc_ffn_down");
    add(a.hc_ffn.up, b.hc_ffn.up, "hc_ffn_up");
    add(a.shexp_gate, b.shexp_gate, "ffn_gate_shexp");
    add(a.shexp_up, b.shexp_up, "ffn_up_shexp");
    add(a.shexp_down, b.shexp_down, "ffn_down_shexp");
  }
  std::fprintf(stderr,
               "[twin] shadow check armed: %zu 4-bit tensors paired with their "
               "fake-quant Q8_0 copy\n",
               shadow_map_.size());
  return true;
}

bool DeviceModel::UploadTwinImpl(const TwinWeights& twin,
                                 const core::GgufReader& twin_reader,
                                 std::string* error_msg,
                                 std::vector<DeviceLayer>& out_layers,
                                 bool primary) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = "decode twin: " + message;
    return false;
  };
  if (!out_layers.empty())
    return fail("already uploaded");
  if (twin.layers.size() != layers_.size())
    return fail("layer count differs from the trunk");
  if (hc_load_q8_active_) {
    // A twin HC tensor would replace a Q8_0 trunk projection with the twin's
    // own type and leave the wide route chosen from layer 0 inconsistent.
    const auto has_hc = [](const HcMixer& mixer) {
      return !mixer.down.empty() || !mixer.up.empty();
    };
    bool twin_hc = has_hc(twin.hc_head);
    for (const auto& l : twin.layers)
      twin_hc = twin_hc || has_hc(l.hc_attn) || has_hc(l.hc_ffn);
    if (twin_hc)
      return fail("GUFO_HC_LOAD_Q8=1 narrowed the trunk HC tensors to Q8_0; a "
                  "twin that replaces HC tensors would mix types");
  }
  const auto regions = twin_reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager)
    return false;
  std::vector<Conversion> conversions;
  const std::size_t before = bytes_;
  Uploader up{*stager,        conversions,  allocations_, bytes_,
              max_half_cols_, max_q8_cols_, max_native_gemm_cols_, max_native_gemm_rows_,
              error_msg};
  std::vector<DeviceLayer> built;
  built.reserve(layers_.size());
  for (std::size_t il = 0; il < layers_.size(); ++il) {
    DeviceLayer d = layers_[il];  // shares the trunk's device memory for every
                                  // tensor that is not twinned
    const LayerWeights& t = twin.layers[il];
    const auto put = [&](DeviceTensor& dst, const TensorRef& src) {
      if (!src.empty()) {
        dst = up.Copy(src);
        dst.decode_twin = true;
      }
    };
    // a stacked trunk tensor is split only when all its members come from the
    // twin (or were never stacked): a kept (8-bit) member of a split group has
    // no separate device tensor
    const bool qg = !t.ssm_qkv.empty() || !t.ssm_gate.empty();
    const bool ag = !t.attn_q.empty() || !t.attn_k.empty() || !t.attn_v.empty();
    if ((qg && (t.ssm_qkv.empty() || t.ssm_gate.empty()) &&
         layers_[il].ssm_qkv.empty()) ||
        (ag && (t.attn_q.empty() || t.attn_k.empty() || t.attn_v.empty()) &&
         layers_[il].attn_q.empty()))
      return fail(
          "a stacked projection group is only partly twinned (keep_q8 must "
          "keep a whole group: attn_qkv + attn_gate, attn_q + attn_k + "
          "attn_v)");
    if (!t.ssm_qkv.empty() || !t.ssm_gate.empty()) {
      d.ssm_in =
          {};  // the trunk stacked qkv + gate (both Q8_0, then both are
               // twinned: Bind requires every Q8_0 tensor); the twin keeps them
               // separate; a non-Q8_0 member keeps its trunk tensor
      put(d.ssm_qkv, t.ssm_qkv);
      put(d.ssm_gate, t.ssm_gate);
    }
    put(d.ssm_out, t.ssm_out);
    if (!t.attn_q.empty() || !t.attn_k.empty() || !t.attn_v.empty()) {
      d.attn_qkv = {};
      put(d.attn_q, t.attn_q);
      put(d.attn_k, t.attn_k);
      put(d.attn_v, t.attn_v);
    }
    put(d.attn_out, t.attn_out);
    put(d.hc_attn.down, t.hc_attn.down);
    put(d.hc_attn.up, t.hc_attn.up);
    put(d.hc_ffn.down, t.hc_ffn.down);
    put(d.hc_ffn.up, t.hc_ffn.up);
    put(d.shexp_gate, t.shexp_gate);
    put(d.shexp_up, t.shexp_up);
    put(d.shexp_down, t.shexp_down);
    if (!up.ok)
      return fail(error_msg != nullptr ? *error_msg : "upload failed");
    built.push_back(std::move(d));
  }
  if (primary) {
    twin_hc_head_ = hc_head_;
    if (!twin.hc_head.down.empty()) {
      twin_hc_head_.down = up.Copy(twin.hc_head.down);
      twin_hc_head_.down.decode_twin = true;
    }
    if (!twin.hc_head.up.empty()) {
      twin_hc_head_.up = up.Copy(twin.hc_head.up);
      twin_hc_head_.up.decode_twin = true;
    }
    if (!up.ok)
      return false;
  }
  if (!stager->Finish(error_msg))
    return false;
  if (!conversions.empty())
    return fail("a twin tensor was converted during upload (forbidden)");
  RetireOptionalStager(stager, "twin");
  for (std::size_t il = 0; il < built.size(); ++il) {
    const LayerWeights& t = twin.layers[il];
    const DeviceLayer& d = built[il];
    const std::pair<const TensorRef*, const DeviceTensor*> pairs[] = {
        {&t.ssm_qkv, &d.ssm_qkv},       {&t.ssm_gate, &d.ssm_gate},
        {&t.ssm_out, &d.ssm_out},       {&t.attn_q, &d.attn_q},
        {&t.attn_k, &d.attn_k},         {&t.attn_v, &d.attn_v},
        {&t.attn_out, &d.attn_out},     {&t.hc_attn.down, &d.hc_attn.down},
        {&t.hc_attn.up, &d.hc_attn.up}, {&t.hc_ffn.down, &d.hc_ffn.down},
        {&t.hc_ffn.up, &d.hc_ffn.up},   {&t.shexp_gate, &d.shexp_gate},
        {&t.shexp_up, &d.shexp_up},     {&t.shexp_down, &d.shexp_down}};
    for (const auto& [src, dev] : pairs)
      if (!src->empty() && dev->type != src->type)
        return fail("a twinned tensor changed type during upload");
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess)
    return fail(std::string("synchronize failed: ") +
                hipGetErrorString(status));
  out_layers = std::move(built);
  if (!primary)
    return true;
  twin_bytes_ = bytes_ - before;
  std::fprintf(stderr,
               "[twin] loaded tensors=%zu resident_4bit_bytes=%zu (incl. 4 KiB "
               "tail pads) mirrors_trunk_bytes=%zu saved_per_pass_bytes=%lld "
               "kept_q8_tensors=%zu kept_q8_bytes=%zu\n",
               twin.tensors, twin_bytes_, twin.bytes_8bit,
               static_cast<long long>(twin.bytes_8bit) -
                   static_cast<long long>(twin.bytes_4bit),
               twin.kept_tensors, twin.kept_bytes);
  return true;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
