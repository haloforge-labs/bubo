#include "src/models/qwen36_a3b/engine.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "qfn_mmq.h"
#include "src/models/qwen/vision/device_input.hpp"
#include "src/models/qwen36_a3b/attention.hpp"
#include "src/models/qwen36_a3b/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

namespace fn = gufo::models::qwen38_flash_next::rocm;
using core::GgmlType;

constexpr std::string_view kArch = "qwen35moe";
constexpr std::uint32_t kAttnSplits = 8;

void Fail(std::string* error, std::string message) {
  if (error != nullptr && error->empty())
    *error = std::move(message);
}

bool Ok(hipError_t err, const char* what, std::string* error) {
  if (err == hipSuccess)
    return true;
  Fail(error, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

fn::WeightType Small(GgmlType t) {
  return static_cast<fn::WeightType>(t);
}

WeightKind Kind(const Tensor& t) {
  switch (t.type) {
    case GgmlType::kQ8_0:
      return WeightKind::kQ8;
    case GgmlType::kBF16:
      return WeightKind::kBf16;
    default:
      return WeightKind::kF32;
  }
}

#ifdef _WIN32
// The 27B loader's approach: one hipMalloc per mapped region (dedicated VRAM
// carve-out), filled by 16 readers through a cached overlapped handle
// (O_CONCURRENT_RANDOM). O_DIRECT reads of a mapped file serialize on NTFS.
bool UploadRegions(std::span<const core::GgufMappedRegion> src,
                   std::vector<void*>& dst, std::string* error_msg) {
  constexpr std::size_t kTail = 4096;
  constexpr std::size_t kChunk = 16ULL << 20;
  constexpr std::size_t kReaders = 16;
  const auto started = std::chrono::steady_clock::now();
  std::mutex error_mutex;
  std::string error;
  std::atomic<bool> failed{false};
  const auto fail = [&](std::string message) {
    std::lock_guard lock(error_mutex);
    if (!failed.exchange(true))
      error = std::move(message);
  };
  dst.assign(src.size(), nullptr);
  std::vector<int> fds(src.size(), -1);
  struct Chunk {
    std::size_t region;
    std::size_t offset;
  };
  std::vector<Chunk> chunks;
  std::size_t total = 0;
  for (std::size_t i = 0; i < src.size() && !failed; ++i) {
    if (hipMalloc(&dst[i], src[i].size + kTail) != hipSuccess) {
      fail("hipMalloc failed for " + std::to_string(src[i].size) + " bytes");
      break;
    }
    (void)hipMemset(static_cast<std::uint8_t*>(dst[i]) + src[i].size, 0, kTail);
    const auto path = "/proc/self/fd/" + std::to_string(src[i].file_descriptor);
    fds[i] = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_CONCURRENT_RANDOM);
    if (fds[i] < 0) {
      fail(std::string("cannot reopen GGUF: ") + std::strerror(errno));
      break;
    }
    for (std::size_t off = 0; off < src[i].size; off += kChunk)
      chunks.push_back({i, off});
    total += src[i].size;
  }
  std::atomic<std::size_t> next{0};
  const auto reader = [&] {
    void* staging = nullptr;
    hipStream_t stream = nullptr;
    if (hipHostMalloc(&staging, kChunk) != hipSuccess ||
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess)
      fail("upload staging allocation failed");
    while (!failed) {
      const auto index = next.fetch_add(1);
      if (index >= chunks.size())
        break;
      const auto [region, offset] = chunks[index];
      const auto bytes = std::min(kChunk, src[region].size - offset);
      for (std::size_t got = 0; got < bytes && !failed;) {
        const auto n = ::pread(fds[region], static_cast<char*>(staging) + got,
                               bytes - got, static_cast<off_t>(offset + got));
        if (n <= 0) {
          fail(std::string("GGUF read failed: ") +
               (n == 0 ? "unexpected end of file" : std::strerror(errno)));
          break;
        }
        got += static_cast<std::size_t>(n);
      }
      if (failed)
        break;
      auto status =
          hipMemcpyAsync(static_cast<std::uint8_t*>(dst[region]) + offset,
                         staging, bytes, hipMemcpyHostToDevice, stream);
      if (status == hipSuccess)
        status = hipStreamSynchronize(stream);
      if (status != hipSuccess)
        fail(std::string("upload failed: ") + hipGetErrorString(status));
    }
    if (stream != nullptr)
      (void)hipStreamDestroy(stream);
    if (staging != nullptr)
      (void)hipHostFree(staging);
  };
  if (!failed) {
    std::vector<std::jthread> readers;
    for (std::size_t i = 1; i < kReaders; ++i)
      readers.emplace_back(reader);
    reader();
  }
  for (const int fd : fds)
    if (fd >= 0)
      (void)::close(fd);
  if (failed) {
    Fail(error_msg, "A3B weight upload failed: " + error);
    return false;
  }
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  const double gib = static_cast<double>(total) / (1ULL << 30);
  std::fprintf(stderr, "a3b: weight upload %.1f GiB in %.1f s (%.2f GiB/s)\n",
               gib, s, s > 0 ? gib / s : 0.0);
  return true;
}
#endif

}  // namespace

std::unique_ptr<Engine> Engine::Load(const std::filesystem::path& path,
                                     const Options& options,
                                     std::string* error) {
  std::unique_ptr<Engine> e(new Engine());
  e->options_ = options;
  e->gguf_ = core::GgufReader::OpenFile(path, error);
  if (!e->gguf_)
    return nullptr;
  if (!e->Bind(error) || !e->Upload(error) || !e->Allocate(error) ||
      (options.fused && !e->AllocateSpec(error)))
    return nullptr;
  e->Reset();
  if (!e->Sync(error))
    return nullptr;
  return e;
}

Engine::~Engine() {
  if (stream_ != nullptr)
    (void)hipStreamSynchronize(stream_);
  FreeWide();
  if (decode_exec_ != nullptr)
    (void)hipGraphExecDestroy(decode_exec_);
  for (auto& [key, exec] : spec_graphs_)
    (void)hipGraphExecDestroy(exec);
  if (stage_host_ != nullptr)
    (void)hipHostFree(stage_host_);
  if (down_host_ != nullptr)
    (void)hipHostFree(down_host_);
  if (draft_host_ != nullptr)
    (void)hipHostFree(draft_host_);
  for (void* p : owned_)
    (void)hipFree(p);
  for (void* p : regions_)
    (void)hipFree(p);
  if (tokens_host_ != nullptr)
    (void)hipHostFree(tokens_host_);
  if (pos_host_ != nullptr)
    (void)hipHostFree(pos_host_);
  if (stream_ != nullptr)
    (void)hipStreamDestroy(stream_);
}

bool Engine::Bind(std::string* error) {
  const auto& g = *gguf_;
  if (g.GetMetadataString("general.architecture") != kArch) {
    Fail(error, "GGUF architecture is not qwen35moe");
    return false;
  }
  bool ok = true;
  const auto u32 = [&](std::string_view key) {
    const auto v =
        g.GetMetadataUint32(std::string(kArch) + "." + std::string(key));
    if (!v) {
      Fail(error, "missing GGUF key " + std::string(key));
      ok = false;
    }
    return v.value_or(0);
  };
  const auto f32 = [&](std::string_view key) {
    const auto v =
        g.GetMetadataFloat32(std::string(kArch) + "." + std::string(key));
    if (!v) {
      Fail(error, "missing GGUF key " + std::string(key));
      ok = false;
    }
    return v.value_or(0.0F);
  };
  c_.layers = u32("block_count");
  c_.hidden = u32("embedding_length");
  c_.heads = u32("attention.head_count");
  c_.kv_heads = u32("attention.head_count_kv");
  c_.head_dim = u32("attention.key_length");
  c_.rotary_dim = u32("rope.dimension_count");
  c_.rope_theta = f32("rope.freq_base");
  c_.eps = f32("attention.layer_norm_rms_epsilon");
  c_.full_attention_interval = u32("full_attention_interval");
  c_.ssm_conv_kernel = u32("ssm.conv_kernel");
  c_.ssm_head_dim = u32("ssm.state_size");
  c_.ssm_k_heads = u32("ssm.group_count");
  c_.ssm_v_heads = u32("ssm.time_step_rank");
  c_.experts = u32("expert_count");
  c_.experts_used = u32("expert_used_count");
  c_.expert_ff = u32("expert_feed_forward_length");
  c_.shared_ff = u32("expert_shared_feed_forward_length");
  if (!ok)
    return false;
  // MTP files append their nextn block(s) after the trunk.
  c_.nextn = g.GetMetadataUint32(std::string(kArch) + ".nextn_predict_layers")
                 .value_or(0);
  if (c_.nextn >= c_.layers) {
    Fail(error, "bad nextn_predict_layers");
    return false;
  }
  c_.layers -= c_.nextn;
  // The FN kernels this runtime borrows fix these geometry points.
  if (c_.head_dim != 256 || c_.ssm_head_dim != 128 || c_.ssm_conv_kernel != 4 ||
      c_.heads % c_.kv_heads != 0 || c_.ssm_v_heads % c_.ssm_k_heads != 0 ||
      c_.experts > 512 || c_.full_attention_interval == 0 ||
      c_.hidden % 32 != 0) {
    Fail(error, "unsupported qwen35moe geometry");
    return false;
  }
  return true;
}

const void* Engine::Device(const core::GgufTensorInfo& t) const {
  const auto regions = gguf_->GetMappedRegions();
  const auto* p = static_cast<const std::uint8_t*>(t.data);
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto* base = static_cast<const std::uint8_t*>(regions[i].data);
    if (p >= base && p + t.size_bytes <= base + regions[i].size)
      return static_cast<const std::uint8_t*>(regions_[i]) + (p - base);
  }
  return nullptr;
}

bool Engine::Get(const std::string& name, Tensor* out, std::string* error) {
  const auto* t = gguf_->FindTensor(name);
  if (t == nullptr) {
    Fail(error, "missing tensor " + name);
    return false;
  }
  const auto type = t->type;
  if (type != GgmlType::kF32 && type != GgmlType::kQ8_0 &&
      type != GgmlType::kBF16) {
    Fail(error, "unsupported type " + std::string(core::ToString(type)) +
                    " for " + name + " (this build handles F32/Q8_0/BF16)");
    return false;
  }
  out->data = Device(*t);
  out->type = type;
  out->cols = static_cast<std::uint32_t>(t->dimensions[0]);
  out->rows = t->dimensions.size() > 1
                  ? static_cast<std::uint32_t>(t->dimensions[1])
                  : 1;
  out->experts = t->dimensions.size() > 2
                     ? static_cast<std::uint32_t>(t->dimensions[2])
                     : 1;
  if (out->data == nullptr) {
    Fail(error, "tensor outside mapped regions: " + name);
    return false;
  }
  return true;
}

void* Engine::StackRows(std::initializer_list<const Tensor*> parts,
                        std::string* error) {
  const GgmlType type = (*parts.begin())->type;
  const std::size_t element = type == GgmlType::kF32 ? 4 : 2;
  std::size_t total = 0;
  for (const auto* p : parts) {
    if (p->type != type ||
        (type != GgmlType::kF32 && type != GgmlType::kBF16)) {
      Fail(error, "stacked projection parts must share an F32/BF16 type");
      return nullptr;
    }
    total += static_cast<std::size_t>(p->rows) * p->cols;
  }
  void* dst = nullptr;
  if (!Ok(hipMalloc(&dst, total * element), "stack alloc", error))
    return nullptr;
  owned_.push_back(dst);
  auto* b = static_cast<std::uint8_t*>(dst);
  for (const auto* p : parts) {
    const std::size_t bytes =
        static_cast<std::size_t>(p->rows) * p->cols * element;
    if (!Ok(hipMemcpy(b, p->data, bytes, hipMemcpyDeviceToDevice), "stack copy",
            error))
      return nullptr;
    b += bytes;
  }
  return dst;
}

bool Engine::Upload(std::string* error) {
  if (!Ok(hipSetDevice(0), "hipSetDevice", error))
    return false;
  if (qfn_mmq_init(0) != 0) {
    Fail(error, "qfn_mmq_init failed");
    return false;
  }
  if (!Ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "stream",
          error))
    return false;
#ifdef _WIN32
  if (!UploadRegions(gguf_->GetMappedRegions(), regions_, error))
    return false;
#else
  Fail(error, "the A3B exploration runtime is Windows-only for now");
  return false;
#endif
  // Bind every tensor against the device copy.
  bool ok = Get("token_embd.weight", &token_embd_, error) &&
            Get("output_norm.weight", &output_norm_, error);
  if (!ok)
    return false;
  if (gguf_->HasTensor("output.weight")) {
    ok = Get("output.weight", &output_, error);
  } else {
    output_ = token_embd_;
  }
  if (!ok)
    return false;
  c_.vocab = output_.rows;
  if (token_embd_.type != GgmlType::kQ8_0 &&
      token_embd_.type != GgmlType::kBF16 &&
      token_embd_.type != GgmlType::kF32) {
    Fail(error, "unsupported embedding type");
    return false;
  }
  layers_.resize(c_.layers);
  for (std::uint32_t i = 0; i < c_.layers && ok; ++i) {
    auto& l = layers_[i];
    const std::string p = "blk." + std::to_string(i) + ".";
    l.linear = c_.IsLinear(i);
    ok = Get(p + "attn_norm.weight", &l.attn_norm, error) &&
         Get(p + "post_attention_norm.weight", &l.post_norm, error);
    if (ok && l.linear) {
      Tensor alpha, beta;
      ok = Get(p + "attn_qkv.weight", &l.ssm_qkv, error) &&
           Get(p + "attn_gate.weight", &l.ssm_z, error) &&
           Get(p + "ssm_conv1d.weight", &l.ssm_conv1d, error) &&
           Get(p + "ssm_a", &l.ssm_a, error) &&
           Get(p + "ssm_dt.bias", &l.ssm_dt, error) &&
           Get(p + "ssm_norm.weight", &l.ssm_norm, error) &&
           Get(p + "ssm_out.weight", &l.ssm_out, error) &&
           Get(p + "ssm_alpha.weight", &alpha, error) &&
           Get(p + "ssm_beta.weight", &beta, error);
      if (ok) {
        l.alpha_beta = static_cast<float*>(StackRows({&alpha, &beta}, error));
        if (alpha.type != GgmlType::kF32) {
          Fail(error, "ssm_alpha/ssm_beta must be F32");
          ok = false;
        }
        ok = l.alpha_beta != nullptr && l.ssm_qkv.rows == c_.ConvChannels() &&
             l.ssm_z.rows == c_.SsmValueDim();
        if (!ok)
          Fail(error, "GDN shape mismatch at " + p);
      }
    } else if (ok) {
      ok = Get(p + "attn_q.weight", &l.attn_q, error) &&
           Get(p + "attn_k.weight", &l.attn_k, error) &&
           Get(p + "attn_v.weight", &l.attn_v, error) &&
           Get(p + "attn_q_norm.weight", &l.attn_q_norm, error) &&
           Get(p + "attn_k_norm.weight", &l.attn_k_norm, error) &&
           Get(p + "attn_output.weight", &l.attn_out, error);
      if (ok &&
          (l.attn_q.rows != 2 * c_.QDim() || l.attn_k.rows != c_.KvDim())) {
        Fail(error, "attention shape mismatch at " + p);
        ok = false;
      }
    }
    Tensor router, sh_gate_inp;
    ok = ok && Get(p + "ffn_gate_inp.weight", &router, error) &&
         Get(p + "ffn_gate_inp_shexp.weight", &sh_gate_inp, error) &&
         Get(p + "ffn_gate_exps.weight", &l.gate_exps, error) &&
         Get(p + "ffn_up_exps.weight", &l.up_exps, error) &&
         Get(p + "ffn_down_exps.weight", &l.down_exps, error) &&
         Get(p + "ffn_gate_shexp.weight", &l.sh_gate, error) &&
         Get(p + "ffn_up_shexp.weight", &l.sh_up, error) &&
         Get(p + "ffn_down_shexp.weight", &l.sh_down, error);
    if (ok) {
      l.router = StackRows({&router, &sh_gate_inp}, error);
      l.router_type = Kind(router);
      ok = l.router != nullptr;
    }
  }
  if (ok && HasMtp())
    ok = BindMtp(error);
  // StackRows's device-to-device hipMemcpy calls go to the null stream, which
  // HIP on Windows may leave queued: the decode stream is non-blocking and
  // never waits for it, so without this the last layers' stacks read zeros.
  return ok && Ok(hipDeviceSynchronize(), "stack copies", error);
}

bool Engine::BindMtp(std::string* error) {
  if (c_.nextn != 1) {
    Fail(error, "only one MTP block is supported");
    return false;
  }
  const std::string p = "blk." + std::to_string(c_.layers) + ".";
  Layer& l = mtp_.block;
  Tensor router, sh_gate_inp;
  bool ok = Get(p + "attn_norm.weight", &l.attn_norm, error) &&
            Get(p + "post_attention_norm.weight", &l.post_norm, error) &&
            Get(p + "attn_q.weight", &l.attn_q, error) &&
            Get(p + "attn_k.weight", &l.attn_k, error) &&
            Get(p + "attn_v.weight", &l.attn_v, error) &&
            Get(p + "attn_q_norm.weight", &l.attn_q_norm, error) &&
            Get(p + "attn_k_norm.weight", &l.attn_k_norm, error) &&
            Get(p + "attn_output.weight", &l.attn_out, error) &&
            Get(p + "ffn_gate_inp.weight", &router, error) &&
            Get(p + "ffn_gate_inp_shexp.weight", &sh_gate_inp, error) &&
            Get(p + "ffn_gate_exps.weight", &l.gate_exps, error) &&
            Get(p + "ffn_up_exps.weight", &l.up_exps, error) &&
            Get(p + "ffn_down_exps.weight", &l.down_exps, error) &&
            Get(p + "ffn_gate_shexp.weight", &l.sh_gate, error) &&
            Get(p + "ffn_up_shexp.weight", &l.sh_up, error) &&
            Get(p + "ffn_down_shexp.weight", &l.sh_down, error) &&
            Get(p + "nextn.eh_proj.weight", &mtp_.eh_proj, error) &&
            Get(p + "nextn.enorm.weight", &mtp_.enorm, error) &&
            Get(p + "nextn.hnorm.weight", &mtp_.hnorm, error);
  if (!ok)
    return false;
  if (gguf_->HasTensor(p + "nextn.shared_head_norm.weight")) {
    ok = Get(p + "nextn.shared_head_norm.weight", &mtp_.head_norm, error);
  } else {
    mtp_.head_norm = output_norm_;
  }
  if (gguf_->HasTensor(p + "nextn.shared_head_head.weight") ||
      gguf_->HasTensor(p + "nextn.embed_tokens.weight")) {
    Fail(error, "MTP blocks with their own head/embedding are not supported");
    return false;
  }
  l.linear = false;
  l.router = StackRows({&router, &sh_gate_inp}, error);
  l.router_type = Kind(router);
  if (!ok || l.router == nullptr)
    return false;
  if (l.attn_q.rows != 2 * c_.QDim() || mtp_.eh_proj.cols != 2 * c_.hidden ||
      mtp_.eh_proj.rows != c_.hidden) {
    Fail(error, "unexpected MTP block shapes");
    return false;
  }
  return true;
}

bool Engine::Allocate(std::string* error) {
  // The tiled V^T cache holds 16-position blocks.
  options_.max_context = (options_.max_context + 15) / 16 * 16;
  const std::size_t B = kMaxRows;
  const std::size_t H = c_.hidden;
  bool ok = true;
  const auto alloc = [&](std::size_t bytes) -> void* {
    void* p = nullptr;
    if (!ok || !Ok(hipMalloc(&p, std::max<std::size_t>(bytes, 256)),
                   "scratch alloc", error)) {
      ok = false;
      return nullptr;
    }
    owned_.push_back(p);
    return p;
  };
  const auto f32 = [&](std::size_t n) {
    return static_cast<float*>(alloc(n * sizeof(float)));
  };
  const std::size_t ch = c_.ConvChannels();
  const std::size_t vd = c_.SsmValueDim();
  const std::size_t qd = c_.QDim();
  const std::size_t U = c_.experts_used;
  const std::size_t ff = std::max(c_.expert_ff, c_.shared_ff);
  res_ = f32(B * H);
  xn_ = f32(B * H);
  blk_ = f32(B * H);
  qkv_ = f32(B * ch);
  z_ = f32(B * vd);
  ab_ = f32(B * 2 * c_.ssm_v_heads);
  conv_scratch_ = f32((B + c_.ssm_conv_kernel) * ch);
  qn_ = f32(B * c_.ssm_k_heads * c_.ssm_head_dim);
  kn_ = f32(B * std::max(c_.ssm_k_heads * c_.ssm_head_dim, 2 * c_.ssm_v_heads));
  raw_ = f32(B * vd);
  gdn_out_ = f32(B * vd);
  qg_ = f32(B * 2 * qd);
  q_ = f32(B * qd);
  gate_ = f32(B * qd);
  k_ = f32(B * c_.KvDim());
  v_ = f32(B * c_.KvDim());
  ctx_ = f32(B * qd);
  partials_ = f32(std::max<std::size_t>(
      B * c_.heads * kAttnSplits * (c_.head_dim + 2),
      AttentionPartialFloats(c_.kv_heads, options_.max_context)));
  router_ = f32(B * (c_.experts + 1));
  weights_ = f32(B * U);
  sh_act_ = f32(B * ff);
  sh_tmp_ = f32(B * ff);
  sh_out_ = f32(B * H);
  e_act_ = f32(B * U * ff);
  e_up_ = f32(B * U * ff);
  e_down_ = f32(B * U * H);
  logits_ = f32(B * c_.vocab);
  const std::size_t max_k = std::max({H, vd, qd, ff});
  xq_ = alloc(qfn_mmq_q8_1_bytes(static_cast<int>(B), static_cast<int>(max_k)));
  aq_ = static_cast<Q8_1Block*>(alloc(B * (max_k / 32) * sizeof(Q8_1Block)));
  xf_ = f32(B * max_k);
  packed_ = f32(B * (2 * qd + 2 * c_.KvDim()));
  act_q_ = static_cast<Q8_1Block*>(
      alloc(B * (U + 1) * (ff / 32) * sizeof(Q8_1Block)));
  ids_ = static_cast<std::int32_t*>(alloc(B * U * sizeof(std::int32_t)));
  tokens_dev_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
  pos_dev_ = static_cast<std::uint32_t*>(alloc(sizeof(std::uint32_t)));
  if (!ok)
    return false;
  if (!Ok(hipHostMalloc(reinterpret_cast<void**>(&tokens_host_),
                        B * sizeof(std::int32_t)),
          "pinned tokens", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&pos_host_),
                        sizeof(std::uint32_t)),
          "pinned pos", error))
    return false;

  conv_state_.assign(c_.layers, nullptr);
  ssm_state_.assign(c_.layers, nullptr);
  k_cache_.assign(c_.layers, nullptr);
  v_cache_.assign(c_.layers, nullptr);
  vt_cache_.assign(c_.layers, nullptr);
  std::size_t state_bytes = 0;
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (layers_[i].linear) {
      const std::size_t conv = (c_.ssm_conv_kernel - 1) * ch;
      const std::size_t st = static_cast<std::size_t>(c_.ssm_v_heads) *
                             c_.ssm_head_dim * c_.ssm_head_dim;
      conv_state_[i] = f32(conv);
      ssm_state_[i] = f32(st);
      state_bytes += (conv + st) * sizeof(float);
    } else {
      const std::size_t kv =
          static_cast<std::size_t>(options_.max_context) * c_.KvDim();
      k_cache_[i] = static_cast<__half*>(alloc(kv * sizeof(__half)));
      v_cache_[i] = static_cast<__half*>(alloc(kv * sizeof(__half)));
      vt_cache_[i] = static_cast<__half*>(alloc(kv * sizeof(__half)));
      state_bytes += 3 * kv * sizeof(__half);
    }
  }
  std::fprintf(stderr,
               "a3b: %u layers, hidden %u, %u/%u experts, vocab %u; "
               "state+KV %.2f GiB at %u context\n",
               c_.layers, c_.hidden, c_.experts_used, c_.experts, c_.vocab,
               static_cast<double>(state_bytes) / (1ULL << 30),
               options_.max_context);

  // Weight bytes each phase reads per decoded token (routed experts at
  // used/experts of their size).
  const auto bytes = [](const Tensor& t) {
    const double n = double(t.rows) * t.cols * t.experts;
    switch (t.type) {
      case GgmlType::kQ8_0:
        return n * 34.0 / 32.0;
      case GgmlType::kBF16:
        return n * 2.0;
      default:
        return n * 4.0;
    }
  };
  const double routed = double(c_.experts_used) / c_.experts;
  for (const auto& l : layers_) {
    if (l.linear) {
      phase_bytes_[kPhGdnProj] += bytes(l.ssm_qkv) + bytes(l.ssm_z) +
                                  2.0 * c_.ssm_v_heads * c_.hidden * 4;
      phase_bytes_[kPhGdnOut] += bytes(l.ssm_out);
    } else {
      phase_bytes_[kPhAttnProj] +=
          bytes(l.attn_q) + bytes(l.attn_k) + bytes(l.attn_v);
      phase_bytes_[kPhAttnOut] += bytes(l.attn_out);
    }
    phase_bytes_[kPhRouter] += (c_.experts + 1.0) * c_.hidden * 4;
    phase_bytes_[kPhShared] +=
        bytes(l.sh_gate) + bytes(l.sh_up) + bytes(l.sh_down);
    phase_bytes_[kPhGateUp] += routed * (bytes(l.gate_exps) + bytes(l.up_exps));
    phase_bytes_[kPhDown] += routed * bytes(l.down_exps);
  }
  phase_bytes_[kPhHead] = bytes(output_);
  return ok;
}

bool Engine::AllocateSpec(std::string* error) {
  const std::size_t B = kMaxRows;
  const std::size_t H = c_.hidden;
  bool ok = true;
  const auto alloc = [&](std::size_t bytes) -> void* {
    void* p = nullptr;
    if (!ok || !Ok(hipMalloc(&p, std::max<std::size_t>(bytes, 256)),
                   "speculative alloc", error)) {
      ok = false;
      return nullptr;
    }
    owned_.push_back(p);
    return p;
  };
  const auto f32 = [&](std::size_t n) {
    return static_cast<float*>(alloc(n * sizeof(float)));
  };
  const std::size_t conv =
      static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
  state_snap_.assign(c_.layers, {});
  conv_snap_.assign(c_.layers, {});
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (!layers_[i].linear)
      continue;
    for (std::uint32_t row = 0; row + 1 < B; ++row) {
      state_snap_[i][row] = f32(fn::GdnRollbackRowFloats(
          row, c_.ssm_k_heads, c_.ssm_v_heads, c_.ssm_head_dim));
      conv_snap_[i][row] = f32(conv);
    }
  }
  target_h_ = f32(B * H);
  vtok_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
  const std::size_t ws = fn::MtpCandidateWorkspaceSize(c_.vocab);
  cand_ws_ = static_cast<std::uint32_t*>(alloc(ws * sizeof(std::uint32_t)));
  cand_scratch_ =
      static_cast<std::uint32_t*>(alloc(ws * sizeof(std::uint32_t)));
  cand_out_ = f32(B * sizeof(Candidates) / sizeof(float));
  stage_dev_ = static_cast<std::uint32_t*>(alloc(kStageWords * 4));
  if (HasMtp()) {
    const std::size_t kv =
        static_cast<std::size_t>(options_.max_context) * c_.KvDim();
    mtp_k_ = static_cast<__half*>(alloc(kv * sizeof(__half)));
    mtp_v_ = static_cast<__half*>(alloc(kv * sizeof(__half)));
    mtp_vt_ = static_cast<__half*>(alloc(kv * sizeof(__half)));
    mtp_h_ = f32(B * H);
    mtp_pending_ = f32(H);
    mtp_hn_ = f32(H);
    res_m_ = f32(B * H);
    emb_m_ = f32(B * H);
    cat_f_ = f32(B * 2 * H);
    cat_q_ = static_cast<Q8_1Block*>(alloc(B * 2 * H / 32 * sizeof(Q8_1Block)));
    logits_m_ = f32(c_.vocab);
    mtp_tok_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
    argmax_scratch_ = alloc(fn::kArgmaxParts * sizeof(fn::ArgmaxCandidate));
  }
  if (!ok)
    return false;
  if (!Ok(hipHostMalloc(reinterpret_cast<void**>(&stage_host_),
                        kStageWords * 4),
          "pinned staging", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&draft_host_),
                        sizeof(Candidates)),
          "pinned draft candidates", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&down_host_),
                        B * sizeof(std::int32_t) + B * sizeof(Candidates)),
          "pinned downloads", error))
    return false;
  if (mtp_pending_ != nullptr &&
      !Ok(hipMemset(mtp_pending_, 0, H * sizeof(float)), "zero", error))
    return false;
  if (HasMtp() && !options_.draft_vocab_ids.empty() &&
      !BuildDraftHead(options_.draft_vocab_ids, error))
    return false;
  if (HasMtp() && options_.draft_vocab_ids.empty() &&
      !options_.draft_vocab.empty() && !LoadDraftVocab(error))
    return false;
  return Ok(hipDeviceSynchronize(), "speculative alloc", error);
}

bool Engine::LoadDraftVocab(std::string* error) {
  std::FILE* f = std::fopen(options_.draft_vocab.c_str(), "rb");
  if (f == nullptr) {
    Fail(error, "cannot open draft vocab " + options_.draft_vocab);
    return false;
  }
  std::vector<std::int32_t> ids;
  long v = 0;
  while (std::fscanf(f, "%ld", &v) == 1)
    ids.push_back(static_cast<std::int32_t>(v));
  std::fclose(f);
  return BuildDraftHead(std::move(ids), error);
}

bool Engine::BuildDraftHead(std::vector<std::int32_t> ids, std::string* error) {
  std::erase_if(ids, [&](std::int32_t id) {
    return id < 0 || id >= static_cast<std::int32_t>(c_.vocab);
  });
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.size() < Candidates::kCount) {
    Fail(error, "draft vocab is too small");
    return false;
  }
  const std::size_t row_bytes =
      gguf_->FindTensor("output.weight") != nullptr
          ? gguf_->FindTensor("output.weight")->size_bytes / c_.vocab
          : gguf_->FindTensor("token_embd.weight")->size_bytes / c_.vocab;
  draft_vocab_ = static_cast<std::uint32_t>(ids.size());
  void* head = nullptr;
  void* map = nullptr;
  if (!Ok(hipMalloc(&head, row_bytes * ids.size()), "draft head", error) ||
      !Ok(hipMalloc(&map, ids.size() * sizeof(std::int32_t)), "draft map",
          error))
    return false;
  owned_.push_back(head);
  owned_.push_back(map);
  draft_head_ = head;
  draft_map_ = static_cast<std::int32_t*>(map);
  if (!Ok(hipMemcpy(map, ids.data(), ids.size() * sizeof(std::int32_t),
                    hipMemcpyHostToDevice),
          "draft map upload", error))
    return false;
  fn::GatherRows(output_.data, head, draft_map_, draft_vocab_, row_bytes,
                 stream_);
  std::fprintf(stderr, "a3b: draft head over %u tokens (%.0f MB)\n",
               draft_vocab_, row_bytes * ids.size() / 1e6);
  return Ok(hipStreamSynchronize(stream_), "draft head gather", error) &&
         Ok(hipDeviceSynchronize(), "draft head gather", error);
}

void Engine::Reset() {
  DFlashReset();
  pos_ = 0;
  mtp_rows_ = 0;
  verify_rows_ = 0;
  step_mtp_ = false;
  if (mtp_pending_ != nullptr)
    (void)hipMemsetAsync(mtp_pending_, 0, c_.hidden * sizeof(float), stream_);
  const std::size_t ch = c_.ConvChannels();
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    if (!layers_[i].linear)
      continue;
    (void)hipMemsetAsync(conv_state_[i], 0,
                         (c_.ssm_conv_kernel - 1) * ch * sizeof(float),
                         stream_);
    (void)hipMemsetAsync(ssm_state_[i], 0,
                         static_cast<std::size_t>(c_.ssm_v_heads) *
                             c_.ssm_head_dim * c_.ssm_head_dim * sizeof(float),
                         stream_);
  }
}

bool Engine::Sync(std::string* error) {
  return Ok(hipStreamSynchronize(stream_), "stream sync", error);
}

const void* Engine::Quantize(const float* x, std::uint32_t n, std::uint32_t k) {
  (void)qfn_mmq_quantize_q8_1(x, xq_, static_cast<int>(n), static_cast<int>(k),
                              stream_);
  return xq_;
}

void Engine::Dense(const Tensor& w, const float* x, float* out, std::uint32_t n,
                   const void* xq) {
  if (w.type == GgmlType::kQ8_0) {
    if (xq == nullptr)
      xq = Quantize(x, n, w.cols);
    (void)qfn_mmq_q8_0_dense_vec_preq(
        w.data, nullptr, xq, out, static_cast<int>(w.rows), static_cast<int>(n),
        static_cast<int>(w.cols), stream_);
    return;
  }
  fn::SmallGemm(w.data, Small(w.type), x, out, n, w.rows, w.cols, stream_);
}

void Engine::LinearAttention(const Layer& l, std::uint32_t idx,
                             std::uint32_t n) {
  Mark(kPhGdnProj);
  const std::uint32_t ch = c_.ConvChannels();
  const std::uint32_t vd = c_.SsmValueDim();
  const void* xq =
      l.ssm_qkv.type == GgmlType::kQ8_0 && l.ssm_z.type == GgmlType::kQ8_0
          ? Quantize(xn_, n, c_.hidden)
          : nullptr;
  if (!Skip(kPhGdnProj)) {
    Dense(l.ssm_qkv, xn_, qkv_, n, xq);
    Dense(l.ssm_z, xn_, z_, n, xq);
    fn::SmallGemm(l.alpha_beta, fn::WeightType::kF32, xn_, ab_, n,
                  2 * c_.ssm_v_heads, c_.hidden, stream_);
  }
  Mark(kPhGdnCore);
  if (!Skip(kPhGdnCore))
    fn::GatedDeltaNet(qkv_, ch, z_, vd, ab_, l.ssm_conv1d.f32(), l.ssm_a.f32(),
                      l.ssm_dt.f32(), l.ssm_norm.f32(), conv_state_[idx],
                      conv_scratch_, qn_, kn_, raw_, ssm_state_[idx], gdn_out_,
                      nullptr, fn::RollbackRows{}, fn::RollbackRows{}, n,
                      c_.ssm_k_heads, c_.ssm_v_heads, c_.ssm_head_dim,
                      c_.ssm_conv_kernel, false, false, c_.eps, stream_);
  // FN gates with sigmoid(z); Qwen3.6 uses silu(z) = z * sigmoid(z).
  if (!options_.fn_gate && !Skip(kPhGdnCore))
    MulInPlace(gdn_out_, z_, static_cast<std::size_t>(n) * vd, stream_);
  Mark(kPhGdnOut);
  if (!Skip(kPhGdnOut))
    Dense(l.ssm_out, gdn_out_, blk_, n);
}

void Engine::Attention(const Layer& l, std::uint32_t idx, std::uint32_t n) {
  Mark(kPhAttnProj);
  const std::uint32_t kv = c_.KvDim();
  const void* xq = l.attn_q.type == GgmlType::kQ8_0 &&
                           l.attn_k.type == GgmlType::kQ8_0 &&
                           l.attn_v.type == GgmlType::kQ8_0
                       ? Quantize(xn_, n, c_.hidden)
                       : nullptr;
  if (!Skip(kPhAttnProj)) {
    Dense(l.attn_q, xn_, qg_, n, xq);
    Dense(l.attn_k, xn_, k_, n, xq);
    Dense(l.attn_v, xn_, v_, n, xq);
  }
  Mark(kPhAttnCore);
  if (!Skip(kPhAttnCore)) {
    fn::UnpackQGate(qg_, 2 * c_.QDim(), q_, gate_, nullptr, nullptr, n,
                    c_.heads, c_.head_dim, 0, stream_);
    fn::RmsNormRows(q_, l.attn_q_norm.f32(), q_, n * c_.heads, c_.head_dim, 1,
                    c_.eps, stream_);
    fn::RmsNormRows(k_, l.attn_k_norm.f32(), k_, n * c_.kv_heads, c_.head_dim,
                    1, c_.eps, stream_);
    fn::Rope(q_, n, c_.heads, c_.head_dim, c_.rotary_dim, pos_dev_,
             c_.rope_theta, stream_, rope_);
    fn::Rope(k_, n, c_.kv_heads, c_.head_dim, c_.rotary_dim, pos_dev_,
             c_.rope_theta, stream_, rope_);
    fn::StoreKv(k_, k_cache_[idx], n, kv, pos_dev_, stream_);
    fn::StoreKv(v_, v_cache_[idx], n, kv, pos_dev_, stream_);
    fn::Attention(q_, k_cache_[idx], v_cache_[idx], nullptr, 0, ctx_, partials_,
                  kAttnSplits, n, pos_dev_, c_.heads, c_.kv_heads, c_.head_dim,
                  4, stream_);
    fn::SigmoidMul(ctx_, gate_, static_cast<std::size_t>(n) * c_.QDim(),
                   stream_);
  }
  Mark(kPhAttnOut);
  if (!Skip(kPhAttnOut))
    Dense(l.attn_out, ctx_, blk_, n);
}

void Engine::Moe(const Layer& l, std::uint32_t n) {
  const std::uint32_t E = c_.experts;
  const std::uint32_t U = c_.experts_used;
  const std::uint32_t H = c_.hidden;
  Mark(kPhRouter);
  if (Skip(kPhRouter)) {
  } else if (options_.fn_router) {
    fn::SmallGemm(l.router, fn::WeightType::kF32, xn_, router_, n, E + 1, H,
                  stream_);
  } else {
    F32Gemv(static_cast<const float*>(l.router), xn_, router_, n, E + 1, H,
            stream_);
  }
  if (!Skip(kPhRouter))
    fn::RouterTopK(router_, E + 1, ids_, weights_, n, E, U, stream_);

  // Shared expert: silu(gate x) * (up x), then down.
  Mark(kPhShared);
  const std::uint32_t sf = l.sh_up.rows;
  if (Skip(kPhShared)) {
  } else if (l.sh_up.type == GgmlType::kQ8_0 &&
             l.sh_gate.type == GgmlType::kQ8_0) {
    const void* xq = Quantize(xn_, n, H);
    (void)qfn_mmq_q8_0_dense_vec_preq(l.sh_up.data, l.sh_gate.data, xq, sh_act_,
                                      static_cast<int>(sf), static_cast<int>(n),
                                      static_cast<int>(H), stream_);
  } else {
    Dense(l.sh_gate, xn_, sh_act_, n);
    Dense(l.sh_up, xn_, sh_tmp_, n);
    fn::Swiglu(sh_act_, sh_tmp_, static_cast<std::size_t>(n) * sf, stream_);
  }
  if (!Skip(kPhShared))
    Dense(l.sh_down, sh_act_, sh_out_, n);

  // Routed experts.
  const auto experts = [&](const Tensor& w, const float* x, float* out,
                           std::uint32_t rows, std::uint32_t used) {
    if (w.type == GgmlType::kQ8_0) {
      (void)qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, x, ids_, out,
                            static_cast<int>(w.rows), static_cast<int>(w.cols),
                            static_cast<int>(rows), static_cast<int>(E),
                            static_cast<int>(used), stream_);
    } else {
      Bf16Experts(w.data, x, ids_, out, rows, used, w.rows, w.cols, stream_);
    }
  };
  const std::uint32_t ff = l.gate_exps.rows;
  Mark(kPhGateUp);
  if (Skip(kPhGateUp)) {
  } else if (l.gate_exps.type == GgmlType::kQ8_0 &&
             l.up_exps.type == GgmlType::kQ8_0) {
    (void)qfn_mmq_moe_gated_vec(
        static_cast<int>(GgmlType::kQ8_0), l.gate_exps.data, l.up_exps.data,
        xn_, ids_, e_act_, static_cast<int>(ff), static_cast<int>(H),
        static_cast<int>(n), static_cast<int>(E), static_cast<int>(U), stream_);
  } else {
    experts(l.gate_exps, xn_, e_act_, n, U);
    experts(l.up_exps, xn_, e_up_, n, U);
    fn::Swiglu(e_act_, e_up_, static_cast<std::size_t>(n) * U * ff, stream_);
  }
  // One (token, slot) row per expert id.
  Mark(kPhDown);
  if (!Skip(kPhDown))
    experts(l.down_exps, e_act_, e_down_, n * U, 1);
  Mark(kPhEpilogue);
  if (!Skip(kPhEpilogue))
    fn::MoeEpilogue(e_down_, weights_, sh_out_, router_ + E, E + 1, blk_, n, U,
                    H, stream_);
}

void Engine::SetSkip(std::uint32_t mask) {
  (void)hipStreamSynchronize(stream_);
  if (decode_exec_ != nullptr)
    (void)hipGraphExecDestroy(decode_exec_);
  decode_exec_ = nullptr;
  decode_warm_ = false;
  options_.skip = mask;
}

void Engine::DropGraphs() {
  (void)hipStreamSynchronize(stream_);
  if (decode_exec_ != nullptr)
    (void)hipGraphExecDestroy(decode_exec_);
  decode_exec_ = nullptr;
  decode_warm_ = false;
  for (auto& [key, exec] : spec_graphs_)
    (void)hipGraphExecDestroy(exec);
  spec_graphs_.clear();
  spec_warm_.clear();
}

void Engine::SetVision(qwen::vision::DeviceInput* input) {
  vision_ = input;
  // The descriptor's address is baked into captured graphs; its contents
  // (positions, delta) follow each request without a recapture.
  const auto* rope = input != nullptr ? input->rope() : nullptr;
  if (rope != rope_) {
    DropGraphs();
    rope_ = rope;
  }
}

void Engine::InjectImages(float* res, std::uint32_t pos, std::uint32_t n) {
  if (vision_ != nullptr)
    vision_->Inject(res, pos, n, c_.hidden, 1, stream_);
}

void Engine::Mark(Phase phase) {
  if (!profiling_)
    return;
  if (options_.profile_sync) {
    (void)hipStreamSynchronize(stream_);
    host_ms_.push_back(std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count());
    marks_.push_back(phase);
    return;
  }
  if (marks_.size() == events_.size()) {
    hipEvent_t e = nullptr;
    (void)hipEventCreate(&e);
    events_.push_back(e);
  }
  (void)hipEventRecord(events_[marks_.size()], stream_);
  marks_.push_back(phase);
}

void Engine::Collect() {
  for (std::size_t i = 0; i + 1 < marks_.size(); ++i) {
    if (options_.profile_sync) {
      phase_ms_[marks_[i]] += host_ms_[i + 1] - host_ms_[i];
      continue;
    }
    float ms = 0;
    if (hipEventElapsedTime(&ms, events_[i], events_[i + 1]) == hipSuccess)
      phase_ms_[marks_[i]] += ms;
  }
  marks_.clear();
  host_ms_.clear();
  ++profiled_;
}

void Engine::PrintProfile() const {
  static constexpr const char* kNames[kPhCount] = {
      "embed",       "norm+add",  "gdn proj",     "gdn core", "gdn out",
      "attn proj",   "attn core", "attn out",     "router",   "shared exp",
      "exp gate/up", "exp down",  "moe epilogue", "head"};
  if (profiled_ == 0)
    return;
  double total = 0, bytes = 0;
  for (int p = 0; p < kPhCount; ++p) {
    total += phase_ms_[p];
    bytes += phase_bytes_[p];
  }
  std::fprintf(stderr, "\na3b profile over %llu one-token forwards:\n",
               static_cast<unsigned long long>(profiled_));
  std::fprintf(stderr, "  %-13s %8s %6s %9s %8s\n", "phase", "ms/tok", "%",
               "MB/tok", "GB/s");
  for (int p = 0; p < kPhCount; ++p) {
    const double ms = phase_ms_[p] / profiled_;
    const double mb = phase_bytes_[p] / 1e6;
    std::fprintf(stderr, "  %-13s %8.3f %5.1f%% %9.1f %8.1f\n", kNames[p], ms,
                 100.0 * phase_ms_[p] / total, mb, ms > 0 ? mb / ms : 0.0);
  }
  const double ms = total / profiled_;
  std::fprintf(stderr, "  %-13s %8.3f %5.1f%% %9.1f %8.1f\n", "total", ms,
               100.0, bytes / 1e6, bytes / 1e6 / ms);
}

namespace {

// Speculative graph keys.
constexpr std::uint64_t kKeyStep = 1ULL << 40;
constexpr std::uint64_t kKeyCommit = 2ULL << 40;
constexpr std::uint64_t kKeyDraftFirst = 3ULL << 40;
constexpr std::uint64_t kKeyDraftNext = 4ULL << 40;
constexpr std::uint64_t kKeyVerify = 5ULL << 40;

}  // namespace

void Engine::FusedGdn(const Layer& l, std::uint32_t idx, std::uint32_t n,
                      bool snapshots) {
  const std::uint32_t H = c_.hidden;
  const std::uint32_t ch = c_.ConvChannels();
  const std::uint32_t vd = c_.SsmValueDim();
  Mark(kPhGdnProj);
  if (!Skip(kPhGdnProj)) {
    const GemvSeg segs[] = {
        {l.ssm_qkv.data, nullptr, qkv_, l.ssm_qkv.rows, Kind(l.ssm_qkv),
         GemvMode::kStore, ch},
        {l.ssm_z.data, nullptr, z_, l.ssm_z.rows, Kind(l.ssm_z),
         GemvMode::kStore, vd},
        {l.alpha_beta, nullptr, ab_, 2 * c_.ssm_v_heads, WeightKind::kF32,
         GemvMode::kStore, 2 * c_.ssm_v_heads}};
    MultiGemv(segs, 3, aq_, xn_, n, H, stream_);
  }
  Mark(kPhGdnCore);
  if (!Skip(kPhGdnCore)) {
    fn::RollbackRows states, convs;
    if (snapshots) {
      for (std::size_t r = 0; r < 7; ++r) {
        states.rows[r] = state_snap_[idx][r];
        convs.rows[r] = conv_snap_[idx][r];
      }
    }
    fn::GatedDeltaNet(qkv_, ch, z_, vd, ab_, l.ssm_conv1d.f32(), l.ssm_a.f32(),
                      l.ssm_dt.f32(), l.ssm_norm.f32(), conv_state_[idx],
                      conv_scratch_, qn_, kn_, raw_, ssm_state_[idx], gdn_out_,
                      nullptr, states, convs, n, c_.ssm_k_heads, c_.ssm_v_heads,
                      c_.ssm_head_dim, c_.ssm_conv_kernel, false, false, c_.eps,
                      stream_);
    // silu(z) = z * sigmoid(z): multiply FN's sigmoid-gated output by z.
    GateQuant(gdn_out_, z_, vd, GateMode::kMul, aq_,
              l.ssm_out.type == GgmlType::kQ8_0 ? nullptr : xf_, n, vd,
              stream_);
  }
  Mark(kPhGdnOut);
  if (!Skip(kPhGdnOut)) {
    const GemvSeg seg{l.ssm_out.data,        nullptr, res_, H, Kind(l.ssm_out),
                      GemvMode::kAccumulate, H};
    MultiGemv(&seg, 1, aq_, xf_, n, vd, stream_);
  }
}

void Engine::FusedAttention(const Layer& l, __half* k_cache, __half* v_cache,
                            __half* vt_cache, const std::uint32_t* pos,
                            float* res, std::uint32_t n) {
  const std::uint32_t H = c_.hidden;
  const std::uint32_t qd = c_.QDim();
  const std::uint32_t kv = c_.KvDim();
  const std::uint32_t stride = 2 * qd + 2 * kv;
  Mark(kPhAttnProj);
  if (!Skip(kPhAttnProj)) {
    const GemvSeg segs[] = {{l.attn_q.data, nullptr, packed_, l.attn_q.rows,
                             Kind(l.attn_q), GemvMode::kStore, stride},
                            {l.attn_k.data, nullptr, packed_ + 2 * qd, kv,
                             Kind(l.attn_k), GemvMode::kStore, stride},
                            {l.attn_v.data, nullptr, packed_ + 2 * qd + kv, kv,
                             Kind(l.attn_v), GemvMode::kStore, stride}};
    MultiGemv(segs, 3, aq_, xn_, n, H, stream_);
  }
  Mark(kPhAttnCore);
  if (!Skip(kPhAttnCore)) {
    (void)fn::PrepareAttention(
        packed_, stride, l.attn_q_norm.f32(), l.attn_k_norm.f32(), q_, gate_,
        k_cache, v_cache, n, c_.heads, c_.kv_heads, c_.head_dim, c_.rotary_dim,
        pos, c_.rope_theta, c_.eps, stream_, rope_);
    StoreVt(v_cache, vt_cache, pos, n, c_.kv_heads, options_.max_context,
            stream_);
    if (options_.fn_attention ||
        !DecodeAttention(q_, k_cache, v_cache, vt_cache, ctx_, partials_, pos,
                         n, c_.heads, c_.kv_heads, c_.head_dim,
                         options_.max_context, stream_))
      fn::Attention(q_, k_cache, v_cache, nullptr, 0, ctx_, partials_,
                    kAttnSplits, n, pos, c_.heads, c_.kv_heads, c_.head_dim, 4,
                    stream_);
    if (std::getenv("A3B_ATTN_CHECK") != nullptr) {
      // Diagnostic (eager only): both kernels on the same inputs.
      const std::size_t count = static_cast<std::size_t>(n) * qd;
      std::vector<float> mine(count), ref(count);
      (void)hipStreamSynchronize(stream_);
      (void)hipMemcpy(mine.data(), ctx_, count * 4, hipMemcpyDeviceToHost);
      fn::Attention(q_, k_cache, v_cache, nullptr, 0, ctx_, partials_,
                    kAttnSplits, n, pos, c_.heads, c_.kv_heads, c_.head_dim, 4,
                    stream_);
      (void)hipStreamSynchronize(stream_);
      (void)hipMemcpy(ref.data(), ctx_, count * 4, hipMemcpyDeviceToHost);
      std::uint32_t p = 0;
      (void)hipMemcpy(&p, pos, 4, hipMemcpyDeviceToHost);
      double worst = 0, scale = 0;
      std::size_t at = 0;
      for (std::size_t i = 0; i < count; ++i) {
        scale = std::max(scale, static_cast<double>(std::fabs(ref[i])));
        const double d = std::fabs(static_cast<double>(mine[i]) - ref[i]);
        if (d > worst)
          worst = d, at = i;
      }
      std::fprintf(stderr,
                   "attn check pos %u n %u: max |diff| %.3g at %zu "
                   "(row %zu head %zu), max |ref| %.3g\n",
                   p, n, worst, at, at / qd, at % qd / c_.head_dim, scale);
      (void)hipMemcpy(ctx_, mine.data(), count * 4, hipMemcpyHostToDevice);
    }
    GateQuant(ctx_, gate_, qd, GateMode::kSigmoid, aq_,
              l.attn_out.type == GgmlType::kQ8_0 ? nullptr : xf_, n, qd,
              stream_);
  }
  Mark(kPhAttnOut);
  if (!Skip(kPhAttnOut)) {
    const GemvSeg seg{l.attn_out.data,       nullptr, res, H, Kind(l.attn_out),
                      GemvMode::kAccumulate, H};
    MultiGemv(&seg, 1, aq_, xf_, n, qd, stream_);
  }
}

void Engine::FusedMoe(const Layer& l, float* res, std::uint32_t n) {
  const std::uint32_t E = c_.experts;
  const std::uint32_t U = c_.experts_used;
  const std::uint32_t H = c_.hidden;
  const std::uint32_t sf = l.sh_up.rows;
  Mark(kPhRouter);
  if (!Skip(kPhRouter)) {
    // Router logits (+ the shared expert's gate logit) and the shared
    // expert's silu(gate) * up in one launch: both only need the norm.
    const GemvSeg segs[] = {{l.router, nullptr, router_, E + 1, l.router_type,
                             GemvMode::kStore, E + 1},
                            {l.sh_gate.data, l.sh_up.data, sh_act_, sf,
                             Kind(l.sh_up), GemvMode::kGated, sf}};
    MultiGemv(segs, Skip(kPhShared) ? 1 : 2, aq_, xn_, n, H, stream_);
    fn::RouterTopK(router_, E + 1, ids_, weights_, n, E, U, stream_);
  }
  Mark(kPhGateUp);
  if (!Skip(kPhGateUp))
    ExpertsGated(l.gate_exps.data, l.up_exps.data, Kind(l.gate_exps), aq_, xn_,
                 ids_, e_act_, n, U, l.gate_exps.rows, H, stream_);
  Mark(kPhDown);
  if (!Skip(kPhDown))
    ExpertsDown(l.down_exps.data, Kind(l.down_exps), l.sh_down.data,
                Kind(l.sh_down), e_act_, sh_act_, act_q_, ids_, weights_,
                router_, E + 1, E, res, n, U, H, l.down_exps.cols, stream_);
}

void Engine::BodyFused(std::uint32_t n, std::uint32_t rows,
                       const std::int32_t* tokens, const std::uint32_t* pos,
                       bool capture, bool snapshots, std::uint32_t inject_pos) {
  const std::uint32_t H = c_.hidden;
  Mark(kPhEmbed);
  fn::EmbedTokens(token_embd_.data, Small(token_embd_.type), tokens, res_, n, H,
                  1, stream_);
  if (inject_pos != kNoInject)
    InjectImages(res_, inject_pos, n);
  const std::uint32_t parts = options_.fused_parts;
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    const Layer& l = layers_[i];
    Mark(kPhNorm);
    if (!Skip(kPhNorm))
      AddNormQuant(res_, nullptr, l.attn_norm.f32(), xn_, aq_, n, H, c_.eps,
                   stream_);
    if (!(parts & (l.linear ? 1u : 2u))) {
      // Diagnostic: the original launchers (current position only).
      if (l.linear) {
        LinearAttention(l, i, n);
      } else {
        Attention(l, i, n);
      }
      AddInPlace(res_, blk_, static_cast<std::size_t>(n) * H, stream_);
    } else if (l.linear) {
      FusedGdn(l, i, n, snapshots);
    } else {
      FusedAttention(l, k_cache_[i], v_cache_[i], vt_cache_[i], pos, res_, n);
    }
    Mark(kPhNorm);
    if (!Skip(kPhNorm))
      AddNormQuant(res_, nullptr, l.post_norm.f32(), xn_, aq_, n, H, c_.eps,
                   stream_);
    if (parts & 4u) {
      FusedMoe(l, res_, n);
    } else {
      Moe(l, n);
      AddInPlace(res_, blk_, static_cast<std::size_t>(n) * H, stream_);
    }
    if (df_)
      DFlashTap(i, res_, n, false);
  }
  if (rows == 0 && !capture)
    return;
  Mark(kPhHead);
  // With `capture` every row is normed: the MTP block seeds on them.
  const std::uint32_t normed = capture ? n : rows;
  AddNormQuant(res_ + static_cast<std::size_t>(n - normed) * H, nullptr,
               output_norm_.f32(), xn_, aq_, normed, H, c_.eps, stream_);
  if (capture)
    fn::CopyDevice(xn_, target_h_, static_cast<std::size_t>(n) * H, stream_);
  if (rows > 0 && !Skip(kPhHead)) {
    const std::uint32_t skip = normed - rows;
    const GemvSeg seg{output_.data,  nullptr,          logits_, output_.rows,
                      Kind(output_), GemvMode::kStore, c_.vocab};
    MultiGemv(&seg, 1, aq_ + static_cast<std::size_t>(skip) * (H / 32),
              xn_ + static_cast<std::size_t>(skip) * H, rows, H, stream_);
  }
  Mark(kPhCount);  // end sentinel
}

void Engine::MtpPass(const std::int32_t* tokens, const float* hidden,
                     std::uint32_t n, const std::uint32_t* pos,
                     std::int32_t* draft_out, bool candidates) {
  const std::uint32_t H = c_.hidden;
  const Layer& l = mtp_.block;
  // eh_proj([enorm(embed(token)) ; hnorm(hidden)]).
  fn::EmbedTokens(token_embd_.data, Small(token_embd_.type), tokens, emb_m_, n,
                  H, 1, stream_);
  AddNormQuant(emb_m_, nullptr, mtp_.enorm.f32(), cat_f_, cat_q_, n, H, c_.eps,
               stream_, 2 * H, 2 * H / 32);
  // AddNormQuant only writes its residual when adding.
  AddNormQuant(const_cast<float*>(hidden), nullptr, mtp_.hnorm.f32(),
               cat_f_ + H, cat_q_ + H / 32, n, H, c_.eps, stream_, 2 * H,
               2 * H / 32);
  const GemvSeg eh{mtp_.eh_proj.data,  nullptr,          res_m_, H,
                   Kind(mtp_.eh_proj), GemvMode::kStore, H};
  MultiGemv(&eh, 1, cat_q_, cat_f_, n, 2 * H, stream_);
  AddNormQuant(res_m_, nullptr, l.attn_norm.f32(), xn_, aq_, n, H, c_.eps,
               stream_);
  FusedAttention(l, mtp_k_, mtp_v_, mtp_vt_, pos, res_m_, n);
  AddNormQuant(res_m_, nullptr, l.post_norm.f32(), xn_, aq_, n, H, c_.eps,
               stream_);
  FusedMoe(l, res_m_, n);
  if (draft_out == nullptr && !candidates)
    return;
  AddNormQuant(res_m_ + static_cast<std::size_t>(n - 1) * H, nullptr,
               mtp_.head_norm.f32(), mtp_hn_, aq_, 1, H, c_.eps, stream_);
  // The draft head: the trunk's, or its --draft-vocab rows (drafts only; the
  // verified distribution always comes from the full head).
  const bool pruned = draft_head_ != nullptr;
  const std::uint32_t vocab = pruned ? draft_vocab_ : c_.vocab;
  const GemvSeg head{pruned ? draft_head_ : output_.data,
                     nullptr,
                     logits_m_,
                     vocab,
                     Kind(output_),
                     GemvMode::kStore,
                     vocab};
  MultiGemv(&head, 1, aq_, mtp_hn_, 1, H, stream_);
  if (draft_out != nullptr) {
    fn::Argmax(logits_m_, static_cast<fn::ArgmaxCandidate*>(argmax_scratch_),
               draft_out, 1, vocab, stream_);
    if (pruned)
      fn::RemapIds(reinterpret_cast<std::uint32_t*>(draft_out), 1, draft_map_,
                   stream_);
  }
  if (candidates) {
    fn::MtpTopCandidates(
        logits_m_, cand_ws_, cand_scratch_,
        reinterpret_cast<float*>(cand_ws_ + Candidates::kCount), vocab,
        stream_);
    if (pruned)
      fn::RemapIds(cand_ws_, Candidates::kCount, draft_map_, stream_);
    fn::CopyMapped(cand_ws_, draft_host_, sizeof(Candidates), stream_);
  }
}

void Engine::TopCandidates(std::uint32_t rows) {
  constexpr std::size_t kWords = sizeof(Candidates) / sizeof(float);
  static_assert(sizeof(Candidates) == 2 * Candidates::kCount * 4);
  for (std::uint32_t r = 0; r < rows; ++r) {
    fn::MtpTopCandidates(
        logits_ + static_cast<std::size_t>(r) * c_.vocab, cand_ws_,
        cand_scratch_, reinterpret_cast<float*>(cand_ws_ + Candidates::kCount),
        c_.vocab, stream_);
    fn::CopyDevice(reinterpret_cast<const float*>(cand_ws_),
                   cand_out_ + r * kWords, kWords, stream_);
  }
}

bool Engine::Run(const std::function<void()>& body, std::uint64_t key,
                 std::string* error) {
  if (!options_.graphs || options_.spec_profile) {
    body();
    return Ok(hipGetLastError(), "speculative launch", error);
  }
  auto it = spec_graphs_.find(key);
  if (it == spec_graphs_.end()) {
    if (spec_warm_.insert(key).second) {
      // The first run is eager, so every lazily sized pool exists.
      body();
      return Ok(hipGetLastError(), "speculative launch", error);
    }
    hipGraph_t g = nullptr;
    hipGraphExec_t exec = nullptr;
    if (!Ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed),
            "begin capture", error))
      return false;
    body();
    if (!Ok(hipStreamEndCapture(stream_, &g), "end capture", error) ||
        !Ok(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0),
            "graph instantiate", error))
      return false;
    (void)hipGraphDestroy(g);
    it = spec_graphs_.emplace(key, exec).first;
  }
  return Ok(hipGraphLaunch(it->second, stream_), "graph launch", error);
}

void Engine::SpecTick(int slot) {
  if (!options_.spec_profile)
    return;
  (void)hipStreamSynchronize(stream_);
  const auto now = std::chrono::steady_clock::now();
  if (slot >= 0)
    spec_ms_[slot] +=
        std::chrono::duration<double, std::milli>(now - spec_mark_).count();
  if (slot == kSpecCandidates)
    ++spec_steps_;
  spec_mark_ = now;
}

void Engine::PrintSpecProfile() const {
  if (spec_steps_ == 0)
    return;
  static constexpr const char* kNames[kSpecSlots] = {
      "mtp catch-up", "mtp chain", "verify trunk", "candidates", "commit"};
  double total = 0;
  for (double ms : spec_ms_)
    total += ms;
  std::fprintf(stderr, "\na3b speculative profile over %llu steps (synced):\n",
               static_cast<unsigned long long>(spec_steps_));
  for (int i = 0; i < kSpecSlots; ++i)
    std::fprintf(stderr, "  %-13s %7.3f ms/step %5.1f%%\n", kNames[i],
                 spec_ms_[i] / spec_steps_, 100.0 * spec_ms_[i] / total);
  std::fprintf(stderr, "  %-13s %7.3f ms/step\n", "total", total / spec_steps_);
}

bool Engine::SpecPrefill(std::span<const std::int32_t> tokens, Candidates* last,
                         std::string* error) {
  if (options_.prefill_chunk > 0)
    return SpecPrefillWide(tokens, last, error);
  const std::uint32_t H = c_.hidden;
  const bool mtp = HasMtp() && mtp_k_ != nullptr;
  if (tokens.empty() ||
      pos_ + tokens.size() + kMaxRows > options_.max_context) {
    Fail(error, "prompt does not fit the context");
    return false;
  }
  const auto end = static_cast<std::uint32_t>(pos_ + tokens.size());
  for (std::size_t off = 0; off < tokens.size(); off += kMaxRows) {
    const auto n = static_cast<std::uint32_t>(
        std::min<std::size_t>(kMaxRows, tokens.size() - off));
    const bool final = off + n == tokens.size();
    // The pinned staging is read when the queued copy runs.
    if (!Sync(error))
      return false;
    std::copy_n(tokens.data() + off, n, tokens_host_);
    *pos_host_ = pos_;
    if (!Ok(hipMemcpyAsync(tokens_dev_, tokens_host_, n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, stream_),
            "token upload", error) ||
        !Ok(hipMemcpyAsync(pos_dev_, pos_host_, sizeof(std::uint32_t),
                           hipMemcpyHostToDevice, stream_),
            "position upload", error))
      return false;
    BodyFused(n, final ? 1 : 0, tokens_dev_, pos_dev_, mtp, false, pos_);
    if (df_ && !DFlashInject(n, pos_, end, false, pos_dev_, error))
      return false;
    if (mtp) {
      // MTP rows at the same positions pair token i with hidden i - 1: the
      // pending row from the previous chunk (zeros at position 0), then this
      // chunk's rows but the last, which becomes the new pending row.
      fn::CopyDevice(mtp_pending_, mtp_h_, H, stream_);
      if (n > 1)
        fn::CopyDevice(target_h_, mtp_h_ + H,
                       static_cast<std::size_t>(n - 1) * H, stream_);
      fn::CopyDevice(target_h_ + static_cast<std::size_t>(n - 1) * H,
                     mtp_pending_, H, stream_);
      MtpPass(tokens_dev_, mtp_h_, n, pos_dev_, nullptr, false);
    }
    pos_ += n;
    if (final)
      TopCandidates(1);
  }
  if (mtp) {
    fn::CopyDevice(mtp_pending_, mtp_h_, H, stream_);
    mtp_rows_ = 1;
  }
  verify_rows_ = 0;
  if (!Ok(hipGetLastError(), "prefill launch", error) ||
      !Ok(hipMemcpyAsync(last, cand_out_, sizeof(Candidates),
                         hipMemcpyDeviceToHost, stream_),
          "candidates download", error))
    return false;
  return Sync(error);
}

bool Engine::SpecStep(std::int32_t x, std::uint32_t draft,
                      std::vector<std::int32_t>* drafts,
                      std::vector<Candidates>* rows, std::string* error) {
  const std::uint32_t H = c_.hidden;
  const bool mtp = draft > 0;
  if (mtp && (mtp_k_ == nullptr || mtp_rows_ == 0)) {
    Fail(error, "MTP drafting needs an MTP file and SpecPrefill");
    return false;
  }
  if (draft + 1 > kMaxRows || pos_ + draft + 1 > options_.max_context) {
    Fail(error, "speculative step does not fit");
    return false;
  }
  const std::uint32_t n = draft + 1;
  const std::uint32_t r = mtp ? mtp_rows_ : 0;
  // Staging: [0] x, [1] trunk position, [2] MTP pass position, [3 + j]
  // chained draft j's position. The step's graph reads it from stage_dev_.
  if (!Sync(error))
    return false;
  stage_host_[0] = static_cast<std::uint32_t>(x);
  stage_host_[1] = pos_;
  stage_host_[2] = pos_ + 1 - r;
  for (std::uint32_t j = 1; j < draft; ++j)
    stage_host_[2 + j] = pos_ + j;
  const auto body = [&] {
    SpecTick(-1);
    fn::CopyMapped(stage_host_, stage_dev_, kStageWords * 4, stream_);
    auto* x_dev = reinterpret_cast<const float*>(stage_dev_);
    fn::CopyDevice(x_dev, reinterpret_cast<float*>(vtok_), 1, stream_);
    if (mtp) {
      // Catch-up rows (accepted drafts, then x) seeded by the trunk's
      // hiddens one position back; the last row drafts d1.
      fn::CopyDevice(x_dev, reinterpret_cast<float*>(mtp_tok_ + r - 1), 1,
                     stream_);
      MtpPass(mtp_tok_, mtp_h_, r, stage_dev_ + 2, vtok_ + 1, false);
      SpecTick(kSpecCatchUp);
      for (std::uint32_t j = 1; j < draft; ++j)
        MtpPass(vtok_ + j, mtp_hn_, 1, stage_dev_ + 2 + j, vtok_ + 1 + j,
                false);
      SpecTick(kSpecChain);
    }
    BodyFused(n, n, vtok_, stage_dev_ + 1, mtp, mtp);
    SpecTick(kSpecVerify);
    TopCandidates(n);
    fn::CopyMapped(vtok_, down_host_, kMaxRows * sizeof(std::int32_t), stream_);
    fn::CopyMapped(cand_out_, down_host_ + kMaxRows * sizeof(std::int32_t),
                   n * sizeof(Candidates), stream_);
    SpecTick(kSpecCandidates);
  };
  if (!Run(body, kKeyStep | (std::uint64_t{draft} << 8) | r, error) ||
      !Sync(error))
    return false;
  const auto* toks = reinterpret_cast<const std::int32_t*>(down_host_);
  drafts->assign(toks + 1, toks + n);
  rows->resize(n);
  std::memcpy(rows->data(), down_host_ + kMaxRows * sizeof(std::int32_t),
              n * sizeof(Candidates));
  verify_rows_ = n;
  step_mtp_ = mtp;
  (void)H;
  return true;
}

bool Engine::SpecCommit(std::uint32_t keep, std::string* error) {
  const std::uint32_t H = c_.hidden;
  const std::uint32_t n = verify_rows_;
  if (n == 0 || keep == 0 || keep > n) {
    Fail(error, "commit outside the pending speculative step");
    return false;
  }
  const bool mtp = step_mtp_;
  const auto body = [&] {
    if (keep < n) {
      const std::size_t conv =
          static_cast<std::size_t>(c_.ssm_conv_kernel - 1) * c_.ConvChannels();
      for (std::uint32_t i = 0; i < c_.layers; ++i) {
        if (!layers_[i].linear)
          continue;
        fn::RollbackRows states;
        for (std::size_t row = 0; row < 7; ++row)
          states.rows[row] = state_snap_[i][row];
        fn::RestoreGdnState(ssm_state_[i], states, keep, c_.ssm_k_heads,
                            c_.ssm_v_heads, stream_);
        fn::CopyDevice(conv_snap_[i][keep - 1], conv_state_[i], conv, stream_);
      }
    }
    if (mtp) {
      // Next catch-up: tokens x_{p+1..p+keep-1} (the accepted drafts; the
      // next x joins at the step) seeded by h_p..h_{p+keep-1}.
      fn::CopyDevice(target_h_, mtp_h_, static_cast<std::size_t>(keep) * H,
                     stream_);
      if (keep > 1)
        fn::CopyDevice(reinterpret_cast<const float*>(vtok_ + 1),
                       reinterpret_cast<float*>(mtp_tok_), keep - 1, stream_);
    }
  };
  SpecTick(-1);
  if (!Run(body,
           kKeyCommit | (std::uint64_t{mtp} << 16) | (std::uint64_t{n} << 8) |
               keep,
           error))
    return false;
  SpecTick(kSpecCommit);
  // The kept rows' features (captured by the verify) join the draft context;
  // stage_dev_[1] holds the verify's first position.
  if (df_ && !DFlashInject(keep, pos_, 0, false, stage_dev_ + 1, error))
    return false;
  pos_ += keep;
  if (mtp)
    mtp_rows_ = keep;
  verify_rows_ = 0;
  step_mtp_ = false;
  return true;
}

bool Engine::DraftFirst(std::int32_t x, Candidates* q, std::string* error) {
  if (mtp_k_ == nullptr || mtp_rows_ == 0 || verify_rows_ != 0) {
    Fail(error, "DraftFirst needs an MTP file, SpecPrefill and a commit");
    return false;
  }
  const std::uint32_t r = mtp_rows_;
  if (!Sync(error))
    return false;
  stage_host_[0] = static_cast<std::uint32_t>(x);
  stage_host_[2] = pos_ + 1 - r;
  const auto body = [&] {
    SpecTick(-1);
    fn::CopyMapped(stage_host_, stage_dev_, kStageWords * 4, stream_);
    // Catch-up rows (accepted drafts, then x) seeded by the trunk's
    // hiddens one position back; the last row proposes d1.
    fn::CopyDevice(reinterpret_cast<const float*>(stage_dev_),
                   reinterpret_cast<float*>(mtp_tok_ + r - 1), 1, stream_);
    MtpPass(mtp_tok_, mtp_h_, r, stage_dev_ + 2, nullptr, true);
    SpecTick(kSpecCatchUp);
  };
  if (!Run(body, kKeyDraftFirst | r, error) || !Sync(error))
    return false;
  std::memcpy(q, draft_host_, sizeof(Candidates));
  step_mtp_ = true;
  draft_depth_ = 1;
  return true;
}

bool Engine::DraftNext(std::int32_t token, Candidates* q, std::string* error) {
  if (!step_mtp_ || draft_depth_ + 1 >= kMaxRows) {
    Fail(error, "DraftNext outside a drafting step");
    return false;
  }
  if (!Sync(error))
    return false;
  stage_host_[3] = pos_ + draft_depth_;
  stage_host_[kStageChainToken] = static_cast<std::uint32_t>(token);
  const auto body = [&] {
    SpecTick(-1);
    fn::CopyMapped(stage_host_, stage_dev_, kStageWords * 4, stream_);
    MtpPass(
        reinterpret_cast<const std::int32_t*>(stage_dev_ + kStageChainToken),
        mtp_hn_, 1, stage_dev_ + 3, nullptr, true);
    SpecTick(kSpecChain);
  };
  if (!Run(body, kKeyDraftNext, error) || !Sync(error))
    return false;
  std::memcpy(q, draft_host_, sizeof(Candidates));
  ++draft_depth_;
  return true;
}

bool Engine::Verify(std::int32_t x, std::span<const std::int32_t> drafts,
                    std::vector<Candidates>* rows, std::string* error,
                    bool catch_up) {
  const auto n = static_cast<std::uint32_t>(drafts.size() + 1);
  if (n > kMaxRows || verify_rows_ != 0 || pos_ + n > options_.max_context) {
    Fail(error, "verify does not fit");
    return false;
  }
  if (!Sync(error))
    return false;
  // The MTP rows still to catch up (none after DraftFirst).
  const std::uint32_t r =
      catch_up && !step_mtp_ && mtp_k_ != nullptr ? mtp_rows_ : 0;
  stage_host_[1] = pos_;
  stage_host_[2] = pos_ + 1 - r;
  stage_host_[kStageVerify] = static_cast<std::uint32_t>(x);
  for (std::uint32_t j = 0; j + 1 < n; ++j)
    stage_host_[kStageVerify + 1 + j] = static_cast<std::uint32_t>(drafts[j]);
  const bool mtp = step_mtp_ || r > 0;
  const auto body = [&] {
    SpecTick(-1);
    fn::CopyMapped(stage_host_, stage_dev_, kStageWords * 4, stream_);
    if (r > 0) {
      fn::CopyDevice(reinterpret_cast<const float*>(stage_dev_ + kStageVerify),
                     reinterpret_cast<float*>(mtp_tok_ + r - 1), 1, stream_);
      MtpPass(mtp_tok_, mtp_h_, r, stage_dev_ + 2, nullptr, false);
      SpecTick(kSpecCatchUp);
    }
    fn::CopyDevice(reinterpret_cast<const float*>(stage_dev_ + kStageVerify),
                   reinterpret_cast<float*>(vtok_), n, stream_);
    BodyFused(n, n, vtok_, stage_dev_ + 1, mtp, n > 1);
    SpecTick(kSpecVerify);
    TopCandidates(n);
    fn::CopyMapped(cand_out_, down_host_ + kMaxRows * sizeof(std::int32_t),
                   n * sizeof(Candidates), stream_);
    SpecTick(kSpecCandidates);
  };
  if (!Run(
          body,
          kKeyVerify | (std::uint64_t{r} << 16) | (std::uint64_t{mtp} << 8) | n,
          error) ||
      !Sync(error))
    return false;
  rows->resize(n);
  std::memcpy(rows->data(), down_host_ + kMaxRows * sizeof(std::int32_t),
              n * sizeof(Candidates));
  verify_rows_ = n;
  step_mtp_ = mtp;
  return true;
}

void Engine::Body(std::uint32_t n, std::uint32_t rows) {
  if (options_.fused) {
    BodyFused(n, rows, tokens_dev_, pos_dev_, false, false);
    return;
  }
  const std::uint32_t H = c_.hidden;
  Mark(kPhEmbed);
  fn::EmbedTokens(token_embd_.data, Small(token_embd_.type), tokens_dev_, res_,
                  n, H, 1, stream_);
  const std::size_t count = static_cast<std::size_t>(n) * H;
  for (std::uint32_t i = 0; i < c_.layers; ++i) {
    const Layer& l = layers_[i];
    Mark(kPhNorm);
    if (!Skip(kPhNorm))
      fn::RmsNormRows(res_, l.attn_norm.f32(), xn_, n, H, 1, c_.eps, stream_);
    if (l.linear) {
      LinearAttention(l, i, n);
    } else {
      Attention(l, i, n);
    }
    Mark(kPhNorm);
    if (!Skip(kPhNorm)) {
      AddInPlace(res_, blk_, count, stream_);
      fn::RmsNormRows(res_, l.post_norm.f32(), xn_, n, H, 1, c_.eps, stream_);
    }
    Moe(l, n);
    Mark(kPhNorm);
    if (!Skip(kPhNorm))
      AddInPlace(res_, blk_, count, stream_);
  }
  if (rows == 0)
    return;
  Mark(kPhHead);
  fn::RmsNormRows(res_ + static_cast<std::size_t>(n - rows) * H,
                  output_norm_.f32(), xn_, rows, H, 1, c_.eps, stream_);
  if (!Skip(kPhHead))
    Dense(output_, xn_, logits_, rows);
  Mark(kPhCount);  // end sentinel
}

bool Engine::Forward(std::span<const std::int32_t> tokens, float* logits,
                     std::string* error, bool all_rows) {
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (n == 0 || n > kMaxRows) {
    Fail(error, "Forward takes 1-8 tokens");
    return false;
  }
  if (pos_ + n > options_.max_context) {
    Fail(error, "context full");
    return false;
  }
  // The pinned staging is read when the queued copy executes, not when it
  // is enqueued: a previous unsynchronized chunk must finish first.
  if (!Sync(error))
    return false;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  *pos_host_ = pos_;
  if (!Ok(hipMemcpyAsync(tokens_dev_, tokens_host_, n * sizeof(std::int32_t),
                         hipMemcpyHostToDevice, stream_),
          "token upload", error) ||
      !Ok(hipMemcpyAsync(pos_dev_, pos_host_, sizeof(std::uint32_t),
                         hipMemcpyHostToDevice, stream_),
          "position upload", error))
    return false;
  const std::uint32_t rows = logits == nullptr ? 0 : (all_rows ? n : 1);
  // One-token decode replays a captured graph: ~1100 eager launches per
  // token otherwise. The first decode runs eagerly so the quantized tier's
  // pools hold their buffers before capture.
  const bool graph =
      options_.graphs && !options_.profile && n == 1 && rows == 1;
  profiling_ = options_.profile && n == 1 && rows == 1;
  marks_.clear();
  host_ms_.clear();
  if (graph && decode_exec_ == nullptr && decode_warm_) {
    hipGraph_t g = nullptr;
    if (!Ok(hipStreamBeginCapture(stream_, hipStreamCaptureModeRelaxed),
            "begin capture", error))
      return false;
    Body(1, 1);
    if (!Ok(hipStreamEndCapture(stream_, &g), "end capture", error) ||
        !Ok(hipGraphInstantiate(&decode_exec_, g, nullptr, nullptr, 0),
            "graph instantiate", error))
      return false;
    (void)hipGraphDestroy(g);
  }
  if (graph && decode_exec_ != nullptr) {
    if (!Ok(hipGraphLaunch(decode_exec_, stream_), "graph launch", error))
      return false;
  } else {
    Body(n, rows);
    decode_warm_ = decode_warm_ || graph;
  }
  pos_ += n;
  if (logits == nullptr) {
    // Flush queued launches (Windows batches them until a query).
    (void)hipStreamQuery(stream_);
    return Ok(hipGetLastError(), "forward launch", error);
  }
  return Ok(hipGetLastError(), "forward launch", error) &&
         Ok(hipMemcpyAsync(
                logits, logits_,
                static_cast<std::size_t>(rows) * c_.vocab * sizeof(float),
                hipMemcpyDeviceToHost, stream_),
            "logits download", error) &&
         Sync(error) && (profiling_ ? (Collect(), true) : true);
}

}  // namespace gufo::models::qwen36_a3b
