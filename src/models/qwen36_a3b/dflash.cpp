// DFlash-2 drafter for the A3B engine (see dflash.hpp).
#include "src/models/qwen36_a3b/dflash.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <variant>

#include "src/models/qwen/hip/kernels/dflash_kernels.hpp"
#include "src/models/qwen36_a3b/dflash_kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen36_a3b {
namespace {

namespace fn = gufo::models::qwen38_flash_next::rocm;
namespace dk = gufo::hip::kernels;
using core::GgmlType;

constexpr std::uint32_t kPosWord = 16;

// A3B_DFLASH_PROFILE: host wall time of the draft passes and injections
// (each synced), printed every 100 drafts.
struct DFlashClock {
  bool on = std::getenv("A3B_DFLASH_PROFILE") != nullptr;
  double draft_ms = 0, inject_ms = 0;
  std::uint64_t drafts = 0, injects = 0;
} g_clock;

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

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

float HalfToFloat(std::uint16_t h) {
  return static_cast<float>(std::bit_cast<_Float16>(h));
}

std::uint16_t FloatToHalf(float f) {
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(f));
}

std::uint16_t FloatToBf16(float f) {
  std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  u += 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
  return static_cast<std::uint16_t>(u >> 16);
}

/// The tensor's values as F32 (F32, F16 or BF16 files).
bool ReadFloats(const core::GgufTensorInfo& t, std::vector<float>* out,
                std::string* error) {
  const std::size_t n = t.ElementCount();
  out->resize(n);
  switch (t.type) {
    case GgmlType::kF32:
      std::memcpy(out->data(), t.data, n * 4);
      return true;
    case GgmlType::kF16: {
      const auto* s = static_cast<const std::uint16_t*>(t.data);
      for (std::size_t i = 0; i < n; ++i)
        (*out)[i] = HalfToFloat(s[i]);
      return true;
    }
    case GgmlType::kBF16: {
      const auto* s = static_cast<const std::uint16_t*>(t.data);
      for (std::size_t i = 0; i < n; ++i)
        (*out)[i] =
            std::bit_cast<float>(static_cast<std::uint32_t>(s[i]) << 16);
      return true;
    }
    default:
      Fail(error,
           "DFlash tensor " + std::string(t.name) + " has an unsupported type");
      return false;
  }
}

/// GGML Q8_0 rows: per 32 values, d = amax / 127 (F16) and round(x / d).
std::vector<std::uint8_t> QuantizeQ8(const std::vector<float>& x,
                                     std::size_t cols) {
  const std::size_t blocks = x.size() / 32;
  std::vector<std::uint8_t> out(blocks * 34);
  (void)cols;
  for (std::size_t b = 0; b < blocks; ++b) {
    const float* v = x.data() + b * 32;
    float amax = 0;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(v[i]));
    const float d = amax / 127.0f;
    const float id = d != 0 ? 1.0f / d : 0.0f;
    std::uint8_t* o = out.data() + b * 34;
    const std::uint16_t dh = FloatToHalf(d);
    std::memcpy(o, &dh, 2);
    for (int i = 0; i < 32; ++i)
      o[2 + i] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>(std::lround(v[i] * id)));
  }
  return out;
}

}  // namespace

DFlash::~DFlash() {
  for (void* p : owned)
    if (p != nullptr)
      (void)hipFree(p);
  if (host != nullptr)
    (void)hipHostFree(host);
  if (pos_host != nullptr)
    (void)hipHostFree(pos_host);
}

bool Engine::LoadDFlash(const std::filesystem::path& path, std::string* error) {
  auto reader = core::GgufReader::OpenFile(path, error);
  if (!reader)
    return false;
  const auto arch = reader->GetMetadataString("general.architecture");
  if (!arch || *arch != "dflash") {
    Fail(error, "not a DFlash GGUF");
    return false;
  }
  auto d = std::make_shared<DFlash>();
  const auto u32 = [&](const char* key, std::uint32_t* out) {
    const auto v = reader->GetMetadataUint32(std::string("dflash.") + key);
    if (!v)
      Fail(error, std::string("DFlash GGUF lacks dflash.") + key);
    else
      *out = *v;
    return v.has_value();
  };
  std::uint32_t n_layers = 0;
  if (!u32("embedding_length", &d->hidden) ||
      !u32("attention.head_count", &d->heads) ||
      !u32("attention.head_count_kv", &d->kv_heads) ||
      !u32("attention.key_length", &d->head_dim) ||
      !u32("feed_forward_length", &d->ff) || !u32("block_count", &n_layers) ||
      !u32("block_size", &d->block_size) ||
      !u32("conv_kernel_size", &d->conv_k) ||
      !u32("conv_group_size", &d->conv_g) ||
      !u32("selector_rank", &d->sel_rank) ||
      !u32("selector_top_k", &d->sel_topk) ||
      !u32("attention.sliding_window", &d->window))
    return false;
  if (!reader->GetMetadataUint32("dflash.rope.dimension_count")) {
    d->rotary = d->head_dim;
  } else {
    d->rotary = *reader->GetMetadataUint32("dflash.rope.dimension_count");
  }
  d->theta = reader->GetMetadataFloat32("dflash.rope.freq_base").value_or(1e7f);
  d->eps = reader->GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon")
               .value_or(1e-6f);
  const auto mask = reader->GetMetadataUint32("tokenizer.ggml.mask_token_id");
  if (!mask) {
    Fail(error, "DFlash GGUF lacks tokenizer.ggml.mask_token_id");
    return false;
  }
  d->mask_token = *mask;
  if (reader->GetMetadataBool("dflash.attention.causal").value_or(false)) {
    Fail(error, "causal DFlash drafts are not supported");
    return false;
  }
  if (d->hidden != c_.hidden || d->hidden % 1024 != 0 || d->heads == 0 ||
      d->heads % d->kv_heads != 0 || d->hidden % d->conv_g != 0 ||
      d->sel_topk > 16 || d->mask_token >= c_.vocab || d->block_size < 2) {
    Fail(error, "DFlash draft does not match the target (hidden " +
                    std::to_string(d->hidden) + " vs " +
                    std::to_string(c_.hidden) + ")");
    return false;
  }
  // Target taps: layer-input indices N = trunk layer N - 1's output.
  if (const auto* m = reader->FindMetadata("dflash.target_layers")) {
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&m->value))
      for (const auto x : *u)
        d->taps.push_back(static_cast<std::uint32_t>(x));
    if (const auto* s = std::get_if<std::vector<std::int64_t>>(&m->value))
      for (const auto x : *s)
        d->taps.push_back(static_cast<std::uint32_t>(x));
  }
  if (d->taps.empty()) {
    Fail(error, "DFlash GGUF lacks dflash.target_layers");
    return false;
  }
  d->tap_slot.assign(c_.layers, -1);
  for (std::size_t i = 0; i < d->taps.size(); ++i) {
    if (d->taps[i] == 0 || d->taps[i] > c_.layers) {
      Fail(error, "DFlash target layer out of range");
      return false;
    }
    d->taps[i] -= 1;
    d->tap_slot[d->taps[i]] = static_cast<int>(i);
  }
  d->capacity = std::min(options_.max_context, d->window);

  const auto device = [&](const void* data, std::size_t bytes) -> void* {
    void* p = nullptr;
    if (!Ok(hipMalloc(&p, std::max<std::size_t>(bytes, 256)), "dflash alloc",
            error))
      return nullptr;
    d->owned.push_back(p);
    if (data != nullptr && !Ok(hipMemcpy(p, data, bytes, hipMemcpyHostToDevice),
                               "dflash upload", error))
      return nullptr;
    return p;
  };
  const auto find = [&](const std::string& name,
                        std::size_t elements) -> const core::GgufTensorInfo* {
    const auto* t = reader->FindTensor(name);
    if (t == nullptr) {
      Fail(error, "DFlash GGUF lacks " + name);
      return nullptr;
    }
    if (elements != 0 && t->ElementCount() != elements) {
      Fail(error, "DFlash tensor " + name + " has the wrong size");
      return nullptr;
    }
    return t;
  };
  std::vector<float> tmp;
  // Matrices: [rows][cols] (GGUF dims {cols, rows}) to Q8_0.
  const auto matrix = [&](const std::string& name, std::uint32_t rows,
                          std::uint32_t cols, Tensor* out) {
    const auto* t = find(name, static_cast<std::size_t>(rows) * cols);
    if (t == nullptr || cols % 32 != 0)
      return false;
    const void* data = nullptr;
    if (t->type == GgmlType::kQ8_0) {
      data = device(t->data, static_cast<std::size_t>(rows) * cols / 32 * 34);
    } else {
      if (!ReadFloats(*t, &tmp, error))
        return false;
      const auto q = QuantizeQ8(tmp, cols);
      data = device(q.data(), q.size());
    }
    if (data == nullptr)
      return false;
    *out = Tensor{data, GgmlType::kQ8_0, cols, rows, 1};
    return true;
  };
  const auto vector = [&](const std::string& name, std::size_t elements,
                          const float** out) {
    const auto* t = find(name, elements);
    if (t == nullptr || !ReadFloats(*t, &tmp, error))
      return false;
    *out = static_cast<const float*>(device(tmp.data(), tmp.size() * 4));
    return *out != nullptr;
  };
  const auto bf16 = [&](const std::string& name, std::size_t elements,
                        const void** out) {
    const auto* t = find(name, elements);
    if (t == nullptr || !ReadFloats(*t, &tmp, error))
      return false;
    std::vector<std::uint16_t> b(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i)
      b[i] = FloatToBf16(tmp[i]);
    *out = device(b.data(), b.size() * 2);
    return *out != nullptr;
  };

  const std::uint32_t H = d->hidden;
  const std::uint32_t qd = d->heads * d->head_dim;
  const std::uint32_t kvd = d->kv_heads * d->head_dim;
  const std::uint32_t F = static_cast<std::uint32_t>(d->taps.size()) * H;
  const std::uint32_t dyn = 2 * d->conv_k * (H / d->conv_g);
  const std::size_t base = static_cast<std::size_t>(H) * d->conv_k * 2;
  const std::size_t codebook = static_cast<std::size_t>(c_.vocab) * d->sel_rank;
  if (!matrix("fc.weight", H, F, &d->fc) ||
      !vector("enc.output_norm.weight", H, &d->enc_norm) ||
      !vector("output_norm.weight", H, &d->out_norm) ||
      !matrix("selector_hidden.weight", d->sel_rank, H, &d->sel_hidden) ||
      !bf16("selector_predecessor.weight", codebook, &d->pred) ||
      !bf16("selector_successor.weight", codebook, &d->succ))
    return false;
  d->layers.resize(n_layers);
  const std::size_t ring = static_cast<std::size_t>(d->capacity) * kvd * 4;
  for (std::uint32_t i = 0; i < n_layers; ++i) {
    DFlashLayer& l = d->layers[i];
    const std::string p = "blk." + std::to_string(i) + ".";
    if (!vector(p + "attn_norm.weight", H, &l.attn_norm) ||
        !vector(p + "ffn_norm.weight", H, &l.ffn_norm) ||
        !vector(p + "attn_q_norm.weight", d->head_dim, &l.q_norm) ||
        !vector(p + "attn_k_norm.weight", d->head_dim, &l.k_norm) ||
        !vector(p + "attn_conv_base", base, &l.attn_conv_base) ||
        !vector(p + "ffn_conv_base", base, &l.ffn_conv_base) ||
        !matrix(p + "attn_q.weight", qd, H, &l.q) ||
        !matrix(p + "attn_k.weight", kvd, H, &l.k) ||
        !matrix(p + "attn_v.weight", kvd, H, &l.v) ||
        !matrix(p + "attn_output.weight", H, qd, &l.o) ||
        !matrix(p + "ffn_gate.weight", d->ff, H, &l.gate) ||
        !matrix(p + "ffn_up.weight", d->ff, H, &l.up) ||
        !matrix(p + "ffn_down.weight", H, d->ff, &l.down) ||
        !matrix(p + "attn_conv_proj.weight", dyn, H, &l.attn_conv_proj) ||
        !matrix(p + "ffn_conv_proj.weight", dyn, H, &l.ffn_conv_proj))
      return false;
    l.ring_k = static_cast<float*>(device(nullptr, ring));
    l.ring_v = static_cast<float*>(device(nullptr, ring));
    if (l.ring_k == nullptr || l.ring_v == nullptr)
      return false;
  }

  // Scratch.
  const std::size_t R = kMaxRows;
  const auto f32 = [&](std::size_t n) {
    return static_cast<float*>(device(nullptr, n * 4));
  };
  const std::size_t max_k = std::max({F, qd, d->ff, H});
  d->feat = f32(R * F);
  if (options_.prefill_chunk > 0)
    d->wide_feat = f32(static_cast<std::size_t>(options_.prefill_chunk) * F);
  d->h = f32(R * H);
  d->xn = f32(R * H);
  d->dyn = f32(R * dyn);
  d->conv = f32(R * H);
  d->q = f32(R * qd);
  d->k = f32(R * kvd);
  d->v = f32(R * kvd);
  d->att = f32(R * qd);
  d->o = f32(R * H);
  d->f = f32(R * d->ff);
  d->sel = f32(R * d->sel_rank);
  d->logits = f32(R * c_.vocab);
  d->conf = f32(R);
  const std::size_t scratch = dk::DFlashSelectorScratchElements(c_.vocab);
  d->partial_scores = f32(R * scratch);
  d->partial_ids =
      static_cast<std::uint32_t*>(device(nullptr, R * scratch * 4));
  d->tok = static_cast<std::uint32_t*>(device(nullptr, 64 * 4));
  d->pos = static_cast<std::uint32_t*>(device(nullptr, 4));
  d->max_groups =
      std::max<std::uint32_t>(options_.prefill_chunk, kMaxRows) / kMaxRows + 1;
  d->pos_list = static_cast<std::uint32_t*>(device(nullptr, d->max_groups * 4));
  d->aq = static_cast<Q8_1Block*>(
      device(nullptr, R * (max_k / 32) * sizeof(Q8_1Block)));
  {
    const std::vector<float> ones(max_k, 1.0f);
    d->ones = static_cast<float*>(device(ones.data(), max_k * 4));
  }
  if (d->feat == nullptr || d->h == nullptr || d->logits == nullptr ||
      d->ones == nullptr || d->aq == nullptr || d->pos_list == nullptr ||
      (options_.prefill_chunk > 0 && d->wide_feat == nullptr))
    return false;
  if (!Ok(hipHostMalloc(reinterpret_cast<void**>(&d->host), 64 * 4),
          "dflash pinned", error) ||
      !Ok(hipHostMalloc(reinterpret_cast<void**>(&d->pos_host),
                        d->max_groups * 4),
          "dflash pinned", error))
    return false;
  // Pruned head: the engine's draft vocab (also the MTP's), its codebooks.
  if (draft_head_ == nullptr && !options_.draft_vocab_ids.empty() &&
      !BuildDraftHead(options_.draft_vocab_ids, error))
    return false;
  if (draft_head_ == nullptr && !options_.draft_vocab.empty() &&
      !LoadDraftVocab(error))
    return false;
  if (draft_head_ != nullptr) {
    const std::size_t row = static_cast<std::size_t>(d->sel_rank) * 2;
    d->pred_p = device(nullptr, (draft_vocab_ + 1) * row);
    d->succ_p = device(nullptr, draft_vocab_ * row);
    if (d->pred_p == nullptr || d->succ_p == nullptr)
      return false;
    fn::GatherRows(d->pred, d->pred_p, draft_map_, draft_vocab_, row, stream_);
    fn::GatherRows(d->succ, d->succ_p, draft_map_, draft_vocab_, row, stream_);
    if (!Ok(hipStreamSynchronize(stream_), "dflash codebook gather", error))
      return false;
  }
  std::size_t bytes = 0;
  for (const auto& t : reader->GetTensors())
    bytes += t.size_bytes;
  std::fprintf(stderr,
               "a3b: DFlash draft %u layers, block %u, %zu taps, window %u, "
               "%.2f GiB file -> Q8_0\n",
               n_layers, d->block_size, d->taps.size(), d->window,
               static_cast<double>(bytes) / (1ULL << 30));
  df_ = std::move(d);
  return true;
}

void Engine::DFlashTap(std::uint32_t layer, const float* res, std::uint32_t n,
                       bool wide) {
  if (!df_)
    return;
  const int slot = df_->tap_slot[layer];
  if (slot < 0)
    return;
  const std::uint32_t H = c_.hidden;
  const std::uint32_t F = static_cast<std::uint32_t>(df_->taps.size()) * H;
  float* feat = wide ? df_->wide_feat : df_->feat;
  TapCopy(res, feat + static_cast<std::size_t>(slot) * H, n, H, F, stream_);
}

bool Engine::DFlashInject(std::uint32_t n, std::uint32_t pos0,
                          std::uint32_t end, bool wide,
                          const std::uint32_t* pos_dev, std::string* error) {
  DFlash& d = *df_;
  const bool timed = g_clock.on && !wide;
  double t0 = 0;
  if (timed) {
    if (!Sync(error))
      return false;
    t0 = NowMs();
  }
  const std::uint32_t H = d.hidden;
  const std::uint32_t F = static_cast<std::uint32_t>(d.taps.size()) * H;
  const std::uint32_t kvd = d.kv_heads * d.head_dim;
  const float* feat = wide ? d.wide_feat : d.feat;
  // Only the last `capacity` positions before `end` are ever read.
  std::uint32_t first = 0;
  if (end > d.capacity && end - d.capacity > pos0)
    first = std::min(n, end - d.capacity - pos0);
  const std::uint32_t groups = (n - first + kMaxRows - 1) / kMaxRows;
  if (groups > 1 || pos_dev == nullptr || first != 0) {
    // Group start positions for the device position reads.
    if (groups > d.max_groups) {
      Fail(error, "DFlash injection chunk too large");
      return false;
    }
    if (!Sync(error))
      return false;
    for (std::uint32_t g = 0; g < groups; ++g)
      d.pos_host[g] = pos0 + first + g * kMaxRows;
    if (!Ok(hipMemcpyAsync(d.pos_list, d.pos_host, groups * 4,
                           hipMemcpyHostToDevice, stream_),
            "dflash positions", error))
      return false;
    pos_dev = nullptr;
  }
  for (std::uint32_t g = 0; g < groups; ++g) {
    const std::uint32_t t0 = first + g * kMaxRows;
    const std::uint32_t r = std::min(kMaxRows, n - t0);
    const std::uint32_t* pos = pos_dev != nullptr ? pos_dev : d.pos_list + g;
    GateQuant(feat + static_cast<std::size_t>(t0) * F, d.ones, 0,
              GateMode::kMul, d.aq, nullptr, r, F, stream_);
    const GemvSeg fc{d.fc.data,       nullptr,          d.o, H,
                     WeightKind::kQ8, GemvMode::kStore, H};
    MultiGemv(&fc, 1, d.aq, nullptr, r, F, stream_);
    AddNormQuant(d.o, nullptr, d.enc_norm, d.xn, d.aq, r, H, d.eps, stream_);
    for (const DFlashLayer& l : d.layers) {
      const GemvSeg kv[2] = {
          {l.k.data, nullptr, d.k, kvd, WeightKind::kQ8, GemvMode::kStore, kvd},
          {l.v.data, nullptr, d.v, kvd, WeightKind::kQ8, GemvMode::kStore,
           kvd}};
      MultiGemv(kv, 2, d.aq, d.xn, r, H, stream_);
      fn::RmsNormRows(d.k, l.k_norm, d.k, r * d.kv_heads, d.head_dim, 1, d.eps,
                      stream_);
      fn::Rope(d.k, r, d.kv_heads, d.head_dim, d.rotary, pos, d.theta, stream_);
      RingStore(d.k, d.v, l.ring_k, l.ring_v, pos, r, kvd, d.capacity, stream_);
    }
  }
  d.injected = pos0 + n;
  if (timed) {
    if (!Sync(error))
      return false;
    g_clock.inject_ms += NowMs() - t0;
    ++g_clock.injects;
  }
  return Ok(hipGetLastError(), "dflash inject", error);
}

std::uint32_t Engine::DFlashMaxDrafts() const {
  return df_ ? std::min(df_->block_size - 1, kMaxRows - 1) : 0;
}

bool Engine::DFlashDraft(std::int32_t x, std::uint32_t count,
                         std::vector<std::int32_t>* drafts,
                         std::vector<float>* probs, std::string* error) {
  drafts->clear();
  if (g_clock.on) {
    if (!Sync(error))
      return false;
    const double t0 = NowMs();
    const bool ok = DFlashDraftPass(x, count, drafts, probs, error);
    g_clock.draft_ms += NowMs() - t0;
    if (++g_clock.drafts % 100 == 0)
      std::fprintf(
          stderr, "\n[dflash] draft %.2f ms, inject %.2f ms (avg)\n",
          g_clock.draft_ms / g_clock.drafts,
          g_clock.inject_ms / std::max<std::uint64_t>(1, g_clock.injects));
    return ok;
  }
  return DFlashDraftPass(x, count, drafts, probs, error);
}

bool Engine::DFlashDraftPass(std::int32_t x, std::uint32_t count,
                             std::vector<std::int32_t>* drafts,
                             std::vector<float>* probs, std::string* error) {
  if (probs != nullptr)
    probs->clear();
  if (!df_) {
    Fail(error, "no DFlash draft loaded");
    return false;
  }
  DFlash& d = *df_;
  count = std::min(count, DFlashMaxDrafts());
  if (pos_ + count + 1 > options_.max_context)
    count = 0;
  if (count == 0)
    return true;
  if (d.injected != pos_) {
    Fail(error, "DFlash context is not at the current position");
    return false;
  }
  const std::uint32_t b = count + 1;
  const std::uint32_t H = d.hidden;
  const std::uint32_t qd = d.heads * d.head_dim;
  const std::uint32_t kvd = d.kv_heads * d.head_dim;
  // The pinned staging is read when the queued copies run.
  if (!Sync(error))
    return false;
  d.host[0] = static_cast<std::uint32_t>(x);
  for (std::uint32_t i = 1; i < b; ++i)
    d.host[i] = d.mask_token;
  d.host[kPosWord] = pos_;
  // The selector's chain: stok[0] is the anchor, in the draft vocab's index
  // space when pruned (row V' of pred_p, filled with the anchor's row below).
  const bool pruned = d.pred_p != nullptr;
  const std::uint32_t vocab = pruned ? draft_vocab_ : c_.vocab;
  std::uint32_t* stok = pruned ? d.tok + 32 : d.tok;
  d.host[48] = draft_vocab_;
  if (!Ok(hipMemcpyAsync(d.tok, d.host, b * 4, hipMemcpyHostToDevice, stream_),
          "dflash tokens", error) ||
      (pruned &&
       !Ok(hipMemcpyAsync(stok, d.host + 48, 4, hipMemcpyHostToDevice, stream_),
           "dflash tokens", error)) ||
      !Ok(hipMemcpyAsync(d.pos, d.host + kPosWord, 4, hipMemcpyHostToDevice,
                         stream_),
          "dflash position", error))
    return false;
  const auto quant = [&](const float* src, std::uint32_t k) {
    GateQuant(src, d.ones, 0, GateMode::kMul, d.aq, nullptr, b, k, stream_);
  };
  const auto gemv = [&](const Tensor& w, float* out, std::uint32_t k) {
    const GemvSeg s{w.data,          nullptr,          out,   w.rows,
                    WeightKind::kQ8, GemvMode::kStore, w.rows};
    MultiGemv(&s, 1, d.aq, nullptr, b, k, stream_);
  };
  const auto conv = [&](const float* in, const float* base,
                        std::uint32_t side) {
    dk::LaunchDFlashGroupedDynamicConv(in, d.dyn, base, d.conv, b, H, d.conv_k,
                                       d.conv_g, side, stream_);
  };
  fn::EmbedTokens(
      token_embd_.data, static_cast<fn::WeightType>(token_embd_.type),
      reinterpret_cast<const std::int32_t*>(d.tok), d.h, b, H, 1, stream_);
  const float* add = nullptr;
  const float scale = 1.0f / std::sqrt(static_cast<float>(d.head_dim));
  for (const DFlashLayer& l : d.layers) {
    // Attention: conv(norm(h)) -> q, k, v; attend over the injected window
    // and the block; conv of the output (same dynamic coefficients).
    AddNormQuant(d.h, add, l.attn_norm, d.xn, d.aq, b, H, d.eps, stream_);
    gemv(l.attn_conv_proj, d.dyn, H);
    conv(d.xn, l.attn_conv_base, 0);
    quant(d.conv, H);
    const GemvSeg qkv[3] = {
        {l.q.data, nullptr, d.q, qd, WeightKind::kQ8, GemvMode::kStore, qd},
        {l.k.data, nullptr, d.k, kvd, WeightKind::kQ8, GemvMode::kStore, kvd},
        {l.v.data, nullptr, d.v, kvd, WeightKind::kQ8, GemvMode::kStore, kvd}};
    MultiGemv(qkv, 3, d.aq, nullptr, b, H, stream_);
    fn::RmsNormRows(d.q, l.q_norm, d.q, b * d.heads, d.head_dim, 1, d.eps,
                    stream_);
    fn::RmsNormRows(d.k, l.k_norm, d.k, b * d.kv_heads, d.head_dim, 1, d.eps,
                    stream_);
    fn::Rope(d.q, b, d.heads, d.head_dim, d.rotary, d.pos, d.theta, stream_);
    fn::Rope(d.k, b, d.kv_heads, d.head_dim, d.rotary, d.pos, d.theta, stream_);
    dk::LaunchDFlashNonCausalAttention(
        d.q, l.ring_k, l.ring_v, d.k, d.v, d.att, pos_, pos_, b, d.window,
        d.heads, d.kv_heads, d.head_dim, scale, stream_, d.capacity);
    quant(d.att, qd);
    gemv(l.o, d.o, qd);
    conv(d.o, l.attn_conv_base, 1);
    // FFN: h += conv; conv(norm(h)) -> silu(gate) * up -> down -> conv.
    AddNormQuant(d.h, d.conv, l.ffn_norm, d.xn, d.aq, b, H, d.eps, stream_);
    gemv(l.ffn_conv_proj, d.dyn, H);
    conv(d.xn, l.ffn_conv_base, 0);
    quant(d.conv, H);
    const GemvSeg gated{l.gate.data,     l.up.data,        d.f, d.ff,
                        WeightKind::kQ8, GemvMode::kGated, d.ff};
    MultiGemv(&gated, 1, d.aq, nullptr, b, H, stream_);
    quant(d.f, d.ff);
    gemv(l.down, d.o, d.ff);
    conv(d.o, l.ffn_conv_base, 1);
    add = d.conv;
  }
  AddNormQuant(d.h, add, d.out_norm, d.xn, d.aq, b, H, d.eps, stream_);
  // Proposal rows 1..count: selector projection and the target's head.
  const Q8_1Block* aq1 = d.aq + H / 32;
  const float* xn1 = d.xn + H;
  const GemvSeg sel{d.sel_hidden.data, nullptr,          d.sel,     d.sel_rank,
                    WeightKind::kQ8,   GemvMode::kStore, d.sel_rank};
  MultiGemv(&sel, 1, aq1, xn1, count, H, stream_);
  const WeightKind head_kind = output_.type == GgmlType::kQ8_0 ? WeightKind::kQ8
                               : output_.type == GgmlType::kBF16
                                   ? WeightKind::kBf16
                                   : WeightKind::kF32;
  const GemvSeg head{pruned ? draft_head_ : output_.data,
                     nullptr,
                     d.logits,
                     vocab,
                     head_kind,
                     GemvMode::kStore,
                     vocab};
  MultiGemv(&head, 1, aq1, xn1, count, H, stream_);
  if (pruned) {
    const std::size_t words = d.sel_rank / 2;  // one BF16 row as floats
    fn::CopyDevice(
        static_cast<const float*>(d.pred) + static_cast<std::size_t>(x) * words,
        static_cast<float*>(d.pred_p) + draft_vocab_ * words, words, stream_);
  }
  const std::array<dk::DFlashSelectorSequence, 1> seq{
      dk::DFlashSelectorSequence{count, 0.0f}};
  dk::LaunchDFlashSelectorBatch(
      d.logits, d.sel, pruned ? d.pred_p : d.pred, pruned ? d.succ_p : d.succ,
      stok, d.conf, d.partial_scores, d.partial_ids, nullptr, nullptr, nullptr,
      seq, vocab, d.sel_rank, d.sel_topk, stream_);
  TokenProb(d.logits, stok + 1, d.conf, count, vocab, stream_);
  if (pruned)
    fn::RemapIds(stok + 1, count, draft_map_, stream_);
  if (!Ok(hipMemcpyAsync(d.host + 1, stok + 1, count * 4, hipMemcpyDeviceToHost,
                         stream_),
          "dflash drafts", error) ||
      !Ok(hipMemcpyAsync(d.host + 32, d.conf, count * 4, hipMemcpyDeviceToHost,
                         stream_),
          "dflash probabilities", error) ||
      !Sync(error))
    return false;
  for (std::uint32_t i = 1; i <= count; ++i)
    drafts->push_back(static_cast<std::int32_t>(d.host[i]));
  if (probs != nullptr)
    for (std::uint32_t i = 0; i < count; ++i)
      probs->push_back(std::bit_cast<float>(d.host[32 + i]));
  return true;
}

void Engine::DFlashReset() {
  if (df_)
    df_->injected = 0;
}

// Prompt checkpoint: the ring holds the last `capacity` context positions,
// which later tokens overwrite, so it is copied whole (with its position).
bool Engine::DFlashSaveCheckpoint(std::string* error) {
  DFlash& d = *df_;
  const std::size_t floats =
      static_cast<std::size_t>(d.capacity) * d.kv_heads * d.head_dim;
  if (d.ckpt_k.empty()) {
    for (std::size_t i = 0; i < d.layers.size(); ++i) {
      void* k = nullptr;
      void* v = nullptr;
      if (!Ok(hipMalloc(&k, floats * 4), "dflash checkpoint", error) ||
          !Ok(hipMalloc(&v, floats * 4), "dflash checkpoint", error))
        return false;
      d.owned.push_back(k);
      d.owned.push_back(v);
      d.ckpt_k.push_back(static_cast<float*>(k));
      d.ckpt_v.push_back(static_cast<float*>(v));
    }
  }
  for (std::size_t i = 0; i < d.layers.size(); ++i) {
    fn::CopyDevice(d.layers[i].ring_k, d.ckpt_k[i], floats, stream_);
    fn::CopyDevice(d.layers[i].ring_v, d.ckpt_v[i], floats, stream_);
  }
  d.ckpt_injected = d.injected;
  return Ok(hipGetLastError(), "dflash checkpoint", error);
}

bool Engine::DFlashRestoreCheckpoint(std::string* error) {
  DFlash& d = *df_;
  if (d.ckpt_k.empty()) {
    Fail(error, "no DFlash checkpoint");
    return false;
  }
  const std::size_t floats =
      static_cast<std::size_t>(d.capacity) * d.kv_heads * d.head_dim;
  for (std::size_t i = 0; i < d.layers.size(); ++i) {
    fn::CopyDevice(d.ckpt_k[i], d.layers[i].ring_k, floats, stream_);
    fn::CopyDevice(d.ckpt_v[i], d.layers[i].ring_v, floats, stream_);
  }
  d.injected = d.ckpt_injected;
  return Ok(hipGetLastError(), "dflash restore", error);
}

}  // namespace gufo::models::qwen36_a3b
