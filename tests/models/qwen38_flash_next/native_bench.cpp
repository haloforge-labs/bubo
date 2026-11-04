#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/native_dispatch.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffers {
  std::vector<void*> owned;
  ~Buffers() {
    for (void* p : owned)
      (void)hipFree(p);
  }
  void* Make(std::size_t bytes) {
    void* p = nullptr;
    Check(hipMalloc(&p, bytes));
    owned.push_back(p);
    Check(hipMemset(p, 0, bytes));
    return p;
  }
};
struct Weight {
  qfn::TensorRef ref;
  void* device;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::puts("usage: native_bench <raw-shard-1.gguf>");
    return 1;
  }
  try {
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader)
      throw std::runtime_error(error);
    auto model = qfn::ModelWeights::Bind(*reader, &error);
    if (!model)
      throw std::runtime_error(error);
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("MMQ initialization failed");
    Buffers memory;
    std::vector<Weight> weights;
    std::size_t bytes = 0;
    std::uint32_t max_k = 0, max_m = 0;
    for (const auto& layer : model->layers) {
      const auto& t = layer.linear ? layer.ssm_out : layer.attn_out;
      if (!qfn::NativeQuantDense(t.type))
        continue;
      void* p = memory.Make(t.SizeBytes() + 4096);
      Check(hipMemcpy(p, t.data, t.SizeBytes(), hipMemcpyHostToDevice));
      weights.push_back({t, p});
      bytes += t.SizeBytes();
      max_k = std::max(max_k, static_cast<std::uint32_t>(t.cols));
      max_m = std::max(max_m, static_cast<std::uint32_t>(t.rows));
    }
    if (weights.empty())
      throw std::runtime_error("model has no native quantized layer output matrices");
    std::vector<float> host_x(std::size_t{max_k} * 64);
    for (std::size_t i = 0; i < host_x.size(); ++i)
      host_x[i] = 0.1F * std::sin(static_cast<float>(i) * 0.031F);
    auto* x = static_cast<float*>(memory.Make(host_x.size() * sizeof(float)));
    auto* out = static_cast<float*>(memory.Make(std::size_t{max_m} * 64 * sizeof(float)));
    void* quant = memory.Make(qfn_mmq_q8_1_bytes(8, max_k));
    auto* ids = static_cast<std::int32_t*>(memory.Make(8 * sizeof(std::int32_t)));
    Check(hipMemcpy(x, host_x.data(), host_x.size() * sizeof(float), hipMemcpyHostToDevice));
    hipEvent_t begin, end;
    Check(hipEventCreate(&begin));
    Check(hipEventCreate(&end));
    const auto cycle = [&](int n) {
      for (const auto& w : weights) {
        const auto& t = w.ref;
        int status;
        if (n <= 8) {
          status = qfn_mmq_quantize_q8_1(x, quant, n, static_cast<int>(t.cols), nullptr);
          if (status == 0)
            status = qfn_mmq_dense_vec_preq(static_cast<int>(t.type), w.device,
                quant, ids, out, static_cast<int>(t.rows), n,
                static_cast<int>(t.cols), nullptr);
        } else {
          status = qfn_mmq_k_dense(static_cast<int>(t.type), w.device, x, out,
              static_cast<int>(t.rows), n, static_cast<int>(t.cols), nullptr);
        }
        if (status != 0)
          throw std::runtime_error("native projection launch failed: " + std::string(t.name));
      }
    };
    std::printf("rotating %zu different layer matrices, weight_bytes=%zu\n", weights.size(), bytes);
    for (int n : {1, 8, 64}) {
      cycle(n);
      Check(hipDeviceSynchronize());
      Check(hipEventRecord(begin));
      for (int repeat = 0; repeat < 3; ++repeat)
        cycle(n);
      Check(hipEventRecord(end));
      Check(hipEventSynchronize(end));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, begin, end));
      const double gbps = 3.0 * static_cast<double>(bytes) / (ms * 1e6);
      if (gbps > 256.0)
        std::printf("n=%d measured GB/s=%.3f; discarded (above 256 GB/s chip peak)\n", n, gbps);
      else
        std::printf("n=%d us/matrix=%.3f weight_GB/s=%.3f\n", n,
                    ms * 1000.0 / (3 * weights.size()), gbps);
    }
    Check(hipEventDestroy(begin));
    Check(hipEventDestroy(end));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
