#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/head_two_runtime.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next;
namespace r = q::rocm;
hipError_t ExpertBenchEvict(void* buffer, std::size_t bytes);
namespace {
void Check(hipError_t code) {
  if (code != hipSuccess)
    throw std::runtime_error(hipGetErrorString(code));
}
struct Memory {
  std::vector<void*> pointers;
  void* Get(std::size_t bytes) {
    void* p = nullptr;
    Check(hipMalloc(&p, bytes));
    pointers.push_back(p);
    return p;
  }
  ~Memory() {
    for (void* p : pointers)
      (void)hipFree(p);
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "head_two_bench <main-shard.gguf> [repeats=9]\n"
                   "Full head versus Q4_K shortlist; K=64,128,256,512.\n";
      return 0;
    }
    if (argc < 2)
      throw std::runtime_error(
          "usage: head_two_bench <main-shard.gguf> [repeats=9]");
    const auto model = std::filesystem::path(argv[1]);
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 9;
    if (repeats < 1 || repeats > 100)
      throw std::runtime_error("repeats must be 1..100");
    std::string error;
    auto reader = gufo::core::GgufReader::OpenFile(model, &error);
    if (!reader)
      throw std::runtime_error(error);
    const auto* head = reader->FindTensor("output.weight");
    if (!head || head->type != gufo::core::GgmlType::kQ8_0 ||
        head->dimensions[0] != 2560)
      throw std::runtime_error("requires the production Q8_0 head");
    const int cols = head->dimensions[0], rows = head->dimensions[1];
    if (qfn_mmq_init(0))
      throw std::runtime_error("MMQ initialization failed");
    Memory memory;
    void* weights = memory.Get(head->size_bytes);
    Check(hipMemcpy(weights, head->data, head->size_bytes,
                    hipMemcpyHostToDevice));
    auto* activations =
        static_cast<float*>(memory.Get(std::size_t(16) * cols * sizeof(float)));
    std::vector<float> host(std::size_t(16) * cols);
    for (std::size_t i = 0; i < host.size(); ++i)
      host[i] = .2f * std::sin(float(i * 7));
    Check(hipMemcpy(activations, host.data(), host.size() * sizeof(float),
                    hipMemcpyHostToDevice));
    void* q8 = memory.Get(std::size_t(8) * ((cols + 511) / 512) * 16 * 36);
    auto* logits =
        static_cast<float*>(memory.Get(std::size_t(8) * rows * sizeof(float)));
    constexpr std::size_t cold_bytes = 256U << 20;
    void* cold = memory.Get(cold_bytes);
    Check(hipMemset(cold, 0, cold_bytes));
    hipEvent_t start, end;
    Check(hipEventCreate(&start));
    Check(hipEventCreate(&end));
    std::uint64_t calls = 0;
    int row_rotation = 0;
    const auto time_call = [&](int width, const std::function<void()>& call) {
      // Alternate input rows and read/change 256 MiB before EVERY timed call.
      // Eviction and activation quantization are outside the head interval.
      const int input_row = calls++ % 9;
      row_rotation = (calls * 65537) % rows;
      if (qfn_mmq_quantize_q8_1(activations + input_row * cols, q8, width, cols,
                                nullptr))
        throw std::runtime_error("quantization failed");
      Check(ExpertBenchEvict(cold, cold_bytes));
      Check(hipEventRecord(start));
      call();
      Check(hipEventRecord(end));
      Check(hipEventSynchronize(end));
      float ms;
      Check(hipEventElapsedTime(&ms, start, end));
      return double(ms);
    };
    const auto measure = [&](const char* variant, int k, int width,
                             double bytes, const std::function<void()>& call) {
      std::vector<double> valid;
      for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto ms = time_call(width, call);
        const double gbps = bytes / (ms * 1e6);
        if (gbps > 0 && gbps <= 256)
          valid.push_back(ms);
        else
          std::cout << "discarded above-peak/nonfinite sample variant="
                    << variant << " K=" << k << " N=" << width << '\n';
      }
      if (valid.empty())
        throw std::runtime_error("no below-peak bench samples");
      std::ranges::sort(valid);
      const auto middle = valid.size() / 2;
      const double ms = valid.size() % 2
                            ? valid[middle]
                            : (valid[middle - 1] + valid[middle]) / 2;
      std::cout << "variant=" << variant << " K=" << k << " N=" << width
                << " median_ms=" << ms << " GB/s=" << bytes / (ms * 1e6)
                << " valid=" << valid.size() << '/' << repeats << " fallback=0"
                << '\n';
    };
    std::cout << std::fixed << std::setprecision(3);
    for (int width = 1; width <= 8; ++width)
      measure("full", 0, width, head->size_bytes, [&] {
        if (qfn_mmq_q8_0_dense_vec_preq(weights, nullptr, q8, logits, rows,
                                        width, cols, nullptr))
          throw std::runtime_error("full head failed");
      });
    for (int k : {64, 128, 256, 512}) {
      std::unique_ptr<r::HeadTwoRuntime> runtime;
      r::DeviceTensor tensor{weights, gufo::core::GgmlType::kQ8_0,
                             unsigned(cols), unsigned(rows), 1};
      if (!r::HeadTwoRuntime::Create(
              tensor, model.parent_path() / "draft_head" / "draft-out-q4k.gguf",
              runtime, &error, k) ||
          !runtime->enabled())
        throw std::runtime_error("Q4_K head unavailable: " + error);
      for (int width = 1; width <= 8; ++width)
        measure("A", k, width,
                runtime->weight_bytes() + double(width) * k * cols / 32 * 34,
                [&] {
                  if (!runtime->Enqueue(q8, logits, width, nullptr, &error,
                                        row_rotation))
                    throw std::runtime_error(error);
                });
    }
    Check(hipEventDestroy(start));
    Check(hipEventDestroy(end));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
