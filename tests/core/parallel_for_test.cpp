#include "src/core/parallel_for.hpp"

#include <atomic>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void Check(bool condition) {
  if (!condition)
    throw std::runtime_error("parallel-for contract failed");
}
}  // namespace

int main() {
  const gufo::core::ParallelForOptions parallel{8, 0};
  for (const std::size_t count : {0U, 1U, 3U, 10007U}) {
    std::vector<std::atomic<unsigned>> visits(count);
    gufo::core::ParallelFor(
        count, [&](std::size_t i) { ++visits[i]; }, parallel);
    for (const auto& value : visits)
      Check(value == 1);
  }
  const auto caller = std::this_thread::get_id();
  gufo::core::ParallelFor(
      4, [&](std::size_t) { Check(std::this_thread::get_id() == caller); });
  for (const std::size_t fault : {0U, 40U}) {
    std::atomic<unsigned> finished{0};
    bool caught = false;
    try {
      gufo::core::ParallelFor(
          64,
          [&](std::size_t i) {
            if (i == fault)
              throw std::runtime_error("worker failure");
            ++finished;
          },
          parallel);
    } catch (const std::runtime_error& e) {
      caught = std::string(e.what()) == "worker failure";
    }
    Check(caught && finished == 56);
  }
  std::mt19937 rng(7123);
  std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
  constexpr std::size_t rows = 1003, columns = 71;
  std::vector<float> matrix(rows * columns), input(columns);
  for (auto& value : matrix)
    value = dist(rng);
  for (auto& value : input)
    value = dist(rng);
  std::vector<float> serial(rows), threaded(rows);
  const auto row_dot = [&](std::size_t row, std::vector<float>& output) {
    float sum = 0;
    for (std::size_t column = 0; column < columns; ++column)
      sum += matrix[row * columns + column] * input[column];
    output[row] = sum;
  };
  for (std::size_t row = 0; row < rows; ++row)
    row_dot(row, serial);
  gufo::core::ParallelFor(
      rows, [&](std::size_t row) { row_dot(row, threaded); }, parallel);
  Check(std::memcmp(serial.data(), threaded.data(),
                    serial.size() * sizeof(float)) == 0);
  std::cout << "PASS: parallel-for 0/1/N<threads/N>>threads, serial fast path, "
               "caller/worker exceptions, randomized byte parity\n";
}
