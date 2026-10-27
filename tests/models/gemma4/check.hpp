#ifndef GUFO_TESTS_MODELS_GEMMA4_CHECK_HPP_
#define GUFO_TESTS_MODELS_GEMMA4_CHECK_HPP_

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gemma4_test {

/// Assertion that stays active in optimized builds.
inline void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template<class Fn>
int Run(Fn&& fn) {
  // Unbuffered output and a terminate message: on Windows an abort ends the
  // process with 0xC0000409 and a piped stdout would lose its last 4 KB, which
  // hid the crash point of moe_ops and projection_ops in the first GPU run.
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::signal(SIGABRT, [](int) {
    std::cerr << "FAIL: abort() called\n";
    std::fflush(stdout);
    std::_Exit(3);
  });
  std::set_terminate([] {
    std::cerr << "FAIL: std::terminate";
    try {
      if (const auto active = std::current_exception()) {
        std::rethrow_exception(active);
      }
    } catch (const std::exception& e) {
      std::cerr << " with active exception: " << e.what();
    } catch (...) {
      std::cerr << " with a non-standard active exception";
    }
    std::cerr << " (noexcept or thread context)\n";
    std::fflush(stdout);
    std::_Exit(2);
  });
#ifdef _WIN32
  _set_invalid_parameter_handler([](const wchar_t* expression,
                                    const wchar_t* function, const wchar_t* file,
                                    unsigned int line, std::uintptr_t) {
    std::wcerr << L"FAIL: CRT invalid parameter: "
               << (expression ? expression : L"?") << L" in "
               << (function ? function : L"?") << L" " << (file ? file : L"?")
               << L":" << line << L"\n";
    std::fflush(stdout);
    std::_Exit(4);
  });
#endif
  try {
    fn();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "FAIL: non-standard exception\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

}  // namespace gemma4_test

#endif  // GUFO_TESTS_MODELS_GEMMA4_CHECK_HPP_
