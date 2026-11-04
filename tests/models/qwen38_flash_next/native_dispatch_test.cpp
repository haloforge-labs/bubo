#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "src/models/qwen38_flash_next/native_dispatch.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using Type = gufo::core::GgmlType;

namespace {
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct Shape {
  const char* name;
  std::uint64_t rows;
  std::uint64_t cols;
};

// Flash-Next trunk shapes (rows x cols) of the pf-native-gemm microbench.
constexpr Shape kShapes[] = {
    {"ssm_out/attn_output 2560x6144", 2560, 6144},
    {"fused ssm in 16384x2560", 16384, 2560},
    {"attn qkv 13312x2560", 13312, 2560},
    {"shared gate/up 640x2560", 640, 2560},
    {"shared down 2560x640", 2560, 640},
};

bool Expected(Type type, const Shape& s) {
  // Measured: the wave64 or blocked kernels beat mmq everywhere except the
  // 640-row shape; 2560x640 has cols % 256 != 0 so only IQ4_NL (32) applies.
  const bool big = s.rows >= 1024;
  switch (type) {
    case Type::kQ4_K:
    case Type::kQ5_K:
    case Type::kQ6_K:
    case Type::kIQ4_XS:
    case Type::kIQ3_S:
      return big && s.cols % 256 == 0 && s.cols >= 1024;
    case Type::kIQ4_NL:
      return big && s.cols % 32 == 0 && s.cols >= 512;
    default:
      return false;
  }
}
}  // namespace

int main() {
  try {
    int checked = 0;
    for (const auto type :
         {Type::kQ4_K, Type::kQ5_K, Type::kQ6_K, Type::kIQ4_XS, Type::kIQ3_S,
          Type::kIQ4_NL, Type::kQ8_0, Type::kBF16, Type::kF16, Type::kF32,
          Type::kQ5_1, Type::kQ2_0, Type::kQ3_K}) {
      for (const auto& s : kShapes) {
        const bool want = Expected(type, s);
        const bool got = qfn::NativeWmmaGemm(type, s.rows, s.cols, true);
        std::printf("type=%-7.*s %-30s prefill=%d decode=%d\n",
                    static_cast<int>(gufo::core::ToString(type).size()),
                    gufo::core::ToString(type).data(), s.name, got,
                    qfn::NativeWmmaGemm(type, s.rows, s.cols, false));
        Require(got == want, std::string("envelope mismatch: ") + s.name);
        // Outside prefill_phase (decode, verify, MTP draft, output head)
        // the predicate is always false.
        Require(!qfn::NativeWmmaGemm(type, s.rows, s.cols, false),
                "predicate true outside prefill_phase");
        ++checked;
      }
    }
    // Types never routed: Q5_1, Q2_0, Q3_K (mmq only; Q3_K has no microbench evidence), non-quant types.
    for (const auto type : {Type::kQ5_1, Type::kQ2_0, Type::kQ3_K, Type::kQ8_0, Type::kBF16})
      Require(!qfn::NativeWmmaGemm(type, 16384, 2560, true),
              "unlisted type accepted");
    // cols classes: 256-superblock types need cols % 256 == 0 (IQ4_NL: 32).
    Require(!qfn::NativeWmmaGemm(Type::kQ6_K, 2560, 2560 + 32, true),
            "Q6_K cols % 256 != 0 accepted");
    Require(qfn::NativeWmmaGemm(Type::kIQ4_NL, 2560, 2560 + 32, true),
            "IQ4_NL cols % 32 == 0 rejected");
    Require(!qfn::NativeWmmaGemm(Type::kIQ4_NL, 2560, 2560 + 16, true),
            "IQ4_NL cols % 32 != 0 accepted");
    // Q6_K envelope is exactly the launcher's wave64 condition.
    Require(!qfn::NativeWmmaGemm(Type::kQ6_K, 1023, 2560, true) &&
                qfn::NativeWmmaGemm(Type::kQ6_K, 1024, 2560, true) &&
                !qfn::NativeWmmaGemm(Type::kQ6_K, 2560, 768, true) &&
                qfn::NativeWmmaGemm(Type::kQ6_K, 2560, 1024, true),
            "Q6_K row/col boundary changed");

    // Width independence: the predicate has no token-count argument. The token
    // gate takes 1..max_batch (also max_batch < 96, which runs on the pad) and
    // every pass launches as at least 96 rows, so the kernel family is the one
    // of a full chunk for any width.
    Require(!qfn::NativeWmmaGemmTokens(0, 4096) &&
                qfn::NativeWmmaGemmTokens(1, 4096) &&
                qfn::NativeWmmaGemmTokens(95, 4096) &&
                qfn::NativeWmmaGemmTokens(96, 4096) &&
                qfn::NativeWmmaGemmTokens(4096, 4096) &&
                !qfn::NativeWmmaGemmTokens(4097, 4096) &&
                qfn::NativeWmmaGemmTokens(64, 64) &&
                !qfn::NativeWmmaGemmTokens(65, 64),
            "token gate boundaries changed");
    for (std::uint32_t n = 1; n <= 4096; ++n) {
      Require(qfn::NativeWmmaGemmTokens(n, 4096), "token gate not monotone");
      const auto launch = qfn::NativeGemmLaunchRows(n);
      Require(launch == (n < 96 ? 96u : n), "launch rows must be max(n, 96)");
    }
    Require(qfn::kNativeGemmWideRows == 96, "wide batch class changed");
    // Same (type, rows, cols) gives the same answer for every chunk width.
    for (const auto type : {Type::kQ6_K, Type::kIQ3_S})
      for (std::uint32_t n : {1u, 9u, 13u, 64u, 95u, 96u, 97u, 127u, 128u, 130u, 4096u}) {
        Require(qfn::NativeWmmaGemmTokens(n, 4096), "width rejected");
        Require(qfn::NativeWmmaGemm(type, 2560, 6144, true),
                "route depends on width");
      }
    std::printf("native_dispatch_test: %d type x shape combinations ok\n", checked);

    // GUFO_PREFILL_NATIVE_GEMM parse table: 0 off, 1 the int8
    // route, 2 half and unset; everything else (the empty string included) is an error.
    struct Parse {
      const char* text;
      int want;
    };
    const Parse kParse[] = {{nullptr, 2}, {"0", 0},    {"1", 1},    {"2", 2},
                            {"", -1},     {"3", -1},    {"-1", -1},  {"01", -1},
                            {"10", -1},   {"22", -1},   {" 1", -1},  {"1 ", -1},
                            {"on", -1},   {"true", -1}, {"half", -1}};
    for (const auto& p : kParse) {
      const int got = qfn::ParseNativeGemmMode(p.text);
      std::printf("parse %-8s -> %d\n", p.text ? p.text : "(unset)", got);
      Require(got == p.want,
              std::string("parse mismatch: ") + (p.text ? p.text : "(unset)"));
    }
    Require(qfn::kNativeGemmModeInvalid == -1, "invalid mode value changed");
    Require(
        std::string(qfn::kNativeGemmModeHelp)
                    .find("GUFO_PREFILL_NATIVE_GEMM") != std::string::npos &&
            std::string(qfn::kNativeGemmModeHelp).find("0, 1 or 2") !=
                std::string::npos,
        "parse error text must name the variable and the values");

    // Value 2 predicate: every type x trunk shape. Only Q4_K, Q5_K and Q6_K
    // with cols % 256 == 0, cols >= 1024 and rows >= 512 take it; the 640 x
    // 2560 gate/up shape does (it stays on mmq at value 1), 2560 x 640 does
    // not; the IQ types and the rest stay on the value 1 route.
    int half_checked = 0;
    for (const auto type :
         {Type::kQ4_K, Type::kQ5_K, Type::kQ6_K, Type::kIQ4_XS, Type::kIQ3_S,
          Type::kIQ4_NL, Type::kQ8_0, Type::kBF16, Type::kF16, Type::kF32,
          Type::kQ5_1, Type::kQ2_0, Type::kQ3_K}) {
      const bool kquant =
          type == Type::kQ4_K || type == Type::kQ5_K || type == Type::kQ6_K;
      for (const auto& s : kShapes) {
        const bool want =
            kquant && s.cols % 256 == 0 && s.cols >= 1024 && s.rows >= 512;
        const bool got = qfn::NativeHalfGemm(type, s.rows, s.cols, true);
        std::printf("half type=%-7.*s %-30s prefill=%d decode=%d\n",
                    static_cast<int>(gufo::core::ToString(type).size()),
                    gufo::core::ToString(type).data(), s.name, got,
                    qfn::NativeHalfGemm(type, s.rows, s.cols, false));
        Require(got == want, std::string("half envelope mismatch: ") + s.name);
        Require(!qfn::NativeHalfGemm(type, s.rows, s.cols, false),
                "half predicate true outside prefill_phase");
        // Superset of value 1 for the three K-quants: nothing the int8 route
        // takes is dropped by the half check.
        if (kquant)
          Require(!qfn::NativeWmmaGemm(type, s.rows, s.cols, true) || got,
                  "half envelope is not a superset of the int8 envelope");
        // Types that are not K-quants never take the half route, and a shape
        // value 1 takes stays the same function of its inputs.
        if (!kquant)
          Require(!got, "non-K-quant type took the half route");
        ++half_checked;
      }
    }
    Require(qfn::NativeHalfGemm(Type::kQ6_K, 640, 2560, true) &&
                qfn::NativeHalfGemm(Type::kQ4_K, 640, 2560, true) &&
                qfn::NativeHalfGemm(Type::kQ5_K, 640, 2560, true),
            "640 x 2560 gate/up must take the half route");
    Require(!qfn::NativeHalfGemm(Type::kQ6_K, 2560, 640, true),
            "2560 x 640 shared down must not take the half route");
    Require(!qfn::NativeHalfGemm(Type::kQ6_K, 511, 2560, true) &&
                qfn::NativeHalfGemm(Type::kQ6_K, 512, 2560, true) &&
                !qfn::NativeHalfGemm(Type::kQ6_K, 2560, 768, true) &&
                qfn::NativeHalfGemm(Type::kQ6_K, 2560, 1024, true) &&
                !qfn::NativeHalfGemm(Type::kQ6_K, 2560, 2560 + 32, true),
            "half row/col boundary changed");
    // The knob values 0 and 1 do not change NativeWmmaGemm: the boundary and
    // IQ cases above still pass; the IQ types are rejected by the half check.
    for (const auto type : {Type::kIQ3_S, Type::kIQ4_XS, Type::kIQ4_NL})
      Require(!qfn::NativeHalfGemm(type, 16384, 2560, true) &&
                  qfn::NativeWmmaGemm(type, 16384, 2560, true),
              "IQ types must stay on the value 1 route");
    std::printf("native_dispatch_test: %d half type x shape combinations ok\n",
                half_checked);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
