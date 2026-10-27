// IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS lookup tables, verbatim from ggml-common.h (generated from the tables of the engine's HIP port,
// checked equal to ggml-org/llama.cpp master, 2026-10-20). CPU oracle for the matching GPU kernels.
#ifndef GUFO_CORE_QUANT_IQ_GRIDS_HPP_
#define GUFO_CORE_QUANT_IQ_GRIDS_HPP_

#include <cstdint>

namespace gufo::quant {

/// 128 sign patterns: the 7-bit index in the block selects a byte whose eighth bit is the parity of the other seven.
[[nodiscard]] const std::uint8_t* SignsIq2xs() noexcept;
/// 256 entries; each packs eight uint8 magnitudes (one group of eight elements).
[[nodiscard]] const std::uint64_t* Iq2xxsGrid() noexcept;
/// 512 entries of eight magnitudes.
[[nodiscard]] const std::uint64_t* Iq2xsGrid() noexcept;
/// 1024 entries of eight magnitudes.
[[nodiscard]] const std::uint64_t* Iq2sGrid() noexcept;
/// 256 entries; each packs four uint8 magnitudes (one group of four elements).
[[nodiscard]] const std::uint32_t* Iq3xxsGrid() noexcept;

}  // namespace gufo::quant

#endif  // GUFO_CORE_QUANT_IQ_GRIDS_HPP_
