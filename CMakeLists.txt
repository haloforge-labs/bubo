cmake_minimum_required(VERSION 3.21)
project(gufo LANGUAGES C CXX)
# Production remains Linux x86-64 on gfx1151; Windows x86-64 is a supported
# native build target sharing the same source tree.
if (NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Windows")
    OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "Gufo requires Linux or Windows on x86-64 with an AMD gfx1151 GPU")
endif()
if (WIN32)
  # Declared before any subdirectory: windows.h must never define the
  # min/max macros that break std::min/std::max and rocPRIM device headers.
  # PSAPI_VERSION=2 resolves GetProcessMemoryInfo through kernel32 directly.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN PSAPI_VERSION=2)
endif()
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
option(BUILD_TESTING "Build correctness tests" OFF)
option(GUFO_BUILD_TOOLS "Build development benchmarks and tuning tools" OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(cmake/Version.cmake)
include(cmake/Npu.cmake)
set(BUILD_MANIFEST_MSVC_TOOLS_VERSION "$ENV{VCToolsVersion}" CACHE INTERNAL "Configured MSVC headers/libraries" FORCE)
set(BUILD_MANIFEST_WINDOWS_SDK_VERSION "$ENV{WindowsSDKVersion}" CACHE INTERNAL "Configured Windows SDK" FORCE)
set(GUFO_BRAND "bubo" CACHE STRING "Product brand: bubo or gufo")
set_property(CACHE GUFO_BRAND PROPERTY STRINGS bubo gufo)
if(GUFO_BRAND STREQUAL "bubo")
  set(GUFO_DEFAULT_PROFILE "default")
elseif(GUFO_BRAND STREQUAL "gufo")
  set(GUFO_DEFAULT_PROFILE "stock")
else()
  message(FATAL_ERROR "GUFO_BRAND must be bubo or gufo")
endif()
if(GUFO_BRAND STREQUAL "bubo")
  set(_gufo_openmp_default OFF)
else()
  set(_gufo_openmp_default ON)
endif()
option(GUFO_ENABLE_OPENMP "Enable OpenMP CPU parallelism" ${_gufo_openmp_default})
if(NOT GUFO_ENABLE_OPENMP)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
  elseif(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/wd4068>)
  endif()
endif()
configure_file(src/cli/brand.hpp.in generated/src/cli/brand.hpp @ONLY)
configure_file(src/cli/brand.json.in generated/brand.json @ONLY)
include_directories(${CMAKE_BINARY_DIR}/generated)
include_directories(${CMAKE_SOURCE_DIR})

# Reuse the compiler cache whenever it is installed, as the CPU check
# derivations do. HIP kernels stay uncached: hipcc drive files and bitcode
# paths need their own hit-rate measurement before they join the cache.
find_program(GUFO_CCACHE_PROGRAM NAMES ccache)
if (GUFO_CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
  # Newer CMake gives clang-cl gcc-style depfiles (-clang:-MD), which ccache
  # rejects as unsupported; /showIncludes is cacheable.
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(GUFO_CCACHE_CLANG_CL ON)
    foreach(lang C CXX)
      set(CMAKE_DEPFILE_FLAGS_${lang} "/showIncludes")
      set(CMAKE_${lang}_DEPFILE_FORMAT msvc)
    endforeach()
  endif()
  message(STATUS "gufo: compiling C and CXX through ${GUFO_CCACHE_PROGRAM}")
else()
  message(STATUS "gufo: ccache not found, compiling without a launcher")
endif()

# Windows host checks build only the OS leaves, the CPU cache library and
# their portable tests. The rest of the tree targets Linux only.
option(GUFO_PLATFORM_ONLY
  "Build only the platform leaves, the CPU cache and their portable tests" OFF)
if (GUFO_PLATFORM_ONLY)
  find_package(Threads REQUIRED)
  add_compile_options(-Wall -Wextra -Wpedantic -UNDEBUG)
  add_subdirectory(src/core/platform)
  add_subdirectory(src/cache)
  include(CTest)
  if (BUILD_TESTING)
    if (NOT WIN32)
      find_package(Python3 REQUIRED COMPONENTS Interpreter)
    endif()
    include(cmake/PlatformTests.cmake)
    add_subdirectory(tests/cache)
  endif()
  return()
endif()
# Linux libstdc++ must be parsed before HIP defines its __noinline__ macro.
if (NOT WIN32)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include \"${CMAKE_SOURCE_DIR}/src/core/hip/libstdcxx_noinline.hpp\">")
endif()
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
if (WIN32)
  # Windows has no system pkg-config; locate the pkgconf tool vcpkg installs.
  # .pc files resolve via CMAKE_PREFIX_PATH, which the vcpkg toolchain already
  # points at the manifest install tree.
  find_program(PKG_CONFIG_EXECUTABLE pkgconf REQUIRED
    HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/pkgconf")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
pkg_check_modules(SOXR REQUIRED IMPORTED_TARGET soxr)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
# Public prefill policy headers use shared SHA-256, including metadata-only
# targets that do not link the inference libraries.
include_directories(SYSTEM ${OPENSSL_INCLUDE_DIR})
set(GUFO_FFMPEG_EXECUTABLE "ffmpeg" CACHE STRING
  "FFmpeg executable used for generated media")
set(GUFO_FFPROBE_EXECUTABLE "ffprobe" CACHE STRING
  "FFprobe executable used to validate generated media")

# CPU-only builds exist for host checks; production targets gfx1151 only.
option(ENGINE_ENABLE_HIP "Build the gfx1151 HIP backend" ON)
set(CMAKE_HIP_ARCHITECTURES gfx1151 CACHE STRING "HIP architecture")
if (ENGINE_ENABLE_HIP AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1151")
  message(FATAL_ERROR "Gufo kernels require CMAKE_HIP_ARCHITECTURES=gfx1151")
endif()
# Standard ROCm installations; custom prefixes use CMAKE_PREFIX_PATH. As a
# system prefix /opt/rocm is searched after CMAKE_PREFIX_PATH in the
# environment, so a development shell's ROCm wins over the host's.
if (EXISTS /opt/rocm)
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH /opt/rocm)
endif()
if (WIN32 AND DEFINED ENV{HIP_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{HIP_PATH})
endif()

if (ENGINE_ENABLE_HIP)
  if (WIN32 AND NOT CMAKE_HIP_FLAGS MATCHES "compat/win32/hip_include")
    # Before enable_language so CMake's compiler probe gets it too; see the
    # header for the MSVC 14.5x <cmath> conflict it works around.
    set(CMAKE_HIP_FLAGS
      "${CMAKE_HIP_FLAGS} -I${CMAKE_SOURCE_DIR}/compat/win32/hip_include"
      CACHE STRING "HIP flags" FORCE)
  endif()
  enable_language(HIP)
  # clang-cl compiles HIP like C++ on Windows, so ccache covers it there.
  if (GUFO_CCACHE_CLANG_CL)
    set(CMAKE_HIP_COMPILER_LAUNCHER "${GUFO_CCACHE_PROGRAM}")
    set(CMAKE_DEPFILE_FLAGS_HIP "/showIncludes")
    set(CMAKE_HIP_DEPFILE_FORMAT msvc)
    message(STATUS "gufo: compiling HIP through ${GUFO_CCACHE_PROGRAM}")
  endif()
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocm-core CONFIG QUIET)
  find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp REQUIRED)
  find_path(ROCPRIM_INCLUDE_DIR rocprim/rocprim.hpp REQUIRED)
  find_path(ROCWMMA_INCLUDE_DIR rocwmma/rocwmma.hpp REQUIRED)
  message(STATUS "gufo: HIP target ${CMAKE_HIP_ARCHITECTURES}")
  if (WIN32)
    # The AMD driver installs an older amdhip64 into System32 that shadows
    # the TheRock runtime on the DLL search path; executables must carry the
    # matching runtime beside them (the exe directory precedes System32).
    # ROCM_PATH comes from cmake/toolchains/windows-therock.cmake (HIP_PATH).
    file(GLOB GUFO_HIP_RUNTIME_DLLS "${ROCM_PATH}/bin/*.dll")
    if (NOT GUFO_HIP_RUNTIME_DLLS)
      message(FATAL_ERROR "No HIP runtime DLLs under ROCM_PATH=${ROCM_PATH}/bin")
    endif()
    message(STATUS "gufo: staging ${GUFO_HIP_RUNTIME_DLLS}")
  endif()
endif()

function(gufo_stage_hip_runtime target)
  if (ENGINE_ENABLE_HIP AND WIN32 AND GUFO_HIP_RUNTIME_DLLS)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${GUFO_HIP_RUNTIME_DLLS} $<TARGET_FILE_DIR:${target}>)
    # rocBLAS/hipBLASLt resolve their Tensile kernel libraries relative to
    # their own DLL location, so the data directories ship beside the exe.
    foreach(data_dir rocblas hipblaslt)
      if (EXISTS "${ROCM_PATH}/bin/${data_dir}")
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_directory
                  "${ROCM_PATH}/bin/${data_dir}"
                  "$<TARGET_FILE_DIR:${target}>/${data_dir}")
      endif()
    endforeach()
  endif()
endfunction()

if (ENGINE_ENABLE_HIP)
  add_subdirectory(src/models/deepseek_v4_flash)
endif()

# Qwen3.8-Flash-Next â€” model-private CPU reference plus ROCm runtime.
add_subdirectory(src/models/qwen38_flash_next)

# Qwen3-TTS â€” CPU reference + HIP kernels (CustomVoice 12Hz).
add_subdirectory(src/models/qwen3_tts)

# Qwen3-ASR â€” model-private 1.7B speech recognition runtime.
add_subdirectory(src/models/qwen3_asr)

# EmbeddingGemma 2 is a model-private BF16 text/image embedding runtime.
add_subdirectory(src/models/embedding_gemma2)

# Compiler warnings and sanitizers
option(GUFO_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(GUFO_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if (GUFO_ENABLE_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()

if (GUFO_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # CMake drives lld-link directly, which skips the clang-cl driver and its
  # runtime selection. Code using __builtin_cpu_supports (the disk CRC
  # dispatch) needs the builtins' __cpu_model at link time.
  # Ask the compiler in use for its runtime directory.
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(gufo_clang_forward "/clang:")
  else()
    set(gufo_clang_forward "")
  endif()
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "${gufo_clang_forward}-print-runtime-dir"
                  OUTPUT_VARIABLE GUFO_CLANG_RT_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  set(GUFO_CLANG_RT_BUILTINS
      "${GUFO_CLANG_RT_DIR}/clang_rt.builtins-x86_64.lib")
  if (EXISTS "${GUFO_CLANG_RT_BUILTINS}")
    add_link_options("${GUFO_CLANG_RT_BUILTINS}")
  else()
    message(WARNING "clang_rt.builtins not found under '${GUFO_CLANG_RT_DIR}'")
  endif()
  # The PDB path the linker writes into each executable and DLL is the build
  # tree's absolute path. Keep the file name only, so a shipped binary does not
  # carry the builder's folders.
  add_link_options("LINKER:/PDBALTPATH:%_PDB%")
  if (ENGINE_ENABLE_HIP)
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "${gufo_clang_forward}-print-resource-dir"
                    OUTPUT_VARIABLE GUFO_CLANG_RESOURCE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(GUFO_HIP_CMATH "${GUFO_CLANG_RESOURCE_DIR}/include/__clang_hip_cmath.h")
    if (EXISTS "${GUFO_HIP_CMATH}")
      file(READ "${GUFO_HIP_CMATH}" GUFO_HIP_CMATH_CONTENT)
      if (NOT GUFO_HIP_CMATH_CONTENT MATCHES "GUFO_MSVC_CMATH_PATCH"
          AND NOT CMAKE_HIP_FLAGS MATCHES "compat/win32/hip_include")
        message(FATAL_ERROR
          "TheRock HIP math headers are not patched for MSVC 14.40+; run: "
          "python tools/windows/patch_sdk_math.py")
      endif()
    endif()
  endif()
endif()

if (WIN32)
  # Platform layer: WinSock (net), process memory counters (logging), BCrypt
  # and the registry (os_info).
  set(GUFO_PLATFORM_LIBS ws2_32 iphlpapi bcrypt advapi32)
endif()

add_subdirectory(src/cache)

if (ENGINE_ENABLE_HIP)
  add_library(gufo_hip_host STATIC src/core/hip/host_allocations.cpp)
  target_link_libraries(gufo_hip_host PUBLIC hip::host gufo_platform)
  # HIP loads ROCr through libamdhip64's own RUNPATH, so a build tree runs
  # the unpatched runtime unless LD_LIBRARY_PATH names the patched one (the
  # Nix package wraps `gufo` for this). With GUFO_ROCR_LIBRARY_DIR (the Nix
  # shells export it) every HIP binary links that ROCr directly: the loader
  # maps it first and HIP reuses it, one soname per process.
  set(GUFO_ROCR_LIBRARY_DIR "" CACHE PATH
      "ROCr runtime directory HIP binaries load (empty: HIP's own)")
  set(gufo_rocr_dir "${GUFO_ROCR_LIBRARY_DIR}")
  if (NOT gufo_rocr_dir AND DEFINED ENV{GUFO_ROCR_LIBRARY_DIR})
    set(gufo_rocr_dir "$ENV{GUFO_ROCR_LIBRARY_DIR}")
  endif()
  if (gufo_rocr_dir AND NOT WIN32)
    find_library(GUFO_ROCR_LIBRARY hsa-runtime64 PATHS "${gufo_rocr_dir}"
                 NO_DEFAULT_PATH NO_CACHE REQUIRED)
    target_link_libraries(gufo_hip_host PUBLIC
                          "${GUFO_ROCR_LIBRARY}")
    message(STATUS "gufo: ROCr runtime ${GUFO_ROCR_LIBRARY}")
  endif()
  target_include_directories(gufo_hip_host PUBLIC ${PROJECT_SOURCE_DIR})
  add_library(gufo_cache_hip STATIC
    src/core/hip/committed_backing.cpp
    src/core/hip/staging.cpp
    src/core/hip/transfer_pool.cpp)
  target_link_libraries(gufo_cache_hip PUBLIC gufo_cache gufo_hip_host hip::host)
  if (BUILD_TESTING)
    target_compile_options(gufo_cache_hip PRIVATE -UNDEBUG)
  endif()
  if (GUFO_BUILD_TOOLS)
    add_executable(cache_idle_spill_bench tests/cache/idle_spill_bench.cpp)
    target_link_libraries(cache_idle_spill_bench PRIVATE gufo_cache_hip)
    add_executable(cache_backing_bench tests/cache/backing_bench.cpp)
    target_link_libraries(cache_backing_bench PRIVATE gufo_cache_hip)
    add_executable(cache_streaming_bench tests/cache/streaming_bench.cpp)
    target_link_libraries(cache_streaming_bench PRIVATE gufo_cache_hip)
  endif()
endif()

add_library(gufo_diagnostics STATIC
  src/core/diagnostics/gpu_queues.cpp
  src/core/diagnostics/linux_sysfs.cpp
  src/core/diagnostics/system_inventory.cpp
  src/core/diagnostics/compatibility.cpp
  src/core/diagnostics/fingerprint.cpp
  src/core/diagnostics/artifact_validator.cpp
  src/core/diagnostics/allocation_benchmark.cpp
  src/core/diagnostics/bandwidth.cpp
  src/core/hip/bandwidth.cpp
  src/core/diagnostics/report.cpp
  src/cli/diagnose/diagnose.cpp
)
target_include_directories(gufo_diagnostics PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(gufo_diagnostics PUBLIC gufo_platform)
target_compile_definitions(gufo_diagnostics PRIVATE
  GUFO_REVISION="${GUFO_REVISION}")

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_diagnostics PRIVATE
    src/core/hip/allocation_benchmark.hip)
  target_compile_definitions(gufo_diagnostics PRIVATE ENGINE_ENABLE_HIP=1)
  if (rocm-core_FOUND)
    target_compile_definitions(gufo_diagnostics PRIVATE
      GUFO_ROCM_VERSION="${rocm-core_VERSION}")
  endif()
  target_link_libraries(gufo_diagnostics PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

add_subdirectory(src/core/platform)

add_library(gufo_core STATIC
  src/core/image.cpp
  src/core/crypto/sha256.cpp
  src/core/gguf_reader.cpp
  src/core/gguf_identity.cpp
  src/core/json_constraint.cpp
  src/core/json_schema_lexeme.cpp
  src/core/json_schema_regex.cpp
  src/core/sampling.cpp
  src/core/platform/memory_guard.cpp
  src/models/minimax_h3/audio_vae.cpp
  src/models/minimax_h3/denoiser.cpp
  src/models/minimax_h3/dit.cpp
  src/models/minimax_h3/generation.cpp
  src/models/minimax_h3/json.cpp
  "$<$<PLATFORM_ID:Windows>:src/models/minimax_h3/media_windows.cpp>"
  "$<$<NOT:$<PLATFORM_ID:Windows>>:src/models/minimax_h3/media_posix.cpp>"
  src/models/minimax_h3/prompt_encoder.cpp
  src/models/minimax_h3/runtime.cpp
  src/models/minimax_h3/sampling.cpp
  src/models/minimax_h3/sha256.cpp
  src/models/minimax_h3/tokenizer.cpp
  src/models/minimax_h3/video_vae.cpp
  src/testing/compare/logit_comparator.cpp
  src/eval/dataset.cpp
  src/eval/extract.cpp
  src/eval/regrade.cpp
  src/core/hip/hipblaslt_plan_database.cpp
  src/core/quant/bf16_q8_0.cpp
  src/core/quant/ggml_dequant.cpp
  src/core/quant/iq_grids.cpp
  src/core/quant/ggml_gemm.cpp
)
target_include_directories(gufo_core PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(gufo_core PRIVATE
  GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}"
  GUFO_FFPROBE_EXECUTABLE="${GUFO_FFPROBE_EXECUTABLE}")
target_link_libraries(gufo_core PUBLIC gufo_platform ICU::uc ICU::i18n
  Threads::Threads PNG::PNG JPEG::JPEG PkgConfig::WEBP CURL::libcurl
  OpenSSL::Crypto)

if(GUFO_ENABLE_NPU)
  # Qwen objects in gufo_core contain NPU references. Give every core
  # consumer the same runtime link closure as the production engine. Use an
  # archive so CPU fakes that provide their own NPU methods do not get forced
  # duplicate object definitions. This does not change kernel code or flags.
  add_library(npu_core_support STATIC $<TARGET_OBJECTS:npu_runtime>)
  if(TARGET npu_bfp_hip)
    target_sources(npu_core_support PRIVATE $<TARGET_OBJECTS:npu_bfp_hip>)
  endif()
  target_link_libraries(npu_core_support PRIVATE npu_xrt OpenSSL::Crypto)
  target_link_libraries(gufo_core INTERFACE npu_core_support)
endif()

add_subdirectory(src/models/qwen)
add_subdirectory(src/models/qwen_image_21)
# Gemma 4 text models (dense and mixture-of-experts): model-private CPU
# reference plus ROCm runtime.
add_subdirectory(src/models/gemma4)
# DiffusionGemma text model (MIT code, attribution in NOTICE): block-diffusion
# runtime, separate from the autoregressive engines.
add_subdirectory(src/models/diffusion_gemma)

target_link_libraries(gufo_qwen3_tts PUBLIC gufo_core)
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_deepseek_v4_flash PUBLIC gufo_core)
endif()

if(GUFO_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(gufo_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (WIN32 AND NOT ENGINE_ENABLE_HIP)
  # CPU server tests use the real host RAM/commit readings too.
  target_sources(gufo_core PRIVATE src/core/platform/memory_readings.cpp)
endif()

if (ENGINE_ENABLE_HIP)
  set_property(SOURCE src/models/minimax_h3/dit.hip
    APPEND PROPERTY COMPILE_OPTIONS -O2)
  target_sources(gufo_core PRIVATE
    src/models/minimax_h3/audio_vae.hip
    src/models/minimax_h3/denoiser.hip
    src/models/minimax_h3/dit.hip
    src/models/minimax_h3/dense.hip
    src/models/minimax_h3/runtime_hip.cpp
    src/models/minimax_h3/prompt_encoder.hip
    src/models/minimax_h3/sampling.hip
    src/models/minimax_h3/video_vae.hip
    src/models/minimax_h3/attention.hip
    src/core/hip/snapshot_copy.hip
    src/core/hip/weight_upload.cpp
    src/core/speculative/speculative_verifier.cpp
  )
  if (WIN32)
    target_sources(gufo_core PRIVATE src/core/platform/device_memory.cpp src/core/platform/memory_readings.cpp)
    target_link_libraries(gufo_core PUBLIC dxgi)
  endif()
  target_compile_definitions(gufo_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_core SYSTEM PRIVATE
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  target_link_libraries(gufo_core PUBLIC gufo_hip_host hip::host roc::hipblas
    roc::hipblaslt roc::rocblas)
endif()

add_library(gufo_llm_cli STATIC
  src/cli/bench/bench.cpp
  src/cli/prompt/prompt.cpp)
target_link_libraries(gufo_llm_cli PUBLIC gufo_core gufo_qwen38_flash_next)
if(ENGINE_ENABLE_HIP)
  target_sources(gufo_llm_cli PRIVATE src/cli/prompt/gemma4_prompt.cpp)
  target_link_libraries(gufo_llm_cli PUBLIC gufo_deepseek_v4_flash
    gufo_qwen38_flash_next gufo_gemma4)
endif()

add_library(gufo_logging STATIC src/cli/serve/logging.cpp src/cli/serve/trace.cpp)
target_include_directories(gufo_logging PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_logging PUBLIC cxx_std_20)
target_link_libraries(gufo_logging PUBLIC Threads::Threads gufo_platform)
target_link_libraries(gufo_qwen38_flash_next PUBLIC gufo_logging)
if(TARGET gufo_qwen38_flash_next_mmq)
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC gufo_logging)
endif()

# The imported DeepSeek V4 Flash runtime routes its informational messages
# through the shared logger instead of writing to stderr directly, so it needs
# the threshold that `--log-level` arms. The Gemma 4 loader warns through it.
if (ENGINE_ENABLE_HIP)
  target_link_libraries(gufo_deepseek_v4_flash PUBLIC gufo_logging)
  target_link_libraries(gufo_gemma4 PUBLIC gufo_logging)
endif()

add_library(gufo_http STATIC
  src/cli/serve/image_api.cpp
  src/cli/serve/audio_asr_api.cpp
  src/cli/serve/asr_service.cpp
  src/cli/serve/embeddings_api.cpp
  src/cli/serve/embedding_service.cpp
  src/cli/serve/audio_tts_api.cpp
  src/cli/serve/tts_service.cpp
  src/cli/serve/video_api.cpp
  src/cli/serve/video_jobs.cpp
  src/cli/serve/http_server.cpp
  src/cli/serve/websocket.cpp
  src/cli/serve/asr_websocket.cpp
  src/cli/serve/tts_websocket.cpp
  src/cli/serve/openai_chat.cpp
  # The DeepSeek tool-call scanner lives with its model; its renderer stays in
  # the engine library, so no translation unit is linked twice.
  src/models/deepseek_v4_flash/tool_calls.cpp
)
target_link_libraries(gufo_http PUBLIC gufo_logging gufo_core gufo_gemma4 gufo_qwen3_tts gufo_qwen3_asr gufo_qwen_image_21 gufo_embedding_gemma2)

set(bubo_command_sources
  src/cli/main.cpp
  src/cli/banner.cpp
  src/cli/eval/eval.cpp
  src/cli/transcribe/transcribe.cpp
  src/cli/video/video.cpp
  src/cli/serve/serve.cpp
  src/cli/serve/continuation_cache.cpp
  src/cli/serve/continuation_disk_store.cpp
  src/cli/serve/component_text_cache.cpp
  src/cli/serve/text_model_runner.cpp
  src/cli/serve/text_generation_scheduler.cpp
  src/cli/serve/inference_backend.cpp
  src/eval/http_client.cpp
)
if (ENGINE_ENABLE_HIP)
  list(APPEND bubo_command_sources src/cli/serve/diffusion_gemma_backend.cpp)
endif()
# Model catalogue and `bubo models` (list, estimate). The catalogue JSON is
# compiled into the program; there is no runtime override.
file(READ "${CMAKE_SOURCE_DIR}/src/cli/model_catalogue.json" BUBO_MODEL_CATALOGUE_JSON)
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/model_catalogue_data.inc"
  CONTENT "R\"CATALOGUE(@BUBO_MODEL_CATALOGUE_JSON@)CATALOGUE\"
" @ONLY)
# Names and IDs are shared with core/prepare without a dependency on CLI
# headers. The download pins remain in release_catalogue.cpp.
foreach(field architecture model_family)
  string(JSON BUBO_CATALOGUE_${field} GET "${BUBO_MODEL_CATALOGUE_JSON}" flash_next ${field})
endforeach()
foreach(set huihui original lowmemory)
  foreach(field id display_name)
    string(JSON BUBO_CATALOGUE_${set}_${field} GET "${BUBO_MODEL_CATALOGUE_JSON}" flash_next ${set} ${field})
  endforeach()
endforeach()
file(READ "${CMAKE_SOURCE_DIR}/src/core/prepare/model_identity.hpp.in" BUBO_MODEL_IDENTITY_TEMPLATE)
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/model_identity.hpp"
  CONTENT "${BUBO_MODEL_IDENTITY_TEMPLATE}" @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/cli/model_catalogue.json"
  "${CMAKE_SOURCE_DIR}/src/core/prepare/model_identity.hpp.in")
set(bubo_prepare_sources
  src/cli/command_table.cpp
  src/cli/machine_output.cpp
  src/cli/instance_control.cpp
  src/cli/serve/lifecycle.cpp
  src/cli/serve/serve_command.cpp
  src/cli/status.cpp
  src/cli/machine_adapter.cpp
  src/cli/machine_warnings.cpp
  src/cli/machine_adapter_system.cpp
  src/cli/machine_adapter_bounded.cpp
  src/cli/machine_adapter_adlx.cpp
  src/cli/support_bundle.cpp
  src/cli/stop.cpp
  src/cli/prepared_default.cpp
  src/cli/model_catalogue.cpp
  src/cli/prepare/machine_prepare.cpp
  src/core/prepare/runtime_notices.cpp
  src/core/prepare/model_notices.cpp
  src/cli/prepare/runtime_prepare.cpp
  src/cli/prepare/runtime_location.cpp
  src/core/prepare/zip_reader.cpp
  src/core/prepare/runtime_table.cpp
  src/cli/prepare/vc_runtime.cpp
  src/cli/prepare/prepare.cpp
  src/cli/prepare/model_prepare.cpp
  src/core/prepare/release_catalogue.cpp
  src/core/prepare/fingerprint_table.cpp
  src/core/prepare/gguf_fingerprint.cpp
  src/core/prepare/release_evidence.cpp
  src/core/prepare/file_ops.cpp
  src/core/prepare/hash_file.cpp
  src/core/prepare/transport.cpp
  src/core/prepare/download.cpp
  src/core/crypto/sha256.cpp
  src/core/quant/ggml_dequant.cpp
  src/core/quant/iq_grids.cpp)
add_library(bubo_prepare STATIC ${bubo_prepare_sources})
# Optional ADLX read-only telemetry. The SDK stays outside this repository and
# its files never enter the package. Without this path the adapter reports
# ADLX as not built in.
set(BUBO_ADLX_SDK_DIR "" CACHE PATH "Installed ADLX SDK directory (the folder that holds Include and ADLXHelper)")
if(WIN32 AND BUBO_ADLX_SDK_DIR)
  target_sources(bubo_prepare PRIVATE
    "${BUBO_ADLX_SDK_DIR}/ADLXHelper/Windows/Cpp/ADLXHelper.cpp"
    "${BUBO_ADLX_SDK_DIR}/Platform/Windows/WinAPIs.cpp")
  target_include_directories(bubo_prepare PRIVATE "${BUBO_ADLX_SDK_DIR}")
  target_compile_definitions(bubo_prepare PRIVATE BUBO_HAS_ADLX=1)
endif()
target_link_libraries(bubo_prepare PUBLIC CURL::libcurl OpenSSL::Crypto ZLIB::ZLIB gufo_logging gufo_platform)
add_library(bubo_models STATIC
  src/cli/models_support.cpp
  src/cli/side_file_check.cpp
  src/cli/serve/model_manifest.cpp
  src/cli/serve/release_profile.cpp)
target_link_libraries(bubo_models PUBLIC bubo_prepare gufo_core gufo_qwen38_flash_next gufo_gemma4)
if(GUFO_ENABLE_NPU)
  target_compile_definitions(bubo_models PRIVATE GUFO_NPU=1)
endif()
target_link_libraries(gufo_http PRIVATE bubo_prepare)
target_link_libraries(gufo_diagnostics PUBLIC bubo_prepare)
if(WIN32)
  target_link_libraries(bubo_prepare PUBLIC Rstrtmgr Advapi32 ws2_32 dxgi)
endif()
add_executable(prepare_runtime_table EXCLUDE_FROM_ALL tools/release/prepare_runtime_table.cpp)
target_link_libraries(prepare_runtime_table PRIVATE bubo_prepare)
set_target_properties(prepare_runtime_table PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(prepare_fingerprints EXCLUDE_FROM_ALL tools/release/prepare_fingerprints.cpp
  src/core/prepare/gguf_fingerprint.cpp src/core/prepare/fingerprint_table.cpp
  src/core/prepare/release_catalogue.cpp src/core/crypto/sha256.cpp
  src/core/quant/ggml_dequant.cpp
  src/core/quant/iq_grids.cpp)
target_link_libraries(prepare_fingerprints PRIVATE OpenSSL::Crypto)
set_target_properties(prepare_fingerprints PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(prepare_gate EXCLUDE_FROM_ALL tools/release/prepare_gate.cpp)
target_link_libraries(prepare_gate PRIVATE bubo_prepare)
# Release configurations always check the production pins/evidence. Dev builds
# still compile with explicit OPEN slots, without pretending to be releasable.
add_custom_target(bubo_prepare_release_gate ALL
  COMMAND ${CMAKE_COMMAND}
    "-DCONFIG=$<CONFIG>" "-DGATE=$<TARGET_FILE:prepare_gate>"
    "-DROOT=${CMAKE_SOURCE_DIR}" "-DLIBRARY=$<TARGET_FILE:bubo_prepare>"
    "-DEVIDENCE=${CMAKE_SOURCE_DIR}/tools/release/prepare-evidence"
    -P "${CMAKE_SOURCE_DIR}/tools/release/prepare-gate.cmake"
  DEPENDS prepare_gate bubo_prepare VERBATIM)
add_custom_target(prepare_release_check
  COMMAND $<TARGET_FILE:prepare_gate> --release
    "${CMAKE_SOURCE_DIR}/tools/release/prepare-evidence"
  DEPENDS prepare_gate bubo_prepare VERBATIM)
set(bubo_inference_target gufo)
if(WIN32 AND GUFO_BRAND STREQUAL "bubo")
  add_library(bubo_engine SHARED ${bubo_command_sources})
  set_target_properties(bubo_engine PROPERTIES PREFIX "" OUTPUT_NAME "bubo-engine")
  target_compile_definitions(bubo_engine PRIVATE BUBO_ENGINE_DLL=1)
  set(bubo_inference_target bubo_engine)
  add_executable(gufo src/cli/bootstrap.cpp)
  target_link_libraries(gufo PRIVATE bubo_prepare)
  add_dependencies(gufo bubo_engine)
else()
  add_executable(gufo ${bubo_command_sources})
endif()
target_link_libraries(${bubo_inference_target} PRIVATE bubo_prepare bubo_models)
add_dependencies(gufo bubo_prepare_release_gate)
target_link_libraries(${bubo_inference_target} PRIVATE gufo_cache gufo_diagnostics gufo_core gufo_llm_cli gufo_http gufo_diffusion_gemma CURL::libcurl)
gufo_stage_hip_runtime(${bubo_inference_target})
set_target_properties(gufo PROPERTIES OUTPUT_NAME "${GUFO_BRAND}")
if(GUFO_ENABLE_NPU)
  npu_attach_runtime(${bubo_inference_target})
  if(ENGINE_ENABLE_HIP)
    target_link_libraries(${bubo_inference_target} PRIVATE npu_bfp_hip)
  endif()
endif()
if(GUFO_BRAND STREQUAL "bubo")
  add_custom_target(bubo DEPENDS gufo)
endif()
target_compile_definitions(${bubo_inference_target} PRIVATE
  GUFO_RELEASE_VERSION="${GUFO_RELEASE_VERSION}"
  GUFO_REVISION="${GUFO_REVISION}"
  GUFO_EVAL_SOURCE_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json"
  GUFO_EVAL_INSTALLED_DATA="${CMAKE_INSTALL_PREFIX}/share/gufo/eval/antirez-ds4.json"
  GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
  GUFO_H3_INSTALLED_SOURCE_MANIFEST="${CMAKE_INSTALL_PREFIX}/share/gufo/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")

if (ENGINE_ENABLE_HIP)
  target_link_libraries(${bubo_inference_target} PRIVATE gufo_cache_hip gufo_deepseek_v4_flash)
endif()
target_link_libraries(${bubo_inference_target} PRIVATE gufo_qwen3_tts gufo_qwen3_asr gufo_embedding_gemma2)

if (GUFO_BRAND STREQUAL "gufo" AND WIN32)
  # Symlinks need Developer Mode or elevation on Windows; ship a copy.
  add_custom_target(gufo-server ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gufo> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server.exe
    DEPENDS gufo
  )
elseif(GUFO_BRAND STREQUAL "gufo")
  add_custom_target(gufo-server ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:gufo> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server
    DEPENDS gufo
  )
endif()
if (BUILD_TESTING OR GUFO_BUILD_TOOLS)
  add_library(gufo_kernel_bench_support STATIC
    src/cli/bench/kernel_bench.cpp)
  target_include_directories(gufo_kernel_bench_support PUBLIC ${CMAKE_SOURCE_DIR})
endif()

if (ENGINE_ENABLE_HIP AND GUFO_BUILD_TOOLS)
  # TheRock's Windows distribution has no rocprofiler-sdk; the markers are
  # optional there and the benchmark runs without them.
  if (WIN32)
    find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h)
    find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx)
  else()
    find_path(ROCTX_INCLUDE_DIR rocprofiler-sdk-roctx/roctx.h REQUIRED)
    find_library(ROCTX_LIBRARY NAMES rocprofiler-sdk-roctx REQUIRED)
  endif()
  add_executable(gufo-kernel-bench
    src/cli/bench/kernel_bench_main.cpp
  )
  target_compile_definitions(gufo-kernel-bench PRIVATE
    GUFO_REVISION="${GUFO_REVISION}")
  target_link_libraries(gufo-kernel-bench PRIVATE
    gufo_kernel_bench_support
    gufo_diagnostics
    gufo_core)
  if (ROCTX_INCLUDE_DIR AND ROCTX_LIBRARY)
    target_include_directories(gufo-kernel-bench SYSTEM PRIVATE
      ${ROCTX_INCLUDE_DIR})
    target_link_libraries(gufo-kernel-bench PRIVATE ${ROCTX_LIBRARY})
  else()
    message(STATUS "gufo: rocprofiler-sdk roctx not found; kernel-bench markers disabled")
    target_compile_definitions(gufo-kernel-bench PRIVATE GUFO_NO_ROCTX=1)
  endif()

  add_executable(tune_hipblaslt tools/bench/tune_hipblaslt.cpp)
  target_link_libraries(tune_hipblaslt PRIVATE gufo_core hip::host)

  add_executable(wmma_layout_test tools/bench/wmma_layout_test.hip)
  target_link_libraries(wmma_layout_test PRIVATE gufo_core hip::device)
endif()

install(TARGETS gufo DESTINATION bin)
# --prefill-mode fast reads these scalars from the executable directory.
install(FILES src/models/qwen38_flash_next/prefill-fold-L3.txt DESTINATION bin)
add_custom_command(TARGET ${bubo_inference_target} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/src/models/qwen38_flash_next/prefill-fold-L3.txt
          $<TARGET_FILE_DIR:${bubo_inference_target}>/prefill-fold-L3.txt)
if(TARGET bubo_engine)
  install(TARGETS bubo_engine RUNTIME DESTINATION bin LIBRARY DESTINATION bin)
endif()
install(FILES
  src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json
  DESTINATION share/gufo/models/minimax_h3)
install(FILES
  tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json
  DESTINATION share/gufo/eval)
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md DESTINATION share/licenses/gufo)
install(DIRECTORY licenses/ DESTINATION share/licenses/gufo/third-party)
install(FILES src/models/deepseek_v4_flash/LICENSE.ds4 DESTINATION share/licenses/gufo/third-party)
if (GUFO_BRAND STREQUAL "gufo" AND WIN32)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/gufo-server.exe DESTINATION bin)
elseif(GUFO_BRAND STREQUAL "gufo")
  install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink gufo \"\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server\")")
endif()
if (ENGINE_ENABLE_HIP AND GUFO_BUILD_TOOLS)
  install(TARGETS gufo-kernel-bench tune_hipblaslt
    DESTINATION bin)
endif()

# CTest integration
include(CTest)

# Test assertions remain active when model code uses production optimization.
if (BUILD_TESTING)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-UNDEBUG)
  elseif (MSVC)
    add_compile_options(/UNDEBUG)
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_subdirectory(tests/prepare)
  # The checkpoint grid also controls GPU-only prefill passes.
  add_executable(npu_prefill_plan_test tests/npu/prefill_plan_test.cpp)
  target_include_directories(npu_prefill_plan_test PRIVATE "${CMAKE_SOURCE_DIR}")
  add_test(NAME npu_prefill_plan_test COMMAND npu_prefill_plan_test)
  set_tests_properties(npu_prefill_plan_test PROPERTIES LABELS "cpu;npu;fake")
  if(GUFO_ENABLE_NPU)
    add_executable(npu_xrt_runtime_test
      tests/npu/xrt_runtime_test.cpp src/npu/xrt_runtime_policy.cpp)
    add_test(NAME npu_xrt_runtime_test COMMAND npu_xrt_runtime_test)
    set_tests_properties(npu_xrt_runtime_test PROPERTIES LABELS "cpu;npu;fake")
    add_executable(npu_xrt_delay_guard_test tests/npu/xrt_delay_guard_test.cpp)
    npu_attach_runtime(npu_xrt_delay_guard_test)
    target_link_libraries(npu_xrt_delay_guard_test PRIVATE bubo_prepare)
    add_test(NAME npu_xrt_delay_guard_test COMMAND npu_xrt_delay_guard_test)
    set_tests_properties(npu_xrt_delay_guard_test PROPERTIES LABELS "cpu;npu")
    # Same adapter and pinned public XRT classes, with tooling-derived fake
    # implementations. No import library/runtime loader is linked into this test.
    add_executable(npu_private_bo_adapter_test
      tests/npu/private_bo_adapter_test.cpp tests/npu/fake_xrt.cpp
      src/npu/private_bo_adapter.cpp)
    target_include_directories(npu_private_bo_adapter_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(npu_private_bo_adapter_test SYSTEM PRIVATE
      "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_private_bo_adapter_test PRIVATE
      XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
    target_link_libraries(npu_private_bo_adapter_test PRIVATE gufo_core)
    set(GUFO_NPU_FAKE_CASES "${CMAKE_BINARY_DIR}/npu-fake-cases"
      CACHE PATH "Prepared bad-artifact fixtures for CPU-only NPU tests")
    add_test(NAME npu_private_bo_adapter_test COMMAND npu_private_bo_adapter_test
      "${GUFO_XRT_TOOLING_ROOT}" "${GUFO_NPU_FAKE_CASES}")
    set_tests_properties(npu_private_bo_adapter_test PROPERTIES LABELS "cpu;npu;fake")
    add_executable(npu_bfp_test tests/npu/bfp_test.cpp src/npu/bfp.cpp)
    target_include_directories(npu_bfp_test PRIVATE "${CMAKE_SOURCE_DIR}")
    add_test(NAME npu_bfp_test COMMAND npu_bfp_test)
    set_tests_properties(npu_bfp_test PROPERTIES LABELS "cpu;npu" TIMEOUT 30)
    add_executable(npu_staging_test tests/npu/staging_test.cpp src/npu/staging.cpp src/npu/bfp.cpp
      src/cli/serve/model_manifest.cpp src/cli/serve/release_profile.cpp
      src/npu/session.cpp src/npu/private_bo_adapter.cpp
      src/npu/runtime_status.cpp src/npu/guard.cpp tests/npu/fake_xrt.cpp)
    target_include_directories(npu_staging_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(npu_staging_test SYSTEM PRIVATE
      "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_staging_test PRIVATE
      XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING GUFO_NPU=1)
    target_link_libraries(npu_staging_test PRIVATE gufo_logging gufo_core bubo_prepare)
    add_test(NAME npu_staging_test COMMAND npu_staging_test)
    set_tests_properties(npu_staging_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_linear_z_test tests/npu/linear_z_test.cpp
      src/npu/linear_z.cpp src/npu/staging.cpp src/npu/bfp.cpp
      src/npu/session.cpp src/npu/private_bo_adapter.cpp
      tests/npu/fake_xrt.cpp)
    target_include_directories(npu_linear_z_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(npu_linear_z_test SYSTEM PRIVATE
      "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_linear_z_test PRIVATE
      XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
    target_link_libraries(npu_linear_z_test PRIVATE Threads::Threads gufo_core gufo_logging)
    add_test(NAME npu_linear_z_test COMMAND npu_linear_z_test "${GUFO_XRT_TOOLING_ROOT}")
    set_tests_properties(npu_linear_z_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_guard_latch_test tests/npu/guard_latch_test.cpp
      src/npu/linear_z.cpp src/npu/staging.cpp src/npu/bfp.cpp
      src/npu/session.cpp src/npu/private_bo_adapter.cpp
      tests/npu/fake_xrt.cpp)
    target_include_directories(npu_guard_latch_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(npu_guard_latch_test SYSTEM PRIVATE
      "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_guard_latch_test PRIVATE
      XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
    target_link_libraries(npu_guard_latch_test PRIVATE Threads::Threads gufo_core gufo_logging)
    add_test(NAME npu_guard_latch_test COMMAND npu_guard_latch_test "${GUFO_XRT_TOOLING_ROOT}")
    set_tests_properties(npu_guard_latch_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_projection_test tests/npu/projection_test.cpp
      src/npu/projection.cpp src/npu/staging.cpp src/npu/bfp.cpp
      src/npu/session.cpp src/npu/private_bo_adapter.cpp
      tests/npu/fake_xrt.cpp)
    target_include_directories(npu_projection_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(npu_projection_test SYSTEM PRIVATE
      "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_projection_test PRIVATE
      XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
    target_link_libraries(npu_projection_test PRIVATE Threads::Threads gufo_core gufo_logging)
    add_test(NAME npu_projection_test COMMAND npu_projection_test "${GUFO_XRT_TOOLING_ROOT}")
    set_tests_properties(npu_projection_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    # Fault telemetry + HAL fault monitor (detection only). CPU tests use fakes;
    # nothing here reads the Windows event log.
    set(GUFO_NPU_FAULT_SOURCES src/npu/fault_telemetry.cpp src/npu/hal_fault_monitor.cpp
      src/npu/chip_power.cpp src/npu/runtime_status.cpp src/npu/mem_trace.cpp
      src/npu/guard.cpp src/npu/platform_info.cpp)
    add_executable(npu_hal_fault_monitor_test tests/npu/hal_fault_monitor_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_hal_fault_monitor_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_hal_fault_monitor_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_hal_fault_monitor_test PRIVATE Threads::Threads gufo_logging)
    add_executable(npu_hal_source_real_test tests/npu/hal_source_real_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_hal_source_real_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_hal_source_real_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_hal_source_real_test PRIVATE Threads::Threads gufo_logging)
    add_executable(npu_chip_power_test tests/npu/chip_power_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_chip_power_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_chip_power_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_chip_power_test PRIVATE Threads::Threads gufo_logging)
    add_executable(npu_guard_test tests/npu/guard_test.cpp ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_guard_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_guard_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_guard_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_guard_test COMMAND npu_guard_test)
    set_tests_properties(npu_guard_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_platform_info_test tests/npu/platform_info_test.cpp ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_platform_info_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_platform_info_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_platform_info_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_platform_info_test COMMAND npu_platform_info_test)
    set_tests_properties(npu_platform_info_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_bo_address_test tests/npu/bo_address_test.cpp)
    target_include_directories(npu_bo_address_test PRIVATE "${CMAKE_SOURCE_DIR}")
    add_test(NAME npu_bo_address_test COMMAND npu_bo_address_test)
    set_tests_properties(npu_bo_address_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_chunk_retry_test tests/npu/chunk_retry_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_chunk_retry_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_chunk_retry_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_chunk_retry_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_chunk_retry_test COMMAND npu_chunk_retry_test)
    set_tests_properties(npu_chunk_retry_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_runtime_status_test tests/npu/runtime_status_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_runtime_status_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_runtime_status_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_runtime_status_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_runtime_status_test COMMAND npu_runtime_status_test)
    set_tests_properties(npu_runtime_status_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_mem_trace_test tests/npu/mem_trace_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_mem_trace_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_mem_trace_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_mem_trace_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_mem_trace_test COMMAND npu_mem_trace_test)
    set_tests_properties(npu_mem_trace_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 60)
    add_executable(npu_mem_trace_real_test tests/npu/mem_trace_real_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_mem_trace_real_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_mem_trace_real_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_mem_trace_real_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_mem_trace_real_test COMMAND npu_mem_trace_real_test)
    set_tests_properties(npu_mem_trace_real_test PROPERTIES LABELS "cpu;npu" TIMEOUT 60)
    add_test(NAME npu_chip_power_test COMMAND npu_chip_power_test)
    set_tests_properties(npu_chip_power_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_chip_power_real_test tests/npu/chip_power_real_test.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_chip_power_real_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(npu_chip_power_real_test PRIVATE GUFO_NPU_TESTING)
    target_link_libraries(npu_chip_power_real_test PRIVATE Threads::Threads gufo_logging)
    add_test(NAME npu_chip_power_real_test COMMAND npu_chip_power_real_test)
    set_tests_properties(npu_chip_power_real_test PROPERTIES LABELS "cpu;npu" TIMEOUT 60)
    add_test(NAME npu_hal_source_real_test COMMAND npu_hal_source_real_test)
    set_tests_properties(npu_hal_source_real_test PROPERTIES LABELS "cpu;npu" TIMEOUT 30)
    add_test(NAME npu_hal_fault_monitor_test COMMAND npu_hal_fault_monitor_test)
    set_tests_properties(npu_hal_fault_monitor_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 30)
    add_executable(npu_host_bench tests/npu/host_bench.cpp tests/npu/fake_xrt.cpp src/npu/private_bo_adapter.cpp src/npu/bfp.cpp)
    target_include_directories(npu_host_bench PRIVATE "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/tests/npu")
    target_include_directories(npu_host_bench SYSTEM PRIVATE "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_host_bench PRIVATE XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
    target_link_libraries(npu_host_bench PRIVATE gufo_core)
    add_executable(npu_session_test tests/npu/session_test.cpp tests/npu/fake_xrt.cpp
      tests/npu/fake_hip.cpp src/npu/linear_z_hip.cpp src/npu/projection.cpp
      src/npu/private_bo_adapter.cpp src/npu/session.cpp src/npu/server_prefill.cpp
      src/npu/linear_z.cpp src/npu/staging.cpp src/npu/bfp.cpp
      ${GUFO_NPU_FAULT_SOURCES})
    target_include_directories(npu_session_test PRIVATE "${CMAKE_SOURCE_DIR}")
    target_include_directories(npu_session_test SYSTEM PRIVATE "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
    target_compile_definitions(npu_session_test PRIVATE XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
    target_link_libraries(npu_session_test PRIVATE gufo_core gufo_logging)
    # SDK headers only: fake_hip.cpp supplies symbols; never link hip::host.
    get_target_property(npu_hip_headers hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(npu_session_test SYSTEM PRIVATE ${npu_hip_headers})
    target_compile_definitions(npu_session_test PRIVATE __HIP_PLATFORM_AMD__=1)
    foreach(npu_unit npu_linear_z_test npu_projection_test npu_guard_latch_test)
      target_sources(${npu_unit} PRIVATE tests/npu/fake_hip.cpp
        src/npu/linear_z_hip.cpp src/npu/server_prefill.cpp ${GUFO_NPU_FAULT_SOURCES})
      target_include_directories(${npu_unit} SYSTEM PRIVATE ${npu_hip_headers})
      target_compile_definitions(${npu_unit} PRIVATE __HIP_PLATFORM_AMD__=1)
    endforeach()
    target_compile_options(npu_session_test PRIVATE -include "${CMAKE_SOURCE_DIR}/tests/npu/file_io_hook.hpp")
    foreach(npu_case clean timeout error layer single stream_timeout invalid
        startup_missing_dir startup_missing_insts startup_missing_xclbin startup_hash
        startup_version startup_runtime_missing startup_device startup_pre_failed denied
        power_unavailable power_turbo hal_event invalid_hal)
      add_test(NAME npu_session_${npu_case}_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/npu/failure_log_test.py" "$<TARGET_FILE:npu_session_test>" "${npu_case}" "${GUFO_XRT_TOOLING_ROOT}")
      set_tests_properties(npu_session_${npu_case}_test PROPERTIES LABELS "cpu;npu;fake" TIMEOUT 120)
    endforeach()
    add_custom_target(npu_cpu_checks DEPENDS npu_session_test npu_private_bo_adapter_test
      npu_prefill_plan_test npu_staging_test npu_linear_z_test npu_projection_test npu_guard_latch_test
      npu_bfp_test npu_host_bench npu_hal_fault_monitor_test npu_hal_source_real_test npu_runtime_status_test npu_bo_address_test npu_chunk_retry_test npu_platform_info_test npu_guard_test npu_mem_trace_test npu_mem_trace_real_test npu_chip_power_test
      npu_chip_power_real_test npu_runtime)
    if(TARGET npu_bfp_hip)
      add_dependencies(npu_cpu_checks npu_bfp_hip)
    endif()
  endif()

  add_subdirectory(tests/cache)

  # Opt-in live process-failure checks, never preloaded into hosted tests or
  # production. No HIP headers are needed for these stable C ABI hooks.
  add_library(device_loss_faults SHARED EXCLUDE_FROM_ALL
    tests/functional/device_loss_faults.cpp)
  target_link_libraries(device_loss_faults PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
  # Linux-only: the gate interposes fsync via dlsym(RTLD_NEXT) and reads
  # /proc/self/fd, neither of which exists on Windows.
  if (NOT WIN32)
  add_library(cache_disk_faults SHARED
    tests/functional/cache_disk_faults.cpp)
  target_link_libraries(cache_disk_faults PRIVATE ${CMAKE_DL_LIBS})
  endif()
  # The fault libraries are LD_PRELOADed into uninstrumented processes (the
  # test's Python, a release server); a sanitized library aborts there with
  # "ASan runtime does not come first".
  if (GUFO_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(preload device_loss_faults cache_disk_faults)
      if (TARGET ${preload})
        target_compile_options(${preload} PRIVATE -fno-sanitize=all)
        target_link_options(${preload} PRIVATE -fno-sanitize=all)
      endif()
    endforeach()
  endif()

  add_subdirectory(tests/models/qwen)
  add_subdirectory(tests/models/qwen27b)
  add_subdirectory(tests/models/qwen36_a3b)
  add_subdirectory(tests/models/gemma4)
  gufo_qwen3_asr_register_tests()
  gufo_qwen_image_21_register_tests()
  gufo_diffusion_gemma_register_tests()

  add_subdirectory(tests/models/embedding_gemma2)

  add_test(NAME gufo_version COMMAND gufo --version)
  gufo_escape_version_regex("${GUFO_BRAND}" brand_regex)
  gufo_escape_version_regex("${GUFO_PRODUCT_VERSION}" product_version_regex)
  gufo_escape_version_regex("${GUFO_RELEASE_VERSION}" engine_version_regex)
  gufo_escape_version_regex("${GUFO_BUILD_REVISION}" revision_regex)
  set_tests_properties(gufo_version PROPERTIES
    LABELS "cpu"
    PASS_REGULAR_EXPRESSION
      "^${brand_regex} ${product_version_regex} \\(${revision_regex}\\), engine gufo ${engine_version_regex}[\r\n]*$")
  add_test(NAME version_sources_test COMMAND ${CMAKE_COMMAND}
    "-DVERSION_MODULE=${CMAKE_SOURCE_DIR}/cmake/Version.cmake"
    "-DTEST_ROOT=${CMAKE_BINARY_DIR}/version-sources-test"
    -P "${CMAKE_SOURCE_DIR}/tests/cli/version_sources_test.cmake")
  set_tests_properties(version_sources_test PROPERTIES LABELS "cpu")

  add_test(NAME gufo_help COMMAND gufo --help)
  set_tests_properties(gufo_help PROPERTIES LABELS "cpu")

  add_test(NAME gufo_probe COMMAND gufo probe)
  set_tests_properties(gufo_probe PROPERTIES LABELS "cpu")

  add_test(NAME strix_diagnose_help COMMAND gufo diagnose --help)
  set_tests_properties(strix_diagnose_help PROPERTIES LABELS "cpu")

  add_test(NAME strix_diagnose_json COMMAND gufo diagnose --json --section platform)
  set_tests_properties(strix_diagnose_json PROPERTIES LABELS "cpu")

  add_test(NAME strix_prompt_help COMMAND gufo prompt --help)
  set_tests_properties(strix_prompt_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_video_help COMMAND gufo video --help)
  set_tests_properties(strix_video_help PROPERTIES
    LABELS "cpu;server;cli;minimax-h3")

  add_test(NAME gufo_transcribe_help COMMAND gufo transcribe --help)
  set_tests_properties(gufo_transcribe_help PROPERTIES
    LABELS "cpu;server;cli;qwen3-asr")

  add_test(NAME gufo_serve_asr_help COMMAND gufo serve asr --help)
  set_tests_properties(gufo_serve_asr_help PROPERTIES
    LABELS "cpu;server;cli;qwen3-asr")
  add_test(NAME gufo_serve_tts_help COMMAND gufo serve tts --help)
  set_tests_properties(gufo_serve_tts_help PROPERTIES
    LABELS "cpu;server;cli;qwen3-tts")
  add_test(NAME gufo_serve_embedding_help COMMAND gufo serve embedding --help)
  set_tests_properties(gufo_serve_embedding_help PROPERTIES
    LABELS "cpu;server;cli;embedding-gemma2")

  add_test(NAME strix_serve_help COMMAND gufo serve --help)
  set_tests_properties(strix_serve_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_prompt_smoke COMMAND gufo prompt "Hello world")
  set_tests_properties(strix_prompt_smoke PROPERTIES LABELS "cpu;server;cli")

  set(GUFO_PRODUCTION_ENV_SPEC "$ENV{GUFO_PRODUCTION_ENV_SPEC}" CACHE FILEPATH "Optional production release settings specification")
  file(TO_CMAKE_PATH "${GUFO_PRODUCTION_ENV_SPEC}" GUFO_PRODUCTION_ENV_SPEC_PATH)
  add_executable(log_async_test tests/cli/log_async_test.cpp src/cli/serve/logging.cpp)
  target_compile_definitions(log_async_test PRIVATE GUFO_LOG_TESTING=1)
  target_link_libraries(log_async_test PRIVATE Threads::Threads gufo_platform)
  add_test(NAME log_async_test COMMAND log_async_test)
  set_tests_properties(log_async_test PROPERTIES LABELS "cpu;cli;release;check-pr" TIMEOUT 45)

  if(WIN32)
    # CPU-only CLI-to-creation counter: no NPU runtime, HIP or XRT dependency.
    add_executable(experimental_policy_host tests/cli/experimental_policy_host.cpp
      src/cli/serve/model_manifest.cpp src/cli/serve/release_profile.cpp)
    target_link_libraries(experimental_policy_host PRIVATE gufo_core gufo_logging bubo_prepare)
    if(GUFO_ENABLE_NPU)
      target_compile_definitions(experimental_policy_host PRIVATE GUFO_NPU=1)
      set(experimental_build_mode ON)
    else()
      set(experimental_build_mode OFF)
    endif()
    add_test(NAME experimental_policy_process_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/cli/experimental_policy_process_test.py"
      "$<TARGET_FILE:experimental_policy_host>" "${experimental_build_mode}")
    set_tests_properties(experimental_policy_process_test PROPERTIES LABELS "cpu;cli;fake" TIMEOUT 120)

  endif()

  add_executable(release_profile_test
    tests/cli/release_profile_test.cpp
    src/cli/serve/model_manifest.cpp
    src/cli/serve/release_profile.cpp)
  target_link_libraries(release_profile_test PRIVATE gufo_core gufo_logging bubo_prepare)
  target_compile_definitions(release_profile_test PRIVATE
    GUFO_BRAND_SCAN_PYTHON="${Python3_EXECUTABLE}"
    GUFO_PRODUCTION_ENV_SPEC="${GUFO_PRODUCTION_ENV_SPEC_PATH}"
    GUFO_RELEASE_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
  if(GUFO_ENABLE_NPU)
    target_compile_definitions(release_profile_test PRIVATE GUFO_NPU=1)
  endif()
  add_test(NAME release_profile_test COMMAND release_profile_test)
  set_tests_properties(release_profile_test PROPERTIES LABELS "cpu;cli;release;check-pr")
  add_test(NAME think_nudge_profile_test COMMAND release_profile_test --think-nudge)
  set_tests_properties(think_nudge_profile_test PROPERTIES LABELS "cpu;cli;release;contract")
  # The program-name literal scan on its own: red on its own until the legacy lines are fixed or allow-listed.
  add_test(NAME brand_literal_scan_test COMMAND release_profile_test --brand-scan)
  set_tests_properties(brand_literal_scan_test PROPERTIES LABELS "cpu;cli;release;brand")

  add_executable(arg_parser_test
    tests/cli/arg_parser_test.cpp
  )
  target_link_libraries(arg_parser_test PRIVATE gufo_core)
  add_test(NAME arg_parser_test COMMAND arg_parser_test)
  set_tests_properties(arg_parser_test PROPERTIES LABELS "cpu;cli")

  add_executable(behaviour_test tests/core/behaviour_test.cpp)
  target_link_libraries(behaviour_test PRIVATE gufo_core)
  add_test(NAME behaviour_test COMMAND behaviour_test)
  set_tests_properties(behaviour_test PROPERTIES LABELS "cpu;sampling;behaviour")

  add_executable(think_marker_entropy_test tests/core/think_marker_entropy_test.cpp)
  target_link_libraries(think_marker_entropy_test PRIVATE gufo_core)
  add_test(NAME think_marker_entropy_test COMMAND think_marker_entropy_test)
  set_tests_properties(think_marker_entropy_test PROPERTIES LABELS "cpu;sampling;behaviour")

  add_library(eog_cost_old OBJECT
    tests/core/eog_cost_adapter.cpp
    src/core/json_constraint.cpp
    src/core/json_schema_lexeme.cpp
    src/core/json_schema_regex.cpp
    tests/fixtures/behaviour_fast_baseline/src/core/sampling.cpp)
  target_include_directories(eog_cost_old BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/behaviour_fast_baseline)
  target_link_libraries(eog_cost_old PRIVATE gufo_core)
  target_compile_definitions(eog_cost_old PRIVATE sampling=sampling_eog_old EOG_COST_OLD=1)
  add_executable(eog_cost_test tests/core/eog_cost_test.cpp
    tests/core/eog_cost_adapter.cpp $<TARGET_OBJECTS:eog_cost_old>)
  target_link_libraries(eog_cost_test PRIVATE gufo_core)

  add_executable(loop_breaker_test tests/core/loop_breaker_test.cpp)
  target_link_libraries(loop_breaker_test PRIVATE gufo_core)
  add_test(NAME loop_breaker_test COMMAND loop_breaker_test)
  set_tests_properties(loop_breaker_test PROPERTIES LABELS "cpu;sampling;behaviour")

  add_executable(tool_argument_loop_test tests/core/tool_argument_loop_test.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/component_text_cache.cpp
    src/cli/serve/text_model_runner.cpp
    src/cli/serve/text_generation_scheduler.cpp)
  target_link_libraries(tool_argument_loop_test PRIVATE gufo_logging gufo_core gufo_cache)
  add_test(NAME tool_argument_loop_test COMMAND tool_argument_loop_test)
  set_tests_properties(tool_argument_loop_test PROPERTIES LABELS "cpu;sampling;behaviour")

  # Local token-id replay only: no fixtures, default build or CTest/CI entry.
  add_executable(loop_breaker_replay EXCLUDE_FROM_ALL tests/core/loop_breaker_replay.cpp)
  target_link_libraries(loop_breaker_replay PRIVATE gufo_core)

  # Separate executables avoid all old/new sampler symbol and type collisions.
  # The baseline source/header/behaviour header are exact c847e6df Git blobs.
  add_executable(behaviour_fast_test
    tests/core/behaviour_fast_test.cpp src/core/sampling.cpp)
  target_link_libraries(behaviour_fast_test PRIVATE gufo_core)
  add_executable(behaviour_fast_baseline
    tests/core/behaviour_fast_test.cpp
    tests/fixtures/behaviour_fast_baseline/src/core/sampling.cpp)
  target_include_directories(behaviour_fast_baseline BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/behaviour_fast_baseline)
  target_link_libraries(behaviour_fast_baseline PRIVATE gufo_core)
  file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/behaviour_fast_baseline/src/core/sampling.cpp
    GUFO_BEHAVIOUR_BASELINE_SHA)
  target_compile_definitions(behaviour_fast_baseline PRIVATE
    GUFO_BEHAVIOUR_BASELINE_SHA="${GUFO_BEHAVIOUR_BASELINE_SHA}")
  # gufo_core is created before the directory's test-only -UNDEBUG option.
  # Compile the frozen sampler with the same effective NDEBUG as the candidate.
  set_source_files_properties(
    src/core/sampling.cpp tests/fixtures/behaviour_fast_baseline/src/core/sampling.cpp
    PROPERTIES COMPILE_OPTIONS "-DNDEBUG")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME behaviour_fast_equivalence
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/core/behaviour_fast_compare.py
      $<TARGET_FILE:behaviour_fast_baseline> $<TARGET_FILE:behaviour_fast_test>
      ${CMAKE_CURRENT_BINARY_DIR}/behaviour-fast-equivalence)
  set_tests_properties(behaviour_fast_equivalence PROPERTIES
    LABELS "cpu;sampling;behaviour" RUN_SERIAL TRUE)

  add_executable(json_constraint_test tests/core/json_constraint_test.cpp)
  target_link_libraries(json_constraint_test PRIVATE gufo_core)
  target_compile_definitions(json_constraint_test PRIVATE
    GUFO_LLAMA_TOOL_GRAMMAR_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama_cpp_tool_grammar.json")
  add_test(NAME json_constraint_test COMMAND json_constraint_test)
  set_tests_properties(json_constraint_test PROPERTIES LABELS "cpu;json;sampling")

  add_executable(json_mask_test tests/core/json_mask_test.cpp)
  target_link_libraries(json_mask_test PRIVATE gufo_core)
  add_test(NAME json_mask_test COMMAND json_mask_test)
  set_tests_properties(json_mask_test PROPERTIES LABELS "cpu;json;sampling")

  add_executable(json_test tests/core/json_test.cpp)
  target_link_libraries(json_test PRIVATE gufo_core)
  add_test(NAME json_test COMMAND json_test)
  set_tests_properties(json_test PROPERTIES LABELS "cpu;json")

  add_executable(heap_placement_test tests/core/heap_placement_test.cpp)
  target_link_libraries(heap_placement_test PRIVATE gufo_core)
  add_test(NAME heap_placement_test COMMAND heap_placement_test)
  set_tests_properties(heap_placement_test PROPERTIES LABELS "cpu;hip;core")

  add_executable(sampling_test tests/core/sampling_test.cpp)
  target_link_libraries(sampling_test PRIVATE gufo_core)
  add_test(NAME sampling_test COMMAND sampling_test)
  set_tests_properties(sampling_test PROPERTIES LABELS "cpu;sampling")
  add_executable(platform_tuning_test tests/core/platform_tuning_test.cpp)
  target_link_libraries(platform_tuning_test PRIVATE gufo_logging gufo_platform)
  add_test(NAME platform_tuning_test COMMAND platform_tuning_test)
  set_tests_properties(platform_tuning_test PROPERTIES LABELS "cpu")

  add_executable(openai_chat_test
    tests/cli/openai_chat_test.cpp
    src/cli/serve/model_manifest.cpp
    src/cli/serve/release_profile.cpp
    src/cli/serve/logging.cpp
    src/cli/serve/openai_chat.cpp
    src/models/deepseek_v4_flash/chat_template.cpp
    src/models/deepseek_v4_flash/tool_calls.cpp
  )
  target_link_libraries(openai_chat_test PRIVATE gufo_core gufo_gemma4 bubo_prepare)
  add_test(NAME openai_chat_test COMMAND openai_chat_test)
  add_test(NAME debug_hooks_settings_test COMMAND openai_chat_test --debug-hooks)
  set_tests_properties(debug_hooks_settings_test PROPERTIES LABELS "cpu;debug-hooks" TIMEOUT 300)
  set_tests_properties(openai_chat_test PROPERTIES
    LABELS "cpu;server;api;openai;streaming;tools")

  add_executable(qwen_incremental_http_test
    tests/cli/qwen_incremental_http_test.cpp
    src/cli/serve/model_manifest.cpp
    src/cli/serve/release_profile.cpp
    src/cli/serve/logging.cpp
    src/cli/serve/openai_chat.cpp
    src/models/deepseek_v4_flash/chat_template.cpp
    src/models/deepseek_v4_flash/tool_calls.cpp)
  target_link_libraries(qwen_incremental_http_test PRIVATE
    gufo_core gufo_gemma4 bubo_prepare)
  add_test(NAME qwen_incremental_http_test COMMAND qwen_incremental_http_test)
  set_tests_properties(qwen_incremental_http_test PROPERTIES
    LABELS "cpu;server;api;openai;tools;tokenization;external-model"
    SKIP_RETURN_CODE 77 TIMEOUT 300)

  add_executable(quote_tracker_test tests/cli/quote_tracker_test.cpp)
  target_link_libraries(quote_tracker_test PRIVATE gufo_core)
  add_test(NAME quote_tracker_test COMMAND quote_tracker_test)
  set_tests_properties(quote_tracker_test PROPERTIES LABELS "cpu;server;api")

  # the trace suite simulates RLIMIT_FSIZE
  if (NOT WIN32)
  add_executable(trace_test tests/cli/trace_test.cpp)
  target_link_libraries(trace_test PRIVATE gufo_logging gufo_platform)
  add_test(NAME trace_test COMMAND trace_test)
  set_tests_properties(trace_test PROPERTIES LABELS "cpu;server;trace")
  endif()

  add_executable(cache_disk_limits_test tests/cli/cache_disk_limits_test.cpp)
  add_test(NAME cache_disk_limits_test COMMAND cache_disk_limits_test)
  set_tests_properties(cache_disk_limits_test PROPERTIES LABELS "cpu;server;cli" TIMEOUT 30)

  # These HTTP/WebSocket suites drive raw POSIX sockets or fork() directly.
  if (NOT WIN32)
  add_executable(http_server_test tests/cli/http_server_test.cpp)
  target_link_libraries(http_server_test PRIVATE gufo_http)
  add_test(NAME http_server_test COMMAND http_server_test)
  add_test(NAME debug_hooks_http_test COMMAND http_server_test --debug-hooks)
  set_tests_properties(debug_hooks_http_test PROPERTIES LABELS "cpu;debug-hooks" TIMEOUT 30)
  set_tests_properties(http_server_test PROPERTIES LABELS "cpu;server;http" TIMEOUT 30)

  add_executable(http_server_signals_test tests/cli/http_server_signals_test.cpp)
  target_link_libraries(http_server_signals_test PRIVATE gufo_http)
  add_test(NAME http_server_signals_test COMMAND http_server_signals_test)
  set_tests_properties(http_server_signals_test
    PROPERTIES LABELS "cpu;server;http" TIMEOUT 60)

  add_executable(audio_websocket_test tests/cli/audio_websocket_test.cpp)
  target_link_libraries(audio_websocket_test PRIVATE gufo_http)
  add_test(NAME audio_websocket_test COMMAND audio_websocket_test)
  set_tests_properties(audio_websocket_test PROPERTIES LABELS "cpu;server;audio" TIMEOUT 30)
  endif()

  add_test(NAME serve_cli_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/serve_test.py
      $<TARGET_FILE:gufo>)
  set_tests_properties(serve_cli_test PROPERTIES LABELS "cpu;server;cli" TIMEOUT 120)

  add_test(NAME kernel_resources_parser_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/kernel_resources_test.py)
  set_tests_properties(kernel_resources_parser_test PROPERTIES
    LABELS "cpu;kernels" TIMEOUT 30)

  add_test(NAME check_docs_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/check_docs_test.py)
  set_tests_properties(check_docs_test PROPERTIES LABELS "cpu;docs" TIMEOUT 30)

  add_test(NAME check_pr_title_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/check_pr_title_test.py)
  set_tests_properties(check_pr_title_test PROPERTIES LABELS "cpu;docs" TIMEOUT 30)

  # It reads the code objects of an ELF gufo binary; the Windows binary is a PE file the script cannot parse.
  if(ENGINE_ENABLE_HIP AND TARGET gufo AND NOT WIN32)
    # A compiler update can make a kernel spill with no source change. Clang 23
    # increased the scratch of the Q8 prefill GEMM from 0 to 480 bytes. This
    # test fails if a kernel uses more scratch than the baseline allows. It
    # reads the code objects in the binary and does not use a GPU.
    get_filename_component(GUFO_HIP_BIN_DIR ${CMAKE_HIP_COMPILER} DIRECTORY)
    add_test(NAME kernel_resources_test
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/check-kernel-resources.py
        $<TARGET_FILE:gufo>
        --baseline ${CMAKE_CURRENT_SOURCE_DIR}/tools/ci/kernel-resources.json
        --bundler ${GUFO_HIP_BIN_DIR}/clang-offload-bundler)
    set_tests_properties(kernel_resources_test PROPERTIES
      LABELS "cpu;kernels" TIMEOUT 120)
  endif()

  add_subdirectory(tests/models/deepseek_v4_flash)
  add_subdirectory(tests/models/qwen38_flash_next)

  if(ENGINE_ENABLE_HIP)
    add_executable(chat_template_hf_token_golden_test
      tests/models/chat_template_hf_token_golden_test.cpp)
    target_link_libraries(chat_template_hf_token_golden_test PRIVATE gufo_core)
    target_compile_definitions(chat_template_hf_token_golden_test PRIVATE
      GUFO_CHAT_TEMPLATE_HF_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json")
    add_test(NAME chat_template_hf_token_golden_test
      COMMAND chat_template_hf_token_golden_test)
    set_tests_properties(chat_template_hf_token_golden_test PROPERTIES
      LABELS "gpu;hip;tokenization;chat_template;qwen;external-model"
      SKIP_RETURN_CODE 77)
  endif()

  add_executable(memory_guard_test tests/cli/memory_guard_test.cpp)
  target_link_libraries(memory_guard_test PRIVATE gufo_core)
  add_test(NAME memory_guard_test COMMAND memory_guard_test)
  set_tests_properties(memory_guard_test PROPERTIES LABELS "cpu;server;memory")

  add_executable(continuation_cache_test
    tests/cli/continuation_cache_test.cpp
    src/cli/serve/continuation_cache.cpp)
  target_link_libraries(continuation_cache_test PRIVATE gufo_core gufo_logging)
  add_test(NAME continuation_cache_test COMMAND continuation_cache_test)
  set_tests_properties(continuation_cache_test PROPERTIES
    LABELS "cpu;server;cache;state;lifecycle")

  add_executable(component_text_cache_test
    tests/cli/component_text_cache_test.cpp
    src/cli/serve/component_text_cache.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/text_model_runner.cpp)
  target_link_libraries(component_text_cache_test PRIVATE gufo_cache_fake gufo_core gufo_logging gufo_qwen38_flash_next)
  add_test(NAME component_text_cache_test COMMAND component_text_cache_test)
  set_tests_properties(component_text_cache_test PROPERTIES LABELS "cpu;server;cache;state;lifecycle")
  add_test(NAME qfn_packed_ingress_publication_test
    COMMAND ${CMAKE_COMMAND} -E env GUFO_TEST_PACKED_INGRESS_ONLY=1
      $<TARGET_FILE:component_text_cache_test>)
  set_tests_properties(qfn_packed_ingress_publication_test PROPERTIES
    LABELS "cpu;server;cache;qwen38_flash_next")
  add_test(NAME qfn_rotated_process_local_disk_test
    COMMAND ${CMAKE_COMMAND} -E env GUFO_TEST_ROTATED_PROCESS_LOCAL_ONLY=disk
      $<TARGET_FILE:component_text_cache_test>)
  set_tests_properties(qfn_rotated_process_local_disk_test PROPERTIES
    LABELS "cpu;server;cache;qwen38_flash_next")
  add_executable(qfn_kv_ingress_validation_test
    tests/cli/qfn_kv_ingress_validation_test.cpp)
  target_link_libraries(qfn_kv_ingress_validation_test PRIVATE
    gufo_qwen38_flash_next gufo_cache)
  target_compile_options(qfn_kv_ingress_validation_test PRIVATE
    -fno-fast-math -ffp-contract=off -frounding-math)
  add_test(NAME qfn_kv_ingress_validation_test COMMAND qfn_kv_ingress_validation_test)
  set_tests_properties(qfn_kv_ingress_validation_test PROPERTIES
    LABELS "cpu;server;cache;qwen38_flash_next")

  add_executable(component_async_replay_test
    tests/cli/component_async_replay_test.cpp
    src/cli/serve/component_text_cache.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/text_model_runner.cpp)
  target_link_libraries(component_async_replay_test PRIVATE gufo_cache_fake gufo_core gufo_logging)
  target_compile_options(component_async_replay_test PRIVATE -UNDEBUG)
  add_test(NAME component_async_replay_test COMMAND component_async_replay_test)
  add_test(NAME component_async_registration_adjacent
    COMMAND component_async_replay_test --registration-layout=adjacent)
  add_test(NAME component_async_registration_gapped
    COMMAND component_async_replay_test --registration-layout=gapped)
  set_tests_properties(component_async_replay_test
    component_async_registration_adjacent component_async_registration_gapped PROPERTIES
    LABELS "cpu;server;cache;state;lifecycle" TIMEOUT 180)

  add_executable(component_sequence_test tests/cli/component_sequence_test.cpp)
  target_link_libraries(component_sequence_test PRIVATE gufo_cache)
  add_test(NAME component_sequence_test COMMAND component_sequence_test)
  add_test(NAME component_sequence_fork_test COMMAND component_sequence_test --fork)
  set_tests_properties(component_sequence_test component_sequence_fork_test PROPERTIES
    LABELS "cpu;cache;prefill;acceptance" TIMEOUT 180)

  add_executable(text_model_runner_test
    tests/cli/text_model_runner_test.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/component_text_cache.cpp
  src/cli/serve/text_model_runner.cpp)
  target_link_libraries(text_model_runner_test PRIVATE gufo_logging gufo_core gufo_cache)
  add_test(NAME text_model_runner_test COMMAND text_model_runner_test)
  # The fake runners admit snapshots through the real host-memory floor; with it
  # on, the test fails whenever this machine has under 4 GiB free. The floor
  # itself is tested with explicit values in continuation_cache_test.
  set_tests_properties(text_model_runner_test PROPERTIES
    LABELS "cpu;server;runner;cache;state;lifecycle"
    ENVIRONMENT "GUFO_CACHE_HOST_FLOOR_MB=0")

  add_executable(prefill_tail_component_test
    tests/cli/prefill_tail_component_test.cpp
    src/cli/serve/component_text_cache.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/text_model_runner.cpp)
  target_link_libraries(prefill_tail_component_test PRIVATE
    gufo_cache_fake gufo_core gufo_logging)
  set(prefill_tail_tests)
  foreach(case IN ITEMS TestExpertTailRequestBoundary TestExpertTailSnapshotEdges
      TestFastWarmSessionRunsNormal TestFastLongPromptThenWarmTurns
      TestFastRestoreInsideCutRegion TestFastThresholdEdge)
    add_test(NAME prefill_tail_component.${case}
      COMMAND prefill_tail_component_test --test ${case})
    list(APPEND prefill_tail_tests prefill_tail_component.${case})
  endforeach()
  foreach(scenario RANGE 0 13)
    add_test(NAME prefill_tail_component.newest_${scenario}
      COMMAND prefill_tail_component_test --tail-newest ${scenario})
    list(APPEND prefill_tail_tests prefill_tail_component.newest_${scenario})
  endforeach()
  foreach(capacity IN ITEMS 4 8)
    add_test(NAME prefill_tail_component.real_sequence_${capacity}
      COMMAND prefill_tail_component_test --tail-real-sequence ${capacity})
    list(APPEND prefill_tail_tests prefill_tail_component.real_sequence_${capacity})
  endforeach()
  add_test(NAME prefill_tail_component.default_golden
    COMMAND prefill_tail_component_test --tail-default-golden)
  list(APPEND prefill_tail_tests prefill_tail_component.default_golden)
  set_tests_properties(${prefill_tail_tests} PROPERTIES
    LABELS "cpu;server;cache;prefill;acceptance"
    ENVIRONMENT "GUFO_CACHE_HOST_FLOOR_MB=0" TIMEOUT 120)

  # the disk-store suite exercises permission faults with chmod and lstat
  if (NOT WIN32)
  add_executable(continuation_disk_store_test
    tests/cli/continuation_disk_store_test.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/component_text_cache.cpp
  src/cli/serve/text_model_runner.cpp
    src/cli/serve/continuation_cache.cpp)
  target_link_libraries(continuation_disk_store_test PRIVATE gufo_logging gufo_core gufo_cache)
  add_test(NAME continuation_disk_store_test
    COMMAND continuation_disk_store_test)
  set_tests_properties(continuation_disk_store_test PROPERTIES
    LABELS "cpu;server;cache;disk;state;lifecycle")
  endif()

  add_executable(logit_sampler_test tests/cli/logit_sampler_test.cpp)
  target_link_libraries(logit_sampler_test PRIVATE gufo_core)
  add_test(NAME logit_sampler_test COMMAND logit_sampler_test)
  add_test(NAME debug_hooks_logprobs_test COMMAND logit_sampler_test --debug-hooks)
  set_tests_properties(debug_hooks_logprobs_test PROPERTIES LABELS "cpu;debug-hooks" TIMEOUT 30)
  set_tests_properties(logit_sampler_test PROPERTIES
    LABELS "cpu;server;sampling")

  add_executable(raw_logprobs_equivalence_test
    tests/cli/raw_logprobs_equivalence_test.cpp)
  target_link_libraries(raw_logprobs_equivalence_test PRIVATE gufo_core)
  add_test(NAME raw_logprobs_equivalence_test COMMAND raw_logprobs_equivalence_test)
  set_tests_properties(raw_logprobs_equivalence_test PROPERTIES
    LABELS "cpu;server;sampling" TIMEOUT 120)

  add_executable(legacy_logprobs_memory_test
    tests/cli/legacy_logprobs_memory_test.cpp)
  target_link_libraries(legacy_logprobs_memory_test PRIVATE gufo_core)
  add_test(NAME legacy_logprobs_memory_test COMMAND legacy_logprobs_memory_test)
  set_tests_properties(legacy_logprobs_memory_test PROPERTIES
    LABELS "cpu;server;sampling" TIMEOUT 120)

  add_executable(text_generation_scheduler_test
    tests/cli/text_generation_scheduler_test.cpp
    src/cli/serve/continuation_cache.cpp
    src/cli/serve/continuation_disk_store.cpp
    src/cli/serve/component_text_cache.cpp
  src/cli/serve/text_model_runner.cpp
    src/cli/serve/text_generation_scheduler.cpp)
  target_link_libraries(text_generation_scheduler_test PRIVATE gufo_logging gufo_core gufo_cache)
  add_test(NAME text_generation_scheduler_test
    COMMAND text_generation_scheduler_test)
  add_test(NAME debug_hooks_timeline_test COMMAND text_generation_scheduler_test --debug-hooks)
  # As text_model_runner_test: the fake runners admit snapshots through the real
  # host-memory floor, so with it on these fail whenever this machine has under
  # 4 GiB free (10-15 07:3x build-int: "snapshot skipped reason=host_memory_floor").
  set_tests_properties(debug_hooks_timeline_test PROPERTIES LABELS "cpu;debug-hooks" TIMEOUT 30
    ENVIRONMENT "GUFO_CACHE_HOST_FLOOR_MB=0")
  set_tests_properties(text_generation_scheduler_test PROPERTIES
    LABELS "cpu;server;scheduler;queue;state;lifecycle;cancellation"
    ENVIRONMENT "GUFO_CACHE_HOST_FLOOR_MB=0")

  add_executable(prompt_cli_test
    tests/cli/prompt_test.cpp
  )
  target_link_libraries(prompt_cli_test PRIVATE gufo_llm_cli)

  add_test(NAME prompt_cli_test COMMAND prompt_cli_test)
  set_tests_properties(prompt_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bench_cli_test
    tests/cli/bench_test.cpp
  )
  target_link_libraries(bench_cli_test PRIVATE gufo_llm_cli)

  add_test(NAME bench_cli_test COMMAND bench_cli_test)
  set_tests_properties(bench_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(video_cli_test
    tests/cli/video_test.cpp
    src/cli/video/video.cpp
  )
  target_link_libraries(video_cli_test PRIVATE gufo_core)
  add_test(NAME video_cli_test COMMAND video_cli_test)
  set_tests_properties(video_cli_test PROPERTIES
    LABELS "cpu;server;cli;minimax-h3;quality")

  add_executable(video_jobs_test
    tests/cli/video_jobs_test.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_jobs_test PRIVATE gufo_logging gufo_core gufo_cache)
  add_test(NAME video_jobs_test COMMAND video_jobs_test)
  set_tests_properties(video_jobs_test PROPERTIES
    LABELS "cpu;server;video;minimax-h3;queue;cancellation;quality"
    SKIP_RETURN_CODE 77)

  add_executable(video_api_test
    tests/cli/video_api_test.cpp
    src/cli/serve/video_api.cpp
    src/cli/serve/video_jobs.cpp
  )
  target_link_libraries(video_api_test PRIVATE gufo_logging gufo_core gufo_cache)
  add_test(NAME video_api_test COMMAND video_api_test)
  set_tests_properties(video_api_test PROPERTIES
    LABELS "cpu;server;api;video;minimax-h3;quality")

  add_test(NAME h3_profile_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_profile_test.py)
  set_tests_properties(h3_profile_test PROPERTIES
    LABELS "cpu;profiling;video;minimax-h3;quality")

  add_test(NAME stage_trace_table_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/stage_trace_table_test.py)
  set_tests_properties(stage_trace_table_test PROPERTIES LABELS "cpu;profiling")

  add_test(NAME h3_preset_quality_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_preset_quality_test.py)
  set_tests_properties(h3_preset_quality_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;quality;video;audio")

  add_test(NAME h3_rng_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_rng_test.py)
  set_tests_properties(h3_rng_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;quality")

  add_test(NAME h3_latent_quality_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_latent_quality_test.py)
  set_tests_properties(h3_latent_quality_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;quality")

  add_test(NAME h3_cache_control_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_cache_control_test.py)
  set_tests_properties(h3_cache_control_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;profiling")

  add_test(NAME h3_profile_report_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/h3_profile_report_test.py)
  set_tests_properties(h3_profile_report_test PROPERTIES
    LABELS "cpu;tools;models;minimax-h3;profiling")

  add_test(NAME serving_bench_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/serving_bench_test.py)
  set_tests_properties(serving_bench_test PROPERTIES
    LABELS "cpu;tools;server;benchmarks;statistics;json"
    SKIP_RETURN_CODE 77)

  add_test(NAME functional_runner_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional/test_runner.py)
  set_tests_properties(functional_runner_test PROPERTIES LABELS "cpu;server;functional")
  add_test(NAME cache_workloads_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional/test_cache_workloads.py)
  set_tests_properties(cache_workloads_test PROPERTIES LABELS "cpu;server;functional")
  # the fault runner needs the Linux-only gate library
  if (NOT WIN32)
  add_test(NAME cache_disk_faults_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/functional/cache_disk_faults_test.py
      $<TARGET_FILE:cache_disk_faults>)
  set_tests_properties(cache_disk_faults_test PROPERTIES LABELS "cpu;server;functional")
  endif()

  add_test(NAME speculative_corpus_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/speculative_corpus_test.py)
  set_tests_properties(speculative_corpus_test PROPERTIES
    LABELS "cpu;tools;benchmarks;speculative;ds4;qwen")

  add_test(NAME prof_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/prof_test.py)
  set_tests_properties(prof_test PROPERTIES
    LABELS "cpu;tools;profiling")

  add_executable(gpu_queues_test tests/diagnostics/gpu_queues_test.cpp)
  target_link_libraries(gpu_queues_test PRIVATE gufo_diagnostics gufo_platform)
  add_test(NAME gpu_queues_test COMMAND gpu_queues_test)
  set_tests_properties(gpu_queues_test PROPERTIES LABELS "cpu;diagnostics;serving")

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE gufo_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

  add_executable(allocation_benchmark_test
    tests/diagnostics/allocation_benchmark_test.cpp)
  target_compile_definitions(allocation_benchmark_test PRIVATE
    TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(allocation_benchmark_test PRIVATE gufo_diagnostics)
  add_test(NAME allocation_benchmark_test COMMAND allocation_benchmark_test)
  set_tests_properties(allocation_benchmark_test PROPERTIES
    LABELS "gpu;hip;gfx1151;diagnostics;allocation;opt-c091")

  # the prefetch fixture uses mkstemp and mmap
  if (NOT WIN32)
  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE gufo_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")
  endif()

  add_executable(image_resize_test tests/core/image_resize_test.cpp)
  target_link_libraries(image_resize_test PRIVATE gufo_core)
  add_test(NAME image_resize_test COMMAND image_resize_test)
  set_tests_properties(image_resize_test PROPERTIES LABELS "cpu;core;image;vision")

  # the digest-cache case restores mtimes with utimensat
  if (NOT WIN32)
  add_executable(gguf_identity_test tests/core/gguf_identity_test.cpp)
  target_link_libraries(gguf_identity_test PRIVATE gufo_core)

  add_test(NAME gguf_identity_test COMMAND gguf_identity_test)
  set_tests_properties(gguf_identity_test PROPERTIES LABELS "cpu;core;gguf")
  endif()

  add_executable(minimax_h3_runtime_test
    tests/models/minimax_h3_runtime_test.cpp)
  target_link_libraries(minimax_h3_runtime_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_runtime_test COMMAND minimax_h3_runtime_test)
  set_tests_properties(minimax_h3_runtime_test PROPERTIES
    LABELS "cpu;models;minimax-h3;safetensors;residency;failure-injection")

  add_executable(qwen3_tts_loader_test
    tests/models/qwen3_tts/loader_test.cpp)
  target_link_libraries(qwen3_tts_loader_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_loader_test COMMAND qwen3_tts_loader_test)
  set_tests_properties(qwen3_tts_loader_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;loader;safetensors"
    SKIP_RETURN_CODE 77)

  add_executable(qwen3_tts_audio_api_test
    tests/models/qwen3_tts/audio_api_test.cpp
    src/cli/serve/audio_tts_api.cpp
    src/cli/serve/tts_service.cpp)
  target_link_libraries(qwen3_tts_audio_api_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_audio_api_test COMMAND qwen3_tts_audio_api_test)
  set_tests_properties(qwen3_tts_audio_api_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;server;api;audio;quality")

  add_executable(qwen3_tts_audio_test
    tests/models/qwen3_tts/audio_test.cpp)
  target_link_libraries(qwen3_tts_audio_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_audio_test COMMAND qwen3_tts_audio_test)
  set_tests_properties(qwen3_tts_audio_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;audio;quality;resampling")

  add_executable(qwen3_tts_config_test
    tests/models/qwen3_tts/config_test.cpp)
  target_link_libraries(qwen3_tts_config_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_config_test COMMAND qwen3_tts_config_test)

  add_executable(qwen3_tts_tokenizer_test
    tests/models/qwen3_tts/tokenizer_test.cpp)
  target_link_libraries(qwen3_tts_tokenizer_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_tokenizer_test COMMAND qwen3_tts_tokenizer_test)
  set_tests_properties(qwen3_tts_tokenizer_test PROPERTIES
    SKIP_RETURN_CODE 77)
  set_tests_properties(qwen3_tts_config_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;config;quality")

  add_executable(qwen3_tts_prompt_test
    tests/models/qwen3_tts/prompt_test.cpp)
  target_link_libraries(qwen3_tts_prompt_test PRIVATE gufo_qwen3_tts)
  add_test(NAME qwen3_tts_prompt_test COMMAND qwen3_tts_prompt_test)
  set_tests_properties(qwen3_tts_prompt_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;prompt;quality;exact")

  add_test(NAME qwen3_tts_reference_artifacts_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/qwen3_tts/reference/verify_artifacts.py
      --contract
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/qwen3_tts/reference_contract.json
      --artifacts
      ${CMAKE_CURRENT_SOURCE_DIR}/artifacts/qwen3_tts)
  set_tests_properties(qwen3_tts_reference_artifacts_test PROPERTIES
    LABELS "cpu;models;qwen3-tts;reference;quality;exact"
    SKIP_RETURN_CODE 77)

  add_executable(minimax_h3_tokenizer_test
    tests/models/minimax_h3_tokenizer_test.cpp)
  target_link_libraries(minimax_h3_tokenizer_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_tokenizer_test COMMAND minimax_h3_tokenizer_test)
  set_tests_properties(minimax_h3_tokenizer_test PROPERTIES
    LABELS "cpu;models;minimax-h3;tokenizer;quality")

  add_executable(minimax_h3_sampling_test
    tests/models/minimax_h3_sampling_test.cpp)
  target_link_libraries(minimax_h3_sampling_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_sampling_test COMMAND minimax_h3_sampling_test)
  set_tests_properties(minimax_h3_sampling_test PROPERTIES
    LABELS "cpu;models;minimax-h3;layout;schedule;rng;quality")

  add_executable(minimax_h3_generation_test
    tests/models/minimax_h3_generation_test.cpp)
  target_link_libraries(minimax_h3_generation_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_generation_test
    COMMAND minimax_h3_generation_test)
  set_tests_properties(minimax_h3_generation_test PROPERTIES
    LABELS "cpu;models;minimax-h3;generation;cli;quality;cancellation")

  add_executable(minimax_h3_video_vae_test
    tests/models/minimax_h3_video_vae_test.cpp)
  target_link_libraries(minimax_h3_video_vae_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_video_vae_plan
    COMMAND minimax_h3_video_vae_test)
  set_tests_properties(minimax_h3_video_vae_plan PROPERTIES
    LABELS "cpu;models;minimax-h3;video;vae;tiling;chunking")

  add_executable(minimax_h3_media_test
    tests/models/minimax_h3_media_test.cpp)
  target_link_libraries(minimax_h3_media_test PRIVATE gufo_core)
  add_test(NAME minimax_h3_media_test COMMAND minimax_h3_media_test)
  set_tests_properties(minimax_h3_media_test PROPERTIES
    LABELS "cpu;models;minimax-h3;video;audio;ffmpeg;mux;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ggml_dequant_test tests/core/ggml_dequant_test.cpp)
  target_link_libraries(ggml_dequant_test PRIVATE gufo_core)

  add_test(NAME ggml_dequant_test COMMAND ggml_dequant_test)
  set_tests_properties(ggml_dequant_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(ggml_iq_dequant_test tests/core/ggml_iq_dequant_test.cpp)
  target_link_libraries(ggml_iq_dequant_test PRIVATE gufo_core)
  add_test(NAME ggml_iq_dequant_test COMMAND ggml_iq_dequant_test)
  set_tests_properties(ggml_iq_dequant_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(ggml_gemm_parity_test tests/core/ggml_gemm_parity_test.cpp)
  target_link_libraries(ggml_gemm_parity_test PRIVATE gufo_core)

  add_test(NAME ggml_gemm_parity_test COMMAND ggml_gemm_parity_test)
  set_tests_properties(ggml_gemm_parity_test PROPERTIES LABELS "cpu;core;quant")

  add_executable(exact_token_test tests/quality/exact_token_test.cpp)
  target_link_libraries(exact_token_test PRIVATE gufo_core)
  add_test(NAME exact_token_test COMMAND exact_token_test)
  set_tests_properties(exact_token_test PROPERTIES LABELS "cpu;quality;tokens")

  add_executable(eval_regrade_test tests/quality/eval_regrade_test.cpp)
  target_link_libraries(eval_regrade_test PRIVATE gufo_core)
  add_test(NAME eval_regrade_test COMMAND eval_regrade_test)
  set_tests_properties(eval_regrade_test PROPERTIES LABELS "cpu;quality;eval")

  add_test(NAME eval_http_test
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/tools/test_eval_http.py
      --gufo $<TARGET_FILE:gufo>
      --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json)
  set_tests_properties(eval_http_test PROPERTIES
    LABELS "cpu;quality;eval;http;cli;server")

  # The eval client fixture drives raw POSIX sockets directly.
  if (NOT WIN32)
  add_executable(eval_http_client_test
    tests/quality/eval_http_client_test.cpp src/eval/http_client.cpp)
  target_include_directories(eval_http_client_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(eval_http_client_test PRIVATE CURL::libcurl Threads::Threads)
  add_test(NAME eval_http_client_test COMMAND eval_http_client_test)
  set_tests_properties(eval_http_client_test PROPERTIES LABELS "cpu;quality;eval;http")
  endif()

  add_executable(kernel_bench_test tests/benchmarks/kernel_bench_test.cpp)
  target_link_libraries(kernel_bench_test PRIVATE gufo_kernel_bench_support)
  add_test(NAME kernel_bench_test COMMAND kernel_bench_test)
  set_tests_properties(kernel_bench_test PROPERTIES
    LABELS "cpu;benchmarks;statistics;json")

  # the publish suite drives descriptors through dup2
  if (NOT WIN32)
  add_executable(platform_publish_test
    tests/core/platform_publish_test.cpp src/core/platform/publish.cpp)
  target_link_libraries(platform_publish_test PRIVATE Threads::Threads)
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(platform_publish_test PRIVATE "-Wl,--wrap=fsync")
  endif()
  add_test(NAME platform_publish_test COMMAND platform_publish_test)
  set_tests_properties(platform_publish_test PROPERTIES LABELS "cpu;platform")
  endif()

  # the prefault fixture maps pages with mmap
  if (NOT WIN32)
  add_executable(platform_prefault_test tests/core/platform_prefault_test.cpp)
  target_link_libraries(platform_prefault_test PRIVATE Threads::Threads)
  add_test(NAME platform_prefault_test COMMAND platform_prefault_test)
  set_tests_properties(platform_prefault_test PROPERTIES LABELS "cpu;platform")
  endif()

  include(cmake/PlatformTests.cmake)

  add_executable(hipblaslt_plan_cache_test
    tests/core/hipblaslt_plan_cache_test.cpp)
  target_link_libraries(hipblaslt_plan_cache_test PRIVATE gufo_core)
  add_test(NAME hipblaslt_plan_cache_test COMMAND hipblaslt_plan_cache_test)
  set_tests_properties(hipblaslt_plan_cache_test PROPERTIES
    LABELS "cpu;hipblaslt;cache")

  if (ENGINE_ENABLE_HIP)
    add_executable(qwen3_tts_synthesis_hip_test
      tests/models/qwen3_tts/synthesis_hip_test.cpp)
    target_link_libraries(qwen3_tts_synthesis_hip_test PRIVATE gufo_qwen3_tts)
    add_test(NAME qwen3_tts_synthesis_hip_test COMMAND qwen3_tts_synthesis_hip_test)
    set_tests_properties(qwen3_tts_synthesis_hip_test PROPERTIES
      LABELS "hip;gpu;models;qwen3-tts;quality;external-model;slow" SKIP_RETURN_CODE 77)
    add_executable(qwen3_tts_speech_decoder_hip_test
      tests/models/qwen3_tts/speech_decoder_hip_test.cpp)
    target_link_libraries(qwen3_tts_speech_decoder_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_speech_decoder_hip_test
      COMMAND qwen3_tts_speech_decoder_hip_test)
    set_tests_properties(qwen3_tts_speech_decoder_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;speech-decoder;audio;quality;external-model;slow;opt-qwen3-tts-device-copy"
      SKIP_RETURN_CODE 77)
    add_executable(qwen3_tts_talker_hip_prefill_test
      tests/models/qwen3_tts/talker_hip_prefill_test.cpp)
    target_link_libraries(qwen3_tts_talker_hip_prefill_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_talker_hip_prefill_test
      COMMAND qwen3_tts_talker_hip_prefill_test)
    set_tests_properties(qwen3_tts_talker_hip_prefill_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;talker;prefill;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(qwen3_tts_variant_talker_hip_test
      tests/models/qwen3_tts/variant_talker_hip_test.cpp)
    target_link_libraries(qwen3_tts_variant_talker_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_variant_talker_hip_test
      COMMAND qwen3_tts_variant_talker_hip_test)
    set_tests_properties(qwen3_tts_variant_talker_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;talker;voice-design;base;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(qwen3_tts_base_encoders_hip_test
      tests/models/qwen3_tts/base_encoders_hip_test.cpp)
    target_link_libraries(qwen3_tts_base_encoders_hip_test PRIVATE
      gufo_qwen3_tts hip::host)
    add_test(NAME qwen3_tts_base_encoders_hip_test
      COMMAND qwen3_tts_base_encoders_hip_test)
    set_tests_properties(qwen3_tts_base_encoders_hip_test PROPERTIES
      LABELS "hip;gpu;gfx1151;models;qwen3-tts;base;speech-encoder;speaker-encoder;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_runtime_hip_test
      tests/models/minimax_h3_runtime_hip_test.cpp)
    target_link_libraries(minimax_h3_runtime_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_runtime_hip_test
      COMMAND minimax_h3_runtime_hip_test)
    set_tests_properties(minimax_h3_runtime_hip_test PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_prompt_encoder_hip_test
      tests/models/minimax_h3_prompt_encoder_hip_test.hip)
    target_compile_definitions(minimax_h3_prompt_encoder_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_prompt_encoder_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_prompt_encoder_analytic
      COMMAND minimax_h3_prompt_encoder_hip_test --analytic)
    set_tests_properties(minimax_h3_prompt_encoder_analytic PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;prompt-encoder;analytic")
    add_test(NAME minimax_h3_prompt_encoder_real
      COMMAND minimax_h3_prompt_encoder_hip_test --real)
    set_tests_properties(minimax_h3_prompt_encoder_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;prompt-encoder;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_sampling_hip_test
      tests/models/minimax_h3_sampling_hip_test.hip)
    target_link_libraries(minimax_h3_sampling_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_sampling_hip_test
      COMMAND minimax_h3_sampling_hip_test)
    set_tests_properties(minimax_h3_sampling_hip_test PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;sampling;euler;analytic"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_dit_hip_test
      tests/models/minimax_h3_dit_hip_test.hip)
    target_compile_definitions(minimax_h3_dit_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_dit_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_dit_analytic
      COMMAND minimax_h3_dit_hip_test --analytic)
    set_tests_properties(minimax_h3_dit_analytic PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;analytic")
    add_test(NAME minimax_h3_dit_block0_real
      COMMAND minimax_h3_dit_hip_test --real)
    set_tests_properties(minimax_h3_dit_block0_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_denoiser_hip_test
      tests/models/minimax_h3_denoiser_hip_test.hip)
    target_compile_definitions(minimax_h3_denoiser_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json")
    target_link_libraries(minimax_h3_denoiser_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_denoiser_smoke
      COMMAND minimax_h3_denoiser_hip_test --smoke)
    set_tests_properties(minimax_h3_denoiser_smoke PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;external-model;slow"
      SKIP_RETURN_CODE 77)
    add_test(NAME minimax_h3_denoiser_oracle
      COMMAND minimax_h3_denoiser_hip_test --oracle-forward)
    set_tests_properties(minimax_h3_denoiser_oracle PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;quality;external-model;slow"
      SKIP_RETURN_CODE 77)
    add_test(NAME minimax_h3_denoiser_production_shape_oracle
      COMMAND minimax_h3_denoiser_hip_test --oracle-512-forward)
    set_tests_properties(minimax_h3_denoiser_production_shape_oracle PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;quality;external-model;slow"
      SKIP_RETURN_CODE 77)
    add_test(NAME minimax_h3_denoiser_reuse_smoke
      COMMAND minimax_h3_denoiser_hip_test --reuse-smoke)
    set_tests_properties(minimax_h3_denoiser_reuse_smoke PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;dit;denoiser;reuse;quality;external-model;slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_video_vae_hip_test
      tests/models/minimax_h3_video_vae_hip_test.hip)
    target_compile_definitions(minimax_h3_video_vae_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
      GUFO_H3_VIDEO_VAE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_h3/video-vae-oracle-v1.json")
    target_link_libraries(minimax_h3_video_vae_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_video_vae_real
      COMMAND minimax_h3_video_vae_hip_test)
    set_tests_properties(minimax_h3_video_vae_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;video;vae;quality;external-model;very-slow"
      SKIP_RETURN_CODE 77)

    add_executable(minimax_h3_audio_vae_hip_test
      tests/models/minimax_h3_audio_vae_hip_test.hip)
    target_compile_definitions(minimax_h3_audio_vae_hip_test PRIVATE
      GUFO_H3_SOURCE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/src/models/minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json"
      GUFO_H3_AUDIO_VAE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/minimax_h3/audio-vae-oracle-v1.json")
    target_link_libraries(minimax_h3_audio_vae_hip_test PRIVATE
      gufo_core hip::host)
    add_test(NAME minimax_h3_audio_vae_primitives
      COMMAND minimax_h3_audio_vae_hip_test --primitives)
    set_tests_properties(minimax_h3_audio_vae_primitives PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;audio;vae;analytic")
    add_test(NAME minimax_h3_audio_vae_real
      COMMAND minimax_h3_audio_vae_hip_test --real)
    set_tests_properties(minimax_h3_audio_vae_real PROPERTIES
      LABELS "gpu;hip;gfx1151;minimax-h3;audio;vae;quality;external-model;very-slow"
      SKIP_RETURN_CODE 77)

    add_executable(hip_memory_visibility_test
      tests/core/hip_memory_visibility_test.hip)
    target_link_libraries(hip_memory_visibility_test PRIVATE gufo_core hip::host)
    add_test(NAME hip_memory_visibility_test COMMAND hip_memory_visibility_test)
    add_test(NAME hip_memory_visibility_restricted_test
      COMMAND hip_memory_visibility_test --deny-long-term-pins)
    add_test(NAME hip_memory_visibility_driver_test COMMAND hip_memory_visibility_test)
    set_tests_properties(hip_memory_visibility_restricted_test PROPERTIES
      SKIP_RETURN_CODE 77)
    set_tests_properties(hip_memory_visibility_test hip_memory_visibility_restricted_test PROPERTIES
      LABELS "gpu;hip;gfx1151"
      ENVIRONMENT "HSA_USERPTR_FOR_PAGED_MEM=1")
    set_tests_properties(hip_memory_visibility_driver_test PROPERTIES
      LABELS "gpu;hip;gfx1151"
      ENVIRONMENT "HSA_USERPTR_FOR_PAGED_MEM=0")

    add_executable(hip_weight_upload_test tests/core/hip_weight_upload_test.cpp)
    target_link_libraries(hip_weight_upload_test PRIVATE gufo_core)
    add_test(NAME hip_weight_upload_test COMMAND hip_weight_upload_test)
    set_tests_properties(hip_weight_upload_test PROPERTIES LABELS "hip;gpu;gfx1151")

    add_executable(hipblaslt_runtime_cache_test
      tests/kernels/hipblaslt_runtime_cache_test.cpp)
    target_link_libraries(hipblaslt_runtime_cache_test PRIVATE
      gufo_core hip::host)
    add_test(NAME hipblaslt_runtime_cache_test
      COMMAND hipblaslt_runtime_cache_test)
    set_tests_properties(hipblaslt_runtime_cache_test PROPERTIES
      LABELS "hip;gpu;hipblaslt;cache")

    add_executable(persistent_deltanet_test
      tests/kernels/persistent_deltanet_test.cpp)
    target_link_libraries(persistent_deltanet_test PRIVATE gufo_core hip::host)
    add_test(NAME persistent_deltanet_test COMMAND persistent_deltanet_test)
    set_tests_properties(persistent_deltanet_test PROPERTIES
      LABELS "hip;gpu;kernels;deltanet;speculative")

    add_executable(gfx1151_acceptance_test tests/benchmarks/gfx1151_acceptance_test.cpp)
    target_link_libraries(gfx1151_acceptance_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME gfx1151_acceptance_test COMMAND gfx1151_acceptance_test)
    set_tests_properties(gfx1151_acceptance_test PROPERTIES LABELS "hip;gpu;benchmarks")

    add_executable(speculative_verification_test tests/core/test_speculative_verification.cpp)
    target_link_libraries(speculative_verification_test PRIVATE gufo_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME speculative_verification_test COMMAND speculative_verification_test)
    set_tests_properties(speculative_verification_test PROPERTIES LABELS "hip;gpu;speculative")
  endif()

  if (ENGINE_ENABLE_HIP)
    add_test(NAME gufo_gpu_probe COMMAND gufo)
    set_tests_properties(gufo_gpu_probe PROPERTIES
      LABELS "platform;probe;hip;gpu")
  else()
    add_test(NAME gufo_cpu_probe COMMAND gufo)
    set_tests_properties(gufo_cpu_probe PROPERTIES
      LABELS "platform;probe;cpu")
  endif()
endif()

if (BUILD_TESTING)
  # Compile the release layer and its CPU regressions as one named group.
  foreach(target IN ITEMS model_manifest_test model_manifest_mutant_stamp_match
      model_manifest_mutant_automatic_hash)
    target_compile_definitions(${target} PRIVATE
      BUBO_MANIFEST_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  endforeach()
  add_executable(brand_test tests/cli/brand_test.cpp)
  add_test(NAME brand_test COMMAND brand_test)
  set_tests_properties(brand_test PROPERTIES LABELS "cpu")
  # A single target also works with PowerShell -File string-array parameters.
  add_custom_target(release_cpu_tests
    DEPENDS gufo brand_test release_profile_test log_async_test platform_tuning_test openai_chat_test arg_parser_test
      behaviour_test think_marker_entropy_test loop_breaker_test tool_argument_loop_test continuation_cache_test
      text_model_runner_test text_generation_scheduler_test)

  include(cmake/Checks.cmake)
endif()

# Qwen3.6-35B-A3B (qwen35moe): its runtime library and the gufo-a3b driver.
add_subdirectory(src/models/qwen36_a3b)
# `bubo serve` runs qwen35moe GGUFs through its text runner
# (src/cli/serve/inference_backend.cpp); OFF builds bubo.exe without it.
option(GUFO_WITH_A3B "bubo serve for Qwen3.6-35B-A3B (qwen35moe) GGUFs" ON)
if(GUFO_WITH_A3B AND TARGET gufo_qwen36_a3b)
  target_link_libraries(${bubo_inference_target} PRIVATE gufo_qwen36_a3b)
  target_compile_definitions(${bubo_inference_target} PRIVATE GUFO_WITH_A3B=1)
  target_compile_definitions(bubo_models PRIVATE GUFO_WITH_A3B=1)
endif()

# A test that starts bubo.exe writes discovery records, prepared-default pointers
# and caches under %LOCALAPPDATA%\Bubo. Tests must not touch the real ones: a
# full ctest run left about 100 records in the owner's folder, and status reads
# every record. Each test that does not set its own LOCALAPPDATA gets a scratch
# folder in the build tree.
function(bubo_isolate_test_state dir)
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(sub IN LISTS subdirs)
    bubo_isolate_test_state("${sub}")
  endforeach()
  get_property(tests DIRECTORY "${dir}" PROPERTY TESTS)
  file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/test-localappdata" scratch)
  foreach(name IN LISTS tests)
    get_property(env TEST "${name}" DIRECTORY "${dir}" PROPERTY ENVIRONMENT)
    if(NOT env MATCHES "(^|;)LOCALAPPDATA=")
      set_property(TEST "${name}" DIRECTORY "${dir}" APPEND PROPERTY
                   ENVIRONMENT "LOCALAPPDATA=${scratch}")
    endif()
  endforeach()
endfunction()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test-localappdata")
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
               bubo_isolate_test_state "${CMAKE_SOURCE_DIR}")
