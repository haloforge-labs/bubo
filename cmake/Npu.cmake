# Windows NPU prefill is opt-in and absent from the build graph when OFF.
option(GUFO_ENABLE_NPU "Build experimental XDNA2 prefill support" OFF)
if(NOT GUFO_ENABLE_NPU)
  return()
endif()
if(NOT WIN32)
  message(FATAL_ERROR "GUFO_ENABLE_NPU currently requires Windows x64")
endif()

# Supply the pinned XRT headers and generated import library.
set(GUFO_XRT_TOOLING_ROOT
  ""
  CACHE PATH "Pinned CPU tooling output with XRT headers and import library")
set(GUFO_XRT_INCLUDE_DIR
  "${GUFO_XRT_TOOLING_ROOT}/XRT/src/runtime_src/core/include"
  CACHE PATH "Pinned XRT public headers")
set(GUFO_XRT_GENERATED_INCLUDE_DIR "${GUFO_XRT_TOOLING_ROOT}/generated"
  CACHE PATH "Tooling-generated XRT version header")
set(GUFO_XRT_IMPORT_LIBRARY "${GUFO_XRT_TOOLING_ROOT}/bin/xrt_coreutil.lib"
  CACHE FILEPATH "Tooling-generated x64 XRT import library")
foreach(required
    "${GUFO_XRT_INCLUDE_DIR}/xrt/xrt_bo.h"
    "${GUFO_XRT_INCLUDE_DIR}/xrt/experimental/xrt_version.h"
    "${GUFO_XRT_GENERATED_INCLUDE_DIR}/xrt/detail/version-slim.h"
    "${GUFO_XRT_IMPORT_LIBRARY}")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "GUFO_ENABLE_NPU missing pinned tooling input: ${required}")
  endif()
endforeach()
get_filename_component(xrt_checkout "${GUFO_XRT_INCLUDE_DIR}/../../../.." ABSOLUTE)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${xrt_checkout}" rev-parse HEAD
  OUTPUT_VARIABLE xrt_commit OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE xrt_rc)
if(NOT xrt_rc EQUAL 0 OR
   NOT xrt_commit STREQUAL "15e6319be8de1e76a6150111a3861729e988fdb5")
  message(FATAL_ERROR "GUFO_ENABLE_NPU requires pinned XRT headers at 15e6319")
endif()

# Snapshot the import library for this configuration.
set(xrt_snapshot "${CMAKE_BINARY_DIR}/npu-xrt/xrt_coreutil.lib")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/npu-xrt")
file(COPY_FILE "${GUFO_XRT_IMPORT_LIBRARY}" "${xrt_snapshot}")
file(SHA256 "${GUFO_XRT_IMPORT_LIBRARY}" xrt_source_sha256)
file(SHA256 "${xrt_snapshot}" xrt_snapshot_sha256)
if(NOT xrt_source_sha256 STREQUAL xrt_snapshot_sha256)
  message(FATAL_ERROR "XRT import library changed during snapshot; reconfigure")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/npu-xrt/import-library.sha256"
  "${xrt_snapshot_sha256}\n")

add_library(npu_runtime OBJECT
  src/npu/xrt_runtime_windows.cpp src/npu/xrt_runtime_policy.cpp
  src/npu/private_bo_adapter.cpp src/npu/session.cpp
  src/npu/server_prefill.cpp src/npu/staging.cpp
  src/npu/bfp.cpp src/npu/linear_z.cpp src/npu/projection.cpp)
target_include_directories(npu_runtime PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(npu_runtime SYSTEM PRIVATE
  "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_runtime PRIVATE _ITERATOR_DEBUG_LEVEL=0)
# Use the same XRT class declarations throughout the runtime.
add_library(npu_xrt INTERFACE)
target_include_directories(npu_xrt SYSTEM INTERFACE
  "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_xrt INTERFACE _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU=1)
target_link_libraries(npu_xrt INTERFACE "${xrt_snapshot}"
  delayimp bcrypt version gufo_logging)
target_link_options(npu_xrt INTERFACE
  "LINKER:/DELAYLOAD:xrt_coreutil.dll" "LINKER:/DELAY:UNLOAD")

function(npu_attach_runtime target)
  target_sources(${target} PRIVATE $<TARGET_OBJECTS:npu_runtime>)
  target_link_libraries(${target} PRIVATE npu_xrt)
endfunction()

# HIP encoding and transfer support.
if(ENGINE_ENABLE_HIP)
  add_library(npu_bfp_hip OBJECT src/npu/bfp.hip.cpp src/npu/linear_z_hip.cpp)
  set_source_files_properties(src/npu/bfp.hip.cpp PROPERTIES LANGUAGE HIP)
  set_target_properties(npu_bfp_hip PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  target_include_directories(npu_bfp_hip PRIVATE "${CMAKE_SOURCE_DIR}")
  target_link_libraries(npu_bfp_hip PRIVATE hip::host)
endif()
