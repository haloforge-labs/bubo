cmake_minimum_required(VERSION 3.21)
# Independent fixtures catch wiring both generated versions to either source.
file(MAKE_DIRECTORY "${TEST_ROOT}/cmake" "${TEST_ROOT}/tools/release")
configure_file("${VERSION_MODULE}" "${TEST_ROOT}/cmake/Version.cmake" COPYONLY)
file(WRITE "${TEST_ROOT}/version.txt" "7.8.9\n")
file(WRITE "${TEST_ROOT}/tools/release/product_version.txt" "1.2.3\n")
set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)
set(GUFO_RELEASE_VERSION "0.6.0" CACHE STRING "stale engine")
set(GUFO_PRODUCT_VERSION "0.6.0" CACHE STRING "stale product")
set(GUFO_REVISION "stale" CACHE STRING "stale revision")
include("${TEST_ROOT}/cmake/Version.cmake")
if(NOT GUFO_RELEASE_VERSION STREQUAL "7.8.9" OR
   NOT GUFO_PRODUCT_VERSION STREQUAL "1.2.3" OR
   NOT GUFO_BUILD_REVISION STREQUAL "unknown")
  message(FATAL_ERROR "Tracked versions or absent-Git fallback were overridden")
endif()
get_filename_component(module_dir "${VERSION_MODULE}" DIRECTORY)
configure_file("${module_dir}/../src/cli/brand.hpp.in"
  "${TEST_ROOT}/brand.hpp" @ONLY)
file(READ "${TEST_ROOT}/brand.hpp" header)
foreach(expected "kProductVersion = \"1.2.3\"" "kEngineVersion = \"7.8.9\"")
  string(FIND "${header}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Generated header did not use independent version sources")
  endif()
endforeach()
# Normal overrides and values from a previous configure must not survive either.
file(WRITE "${TEST_ROOT}/version.txt" "8.9.10\n")
set(GUFO_RELEASE_VERSION "normal-override")
set(GUFO_PRODUCT_VERSION "normal-override")
set(GUFO_REVISION "normal-override")
include("${TEST_ROOT}/cmake/Version.cmake")
if(NOT GUFO_RELEASE_VERSION STREQUAL "8.9.10" OR
   NOT GUFO_PRODUCT_VERSION STREQUAL "1.2.3" OR
   NOT GUFO_REVISION STREQUAL "unknown")
  message(FATAL_ERROR "Engine change affected product or a normal override survived")
endif()
file(WRITE "${TEST_ROOT}/tools/release/product_version.txt" "2.3.4\n")
include("${TEST_ROOT}/cmake/Version.cmake")
if(NOT GUFO_RELEASE_VERSION STREQUAL "8.9.10" OR
   NOT GUFO_PRODUCT_VERSION STREQUAL "2.3.4")
  message(FATAL_ERROR "Product change affected engine or was not refreshed")
endif()
gufo_escape_version_regex("1.2.3+meta(test)[x]?|^$*\\" escaped)
if(NOT "1.2.3+meta(test)[x]?|^$*\\" MATCHES "^${escaped}$" OR
   "1x2x3+meta(test)[x]?|^$*\\" MATCHES "^${escaped}$")
  message(FATAL_ERROR "Version regex did not escape literal metacharacters")
endif()
message(STATUS "PASS: independent version sources, refresh, stale overrides, unknown Git, regex escaping")
