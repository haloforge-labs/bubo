foreach(name prepare_args_test prepare_catalogue_test prepare_gate_test
             prepare_fingerprint_test prepare_dependencies_test prepare_zip_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_compile_definitions(${name} PRIVATE
    BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
list(TRANSFORM bubo_prepare_sources PREPEND "${CMAKE_SOURCE_DIR}/"
  OUTPUT_VARIABLE bubo_prepare_test_sources)
add_library(bubo_prepare_test STATIC ${bubo_prepare_test_sources})
target_link_libraries(bubo_prepare_test PUBLIC CURL::libcurl OpenSSL::Crypto ZLIB::ZLIB)
target_compile_definitions(bubo_prepare_test PUBLIC BUBO_PREPARE_TESTING=1)
if(WIN32)
  target_link_libraries(bubo_prepare_test PUBLIC Rstrtmgr)
endif()
add_executable(prepare_runtime_test prepare_runtime_test.cpp loopback_stub.cpp)
target_link_libraries(prepare_runtime_test PRIVATE bubo_prepare_test)
target_compile_definitions(prepare_runtime_test PRIVATE
  BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
  BUBO_NOTICE_SOURCE="${CMAKE_SOURCE_DIR}/tools/release/runtime-notices")
set_target_properties(prepare_runtime_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(WIN32)
  target_link_libraries(prepare_runtime_test PRIVATE ws2_32)
  add_dependencies(prepare_runtime_test prepare_fixture_engine)
  target_compile_definitions(prepare_runtime_test PRIVATE BUBO_INERT_DLL="$<TARGET_FILE:prepare_fixture_engine>")
endif()
add_test(NAME prepare_runtime_test COMMAND prepare_runtime_test
  "${CMAKE_BINARY_DIR}/j3-ctest-runtime" "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/runtime-consent-source.md")
set_tests_properties(prepare_runtime_test PROPERTIES LABELS "cpu;prepare;release")
include(runtime_mutations.cmake)
foreach(name prepare_transport_test prepare_retry_policy_test prepare_large_resume_test prepare_model_test prepare_loopback_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare_test)
  target_compile_definitions(${name} PRIVATE
    BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  if(name STREQUAL "prepare_model_test" OR name STREQUAL "prepare_loopback_test")
    if(name STREQUAL "prepare_model_test")
      set(kind model)
    else()
      set(kind loopback)
    endif()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND}
      "-DTEST=$<TARGET_FILE:${name}>" "-DBUILD=${CMAKE_BINARY_DIR}"
      "-DKIND=${kind}" "-DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model-consent.txt"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/fixture_test.cmake")
  else()
    add_test(NAME ${name} COMMAND ${name})
  endif()
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
target_sources(prepare_loopback_test PRIVATE loopback_stub.cpp)
target_sources(prepare_large_resume_test PRIVATE loopback_stub.cpp)
target_sources(prepare_model_test PRIVATE loopback_stub.cpp)
if(WIN32)
  target_link_libraries(prepare_loopback_test PRIVATE ws2_32)
  target_link_libraries(prepare_large_resume_test PRIVATE ws2_32)
  target_link_libraries(prepare_model_test PRIVATE ws2_32)
endif()
foreach(name prepare_policy_test prepare_gguf_fingerprint_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
include(model_mutations.cmake)
include(transport_mutations.cmake)
add_test(NAME prepare_help_snapshot
  COMMAND ${CMAKE_COMMAND} "-DBUBO=$<TARGET_FILE:gufo>"
    "-DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/prepare-help.txt"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/help_snapshot.cmake")
set_tests_properties(prepare_help_snapshot PROPERTIES LABELS "cpu;prepare;cli")
if(WIN32 AND GUFO_BRAND STREQUAL "bubo")
  add_library(prepare_fixture_engine SHARED bootstrap_fixture_engine.cpp)
  set_target_properties(prepare_fixture_engine PROPERTIES PREFIX ""
    OUTPUT_NAME "bubo-engine"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/prepare-fixture-engine")
endif()

add_executable(model_manifest_test
  ../serve/model_manifest_test.cpp
  ../../src/cli/serve/model_manifest.cpp
  ../../src/cli/serve/release_profile.cpp)
target_link_libraries(model_manifest_test PRIVATE bubo_prepare_test gufo_core gufo_logging)
set_target_properties(model_manifest_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME model_manifest_test COMMAND model_manifest_test)
set_tests_properties(model_manifest_test PROPERTIES LABELS "cpu;prepare;release;resolver")

add_executable(prepare_integration_test prepare_integration_test.cpp
  ../../src/cli/serve/model_manifest.cpp ../../src/cli/serve/release_profile.cpp)
target_link_libraries(prepare_integration_test PRIVATE bubo_prepare_test gufo_core gufo_logging)
set_target_properties(prepare_integration_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_test(NAME prepare_integration_test COMMAND prepare_integration_test)
set_tests_properties(prepare_integration_test PROPERTIES LABELS "cpu;prepare;release;integration")
add_custom_target(prepare_ci_test
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_ci_test.py"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
add_test(NAME prepare_ci_test
  COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_ci_test.py")
set_tests_properties(prepare_ci_test PROPERTIES WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" LABELS "cpu;prepare;release;ci")
