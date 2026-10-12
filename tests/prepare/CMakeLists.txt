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
