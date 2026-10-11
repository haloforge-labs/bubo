foreach(name prepare_args_test prepare_catalogue_test prepare_gate_test
             prepare_fingerprint_test prepare_dependencies_test)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE bubo_prepare)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  target_compile_definitions(${name} PRIVATE
    BUBO_PREPARE_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "cpu;prepare;release")
endforeach()
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
