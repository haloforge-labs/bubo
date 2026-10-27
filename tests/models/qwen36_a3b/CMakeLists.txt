# Metadata and allocation arithmetic need no HIP runtime or model weights.
add_executable(qwen36_a3b_config_state_bytes_test config_state_bytes_test.cpp)
target_link_libraries(qwen36_a3b_config_state_bytes_test PRIVATE gufo_core)
add_test(NAME qwen36_a3b.config_state_bytes COMMAND qwen36_a3b_config_state_bytes_test)
set_tests_properties(qwen36_a3b.config_state_bytes PROPERTIES
  LABELS "cpu;models;qwen36_a3b;memory;reference")
