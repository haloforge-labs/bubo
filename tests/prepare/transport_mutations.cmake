# Deliberately broken COPIES of the implementation; the intact tests must fail.
# Anchors are mandatory so source drift cannot produce a vacuous mutant.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/core/prepare/transport.cpp"
  "${CMAKE_SOURCE_DIR}/src/core/prepare/download.cpp")
file(READ "${CMAKE_SOURCE_DIR}/src/core/prepare/transport.cpp" policy_source)
set(anchor [=[  return host == suffix ||
         (host.size() > suffix.size() && host.ends_with(suffix) &&
          host[host.size() - suffix.size() - 1] == '.');]=])
string(FIND "${policy_source}" "${anchor}" found)
if(found EQUAL -1)
  message(FATAL_ERROR "host mutation anchor missing")
endif()
string(REPLACE "${anchor}" "  return host.find(suffix) != host.npos;" mutated "${policy_source}")
set(copy "${CMAKE_CURRENT_BINARY_DIR}/policy_mutant.cpp")
file(WRITE "${copy}" "${mutated}")
add_executable(prepare_policy_mutant prepare_policy_test.cpp "${copy}")
target_link_libraries(prepare_policy_mutant PRIVATE bubo_prepare)
set_target_properties(prepare_policy_mutant PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(READ "${CMAKE_SOURCE_DIR}/src/core/prepare/download.cpp" download_source)
foreach(kind hash linked)
  if(kind STREQUAL "hash")
    set(anchor [=[hasher->FinishHex() != entry.sha256]=])
    set(replacement [=[false /* broken: skip streaming SHA equality */]=])
  else()
    set(anchor [=[if ((!linked.empty() && (!IsLowerHex(Unquote(linked), 64) ||
                                         Unquote(linked) != entry.sha256)) ||
                    wrong_size) {]=])
    set(replacement [=[if (false /* broken: ignore origin X-Linked pins */) {]=])
  endif()
  string(FIND "${download_source}" "${anchor}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "${kind} transport mutation anchor missing")
  endif()
  string(REPLACE "${anchor}" "${replacement}" mutated "${download_source}")
  set(copy "${CMAKE_CURRENT_BINARY_DIR}/download_mutant_${kind}.cpp")
  file(WRITE "${copy}" "${mutated}")
  set(name "prepare_transport_mutant_${kind}")
  add_executable(${name} prepare_transport_test.cpp "${copy}")
  target_link_libraries(${name} PRIVATE bubo_prepare_test)
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()
