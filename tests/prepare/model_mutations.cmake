# Each mutant is a COPY of the implementation linked against the same intact
# baseline test. It must fail; merely changing an expected result is forbidden.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/cli/prepare/model_prepare.cpp")
file(READ "${CMAKE_SOURCE_DIR}/src/cli/prepare/model_prepare.cpp" model_source)
set(model_mutation_names fat32 manifest_first no_op consent)
foreach(kind IN LISTS model_mutation_names)
  if(kind STREQUAL "fat32")
    set(anchor [=[volume.filesystem == "FAT32"]=])
    set(replacement [=[false /* deliberately broken FAT32 check */]=])
  elseif(kind STREQUAL "manifest_first")
    set(anchor [=[// mutation-anchor: before-file-loop]=])
    set(replacement [=[// Deliberately broken: readiness before payload completion.
    if (!AtomicWrite(root / "bubo-package.json", Manifest(env.entries), error))
      return Error(err, 7, error);]=])
  elseif(kind STREQUAL "no_op")
    set(anchor [=[// mutation-anchor: initial-readonly-check]=])
    set(replacement [=[// Deliberately broken: write during a read-only no-op.
    if (initial.ready) {
      std::string mutation_error;
      AtomicWrite(root / ".bubo" / "prepare-state.json", "{}\n", mutation_error);
    }]=])
  else()
    set(anchor [=[Wrap(text, "The derived files are hosted by the Bubo project.");]=])
    set(replacement [=[/* Deliberately broken: omit the derived-host disclosure. */]=])
  endif()
  string(FIND "${model_source}" "${anchor}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "model ${kind} mutation anchor missing; refusing a vacuous proof")
  endif()
  string(REPLACE "${anchor}" "${replacement}" mutated "${model_source}")
  set(copy "${CMAKE_CURRENT_BINARY_DIR}/model_mutation_${kind}.cpp")
  file(WRITE "${copy}" "${mutated}")
  set(name "prepare_model_mutant_${kind}")
  add_executable(${name} prepare_model_test.cpp loopback_stub.cpp "${copy}")
  target_link_libraries(${name} PRIVATE bubo_prepare_test)
  if(WIN32)
    target_link_libraries(${name} PRIVATE Ws2_32)
  endif()
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  # The root job provides an unused job-owned fixture path when executing
  # these targets. No test output/root is implicitly placed outside that job.
endforeach()
