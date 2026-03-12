set(CT_TOOL_LINK_LIBRARIES  utility runner mlir-support scheduler ${LIBS} runtime)
if(ENABLE_PYTHON STREQUAL "CPYTHON")
list(APPEND CT_TOOL_LINK_LIBRARIES python3.14)
endif()
#mlir-to-json
add_executable(mlir-to-json mlir-relalg-to-json.cpp)
llvm_update_compile_flags(mlir-to-json)
target_link_libraries(mlir-to-json  PUBLIC ${CT_TOOL_LINK_LIBRARIES})
set_target_properties(mlir-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(mlir-to-json PRIVATE ${PY_HOME}/lib)
    target_compile_definitions(mlir-to-json PRIVATE USE_CPYTHON_RUNTIME=1)
endif()


add_executable(mlir-subop-to-json mlir-subop-to-json.cpp)
llvm_update_compile_flags(mlir-subop-to-json)
target_link_libraries(mlir-subop-to-json  PUBLIC ${CT_TOOL_LINK_LIBRARIES})
set_target_properties(mlir-subop-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(mlir-subop-to-json PRIVATE ${PY_HOME}/lib)
    target_compile_definitions(mlir-subop-to-json PRIVATE USE_CPYTHON_RUNTIME=1)
endif()






#mlir-analyze-snapshots
add_executable(mlir-analyze-snapshots analyze-snapshots.cpp)
llvm_update_compile_flags(mlir-analyze-snapshots)
target_link_libraries(mlir-analyze-snapshots  PUBLIC ${CT_TOOL_LINK_LIBRARIES})
set_target_properties(mlir-analyze-snapshots PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_directories(mlir-analyze-snapshots PRIVATE ${PY_HOME}/lib)
    target_compile_definitions(mlir-analyze-snapshots PRIVATE USE_CPYTHON_RUNTIME=1)
endif()
