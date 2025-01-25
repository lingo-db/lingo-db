
#mlir-to-json
add_executable(mlir-to-json mlir-relalg-to-json.cpp)
llvm_update_compile_flags(mlir-to-json)
target_link_libraries(mlir-to-json  PUBLIC runtime utility runner mlir-support scheduler ${LIBS} PRIVATE Arrow::arrow_static)
set_target_properties(mlir-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")



add_executable(mlir-subop-to-json mlir-subop-to-json.cpp)
llvm_update_compile_flags(mlir-subop-to-json)
target_link_libraries(mlir-subop-to-json PUBLIC runtime utility runner mlir-support scheduler ${LIBS} PRIVATE Arrow::arrow_static)
set_target_properties(mlir-subop-to-json PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")







#mlir-analyze-snapshots
add_executable(mlir-analyze-snapshots analyze-snapshots.cpp)
llvm_update_compile_flags(mlir-analyze-snapshots)
target_link_libraries(mlir-analyze-snapshots  PUBLIC runtime utility runner mlir-support scheduler ${LIBS} PRIVATE Arrow::arrow_static)
set_target_properties(mlir-analyze-snapshots PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

