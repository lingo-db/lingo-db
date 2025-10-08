
add_executable(mlir-pass-profiler mlir-pass-profiler.cpp)
llvm_update_compile_flags(mlir-pass-profiler)
target_link_libraries(mlir-pass-profiler PUBLIC runtime utility runner mlir-support ${LIBS})
set_target_properties(mlir-pass-profiler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
