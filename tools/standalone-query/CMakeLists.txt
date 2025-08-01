# compile-to-cpp
add_executable(compile-to-cpp compile-to-cpp.cpp)
target_link_libraries(compile-to-cpp PUBLIC runner runtime utility mlir-support scheduler)
set_target_properties(compile-to-cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")