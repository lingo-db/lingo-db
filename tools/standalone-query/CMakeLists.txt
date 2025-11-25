# compile-to-cpp
add_executable(compile-to-cpp compile-to-cpp.cpp)
target_link_libraries(compile-to-cpp PUBLIC runner runtime utility mlir-support scheduler)
set_target_properties(compile-to-cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(ENABLE_PYTHON STREQUAL "CPYTHON")
    target_link_libraries(compile-to-cpp PRIVATE python3.14)
    target_link_directories(compile-to-cpp PRIVATE ${PY_HOME}/lib)
    target_compile_definitions(compile-to-cpp PRIVATE USE_CPYTHON_RUNTIME=1)
endif()
