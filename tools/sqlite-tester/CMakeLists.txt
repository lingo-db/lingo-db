add_executable(sqlite-tester sqlite-tester.cpp)
target_link_libraries(sqlite-tester runner runtime mlir-support)
target_link_options(sqlite-tester PUBLIC -Wl,--export-dynamic)
set_target_properties(sqlite-tester PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
