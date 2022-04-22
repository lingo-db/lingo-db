add_executable(sql sql.cpp)
target_link_libraries(sql runner runtime mlir-support)
target_link_options(sql PUBLIC -Wl,--export-dynamic)
set_target_properties(sql PROPERTIES  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
