find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(pybind11)

pybind11_add_module(pymlirdbext mlirdb.cpp)
target_link_libraries(pymlirdbext PUBLIC runner runtime PRIVATE arrow_shared arrow_python_shared)
target_link_options(pymlirdbext PUBLIC -Wl,--export-dynamic)
set_target_properties(pymlirdbext PROPERTIES  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")