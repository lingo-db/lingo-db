add_subdirectory(precompiled)
add_library(runtime OBJECT dynamic_runtime_calls.cpp database.cpp ${CMAKE_SOURCE_DIR}/arrow/cpp/src/arrow/vendored/datetime/tz.cpp)
find_package(Boost REQUIRED)
target_link_libraries(runtime PUBLIC arrow_shared  Boost::headers)
target_link_options(runtime PUBLIC -export-dynamic)

