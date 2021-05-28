add_subdirectory(precompiled)
add_library(runtime OBJECT
        dynamic_calls/arrow_table_builder.cpp
        dynamic_calls/dynamic_gandiva.cpp
        dynamic_calls/dump_values.cpp
        dynamic_calls/arrow_schema.cpp
        dynamic_calls/arrow_table_access.cpp
        dynamic_calls/execution_context.cpp
        dynamic_calls/string.cpp
        dynamic_calls/vector_builder.cpp
        dynamic_calls/sort.cpp
        database.cpp ${CMAKE_SOURCE_DIR}/arrow/cpp/src/arrow/vendored/datetime/tz.cpp)
find_package(Boost REQUIRED)
target_link_libraries(runtime PUBLIC arrow_shared Boost::headers)
target_link_options(runtime PUBLIC -export-dynamic)

