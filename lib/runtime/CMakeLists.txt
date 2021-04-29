add_library(runtime OBJECT runtime.cpp)
target_link_libraries(runtime PUBLIC arrow_shared)
target_link_options(runtime PUBLIC -export-dynamic)

