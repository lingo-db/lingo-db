add_library(mlir-support OBJECT mlir-support.cpp)
target_link_libraries(mlir-support PUBLIC arrow_shared)

