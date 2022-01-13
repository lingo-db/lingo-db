add_library(mlir-support OBJECT parsing.cpp eval.cpp)
target_link_libraries(mlir-support PUBLIC arrow_shared)

