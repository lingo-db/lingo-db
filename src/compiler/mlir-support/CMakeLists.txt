add_library(mlir-support OBJECT parsing.cpp eval.cpp tostring.cpp)
target_link_libraries(mlir-support PRIVATE Arrow::arrow_static)
