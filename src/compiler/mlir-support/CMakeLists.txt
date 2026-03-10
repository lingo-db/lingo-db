add_library(mlir-support OBJECT parsing.cpp eval.cpp tostring.cpp)
#todo: should not be necessary, but it is for Arrow 23
if("abslAlt" IN_LIST ARROW_SYSTEM_DEPENDENCIES)
    target_link_libraries(mlir-support PRIVATE Arrow::arrow_static ArrowCompute::arrow_compute_static absl::strings absl::time absl::bad_optional_access)
else()
    target_link_libraries(mlir-support PRIVATE Arrow::arrow_static ArrowCompute::arrow_compute_static)
endif()
