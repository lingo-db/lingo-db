add_library(runner runner.cpp)
llvm_update_compile_flags(runner)
target_link_libraries(runner PRIVATE ${LIBS} PUBLIC arrow_shared)
