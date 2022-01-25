add_executable(sourcemap sourcemap.cpp)
llvm_update_compile_flags(sourcemap)
target_link_libraries(sourcemap PUBLIC runtime ${LIBS})