add_executable(mlir-to-sql mlir-to-sql.cpp)
llvm_update_compile_flags(mlir-to-sql)
target_link_libraries(mlir-to-sql PUBLIC runtime ${LIBS})