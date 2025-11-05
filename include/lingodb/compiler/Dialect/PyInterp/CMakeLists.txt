
set(LLVM_TARGET_DEFINITIONS PyInterpOps.td)
mlir_tablegen(PyInterpDialect.h.inc -gen-dialect-decls -dialect=py_interp)
mlir_tablegen(PyInterpDialect.cpp.inc -gen-dialect-defs -dialect=py_interp)
mlir_tablegen(PyInterpOps.h.inc -gen-op-decls)
mlir_tablegen(PyInterpOps.cpp.inc -gen-op-defs)
mlir_tablegen(PyInterpOpsTypes.h.inc --gen-typedef-decls --typedefs-dialect=py_interp)
mlir_tablegen(PyInterpOpsTypes.cpp.inc --gen-typedef-defs --typedefs-dialect=py_interp)

add_public_tablegen_target(MLIRPyInterpOpsIncGen)
add_dependencies(build_includes MLIRPyInterpOpsIncGen)
add_mlir_doc(PyInterpOps PyInterpDialect PyInterp/ -gen-dialect-doc  --dialect=py_interp)
