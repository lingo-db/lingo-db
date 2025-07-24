#ifndef LINGODB_COMPILER_DIALECT_PYINTERP_PYINTERPOPS_H
#define LINGODB_COMPILER_DIALECT_PYINTERP_PYINTERPOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "lingodb/compiler/Dialect/PyInterp/PyInterpTypes.h"

#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h.inc"
#endif //LINGODB_COMPILER_DIALECT_PYINTERP_PYINTERPOPS_H
