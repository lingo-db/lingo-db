#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;

namespace lingodb::compiler::dialect::py_interp {

void PyInterpDialect::initialize() {
   addOperations<
#define GET_OP_LIST
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOps.cpp.inc"
      >();
   registerTypes();
}

} // namespace lingodb::compiler::dialect::py_interp
#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.cpp.inc"
