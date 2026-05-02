#include "lingodb/compiler/Dialect/PyInterp/PyInterpTypes.h"
#include "lingodb/compiler/Dialect/PyInterp/PyInterpDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOpsTypes.cpp.inc"

namespace lingodb::compiler::dialect::py_interp {

void PyInterpDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/PyInterp/PyInterpOpsTypes.cpp.inc"
      >();
}

} // namespace lingodb::compiler::dialect::py_interp
