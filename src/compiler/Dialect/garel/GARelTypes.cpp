#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "lingodb/compiler/Dialect/garel/GARelDialect.h"
#include "lingodb/compiler/Dialect/garel/GARelTypes.h"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/garel/GARelOpsTypes.cpp.inc"

namespace garel {

bool isColumnType(mlir::Type t) {
  // Allow i1, i64, f64, index
  return t.isSignlessInteger(1) || t.isSignlessInteger(64) || t.isF64() ||
         t.isIndex();
}

// Need to define this here to avoid depending on IPRTypes in
// IPRDialect and creating a cycle.
void GARelDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/garel/GARelOpsTypes.cpp.inc"
      >();
}

} // namespace garel
