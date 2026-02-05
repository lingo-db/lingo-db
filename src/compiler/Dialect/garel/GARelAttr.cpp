#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

#include "lingodb/compiler/Dialect/garel/GARelAttr.h"
#include "lingodb/compiler/Dialect/garel/GARelDialect.h"

#include "lingodb/compiler/Dialect/garel/GARelEnumAttr.cpp.inc"
#include "lingodb/compiler/Dialect/garel/GARelTypes.h"
#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/garel/GARelAttr.cpp.inc"

namespace garel {

mlir::Type AggregatorAttr::getResultType(mlir::Type inputRel) {
  switch (getFunc()) {
  case AggregateFunc::SUM:
  case AggregateFunc::MIN:
  case AggregateFunc::MAX:
  case AggregateFunc::LOR:
  case AggregateFunc::ARGMIN:
    // NOTE: argmin(arg, val) also uses first input column as output type.
    return llvm::cast<RelationType>(inputRel).getColumns()[getInputs()[0]];
  }
}

mlir::LogicalResult
AggregatorAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                       AggregateFunc func, llvm::ArrayRef<ColumnIdx> inputs) {
  if (func == AggregateFunc::ARGMIN) {
    if (inputs.size() != 2) {
      return emitError() << stringifyAggregateFunc(func)
                         << " expects exactly two inputs (arg, val), got "
                         << inputs.size();
    }
  } else {
    if (inputs.size() != 1) {
      return emitError() << stringifyAggregateFunc(func)
                         << " expects exactly one input, got " << inputs.size();
    }
  }

  return mlir::success();
}

// Need to define this here to avoid depending on GARelAttr in
// GARelDialect and creating a cycle.
void GARelDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/garel/GARelAttr.cpp.inc"
      >();
}

} // namespace garel
