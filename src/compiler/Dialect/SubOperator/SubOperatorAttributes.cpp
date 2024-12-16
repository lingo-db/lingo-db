#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include <llvm/ADT/TypeSwitch.h>

#define GET_ATTRDEF_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.cpp.inc"

void lingodb::compiler::dialect::subop::SubOperatorDialect::registerAttrs() {
   addAttributes<
#define GET_ATTRDEF_LIST
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.cpp.inc"

      >();
}