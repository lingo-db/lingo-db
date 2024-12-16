#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace lingodb::compiler::dialect::db {
void DBDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypes.cpp.inc"
      >();
}

} // namespace lingodb::compiler::dialect::db