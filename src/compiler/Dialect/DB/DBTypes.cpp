#include "lingodb/compiler/Dialect/DB/IR/DBTypes.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>

#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypeInterfaces.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypes.cpp.inc"
namespace lingodb::compiler::dialect::db {
void DBDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/DB/IR/DBOpsTypes.cpp.inc"
      >();
}

bool NullableType::needsManagement() const {
   if (auto managed = mlir::dyn_cast<ManagedType>(getType())) {
      return managed.needsManagement();
   }
   return false;
}

} // namespace lingodb::compiler::dialect::db