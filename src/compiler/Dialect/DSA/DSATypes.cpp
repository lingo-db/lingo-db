#include "lingodb/compiler/Dialect/DSA/IR/DSATypes.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsEnums.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>
using namespace lingodb::compiler::dialect;
mlir::Type dsa::RecordBatchType::getElementType() {
   return dsa::RecordType::get(getContext(), getRowType());
}

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
namespace lingodb::compiler::dialect::dsa {
void DSADialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/DSA/IR/DSAOpsTypes.cpp.inc"
      >();
}

} // namespace lingodb::compiler::dialect::dsa
