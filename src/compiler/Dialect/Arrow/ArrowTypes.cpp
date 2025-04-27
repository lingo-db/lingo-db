#include "lingodb/compiler/Dialect/Arrow/IR/ArrowTypes.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include <llvm/ADT/TypeSwitch.h>
using namespace lingodb::compiler::dialect;

#define GET_TYPEDEF_CLASSES
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOpsTypes.cpp.inc"
namespace lingodb::compiler::dialect::arrow {
void ArrowDialect::registerTypes() {
   addTypes<
#define GET_TYPEDEF_LIST
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowOpsTypes.cpp.inc"
      >();
}

} // namespace lingodb::compiler::dialect::arrow
