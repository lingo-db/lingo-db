#ifndef RelAlg_RelAlgInterfaces
#define RelAlg_RelAlgInterfaces

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <mlir/IR/SymbolTable.h>

static mlir::SymbolRefAttr getScopeNameImpl(mlir::Operation* op) {
   return op->getAttr(mlir::SymbolTable::getSymbolAttrName()).dyn_cast_or_null<mlir::SymbolRefAttr>();
}
namespace mlir::relalg::detail {
std::vector<mlir::relalg::RelationalAttributeDefAttr> getCreations(mlir::Operation* op);
std::vector<mlir::relalg::RelationalAttributeRefAttr> getUsages(mlir::Operation* op);

}
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h.inc"

#endif // RelAlg_RelAlgInterfaces
