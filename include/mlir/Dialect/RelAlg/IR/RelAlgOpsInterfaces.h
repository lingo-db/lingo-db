#ifndef RelAlg_RelAlgInterfaces
#define RelAlg_RelAlgInterfaces

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <mlir/IR/SymbolTable.h>
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgTypes.h"

namespace mlir::relalg::detail {
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getUsedAttributes(mlir::Operation* op);
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getAvailableAttributes(mlir::Operation* op);
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getFreeAttributes(mlir::Operation* op);
llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*,8> getCreatedAttributes(mlir::Operation* op);
bool isDependentJoin(mlir::Operation* op);


}
class Operator;
#define GET_OP_CLASSES
#include "mlir/Dialect/RelAlg/IR/RelAlgOpsInterfaces.h.inc"

#endif // RelAlg_RelAlgInterfaces
