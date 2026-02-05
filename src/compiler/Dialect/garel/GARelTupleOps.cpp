#include <llvm/ADT/STLExtras.h>

#include "lingodb/compiler/Dialect/garel/GARelOps.h"
#include "lingodb/compiler/Dialect/garel/GARelTypes.h"

namespace garel {

mlir::LogicalResult ExtractOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<mlir::Location> location,
    Adaptor adaptor, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  // TODO: Do we need this cast?
  auto tupleType = llvm::cast<TupleType>(adaptor.getTuple().getType());
  auto columnTypes = tupleType.getColumns();
  inferredReturnTypes.push_back(columnTypes[adaptor.getColumn()]);
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  auto columns = getTuple().getType().getColumns();
  if (getColumn() >= getTuple().getType().getColumns().size()) {
    return emitOpError("column ")
           << getColumn() << " not included in tuple " << getTuple().getType();
  }

  return mlir::success();
}

} // namespace garel
