#ifndef MLIR_CONVERSION_DBTOARROWSTD_ARROWTYPES_H
#define MLIR_CONVERSION_DBTOARROWSTD_ARROWTYPES_H
#include <arrow/type_fwd.h>

#include "mlir/Dialect/DB/IR/DBType.h"

namespace mlir::db::codegen {
std::tuple<arrow::Type::type, uint32_t, uint32_t> convertTypeToArrow(mlir::Type type);
} // end namespace mlir::db::codegen
#endif // MLIR_CONVERSION_DBTOARROWSTD_ARROWTYPES_H
