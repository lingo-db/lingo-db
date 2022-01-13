#ifndef MLIR_CONVERSION_DBTOARROWSTD_ARROWTYPES_H
#define MLIR_CONVERSION_DBTOARROWSTD_ARROWTYPES_H
#include "mlir/Dialect/DB/IR/DBType.h"
#include "arrow/type_fwd.h"
namespace mlir::db::codegen {
   std::tuple<arrow::Type::type, uint32_t, uint32_t> convertTypeToArrow(mlir::db::DBType type);
}
#endif //MLIR_CONVERSION_DBTOARROWSTD_ARROWTYPES_H
