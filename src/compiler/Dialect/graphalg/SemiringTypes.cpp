#include <mlir/IR/BuiltinTypes.h>

#include <lingodb/compiler/Dialect/graphalg/GraphAlgTypes.h>
#include <lingodb/compiler/Dialect/graphalg/SemiringTypes.h>

namespace graphalg {

mlir::Type SemiringTypes::forBool(mlir::MLIRContext* ctx) {
   return mlir::IntegerType::get(ctx, 1);
}

mlir::Type SemiringTypes::forInt(mlir::MLIRContext* ctx) {
   return mlir::IntegerType::get(ctx, 64);
}

mlir::Type SemiringTypes::forReal(mlir::MLIRContext* ctx) {
   return mlir::Float64Type::get(ctx);
}

mlir::Type SemiringTypes::forTropInt(mlir::MLIRContext* ctx) {
   return TropI64Type::get(ctx);
}

mlir::Type SemiringTypes::forTropReal(mlir::MLIRContext* ctx) {
   return TropF64Type::get(ctx);
}

mlir::Type SemiringTypes::forTropMaxInt(mlir::MLIRContext* ctx) {
   return TropMaxI64Type::get(ctx);
}

} // namespace graphalg
