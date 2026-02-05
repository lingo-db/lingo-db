#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

namespace graphalg {

/**
 * MLIR types for the semirings supported in graphalg.
 *
 * Naming follows the language specification.
 */
struct SemiringTypes {
   static mlir::Type forBool(mlir::MLIRContext* ctx);
   static mlir::Type forInt(mlir::MLIRContext* ctx);
   static mlir::Type forReal(mlir::MLIRContext* ctx);
   static mlir::Type forTropInt(mlir::MLIRContext* ctx);
   static mlir::Type forTropReal(mlir::MLIRContext* ctx);

   // Note: not part of the specificatin
   static mlir::Type forTropMaxInt(mlir::MLIRContext* ctx);
};

} // namespace graphalg
