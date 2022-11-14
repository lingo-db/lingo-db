#include "mlir/Dialect/SubOperator/Transforms/Passes.h"

void mlir::subop::registerSubOpTransformations() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createEnforceOrderPass();
   });
}