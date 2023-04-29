#include "mlir/Dialect/SubOperator/Transforms/Passes.h"

void mlir::subop::registerSubOpTransformations() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createEnforceOrderPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createNormalizeSubOpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createPullGatherUpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createReuseLocalPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createFoldColumnsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createSpecializeSubOpPass(true);
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createGlobalOptPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createParallelizePass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::subop::createSpecializeParallelPass();
   });
}