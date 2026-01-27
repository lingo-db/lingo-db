#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
using namespace lingodb::compiler::dialect;

void subop::registerSubOpTransformations() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createEnforceOrderPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createNormalizeSubOpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createPullGatherUpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createReuseLocalPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createFoldColumnsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createSpecializeSubOpPass(true);
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createSIPPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createParallelizePass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createSpecializeParallelPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createInlineNestedMapPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createFinalizePass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return subop::createSplitIntoExecutionStepsPass();
   });
}