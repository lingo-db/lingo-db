#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

std::shared_ptr<runtime::Catalog> staticCatalog = {};
void mlir::relalg::setStaticCatalog(std::shared_ptr<runtime::Catalog> catalog) {
   std::cerr << "Warning: setting static catalog, should only be used in combination with mlir-db-opt" << std::endl;
   staticCatalog = catalog;
}
void mlir::relalg::createQueryOptPipeline(mlir::OpPassManager& pm, runtime::Catalog* catalog) {
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createSimplifyAggregationsPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createExtractNestedOperatorsPass());
   pm.addPass(mlir::createCSEPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createInferNotNullConditionsPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createDecomposeLambdasPass());
   pm.addPass(mlir::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createImplicitToExplicitJoinsPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createInferNotNullConditionsPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createDecomposeLambdasPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createPushdownPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createUnnestingPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createColumnFoldingPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createDecomposeLambdasPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createPushdownPass());
   if (catalog) {
      pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createAttachMetaDataPass(*catalog));
   }
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createReduceGroupByKeysPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createExpandTransitiveEqualities());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createOptimizeJoinOrderPass());

   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createCombinePredicatesPass());
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createOptimizeImplementationsPass());
   if (catalog) {
      pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createDetachMetaDataPass());
   }
   pm.addNestedPass<mlir::func::FuncOp>(mlir::relalg::createIntroduceTmpPass());
   pm.addPass(mlir::createCanonicalizerPass());
}
void mlir::relalg::registerQueryOptimizationPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createExtractNestedOperatorsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createDecomposeLambdasPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createImplicitToExplicitJoinsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createUnnestingPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createPushdownPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createOptimizeJoinOrderPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createCombinePredicatesPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createOptimizeImplementationsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createIntroduceTmpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createSimplifyAggregationsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createReduceGroupByKeysPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createExpandTransitiveEqualities();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createColumnFoldingPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createTrackTuplesPass();
   });
   mlir::PassPipelineRegistration<EmptyPipelineOptions>(
      "relalg-query-opt",
      "",
      [](mlir::OpPassManager& pm) { return createQueryOptPipeline(pm, staticCatalog.get()); });
}
