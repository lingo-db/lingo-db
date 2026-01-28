#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "lingodb/compiler/helper.h"

#include <iostream>
namespace {
std::shared_ptr<lingodb::catalog::Catalog> staticCatalog = {};
} // end anonymous namespace
using namespace lingodb::compiler::dialect;
void relalg::setStaticCatalog(std::shared_ptr<lingodb::catalog::Catalog> catalog) {
   std::cerr << "Warning: setting static catalog, should only be used in combination with mlir-db-opt" << std::endl;
   staticCatalog = catalog;
}
void relalg::createQueryOptPipeline(mlir::OpPassManager& pm, lingodb::catalog::Catalog* catalog) {
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createSimplifyAggregationsPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createExtractNestedOperatorsPass());
   pm.addPass(mlir::createCSEPass());
   pm.addPass(lingodb::compiler::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createInferNotNullConditionsPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createDecomposeLambdasPass(true));
   pm.addPass(lingodb::compiler::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createImplicitToExplicitJoinsPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createInferNotNullConditionsPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createDecomposeLambdasPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createPushdownPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createUnnestingPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createColumnFoldingPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createDecomposeLambdasPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createPushdownPass());
   if (catalog) {
      pm.addNestedPass<mlir::func::FuncOp>(relalg::createAttachMetaDataPass(*catalog));
   }
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createReduceGroupByKeysPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createExpandTransitiveEqualities());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createOptimizeJoinOrderPass());

   pm.addNestedPass<mlir::func::FuncOp>(relalg::createCombinePredicatesPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createEliminateNullableTypesPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createOptimizeImplementationsPass());
   if (catalog) {
      pm.addNestedPass<mlir::func::FuncOp>(relalg::createDetachMetaDataPass());
   }
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createCommonSubtreeEliminationPass());
   pm.addPass(lingodb::compiler::createCanonicalizerPass());
   pm.addNestedPass<mlir::func::FuncOp>(relalg::createIntroduceTmpPass());
}
void relalg::registerQueryOptimizationPasses() {
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createExtractNestedOperatorsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createDecomposeLambdasPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createImplicitToExplicitJoinsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createUnnestingPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createPushdownPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createOptimizeJoinOrderPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createCombinePredicatesPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createOptimizeImplementationsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createIntroduceTmpPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createCommonSubtreeEliminationPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createSimplifyAggregationsPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createReduceGroupByKeysPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createExpandTransitiveEqualities();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createColumnFoldingPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createTrackTuplesPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createEliminateNullableTypesPass();
   });

   mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
      "relalg-query-opt",
      "",
      [](mlir::OpPassManager& pm) { return createQueryOptPipeline(pm, staticCatalog.get()); });
}
