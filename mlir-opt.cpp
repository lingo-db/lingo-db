#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"

#include "mlir/Dialect/DB/IR/DBDialect.h"

int main(int argc, char** argv) {
   mlir::registerAllPasses();
   ::mlir::registerPass("relalg-extract-nested-operators", "extract nested operators", []() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createExtractNestedOperatorsPass();
   });
   ::mlir::registerPass("relalg-decompose-lambdas", "extract nested operators", []() -> std::unique_ptr<::mlir::Pass> {
     return mlir::relalg::createDecomposeLambdasPass();
   });
   ::mlir::registerPass("relalg-implicit-to-explicit-joins", "implicit to explicit joins", []() -> std::unique_ptr<::mlir::Pass> {
     return mlir::relalg::createImplicitToExplicitJoinsPass();
   });
   ::mlir::registerPass("relalg-unnesting", "unnest depending joins", []() -> std::unique_ptr<::mlir::Pass> {
     return mlir::relalg::createUnnestingPass();
   });
    ::mlir::registerPass("relalg-pushdown", "pushdown ", []() -> std::unique_ptr<::mlir::Pass> {
        return mlir::relalg::createPushdownPass();
    });
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   return failed(
      mlir::MlirOptMain(argc, argv, "DB dialects optimization driver\n", registry));
}
