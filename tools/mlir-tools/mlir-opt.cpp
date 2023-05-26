#include "execution/BackendPasses.h"
#include "mlir-support/eval.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "runtime/Database.h"
int main(int argc, char** argv) {
   if (argc > 2) {
      if (std::string(argv[1]) == "--use-db") {
         std::shared_ptr<runtime::Database> database = runtime::Database::loadMetaDataAndSamplesFromDir(std::string(argv[2]));
         mlir::relalg::setStaticDB(database);
         char** argvReduced = new char*[argc - 2];
         argvReduced[0] = argv[0];
         for (int i = 3; i < argc; i++) {
            argvReduced[i - 2] = argv[i];
         }
         argc -= 2;
         argv = argvReduced;
      }
   }
   mlir::registerAllPasses();

   mlir::relalg::registerRelAlgToSubOpConversionPasses();
   mlir::relalg::registerQueryOptimizationPasses();
   mlir::db::registerDBConversionPasses();
   mlir::subop::registerSubOpToControlFlowConversionPasses();
   mlir::subop::registerSubOpTransformations();
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::dsa::createLowerToStdPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createDetachMetaDataPass();
   });
   execution::registerBackendPasses();
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::tuples::TupleStreamDialect>();
   registry.insert<mlir::subop::SubOperatorDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();

   registry.insert<mlir::scf::SCFDialect>();

   support::eval::init();

   return failed(
      mlir::MlirOptMain(argc, argv, "DB dialects optimization driver\n", registry));
}
