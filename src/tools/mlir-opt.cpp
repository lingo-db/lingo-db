#include "lingodb/execution/BackendPasses.h"
#include "lingodb/compiler/Conversion/DBToStd/DBToStd.h"
#include "lingodb/compiler/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/compiler/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "lingodb/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DSA/IR/DSADialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "lingodb/runtime/Catalog.h"

#include <lingodb/compiler/Conversion/UtilToLLVM/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
int main(int argc, char** argv) {
   using namespace lingodb::compiler::dialect;
   if (argc > 2) {
      if (std::string(argv[1]) == "--use-db") {
         relalg::setStaticCatalog(lingodb::runtime::DBCatalog::create(lingodb::runtime::Catalog::createEmpty(), std::string(argv[2]), false));
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

   relalg::registerRelAlgToSubOpConversionPasses();
   relalg::registerQueryOptimizationPasses();
   db::registerDBConversionPasses();
   subop::registerSubOpToControlFlowConversionPasses();
   subop::registerSubOpTransformations();
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return dsa::createLowerToStdPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createDetachMetaDataPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return util::createUtilToLLVMPass();
   });
   lingodb::execution::registerBackendPasses();
   mlir::DialectRegistry registry;
   registry.insert<relalg::RelAlgDialect>();
   registry.insert<tuples::TupleStreamDialect>();
   registry.insert<subop::SubOperatorDialect>();
   registry.insert<db::DBDialect>();
   registry.insert<dsa::DSADialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();
   registry.insert<mlir::DLTIDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<util::UtilDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::async::AsyncDialect>();
   registry.insert<mlir::gpu::GPUDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   mlir::registerAllExtensions(registry);
   mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
   mlir::registerAllToLLVMIRTranslations(registry);
   lingodb::compiler::support::eval::init();

   return failed(
      mlir::MlirOptMain(argc, argv, "DB dialects optimization driver\n", registry));
}
