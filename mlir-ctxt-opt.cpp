#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"

#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"

#include "mlir/Conversion/DBToArrowStd/DBToArrowStd.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "runtime/execution_context.h"

int main(int argc, char** argv) {
   if (argc <2) {
      std::cerr << "expected more args" << std::endl;
      return 1;
   }
   char** argvReduced =new char*[argc-1];
   argvReduced[0]=argv[0];
   for(int i=2;i<argc;i++){
      argvReduced[i-1]=argv[i];
   }

   runtime::ExecutionContext context;
   context.id = 42;

   std::cerr << "Loading Database from: " << argv[1] << '\n';
   auto database = runtime::Database::load(std::string(argv[1]));
   context.db = std::move(database);

   ::mlir::registerPass([&context]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createAttachMetaDataPass(*context.db);
   });
   mlir::DialectRegistry registry;
   registry.insert<mlir::relalg::RelAlgDialect>();
   registry.insert<mlir::db::DBDialect>();
   registry.insert<mlir::StandardOpsDialect>();
   registry.insert<mlir::arith::ArithmeticDialect>();

   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<mlir::util::UtilDialect>();
   registry.insert<mlir::scf::SCFDialect>();

   return failed(
      mlir::MlirOptMain(argc-1, argvReduced, "DB dialects optimization driver\n", registry));
}
