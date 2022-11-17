#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>

#include "mlir/Dialect/DB/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/SubOperator/SubOperatorDialect.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"

#include "mlir/Dialect/RelAlg/Passes.h"

#include "mlir/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DSA/IR/DSADialect.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/Dialect/util/UtilDialect.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir-support/eval.h"

#include "mlir/InitAllDialects.h"

#include "runtime/Database.h"

namespace {
struct ToLLVMLoweringPass
   : public mlir::PassWrapper<ToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "tollvm"; }

   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();
   }
   void runOnOperation() final;
};
} // end anonymous namespace

void ToLLVMLoweringPass::runOnOperation() {
   // The first thing to define is the conversion target. This will define the
   // final target for this lowering. For this lowering, we are only targeting
   // the LLVM dialect.
   mlir::LLVMConversionTarget target(getContext());
   target.addLegalOp<mlir::ModuleOp>();

   // During this lowering, we will also be lowering the MemRef types, that are
   // currently being operated on, to a representation in LLVM. To perform this
   // conversion we use a TypeConverter as part of the lowering. This converter
   // details how one type maps to another. This is necessary now that we will be
   // doing more complicated lowerings, involving loop region arguments.
   mlir::LowerToLLVMOptions options(&getContext());
   //options.emitCWrappers = true;
   mlir::LLVMTypeConverter typeConverter(&getContext(), options);
   typeConverter.addSourceMaterialization([&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
      return valueRange.front();
   });
   typeConverter.addTargetMaterialization([&](mlir::OpBuilder&, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
      return valueRange.front();
   });

   // Now that the conversion target has been defined, we need to provide the
   // patterns used for lowering. At this point of the compilation process, we
   // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
   // are already exists a set of patterns to transform `affine` and `std`
   // dialects. These patterns lowering in multiple stages, relying on transitive
   // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
   // patterns must be applied to fully transform an illegal operation into a
   // set of legal ones.
   mlir::RewritePatternSet patterns(&getContext());
   mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
   mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
   populateAffineToStdConversionPatterns(patterns);
   mlir::populateSCFToControlFlowConversionPatterns(patterns);
   mlir::util::populateUtilToLLVMConversionPatterns(typeConverter, patterns);
   mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
   mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
   mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
   mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
   // We want to completely lower to LLVM, so we use a `FullConversion`. This
   // ensures that only legal operations will remain after the conversion.
   auto module = getOperation();
   if (auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main")) {
      mainFunc->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(&getContext()));
   }
   if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
}

int main(int argc, char** argv) {
   if (argc > 2) {
      if (std::string(argv[1]) == "--use-db") {
         std::shared_ptr<runtime::Database> database = runtime::Database::loadFromDir(std::string(argv[2]));
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
      return std::make_unique<ToLLVMLoweringPass>();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return mlir::relalg::createDetachMetaDataPass();
   });

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
   registry.insert<mlir::scf::SCFDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   mlir::registerAllDialects(registry);

   support::eval::init();

   return failed(
      mlir::MlirOptMain(argc, argv, "DB dialects optimization driver\n", registry));
}
