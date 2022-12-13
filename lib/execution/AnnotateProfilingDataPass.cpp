#include <llvm/Support/Format.h>
#include <llvm/Support/FormatVariadic.h>

#include "execution/BackendPasses.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
namespace {
struct InsertPerfAsmPass
   : public mlir::PassWrapper<InsertPerfAsmPass, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertPerfAsmPass)
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect>();
   }
   llvm::StringRef getArgument() const final {
      return "--llvm-insert-profiling-helpers";
   }
   static mlir::Location dropNames(mlir::Location l) {
      if (auto namedLoc = l.dyn_cast<mlir::NameLoc>()) {
         return dropNames(namedLoc.getChildLoc());
      }
      return l;
   }
   void runOnOperation() override {
      getOperation()->walk([](mlir::LLVM::CallOp callOp) {
         size_t loc = 0xdeadbeef;
         if (auto fileLoc = dropNames(callOp.getLoc()).dyn_cast<mlir::FileLineColLoc>()) {
            loc = fileLoc.getLine();
         }
         mlir::OpBuilder b(callOp);
         const auto* asmTp = "mov r15,{0}";
         auto asmDialectAttr =
            mlir::LLVM::AsmDialectAttr::get(b.getContext(), mlir::LLVM::AsmDialect::AD_Intel);
         const auto* asmCstr =
            "";
         auto asmStr = llvm::formatv(asmTp, llvm::format_hex(loc, /*width=*/16)).str();
         b.create<mlir::LLVM::InlineAsmOp>(callOp->getLoc(), mlir::TypeRange(), mlir::ValueRange(), llvm::StringRef(asmStr), llvm::StringRef(asmCstr), true, false, asmDialectAttr, mlir::ArrayAttr());
      });
   }
};
} // namespace

std::unique_ptr<mlir::Pass> execution::createAnnotateProfilingDataPass() {
   return std::make_unique<InsertPerfAsmPass>();
}