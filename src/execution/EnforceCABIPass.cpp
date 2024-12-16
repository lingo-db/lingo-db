#include "lingodb/execution/BackendPasses.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
namespace {
struct EnforceCPPABIPass
   : public mlir::PassWrapper<EnforceCPPABIPass, mlir::OperationPass<mlir::ModuleOp>> {
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnforceCPPABIPass)
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::LLVM::LLVMDialect>();
   }
   llvm::StringRef getArgument() const final {
      return "--llvm-enforce-cpp-abi";
   }
   void runOnOperation() override {
      getOperation().walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
         if (funcOp.getFunctionBody().empty() && funcOp.isPrivate()) {
            auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
            auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();
            size_t numRegs = 0;
            std::vector<size_t> passByMem;
            std::vector<size_t> boolParams;
            for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
               if (funcOp.getArgumentTypes()[i].isInteger(1)) {
                  boolParams.push_back(i);
               }
               auto dataLayout = dataLayoutAnalysis.getAbove(funcOp.getOperation());
               auto typeSize = dataLayout.getTypeSize(funcOp.getArgumentTypes()[i]);
               if (typeSize <= 16) {
                  auto requiredRegs = typeSize <= 8 ? 1 : 2;
                  if (numRegs + requiredRegs > 6) {
                     passByMem.push_back(i);
                  } else {
                     numRegs += requiredRegs;
                  }
               } else {
                  passByMem.push_back(i);
               }
            }
            if (passByMem.empty() && boolParams.empty()) return;
            std::vector<mlir::Type> paramTypes(funcOp.getArgumentTypes().begin(), funcOp.getArgumentTypes().end());
            std::vector<mlir::Type> paramElTypes(funcOp.getArgumentTypes().begin(), funcOp.getArgumentTypes().end());
            for (size_t memId : passByMem) {
               auto originalType = paramTypes[memId];
               paramTypes[memId] = mlir::LLVM::LLVMPointerType::get(&getContext());
               funcOp.setArgAttr(memId, "llvm.byval", mlir::TypeAttr::get(originalType));
            }
            for (size_t paramId : boolParams) {
               funcOp.setArgAttr(paramId, "llvm.zeroext", mlir::UnitAttr::get(&getContext()));
            }
            funcOp.setType(mlir::LLVM::LLVMFunctionType::get(funcOp.getFunctionType().getReturnType(), paramTypes));
            auto uses = mlir::SymbolTable::getSymbolUses(funcOp, moduleOp.getOperation());
            for (auto use : *uses) {
               auto callOp = mlir::cast<mlir::LLVM::CallOp>(use.getUser());
               for (size_t memId : passByMem) {
                  auto userFunc = callOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
                  mlir::OpBuilder builder(userFunc->getContext());
                  builder.setInsertionPointToStart(&userFunc.getBody().front());
                  auto const1 = builder.create<mlir::LLVM::ConstantOp>(callOp.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(1));
                  mlir::Value allocatedElementPtr = builder.create<mlir::LLVM::AllocaOp>(callOp.getLoc(), paramTypes[memId], paramElTypes[memId], const1, 16);
                  mlir::OpBuilder builder2(userFunc->getContext());
                  builder2.setInsertionPoint(callOp);
                  builder2.create<mlir::LLVM::StoreOp>(callOp->getLoc(), callOp.getOperand(memId), allocatedElementPtr);
                  callOp.setOperand(memId, allocatedElementPtr);
               }
            }
         }
      });
   }
};
} // namespace
std::unique_ptr<mlir::Pass> lingodb::execution::createEnforceCABI() {
   return std::make_unique<EnforceCPPABIPass>();
}
