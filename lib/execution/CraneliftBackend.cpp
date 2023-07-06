#include "execution/CraneliftConversions.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/cranelift/CraneliftExecutionEngine.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"

#include "execution/Backend.h"
#include "execution/BackendPasses.h"
#include "execution/CraneliftBackend.h"

#include <chrono>
class CraneliftBackend : public execution::ExecutionBackend {
   void execute(mlir::ModuleOp& moduleOp, runtime::ExecutionContext* executionContext) override {
      auto startLowerToLLVM = std::chrono::high_resolution_clock::now();
      if (auto mainFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>("main")) {
         mlir::OpBuilder builder(moduleOp->getContext());
         builder.setInsertionPointToStart(moduleOp.getBody());
         builder.create<mlir::func::FuncOp>(moduleOp.getLoc(), "rt_set_execution_context", builder.getFunctionType(mlir::TypeRange({mlir::util::RefType::get(moduleOp->getContext(), mlir::IntegerType::get(moduleOp->getContext(), 8))}), mlir::TypeRange()), builder.getStringAttr("private"), mlir::ArrayAttr{}, mlir::ArrayAttr{});
      }

      mlir::PassManager pm2(moduleOp->getContext());
      pm2.enableVerifier(verify);
      pm2.addPass(mlir::createConvertSCFToCFPass());
      pm2.addPass(execution::createDecomposeTuplePass());
      pm2.addPass(mlir::createCanonicalizerPass());
      pm2.addPass(mlir::cranelift::createLowerToCraneliftPass());
      if (mlir::failed(pm2.run(moduleOp))) {
         return;
      }
      auto endLowerToLLVM = std::chrono::high_resolution_clock::now();
      timing["lowerToLLVM"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerToLLVM - startLowerToLLVM).count() / 1000.0;
      auto startJIT = std::chrono::high_resolution_clock::now();

      mlir::cranelift::CraneliftExecutionEngine engine(moduleOp);
      if (!engine.succeeded()) {
         error.emit() << "can not create cranelift execution engine";
         return;
      }
      auto setExecutionContextFn = reinterpret_cast<execution::setExecutionContextFnType>(engine.getFunction("rt_set_execution_context"));
      auto mainFunc = reinterpret_cast<execution::mainFnType>(engine.getFunction("main"));
      auto endJIT = std::chrono::high_resolution_clock::now();

      auto totalJITTime = std::chrono::duration_cast<std::chrono::microseconds>(endJIT - startJIT).count() / 1000.0;
      setExecutionContextFn(executionContext);
      std::vector<double> measuredTimes;
      for (size_t i = 0; i < numRepetitions; i++) {
         auto executionStart = std::chrono::high_resolution_clock::now();
         mainFunc();
         auto executionEnd = std::chrono::high_resolution_clock::now();
         executionContext->reset();
         measuredTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(executionEnd - executionStart).count() / 1000.0);
      }
      timing["llvmCodeGen"] = totalJITTime;

      timing["executionTime"] = (measuredTimes.size() > 1 ? *std::min_element(measuredTimes.begin() + 1, measuredTimes.end()) : measuredTimes[0]);
   }
   bool requiresSnapshotting() override {
      return false;
   }
};
std::unique_ptr<execution::ExecutionBackend> execution::createCraneliftBackend() {
   return std::make_unique<CraneliftBackend>();
}
