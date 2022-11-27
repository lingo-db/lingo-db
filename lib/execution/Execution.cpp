#include "execution/Execution.h"
#include "execution/LLVMBackends.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <chrono>
namespace execution {
static void snapshot(mlir::ModuleOp moduleOp, execution::Error& error, std::string fileName) {
   mlir::PassManager pm(moduleOp->getContext());
   mlir::OpPrintingFlags flags;
   flags.enableDebugInfo(false);
   pm.addPass(mlir::createLocationSnapshotPass(flags, fileName));
   if (pm.run(moduleOp).failed()) {
      error.emit() << "Snapshotting failed";
   }
}
class DefaultQueryOptimizer : public QueryOptimizer {
   void optimize(mlir::ModuleOp& moduleOp) override {
      auto start = std::chrono::high_resolution_clock::now();
      mlir::PassManager pm(moduleOp.getContext());
      pm.enableVerifier(verify);
      pm.addPass(mlir::createInlinerPass());
      pm.addPass(mlir::createSymbolDCEPass());
      mlir::relalg::createQueryOptPipeline(pm, database);
      if (mlir::failed(pm.run(moduleOp))) {
         error.emit() << " Query Optimization failed";
      }
      auto end = std::chrono::high_resolution_clock::now();
      timing["QOpt"] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
   }
};
class RelAlgLoweringStep : public LoweringStep {
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerRelAlg = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerRelAlgPm(moduleOp->getContext());
      lowerRelAlgPm.enableVerifier(verify);
      mlir::relalg::createLowerRelAlgToSubOpPipeline(lowerRelAlgPm);
      if (mlir::failed(lowerRelAlgPm.run(moduleOp))) {
         error.emit() << "Lowering of RelAlg to Sub-Operators failed";
         return;
      }
      auto endLowerRelAlg = std::chrono::high_resolution_clock::now();
      timing["lowerRelAlg"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerRelAlg - startLowerRelAlg).count() / 1000.0;
   }
};
class SubOpLoweringStep : public LoweringStep {
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerSubOp = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerSubOpPm(moduleOp->getContext());
      lowerSubOpPm.enableVerifier(verify);
      mlir::subop::createLowerSubOpPipeline(lowerSubOpPm);
      if (mlir::failed(lowerSubOpPm.run(moduleOp))) {
         error.emit() << "Lowering of Sub-Operators to imperative operations failed";
         return;
      }
      auto endLowerSubOp = std::chrono::high_resolution_clock::now();
      timing["lowerSubOp"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerSubOp - startLowerSubOp).count() / 1000.0;
   }
};
class DefaultImperativeLowering : public LoweringStep {
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerDB = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerDBPm(moduleOp->getContext());
      lowerDBPm.enableVerifier(verify);
      mlir::db::createLowerDBPipeline(lowerDBPm);
      if (mlir::failed(lowerDBPm.run(moduleOp))) {
         error.emit() << "Lowering of imperative db operations failed";
         return;
      }
      auto endLowerDB = std::chrono::high_resolution_clock::now();
      auto startLowerDSA = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerDSAPm(moduleOp->getContext());
      lowerDSAPm.enableVerifier(verify);
      lowerDSAPm.addPass(mlir::dsa::createLowerToStdPass());
      lowerDSAPm.addPass(mlir::createCanonicalizerPass());
      lowerDSAPm.addPass(mlir::createLoopInvariantCodeMotionPass());
      lowerDSAPm.addPass(mlir::createCSEPass());
      if (mlir::failed(lowerDSAPm.run(moduleOp))) {
         error.emit() << "Lowering of dsa failed";
         return;
      }
      auto endLowerDSA = std::chrono::high_resolution_clock::now();
      timing["lowerDB"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerDB - startLowerDB).count() / 1000.0;
      timing["lowerDSA"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerDSA - startLowerDSA).count() / 1000.0;
   }
};
ExecutionMode getExecutionMode() {
   ExecutionMode runMode;
   if (RUN_QUERIES_WITH_PERF) {
      runMode = ExecutionMode::PERF;
   } else {
      runMode = ExecutionMode::DEFAULT;
   }
   if (const char* mode = std::getenv("LINGODB_EXECUTION_MODE")) {
      if (std::string(mode) == "PERF") {
         runMode = ExecutionMode::PERF;
      } else if (std::string(mode) == "DEFAULT") {
         runMode = ExecutionMode::DEFAULT;
      } else if (std::string(mode) == "DEBUGGING") {
         runMode = ExecutionMode::DEBUGGING;
      } else if (std::string(mode) == "SPEED") {
         std::cout << "using speed mode" << std::endl;
         runMode = ExecutionMode::SPEED;
      }
   }
   return runMode;
}

class DefaultQueryExecuter : public QueryExecuter {
   size_t snapShotCounter = 0;
   void handleError(std::string phase, Error& e) {
      if (e) {
         std::cerr << phase << ": " << e.getMessage() << std::endl;
         exit(1);
      }
   }
   void handleTiming(const std::unordered_map<std::string, double>& timing) {
      if (queryExecutionConfig->timingProcessor) {
         queryExecutionConfig->timingProcessor->addTiming(timing);
      }
   }
   void performSnapShot(mlir::ModuleOp moduleOp, std::string fileName = "") {
      if (queryExecutionConfig->executionBackend->requiresSnapshotting()) {
         if (fileName.empty()) {
            fileName = "snapshot-" + std::to_string(snapShotCounter++) + ".mlir";
         }
         Error e;
         snapshot(moduleOp, e, fileName);
         handleError("SNAPSHOT", e);
      }
   }

   public:
   using QueryExecuter::QueryExecuter;
   void execute() override {
      if (!executionContext) {
         std::cerr << "Execution Context is missing" << std::endl;
         exit(1);
      }
      runtime::Database* database = executionContext->getDatabase();

      if (!queryExecutionConfig->frontend) {
         std::cerr << "Frontend is missing" << std::endl;
         exit(1);
      }
      if (!queryExecutionConfig->executionBackend) {
         std::cerr << "Execution Backend is missing" << std::endl;
         exit(1);
      }
      auto& frontend = *queryExecutionConfig->frontend;

      frontend.setDatabase(database);
      if (data) {
         frontend.loadFromString(data.value());
      } else if (file) {
         frontend.loadFromFile(file.value());
      } else {
         std::cerr << "Must provide file or string!" << std::endl;
         exit(1);
      }
      handleError("FRONTEND", frontend.getError());
      mlir::ModuleOp& moduleOp = *queryExecutionConfig->frontend->getModule();
      performSnapShot(moduleOp, "input.mlir");
      if (queryExecutionConfig->queryOptimizer) {
         auto& queryOptimizer = *queryExecutionConfig->queryOptimizer;
         queryOptimizer.setDatabase(database);
         queryOptimizer.optimize(moduleOp);
         handleError("OPTIMIZER", queryOptimizer.getError());
         handleTiming(queryOptimizer.getTiming());
         performSnapShot(moduleOp);
      }
      for (auto& loweringStepPtr : queryExecutionConfig->loweringSteps) {
         auto& loweringStep = *loweringStepPtr;
         loweringStep.setDatabase(database);
         loweringStep.implement(moduleOp);
         handleError("LOWERING", loweringStep.getError());
         handleTiming(loweringStep.getTiming());
         performSnapShot(moduleOp);
      }

      auto& executionBackend = *queryExecutionConfig->executionBackend;
      executionBackend.setSnapShotCounter(snapShotCounter);
      executionBackend.execute(moduleOp, executionContext);
      handleError("BACKEND", executionBackend.getError());
      handleTiming(executionBackend.getTiming());
      if (queryExecutionConfig->resultProcessor) {
         auto& resultProcessor = *queryExecutionConfig->resultProcessor;
         resultProcessor.process(executionContext);
      }
      if (queryExecutionConfig->timingProcessor) {
         queryExecutionConfig->timingProcessor->process();
      }
   }
};
std::unique_ptr<QueryExecutionConfig> createQueryExecutionConfig(execution::ExecutionMode runMode, bool sqlInput) {
   auto config = std::make_unique<QueryExecutionConfig>();
   if (sqlInput) {
      config->frontend = createSQLFrontend();
   } else {
      config->frontend = createMLIRFrontend();
   }
   config->queryOptimizer = std::make_unique<DefaultQueryOptimizer>();
   config->loweringSteps.emplace_back(std::make_unique<RelAlgLoweringStep>());
   config->loweringSteps.emplace_back(std::make_unique<SubOpLoweringStep>());
   config->loweringSteps.emplace_back(std::make_unique<DefaultImperativeLowering>());
   if (runMode == ExecutionMode::DEBUGGING) {
      config->executionBackend = createLLVMDebugBackend();
   } else if (runMode == ExecutionMode::PERF) {
      config->executionBackend = createLLVMProfilingBackend();
   } else {
      config->executionBackend = createDefaultLLVMBackend();
   }
   config->resultProcessor = execution::createTablePrinter();
   if (runMode == ExecutionMode::SPEED) {
      config->queryOptimizer->disableVerification();
      config->executionBackend->disableVerification();
      for (auto& loweringStep : config->loweringSteps) {
         loweringStep->disableVerification();
      }
      config->executionBackend->disableVerification();
   }
   return config;
}
std::unique_ptr<QueryExecuter> QueryExecuter::createDefaultExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig) {
   return std::make_unique<DefaultQueryExecuter>(std::move(queryExecutionConfig));
}

} // namespace execution