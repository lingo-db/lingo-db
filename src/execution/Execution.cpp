#include "lingodb/execution/Execution.h"

#include "lingodb/compiler/Conversion/DBToStd/DBToStd.h"
#include "lingodb/compiler/Conversion/DSAToStd/DSAToStd.h"
#include "lingodb/compiler/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "lingodb/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOps.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/execution/CBackend.h"
#include "lingodb/execution/LLVMBackends.h"
#include "lingodb/utility/Setting.h"
#include "lingodb/utility/Tracer.h"

#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"

#include "json.h"

#include <chrono>
#include <sstream>
#include <unordered_set>

namespace {
utility::GlobalSetting<std::string> executionModeSetting("system.execution_mode", "DEFAULT");
utility::GlobalSetting<std::string> subopOptPassesSetting("system.subop.opt", "GlobalOpt,ReuseLocal,Specialize,PullGatherUp,Compression");
utility::Tracer::Event queryOptimizationEvent("Compilation", "Query Opt.");
utility::Tracer::Event lowerRelalgEvent("Compilation", "Lower RelAlg");
utility::Tracer::Event lowerSubOpEvent("Compilation", "Lower SubOp");
utility::Tracer::Event lowerImperativeEvent("Compilation", "Lower DB");
utility::Tracer::Event loadIndicesEvent("Compilation", "Lower DB");
} // end anonymous namespace
namespace lingodb::execution {
using namespace lingodb::compiler::dialect;
class DefaultQueryOptimizer : public QueryOptimizer {
   void optimize(mlir::ModuleOp& moduleOp) override {
      auto start = std::chrono::high_resolution_clock::now();
      utility::Tracer::Trace trace(queryOptimizationEvent);
      mlir::PassManager pm(moduleOp.getContext());
      pm.enableVerifier(verify);
      addLingoDBInstrumentation(pm, getSerializationState());
      //pm.addPass(mlir::createInlinerPass());
      //pm.addPass(mlir::createSymbolDCEPass());
      relalg::createQueryOptPipeline(pm, catalog);
      if (mlir::failed(pm.run(moduleOp))) {
         error.emit() << " Query Optimization failed";
      }
      auto end = std::chrono::high_resolution_clock::now();
      timing["QOpt"] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
   }
};
class RelAlgLoweringStep : public LoweringStep {
   std::string getShortName() const override {
      return "subop";
   }
   void implement(mlir::ModuleOp& moduleOp) override {
      utility::Tracer::Trace trace(lowerRelalgEvent);

      auto startLowerRelAlg = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerRelAlgPm(moduleOp->getContext());
      lowerRelAlgPm.enableVerifier(verify);
      addLingoDBInstrumentation(lowerRelAlgPm, getSerializationState());
      relalg::createLowerRelAlgToSubOpPipeline(lowerRelAlgPm);
      if (mlir::failed(lowerRelAlgPm.run(moduleOp))) {
         error.emit() << "Lowering of RelAlg to Sub-Operators failed";
         return;
      }
      auto endLowerRelAlg = std::chrono::high_resolution_clock::now();
      timing["lowerRelAlg"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerRelAlg - startLowerRelAlg).count() / 1000.0;
      utility::Tracer::Trace indexLoadingTrace(loadIndicesEvent);
      // Load the required tables/indices for the query
      moduleOp.walk([&](mlir::Operation* op) {
         if (auto getExternalOp = mlir::dyn_cast_or_null<subop::GetExternalOp>(*op)) {
            auto* catalog = getCatalog();
            auto json = nlohmann::json::parse(getExternalOp.getDescr().str());
            auto tableName = json.value("table", "");
            bool addIndex = false;
            if (!tableName.size()) {
               addIndex = json.contains("index");
               if (!addIndex) return;
               tableName = json["relation"];
            }
            // Load table
            if (auto relation = catalog->findRelation(tableName)) {
               relation->loadData();
               if (addIndex) {
                  relation->getIndex(json["index"])->ensureLoaded();
               }
            }
         }
      });
   }
};
class SubOpLoweringStep : public LoweringStep {
   std::string getShortName() const override {
      return "hl-imperative";
   }
   void implement(mlir::ModuleOp& moduleOp) override {
      utility::Tracer::Trace trace(lowerSubOpEvent);
      auto startLowerSubOp = std::chrono::high_resolution_clock::now();
      mlir::PassManager optSubOpPm(moduleOp->getContext());
      optSubOpPm.enableVerifier(verify);
      addLingoDBInstrumentation(optSubOpPm, getSerializationState());
      std::unordered_set<std::string> enabledPasses = {};

      enabledPasses.clear();
      std::stringstream configList(subopOptPassesSetting.getValue());
      std::string optPass;
      while (std::getline(configList, optPass, ',')) {
         enabledPasses.insert(optPass);
      }
      if (enabledPasses.contains("GlobalOpt"))
         optSubOpPm.addPass(subop::createGlobalOptPass());
      optSubOpPm.addPass(subop::createFoldColumnsPass());
      if (enabledPasses.contains("ReuseLocal"))
         optSubOpPm.addPass(subop::createReuseLocalPass());
      optSubOpPm.addPass(subop::createSpecializeSubOpPass(enabledPasses.contains("Specialize")));
      optSubOpPm.addPass(subop::createNormalizeSubOpPass());
      if (enabledPasses.contains("PullGatherUp"))
         optSubOpPm.addPass(subop::createPullGatherUpPass());
      optSubOpPm.addPass(subop::createEnforceOrderPass());
      optSubOpPm.addPass(subop::createInlineNestedMapPass());
      optSubOpPm.addPass(subop::createFinalizePass());
      optSubOpPm.addPass(subop::createSplitIntoExecutionStepsPass());
      if (!moduleOp->hasAttr("subop.sequential")) {
         optSubOpPm.addNestedPass<mlir::func::FuncOp>(subop::createParallelizePass());
         optSubOpPm.addPass(subop::createSpecializeParallelPass());
      }
      optSubOpPm.addPass(subop::createPrepareLoweringPass());
      if (mlir::failed(optSubOpPm.run(moduleOp))) {
         error.emit() << "Lowering of Sub-Operators to imperative operations failed";
         return;
      }
      snapshotImportantStep("subop-opt", moduleOp, getSerializationState());

      mlir::PassManager lowerSubOpPm(moduleOp->getContext());
      lowerSubOpPm.enableVerifier(verify);
      addLingoDBInstrumentation(lowerSubOpPm, getSerializationState());

      subop::setCompressionEnabled(enabledPasses.contains("Compression"));
      lowerSubOpPm.addPass(subop::createLowerSubOpPass());
      lowerSubOpPm.addPass(mlir::createCanonicalizerPass());
      lowerSubOpPm.addPass(mlir::createCSEPass());
      if (mlir::failed(lowerSubOpPm.run(moduleOp))) {
         error.emit() << "Lowering of Sub-Operators to imperative operations failed";
         return;
      }
      auto endLowerSubOp = std::chrono::high_resolution_clock::now();
      timing["lowerSubOp"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowerSubOp - startLowerSubOp).count() / 1000.0;
   }
};
class DefaultImperativeLowering : public LoweringStep {
   std::string getShortName() const override {
      return "ll-imperative";
   }
   void implement(mlir::ModuleOp& moduleOp) override {
      utility::Tracer::Trace trace(lowerImperativeEvent);
      auto startLowerDB = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerDBPm(moduleOp->getContext());
      lowerDBPm.enableVerifier(verify);
      addLingoDBInstrumentation(lowerDBPm, getSerializationState());
      db::createLowerDBPipeline(lowerDBPm);
      if (mlir::failed(lowerDBPm.run(moduleOp))) {
         error.emit() << "Lowering of imperative db operations failed";
         return;
      }
      auto endLowerDB = std::chrono::high_resolution_clock::now();
      auto startLowerDSA = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerDSAPm(moduleOp->getContext());
      lowerDSAPm.enableVerifier(verify);
      addLingoDBInstrumentation(lowerDSAPm, getSerializationState());
      lowerDSAPm.addPass(dsa::createLowerToStdPass());
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
   std::string mode = executionModeSetting.getValue();
   if (std::string(mode) == "PERF") {
      runMode = ExecutionMode::PERF;
   } else if (std::string(mode) == "CHEAP") {
      runMode = ExecutionMode::CHEAP;
   } else if (std::string(mode) == "EXTREME_CHEAP") {
      runMode = ExecutionMode::EXTREME_CHEAP;
   } else if (std::string(mode) == "DEFAULT") {
      runMode = ExecutionMode::DEFAULT;
   } else if (std::string(mode) == "DEBUGGING") {
      runMode = ExecutionMode::DEBUGGING;
   } else if (std::string(mode) == "SPEED") {
      std::cout << "using speed mode" << std::endl;
      runMode = ExecutionMode::SPEED;
   } else if (std::string(mode) == "C") {
      runMode = ExecutionMode::C;
   } else if (std::string(mode) == "GPU") {
#if GPU_ENABLED == 1
      runMode = ExecutionMode::GPU;
#endif
   } else if (std::string(mode) == "NONE") {
      runMode = ExecutionMode::NONE;
   }

   return runMode;
}

class DefaultQueryExecuter : public QueryExecuter {
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

   public:
   using QueryExecuter::QueryExecuter;
   void execute() override {
      if (!executionContext) {
         std::cerr << "Execution Context is missing" << std::endl;
         exit(1);
      }
      auto* catalog = executionContext->getSession().getCatalog().get();

      if (!queryExecutionConfig->frontend) {
         std::cerr << "Frontend is missing" << std::endl;
         exit(1);
      }
      auto& frontend = *queryExecutionConfig->frontend;

      frontend.setCatalog(catalog);
      if (data) {
         frontend.loadFromString(data.value());
      } else if (file) {
         frontend.loadFromFile(file.value());
      } else {
         std::cerr << "Must provide file or string!" << std::endl;
         exit(1);
      }
      auto serializationState = std::make_shared<SnapshotState>();
      serializationState->serialize = true;

      handleError("FRONTEND", frontend.getError());
      mlir::ModuleOp& moduleOp = *queryExecutionConfig->frontend->getModule();
      snapshotImportantStep("canonical", moduleOp, serializationState);
      if (queryExecutionConfig->queryOptimizer) {
         auto& queryOptimizer = *queryExecutionConfig->queryOptimizer;
         queryOptimizer.setCatalog(catalog);
         queryOptimizer.setSerializationState(serializationState);
         queryOptimizer.optimize(moduleOp);
         handleError("OPTIMIZER", queryOptimizer.getError());
         handleTiming(queryOptimizer.getTiming());
         if (queryExecutionConfig->trackTupleCount) {
            mlir::PassManager pm(moduleOp.getContext());
            pm.addPass(relalg::createTrackTuplesPass());
            if (pm.run(moduleOp).failed()) {
               Error e;
               e.emit() << "createTrackTuplesPass failed";
               handleError("TUPLE_TRACKING", e);
            }
         }
         snapshotImportantStep("qopt", moduleOp, serializationState);
      }

      bool parallelismEnabled = scheduler::getNumWorkers() != 1 && queryExecutionConfig->parallel;
      if (!frontend.isParallelismAllowed() || !parallelismEnabled) {
         moduleOp->setAttr("subop.sequential", mlir::UnitAttr::get(moduleOp->getContext()));
         //numThreads = 1;
      }
      for (auto& loweringStepPtr : queryExecutionConfig->loweringSteps) {
         auto& loweringStep = *loweringStepPtr;
         loweringStep.setCatalog(catalog);
         loweringStep.setSerializationState(serializationState);
         loweringStep.implement(moduleOp);
         snapshotImportantStep(loweringStep.getShortName(), moduleOp, serializationState);
         handleError("LOWERING", loweringStep.getError());
         handleTiming(loweringStep.getTiming());
      }
      if (queryExecutionConfig->executionBackend) {
         auto& executionBackend = *queryExecutionConfig->executionBackend;
         executionBackend.setSerializationState(serializationState);
         executionBackend.execute(moduleOp, executionContext.get());
#ifdef TRACER
         utility::Tracer::dump();
#endif
         handleError("BACKEND", executionBackend.getError());
         handleTiming(executionBackend.getTiming());
         if (queryExecutionConfig->resultProcessor) {
            auto& resultProcessor = *queryExecutionConfig->resultProcessor;
            resultProcessor.process(executionContext.get());
         }
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
#if defined(ASAN_ACTIVE)
   if(runMode == ExecutionMode::DEBUGGING) {
      std::cerr << "ASAN is not supported in DEBUGGING mode. Switching to C mode" << std::endl;
      runMode = ExecutionMode::C;
   }
#endif
   if (runMode == ExecutionMode::DEBUGGING) {
      config->executionBackend = createLLVMDebugBackend();
   } else if (runMode == ExecutionMode::C) {
      config->executionBackend = createCBackend();
   } else if (runMode == ExecutionMode::PERF) {
      config->executionBackend = createLLVMProfilingBackend();
   } else if (runMode == ExecutionMode::GPU) {
#if GPU_ENABLED == 1
      config->executionBackend = createGPULLVMBackend();
#endif
   } else if (runMode != ExecutionMode::NONE) {
      config->executionBackend = createDefaultLLVMBackend();
   }
   config->resultProcessor = execution::createTablePrinter();
   if (runMode == ExecutionMode::SPEED || runMode == ExecutionMode::EXTREME_CHEAP) {
      config->queryOptimizer->disableVerification();
      config->executionBackend->disableVerification();
      for (auto& loweringStep : config->loweringSteps) {
         loweringStep->disableVerification();
      }
      config->executionBackend->disableVerification();
   }
   return config;
}
std::unique_ptr<QueryExecuter> QueryExecuter::createDefaultExecuter(std::unique_ptr<QueryExecutionConfig> queryExecutionConfig, runtime::Session& session) {
   return std::make_unique<DefaultQueryExecuter>(std::move(queryExecutionConfig), session.createExecutionContext());
}

} // namespace lingodb::execution
