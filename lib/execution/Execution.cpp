#include "execution/Execution.h"
#include "execution/CBackend.h"
#include "execution/CraneliftBackend.h"
#include "execution/LLVMBackends.h"
#include "json.h"
#include "mlir/Conversion/DBToStd/DBToStd.h"
#include "mlir/Conversion/DSAToStd/DSAToStd.h"
#include "mlir/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "mlir/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SubOperator/SubOperatorOps.h"
#include "mlir/Dialect/SubOperator/Transforms/Passes.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "utility/Tracer.h"

#include <chrono>
#include <sstream>
#include <unordered_set>

#include <oneapi/tbb.h>
namespace {
static void snapshot(mlir::ModuleOp moduleOp, execution::Error& error, std::string fileName) {
   mlir::PassManager pm(moduleOp->getContext());
   mlir::OpPrintingFlags flags;
   flags.shouldPrintDebugInfo();
   flags.enableDebugInfo(true, false);
   pm.addPass(mlir::createLocationSnapshotPass(flags, fileName));
   if (pm.run(moduleOp).failed()) {
      error.emit() << "Snapshotting failed";
   }
}
} // namespace
namespace execution {
class DefaultQueryOptimizer : public QueryOptimizer {
   void optimize(mlir::ModuleOp& moduleOp) override {
      auto start = std::chrono::high_resolution_clock::now();
      mlir::PassManager pm(moduleOp.getContext());
      pm.enableVerifier(verify);
      //pm.addPass(mlir::createInlinerPass());
      //pm.addPass(mlir::createSymbolDCEPass());
      mlir::relalg::createQueryOptPipeline(pm, catalog);
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

      // Load the required tables/indices for the query
      moduleOp.walk([&](mlir::Operation* op) {
         if (auto getExternalOp = mlir::dyn_cast_or_null<mlir::subop::GetExternalOp>(*op)) {
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
   void implement(mlir::ModuleOp& moduleOp) override {
      auto startLowerSubOp = std::chrono::high_resolution_clock::now();
      mlir::PassManager lowerSubOpPm(moduleOp->getContext());
      lowerSubOpPm.enableVerifier(verify);
      std::unordered_set<std::string> enabledPasses = {"GlobalOpt", "ReuseLocal", "Specialize", "PullGatherUp", "Compression"};
      if (const char* mode = std::getenv("LINGODB_SUBOP_OPTS")) {
         enabledPasses.clear();
         std::stringstream configList(mode);
         std::string optPass;
         while (std::getline(configList, optPass, ',')) {
            enabledPasses.insert(optPass);
         }
      }
      if (enabledPasses.contains("GlobalOpt"))
         lowerSubOpPm.addPass(mlir::subop::createGlobalOptPass());
      lowerSubOpPm.addPass(mlir::subop::createFoldColumnsPass());
      if (enabledPasses.contains("ReuseLocal"))
         lowerSubOpPm.addPass(mlir::subop::createReuseLocalPass());
      lowerSubOpPm.addPass(mlir::subop::createSpecializeSubOpPass(enabledPasses.contains("Specialize")));
      lowerSubOpPm.addPass(mlir::subop::createNormalizeSubOpPass());
      if (enabledPasses.contains("PullGatherUp"))
         lowerSubOpPm.addPass(mlir::subop::createPullGatherUpPass());
      if (!moduleOp->hasAttr("subop.sequential")) {
         lowerSubOpPm.addPass(mlir::subop::createParallelizePass());
         lowerSubOpPm.addPass(mlir::subop::createSpecializeParallelPass());
      }
      lowerSubOpPm.addPass(mlir::subop::createEnforceOrderPass());
      mlir::subop::setCompressionEnabled(enabledPasses.contains("Compression"));
      lowerSubOpPm.addPass(mlir::subop::createLowerSubOpPass());
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
      auto* catalog = executionContext->getSession().getCatalog().get();

      if (!queryExecutionConfig->frontend) {
         std::cerr << "Frontend is missing" << std::endl;
         exit(1);
      }
      if (!queryExecutionConfig->executionBackend) {
         std::cerr << "Execution Backend is missing" << std::endl;
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
      handleError("FRONTEND", frontend.getError());
      mlir::ModuleOp& moduleOp = *queryExecutionConfig->frontend->getModule();
      performSnapShot(moduleOp, "input.mlir");
      if (queryExecutionConfig->queryOptimizer) {
         auto& queryOptimizer = *queryExecutionConfig->queryOptimizer;
         queryOptimizer.setCatalog(catalog);
         queryOptimizer.optimize(moduleOp);
         handleError("OPTIMIZER", queryOptimizer.getError());
         handleTiming(queryOptimizer.getTiming());
         if (queryExecutionConfig->trackTupleCount) {
            mlir::PassManager pm(moduleOp.getContext());
            pm.addPass(mlir::relalg::createTrackTuplesPass());
            if (pm.run(moduleOp).failed()) {
               Error e;
               e.emit() << "createTrackTuplesPass failed";
               handleError("TUPLE_TRACKING", e);
            }
         }
         performSnapShot(moduleOp);
      }
      bool parallelismEnabled = queryExecutionConfig->parallel;
      size_t numThreads = tbb::info::default_concurrency() / 2;
      if (const char* mode = std::getenv("LINGODB_PARALLELISM")) {
         if (std::string(mode) == "OFF") {
            parallelismEnabled = false;
         } else {
            numThreads = std::stol(mode);
         }
      }
      if (!frontend.isParallelismAllowed() || !parallelismEnabled) {
         moduleOp->setAttr("subop.sequential", mlir::UnitAttr::get(moduleOp->getContext()));
         numThreads = 1;
      }
      for (auto& loweringStepPtr : queryExecutionConfig->loweringSteps) {
         auto& loweringStep = *loweringStepPtr;
         loweringStep.setCatalog(catalog);
         loweringStep.implement(moduleOp);
         handleError("LOWERING", loweringStep.getError());
         handleTiming(loweringStep.getTiming());
         performSnapShot(moduleOp);
      }

      auto& executionBackend = *queryExecutionConfig->executionBackend;
      executionBackend.setSnapShotCounter(snapShotCounter);

      tbb::global_control c(tbb::global_control::max_allowed_parallelism, numThreads);
      int sum = oneapi::tbb::parallel_reduce(
         oneapi::tbb::blocked_range<int>(1, 100000), 0,
         [](oneapi::tbb::blocked_range<int> const& r, int init) -> int {
            for (int v = r.begin(); v != r.end(); v++) {
               init += v;
            }
            return init;
         },
         [](int lhs, int rhs) -> int {
            return lhs + rhs;
         });
      if (sum < 0) { exit(0); }
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
   }
   if (runMode == ExecutionMode::C) {
      config->executionBackend = createCBackend();
   } else if (runMode == ExecutionMode::PERF) {
      config->executionBackend = createLLVMProfilingBackend();
   } else if (runMode == ExecutionMode::CHEAP || runMode == ExecutionMode::EXTREME_CHEAP) {
#if CRANELIFT_ENABLED == 1
      config->executionBackend = createCraneliftBackend();
#else
      config->executionBackend = createDefaultLLVMBackend();
#endif
   } else {
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
   return std::make_unique<DefaultQueryExecuter>(std::move(queryExecutionConfig), std::move(session.createExecutionContext()));
}

} // namespace execution
