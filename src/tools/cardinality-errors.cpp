#include "features.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/Execution.h"
#include "lingodb/execution/ResultProcessing.h"
#include "lingodb/runtime/ExecutionContext.h"
#include "lingodb/runtime/Session.h"
#include "lingodb/scheduler/Scheduler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using namespace lingodb;
using namespace lingodb::compiler::dialect;

// Optimizer wrapper: runs the standard query-opt pipeline plus TrackTuples,
// then clones the post-instrumentation module so we can read estimated rows
// after lowering has already mutated the original.
class CaptureCloneOptimizer : public execution::QueryOptimizer {
   mlir::ModuleOp& clonedModule;

   public:
   explicit CaptureCloneOptimizer(mlir::ModuleOp& clonedModule) : clonedModule(clonedModule) {}
   void optimize(mlir::ModuleOp& moduleOp) override {
      mlir::PassManager pm(moduleOp.getContext());
      pm.enableVerifier(verify);
      relalg::createQueryOptPipeline(pm, catalog);
      pm.addPass(relalg::createTrackTuplesPass());
      if (mlir::failed(pm.run(moduleOp))) {
         error.emit() << "Query optimization failed";
         return;
      }
      clonedModule = mlir::cast<mlir::ModuleOp>(moduleOp->clone());
   }
};

class TupleCountResultProcessor : public execution::ResultProcessor {
   std::unordered_map<uint32_t, int64_t>& tupleCounts;

   public:
   explicit TupleCountResultProcessor(std::unordered_map<uint32_t, int64_t>& tupleCounts) : tupleCounts(tupleCounts) {}
   void process(runtime::ExecutionContext* ctx) override {
      tupleCounts = ctx->getTupleCounts();
   }
};

struct OpStat {
   std::string opName;
   double estimated;
   int64_t real;
   double qError;
};

// Standard Moerkotte q-error with max(.,1) smoothing to handle zero rows.
double computeQError(double est, int64_t real) {
   if (std::isnan(est)) return std::numeric_limits<double>::quiet_NaN();
   double e = std::max(est, 1.0);
   double r = std::max(static_cast<double>(real), 1.0);
   return std::max(e / r, r / e);
}

double readRows(mlir::Operation* op) {
   auto attr = op->getAttr("rows");
   if (!attr) return std::numeric_limits<double>::quiet_NaN();
   if (auto f = mlir::dyn_cast<mlir::FloatAttr>(attr)) return f.getValueAsDouble();
   if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr)) return static_cast<double>(i.getInt());
   return std::numeric_limits<double>::quiet_NaN();
}

void report(mlir::ModuleOp moduleOp, const std::unordered_map<uint32_t, int64_t>& tupleCounts) {
   std::vector<OpStat> stats;
   moduleOp.walk([&](mlir::Operation* op) {
      if (op->getName().getDialectNamespace() != "relalg") return;
      if (mlir::isa<relalg::TrackTuplesOP, relalg::QueryOp, relalg::QueryReturnOp>(op)) return;

      mlir::Value tupleStream;
      for (auto r : op->getResults()) {
         if (mlir::isa<tuples::TupleStreamType>(r.getType())) {
            tupleStream = r;
            break;
         }
      }
      if (!tupleStream) return;

      relalg::TrackTuplesOP trackOp;
      for (auto* user : tupleStream.getUsers()) {
         if (auto t = mlir::dyn_cast<relalg::TrackTuplesOP>(user)) {
            trackOp = t;
            break;
         }
      }
      if (!trackOp) return;

      auto it = tupleCounts.find(trackOp.getResultId());
      if (it == tupleCounts.end()) return;

      OpStat s;
      auto fullName = op->getName().getStringRef();
      s.opName = fullName.starts_with("relalg.") ? fullName.drop_front(7).str() : fullName.str();
      s.estimated = readRows(op);
      s.real = it->second;
      s.qError = computeQError(s.estimated, s.real);
      stats.push_back(std::move(s));
   });

   if (stats.empty()) {
      llvm::outs() << "No tracked relalg operators found.\n";
      return;
   }

   std::map<std::string, std::vector<OpStat>> groups;
   for (auto& s : stats) groups[s.opName].push_back(s);

   for (auto& [name, ops] : groups) {
      llvm::outs() << "=== " << name << " (" << ops.size() << ") ===\n";
      for (auto& s : ops) {
         if (std::isnan(s.estimated)) {
            llvm::outs() << llvm::format("  est=         n/a  real=%12lld  q-error=    n/a\n",
                                         static_cast<long long>(s.real));
         } else {
            llvm::outs() << llvm::format("  est=%12.1f  real=%12lld  q-error=%7.2f\n",
                                         s.estimated, static_cast<long long>(s.real), s.qError);
         }
      }
      llvm::outs() << "\n";
   }

   llvm::outs() << "=== summary ===\n";
   double overallSumLog = 0;
   double overallMax = 0;
   int overallN = 0;
   for (auto& [name, ops] : groups) {
      double sumLog = 0;
      double maxQ = 0;
      int n = 0;
      for (auto& s : ops) {
         if (std::isfinite(s.qError)) {
            sumLog += std::log(s.qError);
            maxQ = std::max(maxQ, s.qError);
            ++n;
         }
      }
      double geo = n > 0 ? std::exp(sumLog / n) : std::numeric_limits<double>::quiet_NaN();
      llvm::outs() << llvm::format("  %-15s  n=%3d  geomean=%7.2f  max=%7.2f\n",
                                   name.c_str(), n, geo, maxQ);
      overallSumLog += sumLog;
      overallMax = std::max(overallMax, maxQ);
      overallN += n;
   }
   double overallGeo = overallN > 0 ? std::exp(overallSumLog / overallN) : std::numeric_limits<double>::quiet_NaN();
   const char* overallLabel = "[overall]";
   llvm::outs() << llvm::format("  %-15s  n=%3d  geomean=%7.2f  max=%7.2f\n",
                                overallLabel, overallN, overallGeo, overallMax);
}
} // namespace

int main(int argc, char** argv) {
   if (argc == 2 && std::string(argv[1]) == "--features") {
      printFeatures();
      return 0;
   }
   if (argc < 3) {
      std::cerr << "USAGE: cardinality-errors <file.sql> <database>\n";
      return 1;
   }
   std::string sqlFile = argv[1];
   std::string dbDir = argv[2];

   auto session = runtime::Session::createSession(dbDir, false);
   compiler::support::eval::init();

   std::unordered_map<uint32_t, int64_t> tupleCounts;
   mlir::ModuleOp clonedModule = nullptr;

   auto config = execution::createQueryExecutionConfig(execution::ExecutionMode::CHEAP, true);
   config->timingProcessor = {};
   config->queryOptimizer = std::make_unique<CaptureCloneOptimizer>(clonedModule);
   config->resultProcessor = std::make_unique<TupleCountResultProcessor>(tupleCounts);
   // We add TrackTuples ourselves inside CaptureCloneOptimizer, so disable the
   // executer-side insertion to avoid duplicate tracking ops.
   config->trackTupleCount = false;

   auto scheduler = scheduler::startScheduler();
   auto executer = execution::QueryExecuter::createDefaultExecuter(std::move(config), *session);
   executer->fromFile(sqlFile);

   scheduler::awaitEntryTask(std::make_unique<execution::QueryExecutionTask>(
      std::move(executer),
      [&]() {
         if (clonedModule) {
            report(clonedModule, tupleCounts);
         } else {
            std::cerr << "no module captured (optimizer did not run?)\n";
         }
      }));
   return 0;
}
