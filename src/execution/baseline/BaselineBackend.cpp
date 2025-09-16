#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif

#include "CompilerX64.hpp"
#include "Loader.hpp"

#include "lingodb/execution/BackendPasses.h"
#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/utility/Setting.h"
#include "lingodb/utility/Tracer.h"

#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Transforms/Passes.h>

namespace lingodb::execution::baseline {
using namespace compiler;

// init IRAdaptor static vars
IRAdaptor::IRFuncRef IRAdaptor::INVALID_FUNC_REF = nullptr;
IRAdaptor::IRValueRef IRAdaptor::INVALID_VALUE_REF = mlir::Value();

namespace {
utility::GlobalSetting<std::string> baselineDebugFileOut("system.compilation.baseline_object_out", "");
} // namespace

class BaselineBackend : public ExecutionBackend {
   // lower mlir IR to a form that can be compiled by tpde
   // currently mostly does a SCF to CF conversion
   bool lower(mlir::ModuleOp& moduleOp,
              const std::shared_ptr<SnapshotState>& serializationState) {
      mlir::PassManager pm2(moduleOp->getContext());
      pm2.enableVerifier(verify);
      addLingoDBInstrumentation(pm2, serializationState);
      pm2.addPass(mlir::createConvertSCFToCFPass());
      pm2.addPass(createDecomposeTuplePass());
      pm2.addPass(mlir::createCanonicalizerPass());
      pm2.addPass(mlir::createCSEPass()); // TODO: evaluate whether we need this
      if (mlir::failed(pm2.run(moduleOp))) {
         return false;
      }
      return true;
   }

   void execute(mlir::ModuleOp& moduleOp, lingodb::runtime::ExecutionContext* executionContext) override {
      auto startLowering = std::chrono::high_resolution_clock::now();
      if (!lower(moduleOp, getSerializationState())) {
         error.emit() << "Could not lower module for baseline compilation";
         return;
      }
      auto endLowering = std::chrono::high_resolution_clock::now();
      timing["baselineLowering"] = std::chrono::duration_cast<std::chrono::microseconds>(endLowering - startLowering).count() / 1000.0;

      static SpdLogSpoof logSpoof;
#if defined(__x86_64__)
      IRCompilerX64 compiler{std::make_unique<IRAdaptor>(&moduleOp, error)};
#else
#error "Baseline backend is only supported on x86_64 architecture."
#endif
      logSpoof.enter();
      const auto baselineCodeGenStart = std::chrono::high_resolution_clock::now();
      if (!compiler.compile() || compiler.adaptor->getError()) {
         error.emit() << "Could not compile query module:\n"
                      << logSpoof.logs() << "\n"
                      << compiler.adaptor->getError().emit().str() << "\n"
                      << compiler.getError().emit().str() << "\n";
         return;
      }
      const auto baselineCodeGenEnd = std::chrono::high_resolution_clock::now();
      logSpoof.exit();

      const auto baselineEmitStart = std::chrono::high_resolution_clock::now();
      std::unique_ptr<DynamicLoader> loader;
      if (!baselineDebugFileOut.getValue().empty()) {
         loader = std::make_unique<DebugLoader<IRCompilerX64::Assembler>>(compiler.assembler, error,
                                                                          baselineDebugFileOut.getValue());
      } else {
         if (!compiler.localFuncMap.contains("main")) {
            error.emit() << "No main function found in query module. Please ensure that the module has a "
                            "function named 'main'.\n";
            return;
         }
         const uint32_t mainFuncIdx = compiler.localFuncMap["main"];
         if (mainFuncIdx >= compiler.func_syms.size()) {
            error.emit() << "Main function index out of bounds: " << mainFuncIdx << " >= " << compiler.func_syms.size() << "\n";
            return;
         }
         loader = std::make_unique<InMemoryLoader>(compiler.assembler, error, compiler.func_syms[mainFuncIdx]);
      }
      if (loader->hasError) return;
      auto mainFunc = loader->getMainFunction();
      if (loader->hasError) return;
      const auto baselineEmitEnd = std::chrono::high_resolution_clock::now();

      utility::Tracer::Event execution("Execution", "run");
      utility::Tracer::Trace trace(execution);
      const auto executionStart = std::chrono::high_resolution_clock::now();
      mainFunc();
      const auto executionEnd = std::chrono::high_resolution_clock::now();
      trace.stop();
      loader->teardown();

      timing["baselineCodeGen"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                     baselineCodeGenEnd - baselineCodeGenStart)
                                     .count() /
         1000.0;
      timing["baselineEmit"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                  baselineEmitEnd - baselineEmitStart)
                                  .count() /
         1000.0;
      timing["executionTime"] = std::chrono::duration_cast<std::chrono::microseconds>(
                                   executionEnd - executionStart)
                                   .count() /
         1000.0;
   }
};
} // namespace lingodb::execution::baseline

std::unique_ptr<lingodb::execution::ExecutionBackend> lingodb::execution::createBaselineBackend() { // NOLINT (misc-use-internal-linkage)
   return std::make_unique<baseline::BaselineBackend>();
}
#endif
