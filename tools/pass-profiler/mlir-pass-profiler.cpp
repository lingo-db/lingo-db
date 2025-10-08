#include "lingodb/compiler/Conversion/ArrowToStd/ArrowToStd.h"
#include "lingodb/compiler/Conversion/DBToStd/DBToStd.h"
#include "lingodb/compiler/Conversion/RelAlgToSubOp/RelAlgToSubOpPass.h"
#include "lingodb/compiler/Conversion/SubOpToControlFlow/SubOpToControlFlowPass.h"
#include "lingodb/compiler/Dialect/Arrow/IR/ArrowDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorDialect.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "lingodb/compiler/Dialect/util/UtilDialect.h"
#include "lingodb/compiler/mlir-support/eval.h"
#include "lingodb/execution/BackendPasses.h"
#include "lingodb/execution/Execution.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <chrono>
#include <csignal>
#include <vector>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <lingodb/compiler/Conversion/UtilToLLVM/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>

// should be declared as a part of unistd.h, but fails on macos 15
extern char** environ;

namespace cl = llvm::cl;
namespace {
cl::opt<std::string> inputFilename(cl::Positional,
                                   cl::desc("<input mlir file>"),
                                   cl::init("-"),
                                   cl::value_desc("filename"));

cl::opt<std::string> passName("pass",
                              cl::desc("Pass pipeline to profile (e.g., 'func.func(relalg-pushdown)' or 'relalg-column-folding')"),
                              cl::init("func.func(relalg-pushdown)"),
                              cl::value_desc("pass-pipeline"));

cl::opt<int> numClones("num-clones",
                       cl::desc("Number of module clones (default: 100)"),
                       cl::init(100),
                       cl::value_desc("N"));

cl::opt<std::string> perfFile("perf-output",
                              cl::desc("Perf output file (default: perf.data)"),
                              cl::init("perf.data"),
                              cl::value_desc("filename"));

cl::opt<std::string> perfBinary("perf-binary",
                                cl::desc("Perf binary path (default: /usr/bin/perf)"),
                                cl::init("/usr/bin/perf"),
                                cl::value_desc("path"));

cl::opt<bool> enablePerf("enable-perf",
                         cl::desc("Enable perf profiling (default: false)"),
                         cl::init(false));

using namespace lingodb::compiler::dialect;

void registerAllPasses() {
   // Register all standard MLIR passes
   mlir::registerAllPasses();

   // Register lingodb-specific passes
   relalg::registerRelAlgToSubOpConversionPasses();
   relalg::registerQueryOptimizationPasses();
   db::registerDBConversionPasses();
   subop::registerSubOpToControlFlowConversionPasses();
   subop::registerSubOpTransformations();

   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return lingodb::compiler::dialect::arrow::createLowerToStdPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return relalg::createDetachMetaDataPass();
   });
   ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
      return util::createUtilToLLVMPass();
   });
   lingodb::execution::registerBackendPasses();
}

void registerAllDialects(mlir::DialectRegistry& registry) {
   registry.insert<relalg::RelAlgDialect>();
   registry.insert<tuples::TupleStreamDialect>();
   registry.insert<subop::SubOperatorDialect>();
   registry.insert<db::DBDialect>();
   registry.insert<lingodb::compiler::dialect::arrow::ArrowDialect>();
   registry.insert<mlir::func::FuncDialect>();
   registry.insert<mlir::arith::ArithDialect>();
   registry.insert<mlir::DLTIDialect>();
   registry.insert<mlir::memref::MemRefDialect>();
   registry.insert<util::UtilDialect>();
   registry.insert<mlir::cf::ControlFlowDialect>();
   registry.insert<mlir::LLVM::LLVMDialect>();
   registry.insert<mlir::async::AsyncDialect>();
   registry.insert<mlir::gpu::GPUDialect>();
   registry.insert<mlir::scf::SCFDialect>();
   mlir::registerAllExtensions(registry);
   mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
   mlir::registerAllToLLVMIRTranslations(registry);
}

int loadMLIR(mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module) {
   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
   if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
      return -1;
   }

   // Parse the input mlir.
   llvm::SourceMgr sourceMgr;
   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
   if (!module) {
      llvm::errs() << "Error can't load file " << inputFilename << "\n";
      return 3;
   }
   return 0;
}

pid_t runPerfRecord() {
   pid_t childPid = 0;
   auto parentPid = std::to_string(getpid());
   auto perfBinaryStr = perfBinary.getValue();
   auto perfFileStr = perfFile.getValue();
   const char* argV[] = {perfBinaryStr.c_str(), "record", "-o", perfFileStr.c_str(),
                         "-c", "5000", "-p", parentPid.c_str(), nullptr};
   auto status = posix_spawn(&childPid, perfBinaryStr.c_str(), nullptr, nullptr,
                             const_cast<char**>(argV), environ);
   sleep(1); // Give perf time to start
   if (status != 0) {
      llvm::errs() << "Launching of perf failed: " << status << "\n";
      return -1;
   }
   return childPid;
}

} // namespace

int main(int argc, char** argv) {
   // Register all passes before parsing command line options
   // This allows --help to show all available passes
   registerAllPasses();

   cl::ParseCommandLineOptions(argc, argv, "MLIR Pass Profiler\n");

   mlir::DialectRegistry registry;
   registerAllDialects(registry);

   mlir::MLIRContext context(registry);
   lingodb::execution::initializeContext(context);
   lingodb::compiler::support::eval::init();

   mlir::OwningOpRef<mlir::ModuleOp> module;
   llvm::SourceMgr sourceMgr;
   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

   if (int error = loadMLIR(context, module))
      return error;

   llvm::outs() << "Loaded MLIR module from: " << inputFilename << "\n";

   // Create module clones
   llvm::outs() << "Creating " << numClones << " module clones...\n";
   std::vector<mlir::OwningOpRef<mlir::ModuleOp>> modules;
   modules.reserve(numClones);

   for (int i = 0; i < numClones; i++) {
      modules.push_back(module->clone());
   }

   llvm::outs() << "Created " << modules.size() << " clones\n";

   llvm::outs() << "Profiling pass: " << passName << "\n";

   // Create and configure pass managers outside of measurement
   llvm::outs() << "Setting up pass managers...\n";
   std::vector<std::unique_ptr<mlir::PassManager>> passManagers;
   passManagers.reserve(modules.size());

   for (auto& mod : modules) {
      auto pm = std::make_unique<mlir::PassManager>(mod->getContext());
      pm->enableVerifier(false); // Disable for performance

      // Parse and add the pass using the pass registry
      if (mlir::failed(mlir::parsePassPipeline(passName, *pm))) {
         llvm::errs() << "Failed to parse pass pipeline: " << passName << "\n";
         llvm::errs() << "Use --help to see available passes\n";
         return 1;
      }

      passManagers.push_back(std::move(pm));
   }

   pid_t perfPid = -1;
   if (enablePerf) {
      llvm::outs() << "Starting perf recording...\n";
      perfPid = runPerfRecord();
      if (perfPid < 0) {
         llvm::errs() << "Failed to start perf\n";
         return 1;
      }
   }

   // Apply pass to all modules and measure time (pass managers already configured)
   auto startTime = std::chrono::high_resolution_clock::now();

   for (size_t i = 0; i < modules.size(); i++) {
      if (mlir::failed(passManagers[i]->run(modules[i].get()))) {
         llvm::errs() << "Pass failed on module " << i << "\n";
         if (perfPid > 0) {
            kill(perfPid, SIGINT);
         }
         return 1;
      }
   }

   auto endTime = std::chrono::high_resolution_clock::now();

   // Stop perf recording
   if (perfPid > 0) {
      kill(perfPid, SIGINT);
      sleep(1); // Give perf time to finish
   }

   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

   llvm::outs() << "\n=== Results ===\n";
   llvm::outs() << "Pass: " << passName << "\n";
   llvm::outs() << "Number of modules: " << numClones << "\n";
   llvm::outs() << "Total time: " << duration << " ms\n";
   llvm::outs() << "Average time per module: " << (double) duration / numClones << " ms\n";
   if (enablePerf) {
      llvm::outs() << "Perf data written to: " << perfFile << "\n";
   }

   return 0;
}
