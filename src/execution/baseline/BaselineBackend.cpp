#if BASELINE_ENABLED == 1
#if !defined(__linux__)
#error "Baseline backend is only supported on Linux systems."
#endif
#if defined(__x86_64__)
#include "CompilerX64.hpp"
#elif defined(__aarch64__)
#include "CompilerA64.hpp"
#endif
#include "Loader.hpp"

#include "lingodb/compiler/Dialect/util/UtilOps.h"
#include "lingodb/compiler/helper.h"
#include "lingodb/execution/BackendPasses.h"
#include "lingodb/execution/BaselineBackend.h"
#include "lingodb/utility/Setting.h"
#include "lingodb/utility/Tracer.h"

#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/Passes.h>

namespace lingodb::execution::baseline {
using namespace compiler;

// init IRAdaptor static vars
IRAdaptor::IRFuncRef IRAdaptor::INVALID_FUNC_REF = nullptr;
IRAdaptor::IRValueRef IRAdaptor::INVALID_VALUE_REF = mlir::Value();

namespace {
utility::GlobalSetting<std::string> baselineDebugFileOut("system.compilation.baseline_object_out", "");

// The baseline compiler loads/stores scalar values only. A tuple-typed util.load/util.store (e.g. the closure argument
// that the try_wrapped_fn of db.try_except loads and passes on as a whole) is split into one load/store per element.
mlir::Value loadTuple(mlir::OpBuilder& builder, mlir::Location loc, mlir::TupleType tupleType, mlir::Value ref) {
   llvm::SmallVector<mlir::Value> elements;
   for (auto [i, elementType] : llvm::enumerate(tupleType.getTypes())) {
      auto elementRef = builder.create<dialect::util::TupleElementPtrOp>(loc, dialect::util::RefType::get(elementType), ref, i);
      if (auto nestedTupleType = mlir::dyn_cast<mlir::TupleType>(elementType)) {
         elements.push_back(loadTuple(builder, loc, nestedTupleType, elementRef));
      } else {
         elements.push_back(builder.create<dialect::util::LoadOp>(loc, elementType, elementRef, mlir::Value()));
      }
   }
   return builder.create<dialect::util::PackOp>(loc, tupleType, elements);
}
void storeTuple(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value tuple, mlir::Value ref) {
   auto tupleType = mlir::cast<mlir::TupleType>(tuple.getType());
   for (auto [i, elementType] : llvm::enumerate(tupleType.getTypes())) {
      auto elementRef = builder.create<dialect::util::TupleElementPtrOp>(loc, dialect::util::RefType::get(elementType), ref, i);
      mlir::Value element = builder.create<dialect::util::GetTupleOp>(loc, elementType, tuple, i);
      if (mlir::isa<mlir::TupleType>(elementType)) {
         storeTuple(builder, loc, element, elementRef);
      } else {
         builder.create<dialect::util::StoreOp>(loc, element, elementRef, mlir::Value());
      }
   }
}
mlir::Value elementRefForIndex(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value ref, mlir::Value idx) {
   return idx ? builder.create<dialect::util::ArrayElementPtrOp>(loc, ref.getType(), ref, idx).getResult() : ref;
}
class SplitTupleLoad : public mlir::OpRewritePattern<dialect::util::LoadOp> {
   using OpRewritePattern::OpRewritePattern;
   mlir::LogicalResult matchAndRewrite(dialect::util::LoadOp op, mlir::PatternRewriter& rewriter) const override {
      auto tupleType = mlir::dyn_cast<mlir::TupleType>(op.getVal().getType());
      if (!tupleType) return mlir::failure();
      auto ref = elementRefForIndex(rewriter, op.getLoc(), op.getRef(), op.getIdx());
      rewriter.replaceOp(op, loadTuple(rewriter, op.getLoc(), tupleType, ref));
      return mlir::success();
   }
};
class SplitTupleStore : public mlir::OpRewritePattern<dialect::util::StoreOp> {
   using OpRewritePattern::OpRewritePattern;
   mlir::LogicalResult matchAndRewrite(dialect::util::StoreOp op, mlir::PatternRewriter& rewriter) const override {
      if (!mlir::isa<mlir::TupleType>(op.getVal().getType())) return mlir::failure();
      auto ref = elementRefForIndex(rewriter, op.getLoc(), op.getRef(), op.getIdx());
      storeTuple(rewriter, op.getLoc(), op.getVal(), ref);
      rewriter.eraseOp(op);
      return mlir::success();
   }
};

class LegalizeForBackend : public mlir::PassWrapper<LegalizeForBackend, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "baseline-legalize"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeForBackend)
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         //patterns.insert<EliminateNullCmp>(&getContext());

         dialect::util::UnPackOp::getCanonicalizationPatterns(patterns, patterns.getContext());
         dialect::util::GetTupleOp::getCanonicalizationPatterns(patterns, patterns.getContext());
         dialect::util::StoreOp::getCanonicalizationPatterns(patterns, patterns.getContext());
         dialect::util::UndefOp::getCanonicalizationPatterns(patterns, patterns.getContext());
         dialect::util::StoreElementOp::getCanonicalizationPatterns(patterns, patterns.getContext());
         patterns.insert<SplitTupleLoad, SplitTupleStore>(&getContext());

         if (lingodb::compiler::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }
   }
};

} // namespace

class BaselineBackend : public ExecutionBackend {
   // lower mlir IR to a form that can be compiled by tpde
   // currently mostly does a SCF to CF conversion
   bool lower(mlir::ModuleOp& moduleOp,
              const std::shared_ptr<SnapshotState>& serializationState) {
      mlir::PassManager pm2(moduleOp->getContext());
      pm2.enableVerifier(verify);
      addLingoDBInstrumentation(pm2, serializationState);
      pm2.addPass(std::make_unique<LegalizeForBackend>());
      //pm2.addPass(lingodb::compiler::createCanonicalizerPass());
      pm2.addPass(mlir::createConvertSCFToCFPass());
      // type conversions of earlier lowerings (e.g. !arrow.table <-> !util.ref<i8> in lower-arrow/lower-py-interp) leave
      // pairs of casts that cancel out; the LLVM backends remove them the same way
      pm2.addPass(mlir::createReconcileUnrealizedCastsPass());
      if (mlir::failed(pm2.run(moduleOp))) {
         return false;
      }
      return true;
   }
   bool isLLVMBased() const override { return false; }

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
#elif defined(__aarch64__)
      IRCompilerA64 compiler{std::make_unique<IRAdaptor>(&moduleOp, error)};
#else
#error "Baseline backend is only supported on x86_64 or aarch64 architectures."
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
#if defined(__x86_64__)
         loader = std::make_unique<DebugLoader<IRCompilerX64::Assembler>>(compiler.assembler, error,
                                                                          baselineDebugFileOut.getValue());
#elif defined(__aarch64__)
         loader = std::make_unique<DebugLoader<IRCompilerA64::Assembler>>(compiler.assembler, error,
                                                                          baselineDebugFileOut.getValue());
#else
#error "Baseline backend is only supported on x86_64 or aarch64 architectures."
#endif
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
