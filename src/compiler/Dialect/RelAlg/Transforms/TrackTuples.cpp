#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"

namespace {
using namespace lingodb::compiler::dialect;

class InsertTrackTuplesAfterRelalg : public mlir::RewritePattern {
   public:
   InsertTrackTuplesAfterRelalg(mlir::MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
   mlir::LogicalResult match(mlir::Operation* op) const override {
      // Only track tuples for relalg operations
      if (op->getDialect()->getNamespace() != "relalg") {
         return mlir::failure();
      }
      // Check if op emits a tuple stream
      bool isApplicableType = llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
         return mlir::isa<tuples::TupleStreamType>(type);
      });

      if (!isApplicableType) {
         return mlir::failure();
      }

      // Check if rewrite pattern has already been applied
      bool notAlreadyRewritten = std::count_if(op->getUsers().begin(), op->getUsers().end(), [](mlir::Operation* user) {
                                    return mlir::dyn_cast_or_null<relalg::TrackTuplesOP>(user) != nullptr;
                                 }) == 0;

      bool isVirtual = op->hasAttr("virtual");
      return notAlreadyRewritten && !isVirtual ? mlir::success() : mlir::failure();
   }
   void rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      // keep track of number of used tupleCounters
      static uint32_t currentTupleCounter = 0;

      // Introduce a TrackTuplesOp for every result tupleStream of op
      rewriter.setInsertionPointAfter(op);
      for (auto result : op->getResults()) {
         if (mlir::isa<tuples::TupleStreamType>(result.getType())) {
            rewriter.create<relalg::TrackTuplesOP>(op->getLoc(), result, currentTupleCounter++);
         }
      }
   }
};

class TrackTuples : public mlir::PassWrapper<TrackTuples, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-track-tuples"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TrackTuples)
   private:
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
   }

   public:
   void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.insert<InsertTrackTuplesAfterRelalg>(&getContext());
      if (mlir::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
         signalPassFailure();
      }
   }
};
} // end anonymous namespace


std::unique_ptr<mlir::Pass> relalg::createTrackTuplesPass() { return std::make_unique<TrackTuples>(); }
