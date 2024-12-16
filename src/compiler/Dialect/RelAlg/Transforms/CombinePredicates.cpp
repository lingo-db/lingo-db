#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
using namespace lingodb::compiler::dialect;

class CombinePredicates : public mlir::PassWrapper<CombinePredicates, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-combine-predicates"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CombinePredicates)

   void combine(PredicateOperator higher, PredicateOperator lower) {
      using namespace mlir;
      auto higherTerminator = mlir::dyn_cast_or_null<tuples::ReturnOp>(higher.getPredicateBlock().getTerminator());
      auto lowerTerminator = mlir::dyn_cast_or_null<tuples::ReturnOp>(lower.getPredicateBlock().getTerminator());

      Value higherPredVal = higherTerminator.getResults()[0];
      Value lowerPredVal = lowerTerminator.getResults()[0];

      OpBuilder builder(lower);
      mlir::IRMapping mapping;
      mapping.map(higher.getPredicateArgument(), lower.getPredicateArgument());
      builder.setInsertionPointToEnd(&lower.getPredicateBlock());
      relalg::detail::inlineOpIntoBlock(higherPredVal.getDefiningOp(), higherPredVal.getDefiningOp()->getParentOp(), &lower.getPredicateBlock(), mapping);
      auto nullable = mlir::isa<db::NullableType>(higherPredVal.getType()) || mlir::isa<db::NullableType>(lowerPredVal.getType());
      mlir::Type restype = builder.getI1Type();
      if (nullable) {
         restype = db::NullableType::get(builder.getContext(), restype);
      }
      mlir::Value combined = builder.create<db::AndOp>(higher->getLoc(), restype, ValueRange{lowerPredVal, mapping.lookup(higherPredVal)});
      builder.create<tuples::ReturnOp>(higher->getLoc(), combined);
      lowerTerminator->erase();
   }

   void runOnOperation() override {
      getOperation().walk([&](relalg::SelectionOp op) {
         mlir::Value lower = op.getRel();
         bool canCombine = mlir::isa<relalg::InnerJoinOp>(lower.getDefiningOp());
         if (canCombine&&lower.hasOneUse()) {
            combine(op, mlir::cast<PredicateOperator>(lower.getDefiningOp()));
            op.replaceAllUsesWith(lower);
            op->erase();
         }
      });
   }
};
} // end anonymous namespace


std::unique_ptr<mlir::Pass> relalg::createCombinePredicatesPass() { return std::make_unique<CombinePredicates>(); }
