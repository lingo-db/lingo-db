#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class CombinePredicates : public mlir::PassWrapper<CombinePredicates, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-combine-predicates"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CombinePredicates)

   void combine(PredicateOperator higher, PredicateOperator lower) {
      using namespace mlir;
      auto higherTerminator = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(higher.getPredicateBlock().getTerminator());
      auto lowerTerminator = mlir::dyn_cast_or_null<mlir::tuples::ReturnOp>(lower.getPredicateBlock().getTerminator());

      Value higherPredVal = higherTerminator.getResults()[0];
      Value lowerPredVal = lowerTerminator.getResults()[0];

      OpBuilder builder(lower);
      mlir::IRMapping mapping;
      mapping.map(higher.getPredicateArgument(), lower.getPredicateArgument());
      builder.setInsertionPointToEnd(&lower.getPredicateBlock());
      mlir::relalg::detail::inlineOpIntoBlock(higherPredVal.getDefiningOp(), higherPredVal.getDefiningOp()->getParentOp(), &lower.getPredicateBlock(), mapping);
      auto nullable = higherPredVal.getType().isa<mlir::db::NullableType>() || lowerPredVal.getType().isa<mlir::db::NullableType>();
      mlir::Type restype = builder.getI1Type();
      if (nullable) {
         restype = mlir::db::NullableType::get(builder.getContext(), restype);
      }
      mlir::Value combined = builder.create<mlir::db::AndOp>(higher->getLoc(), restype, ValueRange{lowerPredVal, mapping.lookup(higherPredVal)});
      builder.create<mlir::tuples::ReturnOp>(higher->getLoc(), combined);
      lowerTerminator->erase();
   }

   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::SelectionOp op) {
         mlir::Value lower = op.getRel();
         bool canCombine = mlir::isa<mlir::relalg::InnerJoinOp>(lower.getDefiningOp());
         if (canCombine&&lower.hasOneUse()) {
            combine(op, mlir::cast<PredicateOperator>(lower.getDefiningOp()));
            op.replaceAllUsesWith(lower);
            op->erase();
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createCombinePredicatesPass() { return std::make_unique<CombinePredicates>(); }
} // end namespace relalg
} // end namespace mlir