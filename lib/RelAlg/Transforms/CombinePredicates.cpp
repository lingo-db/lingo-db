#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class CombinePredicates : public mlir::PassWrapper<CombinePredicates, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-combine-predicates"; }

   public:
   void combine(PredicateOperator higher, PredicateOperator lower) {
      using namespace mlir;
      auto higherTerminator = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(higher.getPredicateBlock().getTerminator());
      auto lowerTerminator = mlir::dyn_cast_or_null<mlir::relalg::ReturnOp>(lower.getPredicateBlock().getTerminator());

      Value higherPredVal = higherTerminator.results()[0];
      Value lowerPredVal = lowerTerminator.results()[0];

      OpBuilder builder(lower);
      mlir::BlockAndValueMapping mapping;
      mapping.map(higher.getPredicateArgument(), lower.getPredicateArgument());
      builder.setInsertionPointToEnd(&lower.getPredicateBlock());
      mlir::relalg::detail::inlineOpIntoBlock(higherPredVal.getDefiningOp(), higherPredVal.getDefiningOp()->getParentOp(), lower.getOperation(), &lower.getPredicateBlock(), mapping);
      auto nullable = higherPredVal.getType().isa<mlir::db::NullableType>() || lowerPredVal.getType().isa<mlir::db::NullableType>();
      mlir::Type restype = builder.getI1Type();
      if (nullable) {
         restype = mlir::db::NullableType::get(builder.getContext(), restype);
      }
      mlir::Value combined = builder.create<mlir::db::AndOp>(higher->getLoc(), restype, ValueRange{lowerPredVal, mapping.lookup(higherPredVal)});
      builder.create<mlir::relalg::ReturnOp>(higher->getLoc(), combined);
      lowerTerminator->erase();
   }

   void runOnOperation() override {
      getOperation().walk([&](mlir::relalg::SelectionOp op) {
         mlir::Value lower = op.rel();
         bool canCombine = mlir::isa<mlir::relalg::SelectionOp>(lower.getDefiningOp()) || mlir::isa<mlir::relalg::InnerJoinOp>(lower.getDefiningOp());
         if (canCombine) {
            combine(op, lower.getDefiningOp());
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