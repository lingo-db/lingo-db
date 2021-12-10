#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class CombinePredicates : public mlir::PassWrapper<CombinePredicates, mlir::FunctionPass> {
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
      auto nullable=higherPredVal.getType().cast<mlir::db::DBType>().isNullable()||lowerPredVal.getType().cast<mlir::db::DBType>().isNullable();
      mlir::Value combined=builder.create<mlir::db::AndOp>(higher->getLoc(), mlir::db::BoolType::get(builder.getContext(),nullable),ValueRange{lowerPredVal,mapping.lookup(higherPredVal)});
      builder.create<mlir::relalg::ReturnOp>(higher->getLoc(), combined);
      lowerTerminator->remove();
      lowerTerminator->destroy();
   }

   void runOnFunction() override {
      getFunction().walk([&](mlir::relalg::SelectionOp op) {
         mlir::Value lower = op.rel();
         bool canCombine = mlir::isa<mlir::relalg::SelectionOp>(lower.getDefiningOp()) || mlir::isa<mlir::relalg::InnerJoinOp>(lower.getDefiningOp());
         if (canCombine) {
            combine(op,lower.getDefiningOp());
            op.replaceAllUsesWith(lower);
            op->remove();
            op->destroy();
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