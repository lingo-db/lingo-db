#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/Dialect/RelAlg/Transforms/ColumnCreatorAnalysis.h"
#include "mlir/Dialect/TupleStream/TupleStreamOps.h"

namespace {
class InferNotNullConditions : public mlir::PassWrapper<InferNotNullConditions, mlir::OperationPass<mlir::func::FuncOp>> {
   static bool isAndedResult(mlir::Operation* op, bool first = true) {
      if (mlir::isa<mlir::tuples::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<mlir::db::AndOp>(op) || first) {
         for (auto* user : op->getUsers()) {
            if (!isAndedResult(user, false)) return false;
         }
         return true;
      } else {
         return false;
      }
   }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferNotNullConditions)
   virtual llvm::StringRef getArgument() const override { return "relalg-infer-not-null"; }
   void addNullCheck(mlir::relalg::SelectionOp selection, mlir::Value v, mlir::relalg::ColumnCreatorAnalysis columnCreatorAnalysis) {
      if (v.getType().isa<mlir::db::NullableType>()) {
         if (auto getColumnOp = mlir::dyn_cast_or_null<mlir::tuples::GetColumnOp>(v.getDefiningOp())) {
            const auto *c = &getColumnOp.getAttr().getColumn();
            if (!columnCreatorAnalysis.getCreator(c).canColumnReach({}, selection, c)) return;
            auto returnOp = mlir::cast<mlir::tuples::ReturnOp>(selection.getPredicate().front().getTerminator());
            mlir::OpBuilder builder(returnOp);
            auto isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), v);
            auto isNotNull = builder.create<mlir::db::NotOp>(builder.getUnknownLoc(), isNull);
            auto anded = builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), mlir::ValueRange{returnOp.getResults()[0], isNotNull});
            returnOp.setOperand(0, anded);
         }
      }
   }
   void runOnOperation() override {
      mlir::relalg::ColumnCreatorAnalysis columnCreatorAnalysis(getOperation());
      getOperation().walk([&](mlir::db::CmpOp cmpOp) {
         if (auto selectionOp = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(cmpOp->getParentOp())) {
            if (isAndedResult(cmpOp.getOperation())) {
               addNullCheck(selectionOp, cmpOp.getLeft(), columnCreatorAnalysis);
               addNullCheck(selectionOp, cmpOp.getRight(), columnCreatorAnalysis);
            }
         }
      });
      getOperation().walk([&](mlir::db::BetweenOp betweenOp) {
         if (!betweenOp.getLower().getType().isa<mlir::db::NullableType>() && !betweenOp.getUpper().getType().isa<mlir::db::NullableType>()) {
            if (auto selectionOp = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(betweenOp->getParentOp())) {
               if (isAndedResult(betweenOp.getOperation())) {
                  addNullCheck(selectionOp, betweenOp.getVal(), columnCreatorAnalysis);
               }
            }
         }
      });
      getOperation().walk([&](mlir::db::OneOfOp oneOfOp) {
         bool compareOnlyWithNonNullable = llvm::all_of(oneOfOp.getVals(), [](mlir::Value val) { return !val.getType().isa<mlir::db::NullableType>(); });
         if (compareOnlyWithNonNullable) {
            if (auto selectionOp = mlir::dyn_cast_or_null<mlir::relalg::SelectionOp>(oneOfOp->getParentOp())) {
               if (isAndedResult(oneOfOp.getOperation())) {
                  addNullCheck(selectionOp, oneOfOp.getVal(), columnCreatorAnalysis);
               }
            }
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createInferNotNullConditionsPass() { return std::make_unique<InferNotNullConditions>(); }
} // end namespace relalg
} // end namespace mlir