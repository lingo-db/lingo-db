#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/RelAlg/Transforms/ColumnCreatorAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

namespace {
using namespace lingodb::compiler::dialect;

class InferNotNullConditions : public mlir::PassWrapper<InferNotNullConditions, mlir::OperationPass<mlir::func::FuncOp>> {
   static bool isAndedResult(mlir::Operation* op, bool first = true) {
      if (mlir::isa<tuples::ReturnOp>(op)) {
         return true;
      }
      if (mlir::isa<db::AndOp>(op) || first) {
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
   void addNullCheck(relalg::SelectionOp selection, mlir::Value v, relalg::ColumnCreatorAnalysis columnCreatorAnalysis) {
      if (mlir::isa<db::NullableType>(v.getType())) {
         if (auto getColumnOp = mlir::dyn_cast_or_null<tuples::GetColumnOp>(v.getDefiningOp())) {
            const auto *c = &getColumnOp.getAttr().getColumn();
            if (!columnCreatorAnalysis.getCreator(c).canColumnReach({}, selection, c)) return;
            auto returnOp = mlir::cast<tuples::ReturnOp>(selection.getPredicate().front().getTerminator());
            mlir::OpBuilder builder(returnOp);
            auto isNull = builder.create<db::IsNullOp>(builder.getUnknownLoc(), v);
            auto isNotNull = builder.create<db::NotOp>(builder.getUnknownLoc(), isNull);
            auto anded = builder.create<db::AndOp>(builder.getUnknownLoc(), mlir::ValueRange{returnOp.getResults()[0], isNotNull});
            returnOp.setOperand(0, anded);
         }
      }
   }
   void runOnOperation() override {
      relalg::ColumnCreatorAnalysis columnCreatorAnalysis(getOperation());
      getOperation().walk([&](db::CmpOp cmpOp) {
         if (auto selectionOp = mlir::dyn_cast_or_null<relalg::SelectionOp>(cmpOp->getParentOp())) {
            if (isAndedResult(cmpOp.getOperation())) {
               addNullCheck(selectionOp, cmpOp.getLeft(), columnCreatorAnalysis);
               addNullCheck(selectionOp, cmpOp.getRight(), columnCreatorAnalysis);
            }
         }
      });
      getOperation().walk([&](db::BetweenOp betweenOp) {
         if (!mlir::isa<db::NullableType>(betweenOp.getLower().getType()) && !mlir::isa<db::NullableType>(betweenOp.getUpper().getType())) {
            if (auto selectionOp = mlir::dyn_cast_or_null<relalg::SelectionOp>(betweenOp->getParentOp())) {
               if (isAndedResult(betweenOp.getOperation())) {
                  addNullCheck(selectionOp, betweenOp.getVal(), columnCreatorAnalysis);
               }
            }
         }
      });
      getOperation().walk([&](db::OneOfOp oneOfOp) {
         bool compareOnlyWithNonNullable = llvm::all_of(oneOfOp.getVals(), [](mlir::Value val) { return !mlir::isa<db::NullableType>(val.getType()); });
         if (compareOnlyWithNonNullable) {
            if (auto selectionOp = mlir::dyn_cast_or_null<relalg::SelectionOp>(oneOfOp->getParentOp())) {
               if (isAndedResult(oneOfOp.getOperation())) {
                  addNullCheck(selectionOp, oneOfOp.getVal(), columnCreatorAnalysis);
               }
            }
         }
      });
   }
};
} // end anonymous namespace


std::unique_ptr<mlir::Pass> relalg::createInferNotNullConditionsPass() { return std::make_unique<InferNotNullConditions>(); }
