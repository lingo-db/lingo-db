#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class IntroduceTmp : public mlir::PassWrapper<IntroduceTmp, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-introduce-tmp"; }
   public:
   mlir::relalg::ColumnSet getUsed(mlir::Operation* op) {
      if (auto asOperator = mlir::dyn_cast_or_null<Operator>(op)) {
         auto cols = asOperator.getUsedColumns();
         for (auto *user : asOperator.asRelation().getUsers()) {
            cols.insert(getUsed(user));
         }
         return cols;
      } else if (auto matOp = mlir::dyn_cast_or_null<mlir::relalg::MaterializeOp>(op)) {
         return mlir::relalg::ColumnSet::fromArrayAttr(matOp.cols());
      }
      return {};
   }
   void runOnOperation() override {
      getOperation().walk([&](Operator op) {
         auto users = op->getUsers();
         if (!users.empty() && ++users.begin() != users.end()) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointAfter(op.getOperation());
            mlir::relalg::ColumnSet usedAttributes;
            for (auto *user : users) {
               usedAttributes.insert(getUsed(user));
            }
            usedAttributes=usedAttributes.intersect(op.getAvailableColumns());
            auto tmp = builder.create<mlir::relalg::TmpOp>(op->getLoc(), op.asRelation().getType(), op.asRelation(), usedAttributes.asRefArrayAttr(&getContext()));

            op.asRelation().replaceUsesWithIf(tmp.getResult(), [&](mlir::OpOperand& operand) { return operand.getOwner() != tmp.getOperation(); });
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createIntroduceTmpPass() { return std::make_unique<IntroduceTmp>(); }
} // end namespace relalg
} // end namespace mlir