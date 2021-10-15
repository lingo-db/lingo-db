#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class IntroduceTmp : public mlir::PassWrapper<IntroduceTmp, mlir::FunctionPass> {
   virtual llvm::StringRef getArgument() const override { return "relalg-introduce-tmp"; }
   public:
   mlir::relalg::Attributes getUsed(mlir::Operation* op) {
      if (auto asOperator = mlir::dyn_cast_or_null<Operator>(op)) {
         auto attrs = asOperator.getUsedAttributes();
         for (auto *user : asOperator.asRelation().getUsers()) {
            attrs.insert(getUsed(user));
         }
         return attrs;
      } else if (auto matOp = mlir::dyn_cast_or_null<mlir::relalg::MaterializeOp>(op)) {
         return mlir::relalg::Attributes::fromArrayAttr(matOp.attrs());
      }
      return {};
   }
   void runOnFunction() override {
      getFunction().walk([&](Operator op) {
         auto users = op->getUsers();
         if (!users.empty() && ++users.begin() != users.end()) {
            mlir::OpBuilder builder(&getContext());
            builder.setInsertionPointAfter(op.getOperation());
            mlir::relalg::Attributes usedAttributes;
            for (auto *user : users) {
               usedAttributes.insert(getUsed(user));
            }
            usedAttributes=usedAttributes.intersect(op.getAvailableAttributes());
            auto tmp = builder.create<mlir::relalg::TmpOp>(builder.getUnknownLoc(), op.asRelation().getType(), op.asRelation(), usedAttributes.asRefArrayAttr(&getContext()));

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