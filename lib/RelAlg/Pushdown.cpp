#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/RelAlg/Attributes.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

class Pushdown : public mlir::PassWrapper<Pushdown, mlir::FunctionPass> {
   Operator pushdown(Operator topush, Operator curr) {
      UnaryOperator topushUnary = mlir::dyn_cast_or_null<UnaryOperator>(topush.getOperation());
      mlir::relalg::Attributes usedAttributes = topush.getUsedAttributes();
      auto res = ::llvm::TypeSwitch<mlir::Operation*, Operator>(curr.getOperation())
                    .Case<UnaryOperator>([&](UnaryOperator unaryOperator) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(unaryOperator.getOperation());
                       auto child = mlir::dyn_cast_or_null<Operator>(unaryOperator.child());
                       auto availableChild = child.getAvailableAttributes();
                       if (topushUnary.reorderable(unaryOperator) && usedAttributes.is_subset_of(availableChild)) {
                          topush->moveBefore(asOp.getOperation());
                          asOp.setChildren({pushdown(topush, child)});
                          return asOp;
                       }
                       topush.setChildren({asOp});
                       return topush;
                    })
                    .Case<BinaryOperator>([&](BinaryOperator binop) {
                       Operator asOp = mlir::dyn_cast_or_null<Operator>(binop.getOperation());
                       auto left = mlir::dyn_cast_or_null<Operator>(binop.leftChild());
                       auto right = mlir::dyn_cast_or_null<Operator>(binop.rightChild());

                       bool swapped = false;
                       if (mlir::relalg::detail::isJoin(binop.getOperation())) {
                          if (!mlir::isa<mlir::relalg::InnerJoinOp>(binop.getOperation()) && !mlir::isa<mlir::relalg::FullOuterJoinOp>(binop.getOperation())) {
                             mlir::relalg::JoinDirection joinDirection = mlir::relalg::symbolizeJoinDirection(binop->getAttr("join_direction").dyn_cast_or_null<mlir::IntegerAttr>().getInt()).getValue();
                             if (joinDirection != mlir::relalg::JoinDirection::left) {
                                std::swap(left, right);
                                swapped = true;
                             }
                          }
                       }
                       auto availableLeft = left.getAvailableAttributes();
                       auto availableRight = right.getAvailableAttributes();
                       auto pushableLeft = topushUnary.lPushable(binop) && usedAttributes.is_subset_of(availableLeft);
                       auto pushableRight = topushUnary.rPushable(binop) && usedAttributes.is_subset_of(availableRight);
                       if (!pushableLeft && !pushableRight) {
                          topush.setChildren({asOp});
                          return topush;
                       } else if (pushableLeft) {
                          topush->moveBefore(asOp.getOperation());
                          left = pushdown(topush, left);
                       } else if (pushableRight) {
                          topush->moveBefore(asOp.getOperation());
                          right = pushdown(topush, right);
                       }
                       if (swapped) {
                          std::swap(left, right);
                       }
                       asOp.setChildren({left, right});
                       return asOp;
                    })
                    .Default([&](Operator others) {
                       topush.setChildren({others});
                       return topush;
                    });
      return res;
   }

   void runOnFunction() override {
      using namespace mlir;
      getFunction()->walk([&](mlir::relalg::SelectionOp sel) {
         SmallPtrSet<mlir::Operation*, 4> users;
         for (auto* u : sel->getUsers()) {
            users.insert(u);
         }
         Operator pushedDown = pushdown(sel, sel.getChildren()[0]);
         if (sel.getOperation() != pushedDown.getOperation()) {
            sel.getResult().replaceUsesWithIf(pushedDown->getResult(0), [&](mlir::OpOperand& operand) {
               return users.contains(operand.getOwner());
            });
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createPushdownPass() { return std::make_unique<Pushdown>(); }
} // end namespace relalg
} // end namespace mlir