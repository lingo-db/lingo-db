
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>

namespace {

class Pushdown : public mlir::PassWrapper<Pushdown, mlir::FunctionPass> {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;

   bool intersects(const attribute_set& a, const attribute_set& b) {
      for (auto* x : a) {
         if (b.contains(x)) {
            return true;
         }
      }
      return false;
   }

   bool subset(const attribute_set& a, const attribute_set& b) {
      for (auto* x : a) {
         if (!b.contains(x)) {
            return false;
         }
      }
      return true;
   }

   void print(const attribute_set& a) {
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      for (auto* x : a) {
         auto [scope, name] = attributeManager.getName(x);
         llvm::dbgs() << x << "(" << scope << "," << name << "),";
      }
   }

   Operator pushdown(Operator topush, Operator curr) {
      attribute_set usedAttributes = topush.getUsedAttributes();
      auto res = ::llvm::TypeSwitch<mlir::Operation*, Operator>(curr.getOperation())
                    .Case<mlir::relalg::CrossProductOp>([&](Operator cp) {
                       auto children = cp.getChildren();
                       if (subset(usedAttributes, children[0].getAvailableAttributes())) {
                          topush->moveBefore(cp.getOperation());
                          children[0] = pushdown(topush, children[0]);
                          cp.setChildren(children);
                          return cp;
                       } else if (subset(usedAttributes, children[1].getAvailableAttributes())) {
                          topush->moveBefore(cp.getOperation());
                          children[1] = pushdown(topush, children[1]);
                          cp.setChildren(children);
                          return cp;
                       } else {
                          topush.setChildren({curr});
                          return topush;
                       }
                    })
                    .Case<BinaryOperator>([&](BinaryOperator binop) {
                       Operator opjoin = mlir::dyn_cast_or_null<Operator>(binop.getOperation());
                       if (mlir::relalg::detail::isJoin(binop.getOperation())) {
                          auto left = mlir::dyn_cast_or_null<Operator>(binop.leftChild());
                          auto right = mlir::dyn_cast_or_null<Operator>(binop.rightChild());
                          if (!mlir::isa<mlir::relalg::InnerJoinOp>(binop.getOperation())) {
                             mlir::relalg::JoinDirection joinDirection = mlir::relalg::symbolizeJoinDirection(
                                                                            binop->getAttr(
                                                                                    "join_direction")
                                                                               .dyn_cast_or_null<mlir::IntegerAttr>()
                                                                               .getInt())
                                                                            .getValue();
                             switch (joinDirection) {
                                case mlir::relalg::JoinDirection::left:
                                   if (subset(usedAttributes, left.getAvailableAttributes())) {
                                      topush->moveBefore(opjoin.getOperation());
                                      left = pushdown(topush, left);
                                      opjoin.setChildren({left, right});
                                      return opjoin;
                                   }
                                   [[fallthrough]];
                                case mlir::relalg::JoinDirection::right:
                                   if (subset(usedAttributes, right.getAvailableAttributes())) {
                                      topush->moveBefore(opjoin.getOperation());
                                      right = pushdown(topush, right);
                                      opjoin.setChildren({left, right});
                                      return opjoin;
                                   }
                                   [[fallthrough]];
                                default:
                                   topush.setChildren({curr});
                                   return topush;
                             }
                          } else {
                             auto children = opjoin.getChildren();
                             if (subset(usedAttributes, children[0].getAvailableAttributes())) {
                                topush->moveBefore(opjoin.getOperation());
                                children[0] = pushdown(topush, children[0]);
                                opjoin.setChildren(children);
                                return opjoin;
                             } else if (subset(usedAttributes, children[1].getAvailableAttributes())) {
                                topush->moveBefore(opjoin.getOperation());
                                children[1] = pushdown(topush, children[1]);
                                opjoin.setChildren(children);
                                return opjoin;
                             } else {
                                topush.setChildren({curr});
                                return topush;
                             }
                          }
                       } else {
                          topush.setChildren({opjoin});
                          return topush;
                       }
                    })
                    .Case<mlir::relalg::SelectionOp>([&](Operator sel) {
                       topush->moveBefore(sel.getOperation());
                       sel.setChildren({pushdown(topush, sel.getChildren()[0])});
                       return sel;
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
            //sel.replaceAllUsesWith(pushed_down.getOperation());
            sel.getResult().replaceUsesWithIf(pushedDown->getResult(0), [&](mlir::OpOperand& operand) {
               return users.contains(operand.getOwner());
            });
         }

         //return WalkResult::interrupt();
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createPushdownPass() { return std::make_unique<Pushdown>(); }
} // end namespace relalg
} // end namespace mlir