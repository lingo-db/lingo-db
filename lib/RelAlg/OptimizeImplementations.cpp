#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
class OptimizeImplementations : public mlir::PassWrapper<OptimizeImplementations, mlir::FunctionPass> {
   public:
   bool hashImplPossible(mlir::Block* block, mlir::relalg::Attributes availableLeft, mlir::relalg::Attributes availableRight) { //todo: does not work always
      llvm::DenseMap<mlir::Value, mlir::relalg::Attributes> required;
      mlir::relalg::Attributes leftKeys, rightKeys;
      std::vector<mlir::Type> types;
      bool res = false;
      block->walk([&](mlir::Operation* op) {
         if (auto getAttr = mlir::dyn_cast_or_null<mlir::relalg::GetAttrOp>(op)) {
            required.insert({getAttr.getResult(), mlir::relalg::Attributes::from(getAttr.attr())});
         } else if (auto cmpOp = mlir::dyn_cast_or_null<mlir::db::CmpOp>(op)) {
            if (cmpOp.predicate() == mlir::db::DBCmpPredicate::eq) {
               auto leftAttributes = required[cmpOp.left()];
               auto rightAttributes = required[cmpOp.right()];
               if (leftAttributes.isSubsetOf(availableLeft) && rightAttributes.isSubsetOf(availableRight)) {
                  res = true;
               } else if (leftAttributes.isSubsetOf(availableRight) && rightAttributes.isSubsetOf(availableLeft)) {
                  res = true;
               }
            }
         } else {
            mlir::relalg::Attributes attributes;
            for (auto operand : op->getOperands()) {
               if (required.count(operand)) {
                  attributes.insert(required[operand]);
               }
            }
            for (auto result : op->getResults()) {
               required.insert({result, attributes});
            }
         }
      });
      return res;
   }
   void runOnFunction() override {
      getFunction().walk([&](Operator op) {
         auto impl = ::llvm::TypeSwitch<mlir::Operation*, std::string>(op.getOperation())
                        .Case<mlir::relalg::InnerJoinOp, mlir::relalg::SemiJoinOp, mlir::relalg::AntiSemiJoinOp, mlir::relalg::SingleJoinOp, mlir::relalg::MarkJoinOp>([&](PredicateOperator predicateOperator) {
                           auto binOp = mlir::cast<BinaryOperator>(predicateOperator.getOperation());
                           auto left = mlir::cast<Operator>(binOp.leftChild());
                           auto right = mlir::cast<Operator>(binOp.rightChild());
                           if (hashImplPossible(&predicateOperator.getPredicateBlock(), left.getAvailableAttributes(), right.getAvailableAttributes())) {
                              return "hash";
                           } else {
                              return "";
                           }
                        })
                        .Case<mlir::relalg::OuterJoinOp>([&](mlir::relalg::OuterJoinOp op) { return ""; })
                        .Case<mlir::relalg::FullOuterJoinOp>([&](mlir::relalg::FullOuterJoinOp op) { return ""; })
                        .Default([&](auto x) {
                           return "";
                        });
         if (!impl.empty()) {
            op->setAttr("impl", mlir::StringAttr::get(op.getContext(), impl));
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createOptimizeImplementationsPass() { return std::make_unique<OptimizeImplementations>(); }
} // end namespace relalg
} // end namespace mlir