
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <unordered_set>

namespace {

class ImplicitToExplicitJoins : public mlir::PassWrapper<ImplicitToExplicitJoins, mlir::FunctionPass> {
   llvm::SmallVector<mlir::Operation*> toDestroy;
   void addRequirements(mlir::Operation* op, mlir::Block* b, llvm::SmallVector<mlir::Operation*, 8>& extracted, llvm::SmallPtrSet<mlir::Operation*, 8>& alreadyPresent) {
      if (!op)
         return;
      if (op->getBlock() != b)
         return;
      if (alreadyPresent.contains(op))
         return;
      for (auto operand : op->getOperands()) {
         addRequirements(operand.getDefiningOp(), b, extracted, alreadyPresent);
      }
      alreadyPresent.insert(op);
      extracted.push_back(op);
   }
   bool isDirectSelection(mlir::Operation* op, bool& negated) {
      if (!mlir::isa<mlir::relalg::SelectionOp>(op->getParentOp())) {
         return false;
      }
      auto users = op->getUsers();
      if (users.begin() == users.end() || ++users.begin() != users.end()) {
         return false;
      }
      mlir::Operation* user = *users.begin();
      if (mlir::isa<mlir::db::NotOp>(user)) {
         auto users = user->getUsers();
         if (users.begin() == users.end() || ++users.begin() != users.end()) {
            return false;
         }
         negated = true;
         user = *users.begin();
      }
      if (mlir::isa<mlir::relalg::ReturnOp>(user)) {
         return true;
      }
      return false;
   }
   void handle(TupleLamdaOperator surroundingOperator, mlir::Operation* op, Operator relOperator, std::function<void(PredicateOperator)> apply) {
      using namespace mlir;
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      bool negated = false;
      bool directSelection = isDirectSelection(op, negated);
      Value treeVal = surroundingOperator->getOperand(0);

      //get attribute f relation to search in
      OpBuilder builder(surroundingOperator);
      auto relType = mlir::relalg::RelationType::get(&getContext());
      auto leftDir = mlir::relalg::JoinDirection::left;
      auto loc = builder.getUnknownLoc();
      if (directSelection && negated) {
         PredicateOperator mjop;
         if (negated) {
            mjop = builder.create<relalg::AntiSemiJoinOp>(loc, relType, leftDir, treeVal, relOperator.asRelation());
         } else {
            mjop = builder.create<relalg::SemiJoinOp>(loc, relType, leftDir, treeVal, relOperator.asRelation());
         }
         mjop.initPredicate();
         apply(mjop);
         surroundingOperator->replaceAllUsesWith(mjop.getOperation());
         surroundingOperator->remove();
         toDestroy.push_back(surroundingOperator);
      } else {
         std::string scopeName = attributeManager.getUniqueScope("markjoin");
         std::string attributeName = "markattr";
         attributeManager.setCurrentScope(scopeName);
         relalg::RelationalAttributeDefAttr markAttrDef = attributeManager.createDef(attributeName);
         auto& ra = markAttrDef.getRelationalAttribute();
         ra.type = mlir::db::BoolType::get(&getContext());
         PredicateOperator mjop = builder.create<relalg::MarkJoinOp>(loc, relType, leftDir, scopeName, markAttrDef, treeVal, relOperator.asRelation());
         mjop.initPredicate();
         apply(mjop);
         attributeManager.setCurrentScope(scopeName);
         relalg::RelationalAttributeRefAttr markAttrRef = attributeManager.createRef(scopeName, attributeName);
         builder.setInsertionPoint(op);
         auto replacement = builder.create<relalg::GetAttrOp>(loc, db::BoolType::get(builder.getContext()), markAttrRef, surroundingOperator.getLambdaRegion().getArgument(0));
         op->replaceAllUsesWith(replacement);
         op->remove();
         op->destroy();
         surroundingOperator->setOperand(0, mjop->getResult(0));
      }
   }
   void runOnFunction() override {
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      using namespace mlir;
      getFunction().walk([&](mlir::Operation* op) {
         TupleLamdaOperator surroundingOperator = op->getParentOfType<TupleLamdaOperator>();
         if (!surroundingOperator) {
            return;
         }
         Value treeVal = surroundingOperator->getOperand(0);
         if (auto getscalarop = mlir::dyn_cast_or_null<mlir::relalg::GetScalarOp>(op)) {
            OpBuilder builder(surroundingOperator);
            auto mjop = builder.create<relalg::SingleJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, treeVal, getscalarop.rel());
            mjop.initPredicate();
            builder.setInsertionPoint(getscalarop);
            Operation* replacement = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), getscalarop.attr().getRelationalAttribute().type, getscalarop.attr(), surroundingOperator.getLambdaRegion().getArgument(0));
            getscalarop.replaceAllUsesWith(replacement);
            getscalarop->remove();
            getscalarop->destroy();
            treeVal = mjop;
            surroundingOperator->setOperand(0, treeVal);
         } else if (auto existsop = mlir::dyn_cast_or_null<mlir::relalg::ExistsOp>(op)) {
            handle(surroundingOperator, op, existsop.rel().getDefiningOp(), [](auto) {});
         } else if (auto inop = mlir::dyn_cast_or_null<mlir::relalg::InOp>(op)) {
            Operator relOperator = inop.rel().getDefiningOp();
            //get attribute of relation to search in
            auto* attr = *relOperator.getAvailableAttributes().begin();
            auto searchInAttr = attributeManager.createRef(attr);
            handle(surroundingOperator, op, relOperator, [&](PredicateOperator mjop) {
               mjop.addPredicate([&](Value tuple, OpBuilder& builder) {
                  llvm::SmallVector<mlir::Operation*, 8> extracted;
                  llvm::SmallPtrSet<mlir::Operation*, 8> alreadyPresent;
                  addRequirements(inop.val().getDefiningOp(), &surroundingOperator.getLambdaBlock(), extracted, alreadyPresent);
                  mlir::BlockAndValueMapping mapping;
                  auto* terminator = mjop.getPredicateBlock().getTerminator();
                  mapping.map(surroundingOperator.getLambdaArgument(), mjop.getPredicateArgument());
                  for (auto* op : extracted) {
                     auto* cloneOp = builder.clone(*op, mapping);
                     cloneOp->moveBefore(terminator);
                  }
                  auto val = mapping.lookup(inop.val());
                  auto otherVal = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), searchInAttr.getRelationalAttribute().type, searchInAttr, tuple);
                  Value predicate = builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq, val, otherVal);
                  return predicate;
               });
            });
         }
      });
      for (auto* op : toDestroy) {
         op->destroy();
      }
      toDestroy.clear();
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createImplicitToExplicitJoinsPass() { return std::make_unique<ImplicitToExplicitJoins>(); }
} // end namespace relalg
} // end namespace mlir