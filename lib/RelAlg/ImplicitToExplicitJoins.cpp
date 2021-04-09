
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <list>
#include <unordered_set>

namespace {

class ImplicitToExplicitJoins : public mlir::PassWrapper<ImplicitToExplicitJoins, mlir::FunctionPass> {
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
   mlir::Value extract(mlir::Value v, TupleLamdaOperator parent, TupleLamdaOperator newParent) {
      using namespace mlir;
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      std::string scopeName = attributeManager.getUniqueScope("extracted_map");
      attributeManager.setCurrentScope(scopeName);
      llvm::SmallVector<mlir::Operation*, 8> extracted;
      llvm::SmallPtrSet<mlir::Operation*, 8> alreadyPresent;
      addRequirements(v.getDefiningOp(), &parent.getLambdaBlock(), extracted, alreadyPresent);
      OpBuilder builder(parent.getOperation());
      mlir::BlockAndValueMapping mapping;

      mapping.map(parent.getLambdaArgument(), newParent.getLambdaArgument());
      builder.setInsertionPointToStart(&newParent.getLambdaBlock());
      auto returnop = builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
      builder.setInsertionPointToStart(&newParent.getLambdaBlock());
      for (auto* op : extracted) {
         auto* cloneOp = builder.clone(*op, mapping);
         cloneOp->moveBefore(returnop);
      }
      builder.setInsertionPoint(returnop);
      return mapping.lookup(v);
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
   void runOnFunction() override {
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      using namespace mlir;
      Type tupleType = mlir::relalg::TupleType::get(&getContext());
      SmallVector<mlir::Operation*> toDestroy;
      getFunction().walk([&](mlir::Operation* op) {
         TupleLamdaOperator surroundingOperator = op->getParentOfType<TupleLamdaOperator>();
         if (!surroundingOperator) {
            return;
         }
         bool negated = false;
         bool directSelection = isDirectSelection(op, negated);
         Value treeVal = surroundingOperator->getOperand(0);
         if (auto getscalarop = mlir::dyn_cast_or_null<mlir::relalg::GetScalarOp>(op)) {
            OpBuilder builder(surroundingOperator);
            auto mjop = builder.create<relalg::SingleJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, treeVal, getscalarop.rel());
            mjop.getRegion().push_back(new Block);
            mjop.getLambdaBlock().addArgument(tupleType);
            builder.setInsertionPointToStart(&mjop.getRegion().front());
            builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
            builder.setInsertionPoint(getscalarop);
            Operation* replacement = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), getscalarop.attr().getRelationalAttribute().type, getscalarop.attr(), surroundingOperator.getLambdaRegion().getArgument(0));
            getscalarop.replaceAllUsesWith(replacement);
            getscalarop->remove();
            getscalarop->destroy();
            treeVal = mjop;
            surroundingOperator->setOperand(0, treeVal);
         } else if (auto existsop = mlir::dyn_cast_or_null<mlir::relalg::ExistsOp>(op)) {
            OpBuilder builder(surroundingOperator);
            std::string scopeName = attributeManager.getUniqueScope("markjoin");
            std::string attributeName = "markattr";
            attributeManager.setCurrentScope(scopeName);

            TupleLamdaOperator mjop;
            if (directSelection && negated) {
               mjop = builder.create<relalg::AntiSemiJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, treeVal, existsop.rel());
            } else if (directSelection && !negated) {
               mjop = builder.create<relalg::SemiJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, treeVal, existsop.rel());
            } else {
               relalg::RelationalAttributeDefAttr defAttr = attributeManager.createDef(attributeName);
               auto& ra = defAttr.getRelationalAttribute();
               ra.type = mlir::db::BoolType::get(&getContext());
               mjop = builder.create<relalg::MarkJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, scopeName, defAttr, treeVal, existsop.rel());
            }
            mjop.getLambdaRegion().push_back(new Block);
            mjop.getLambdaBlock().addArgument(tupleType);
            builder.setInsertionPointToStart(&mjop.getLambdaBlock());
            builder.create<relalg::ReturnOp>(builder.getUnknownLoc());
            builder.setInsertionPoint(existsop);
            if (!directSelection) {
               relalg::RelationalAttributeRefAttr refAttr = attributeManager.createRef(scopeName, attributeName);
               auto replacement = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), db::BoolType::get(builder.getContext()), refAttr, surroundingOperator.getLambdaRegion().getArgument(0));
               existsop->replaceAllUsesWith(replacement);
               existsop->remove();
               existsop->destroy();
               treeVal = mjop->getResult(0);
               surroundingOperator->setOperand(0, treeVal);
            } else {
               surroundingOperator->replaceAllUsesWith(mjop.getOperation());
               surroundingOperator->remove();
               toDestroy.push_back(surroundingOperator);
            }

         } else if (auto inop = mlir::dyn_cast_or_null<mlir::relalg::InOp>(op)) {
            //get attribute of relation to search in
            Operator relOperator = inop.rel().getDefiningOp();
            auto availableAttrs = relOperator.getAvailableAttributes();
            assert(availableAttrs.size() == 1);
            auto* attr = *availableAttrs.begin();
            auto searchInAttr = attributeManager.createRef(attr);
            //get attribute f relation to search in
            OpBuilder builder(surroundingOperator);
            std::string scopeName = attributeManager.getUniqueScope("markjoin");
            std::string attributeName = "markattr";
            attributeManager.setCurrentScope(scopeName);
            TupleLamdaOperator mjop;
            if (directSelection && negated) {
               mjop = builder.create<relalg::AntiSemiJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, treeVal, inop.rel());
            } else if (directSelection && !negated) {
               mjop = builder.create<relalg::SemiJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, treeVal, inop.rel());
            } else {
               relalg::RelationalAttributeDefAttr markAttrDef = attributeManager.createDef(attributeName);
               auto& ra = markAttrDef.getRelationalAttribute();
               ra.type = mlir::db::BoolType::get(&getContext());
               mjop = builder.create<relalg::MarkJoinOp>(builder.getUnknownLoc(), mlir::relalg::RelationType::get(builder.getContext()), mlir::relalg::JoinDirection::left, scopeName, markAttrDef, treeVal, inop.rel());
            }
            mjop.getLambdaRegion().push_back(new Block);
            mjop.getLambdaBlock().addArgument(tupleType);
            Value val = extract(inop.val(), surroundingOperator, mjop);
            builder.setInsertionPoint(mjop.getLambdaBlock().getTerminator());
            auto otherVal = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), searchInAttr.getRelationalAttribute().type, searchInAttr, mjop.getLambdaArgument());
            bool nullable = val.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() || otherVal.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable();
            Value predicate = builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(&getContext(), nullable), mlir::db::DBCmpPredicate::eq, val, otherVal);
            auto* previousReturn = mjop.getLambdaBlock().getTerminator();
            builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), predicate);
            previousReturn->remove();
            previousReturn->destroy();
            builder.setInsertionPoint(inop);
            if (!directSelection) {
               attributeManager.setCurrentScope(scopeName);
               relalg::RelationalAttributeRefAttr markAttrRef = attributeManager.createRef(scopeName, attributeName);
               auto replacement = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), db::BoolType::get(builder.getContext()), markAttrRef, surroundingOperator.getLambdaRegion().getArgument(0));
               inop->replaceAllUsesWith(replacement);
               inop->remove();
               inop->destroy();
               treeVal = mjop->getResult(0);
               surroundingOperator->setOperand(0, treeVal);
            } else {
               surroundingOperator->replaceAllUsesWith(mjop.getOperation());
               surroundingOperator->remove();
               toDestroy.push_back(surroundingOperator);
            }
         }
      });
      for (auto* op : toDestroy) {
         op->destroy();
      }
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createImplicitToExplicitJoinsPass() { return std::make_unique<ImplicitToExplicitJoins>(); }
} // end namespace relalg
} // end namespace mlir