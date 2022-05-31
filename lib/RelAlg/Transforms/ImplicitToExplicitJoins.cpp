#include "mlir/Dialect/DB/IR/DBDialect.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

class ImplicitToExplicitJoins : public mlir::PassWrapper<ImplicitToExplicitJoins, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-implicit-to-explicit-joins"; }
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::db::DBDialect>();
   }
   llvm::SmallVector<mlir::Operation*> toDestroy;
   void handleScalarBoolOp(mlir::Location loc, TupleLamdaOperator surroundingOperator, mlir::Operation* op, Operator relOperator, std::function<void(PredicateOperator)> apply) {
      using namespace mlir;
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
      bool negated = false;
      bool directSelection = false;
      if (mlir::isa<mlir::relalg::SelectionOp>(op->getParentOp())) {
         auto users = op->getUsers();
         if (users.begin() != users.end() && ++users.begin() == users.end()) {
            mlir::Operation* user = *users.begin();
            if (mlir::isa<mlir::db::NotOp>(user)) {
               auto negationUsers = user->getUsers();
               if (negationUsers.begin() != negationUsers.end() && ++negationUsers.begin() == negationUsers.end()) {
                  negated = true;
                  user = *negationUsers.begin();
               }
            }
            if (mlir::isa<mlir::relalg::ReturnOp>(user)) {
               directSelection = true;
            }
         }
      }
      Value treeVal = surroundingOperator->getOperand(0);

      //get attribute f relation to search in
      OpBuilder builder(surroundingOperator);
      auto relType = mlir::relalg::TupleStreamType::get(&getContext());
      if (directSelection) {
         PredicateOperator semijoin;
         if (negated) {
            semijoin = builder.create<relalg::AntiSemiJoinOp>(loc, relType, treeVal, relOperator.asRelation());
         } else {
            semijoin = builder.create<relalg::SemiJoinOp>(loc, relType, treeVal, relOperator.asRelation());
         }
         semijoin.initPredicate();
         apply(semijoin);
         surroundingOperator->replaceAllUsesWith(semijoin.getOperation());
         surroundingOperator->remove();
         toDestroy.push_back(surroundingOperator);
      } else {
         std::string scopeName = attributeManager.getUniqueScope("markjoin");
         std::string attributeName = "markattr";
         relalg::ColumnDefAttr markAttrDef = attributeManager.createDef(scopeName, attributeName);
         auto& ra = markAttrDef.getColumn();
         ra.type = builder.getI1Type();
         PredicateOperator markJoin = builder.create<relalg::MarkJoinOp>(loc, relType, markAttrDef, treeVal, relOperator.asRelation());
         markJoin.initPredicate();
         apply(markJoin);
         relalg::ColumnRefAttr markAttrRef = attributeManager.createRef(scopeName, attributeName);
         builder.setInsertionPoint(op);
         auto replacement = builder.create<relalg::GetColumnOp>(loc, builder.getI1Type(), markAttrRef, surroundingOperator.getLambdaRegion().getArgument(0));
         op->replaceAllUsesWith(replacement);
         op->erase();
         surroundingOperator->setOperand(0, markJoin->getResult(0));
      }
   }
   void runOnOperation() override {
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getColumnManager();
      using namespace mlir;
      getOperation().walk([&](mlir::Operation* op) {
         TupleLamdaOperator surroundingOperator = op->getParentOfType<TupleLamdaOperator>();
         if (!surroundingOperator) {
            return;
         }
         Value treeVal = surroundingOperator->getOperand(0);
         if (auto getscalarop = mlir::dyn_cast_or_null<mlir::relalg::GetScalarOp>(op)) {
            OpBuilder builder(surroundingOperator);
            std::string scopeName = attributeManager.getUniqueScope("singlejoin");
            std::string attributeName = "sjattr";
            auto before = getscalarop.attr();
            auto fromExisting = ArrayAttr::get(&getContext(), {before});

            auto newAttrType = getscalarop.getType();
            auto newDef = attributeManager.createDef(scopeName, attributeName, fromExisting);
            if (!newAttrType.isa<mlir::db::NullableType>()) {
               newAttrType = mlir::db::NullableType::get(builder.getContext(), newAttrType);
            }
            newDef.getColumn().type = newAttrType;

            auto mapping = ArrayAttr::get(&getContext(), {newDef});
            auto singleJoin = builder.create<relalg::SingleJoinOp>(getscalarop->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), treeVal, getscalarop.rel(), mapping);
            singleJoin.initPredicate();
            builder.setInsertionPoint(getscalarop);
            mlir::Value replacement = builder.create<relalg::GetColumnOp>(getscalarop->getLoc(), newAttrType, attributeManager.createRef(scopeName, attributeName), surroundingOperator.getLambdaRegion().getArgument(0));
            getscalarop.replaceAllUsesWith(replacement);
            getscalarop->erase();
            treeVal = singleJoin;
            surroundingOperator->setOperand(0, treeVal);
         } else if (auto getlistop = mlir::dyn_cast_or_null<mlir::relalg::GetListOp>(op)) {
            OpBuilder builder(surroundingOperator);
            std::string scopeName = attributeManager.getUniqueScope("collectionjoin");
            std::string attributeName = "collattr";
            auto fromAttrs = getlistop.cols();

            auto newDef = attributeManager.createDef(scopeName, attributeName);
            newDef.getColumn().type = getlistop.getType();
            auto collectionJoin = builder.create<relalg::CollectionJoinOp>(getlistop->getLoc(), mlir::relalg::TupleStreamType::get(builder.getContext()), fromAttrs, newDef, treeVal, getlistop.rel());
            collectionJoin.initPredicate();
            builder.setInsertionPoint(getlistop);
            Operation* replacement = builder.create<relalg::GetColumnOp>(getlistop->getLoc(), getlistop.getType(), attributeManager.createRef(scopeName, attributeName), surroundingOperator.getLambdaRegion().getArgument(0));
            getlistop.replaceAllUsesWith(replacement);
            getlistop->erase();
            treeVal = collectionJoin;
            surroundingOperator->setOperand(0, treeVal);
         } else if (auto existsop = mlir::dyn_cast_or_null<mlir::relalg::ExistsOp>(op)) {
            handleScalarBoolOp(existsop->getLoc(), surroundingOperator, op, existsop.rel().getDefiningOp(), [](auto) {});
         } else if (auto inop = mlir::dyn_cast_or_null<mlir::relalg::InOp>(op)) {
            Operator relOperator = inop.rel().getDefiningOp();
            //get attribute of relation to search in
            const auto* attr = *relOperator.getAvailableColumns().begin();
            auto searchInAttr = attributeManager.createRef(attr);
            handleScalarBoolOp(inop->getLoc(), surroundingOperator, op, relOperator, [&](PredicateOperator predicateOperator) {
               predicateOperator.addPredicate([&](Value tuple, OpBuilder& builder) {
                  mlir::BlockAndValueMapping mapping;
                  mapping.map(surroundingOperator.getLambdaArgument(), predicateOperator.getPredicateArgument());
                  mlir::relalg::detail::inlineOpIntoBlock(inop.val().getDefiningOp(), surroundingOperator.getOperation(), predicateOperator.getOperation(), &predicateOperator.getPredicateBlock(), mapping);
                  auto val = mapping.lookup(inop.val());
                  auto otherVal = builder.create<relalg::GetColumnOp>(inop->getLoc(), searchInAttr.getColumn().type, searchInAttr, tuple);
                  Value predicate = builder.create<mlir::db::CmpOp>(inop->getLoc(), mlir::db::DBCmpPredicate::eq, val, otherVal);
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