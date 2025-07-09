#include "lingodb/compiler/Dialect/DB/IR/DBDialect.h"
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {
using namespace lingodb::compiler::dialect;

class ImplicitToExplicitJoins : public mlir::PassWrapper<ImplicitToExplicitJoins, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-implicit-to-explicit-joins"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImplicitToExplicitJoins)
   private:
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<db::DBDialect>();
   }
   llvm::SmallVector<mlir::Operation*> toDestroy;
   void handleScalarBoolOp(mlir::Location loc, TupleLamdaOperator surroundingOperator, mlir::Operation* op, Operator relOperator, std::function<void(PredicateOperator)> apply) {
      using namespace mlir;
      auto& attributeManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      bool negated = false;
      bool directSelection = false;
      if (mlir::isa<relalg::SelectionOp>(op->getParentOp())) {
         auto users = op->getUsers();
         if (users.begin() != users.end() && ++users.begin() == users.end()) {
            mlir::Operation* user = *users.begin();
            if (mlir::isa<db::NotOp>(user)) {
               auto negationUsers = user->getUsers();
               if (negationUsers.begin() != negationUsers.end() && ++negationUsers.begin() == negationUsers.end()) {
                  negated = true;
                  user = *negationUsers.begin();
               }
            }
            if (mlir::isa<tuples::ReturnOp>(user)) {
               directSelection = true;
            }
         }
      }
      Value treeVal = surroundingOperator->getOperand(0);

      //get attribute f relation to search in
      OpBuilder builder(surroundingOperator);
      auto relType = tuples::TupleStreamType::get(&getContext());
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
         tuples::ColumnDefAttr markAttrDef = attributeManager.createDef(scopeName, attributeName);
         auto& ra = markAttrDef.getColumn();
         ra.type = builder.getI1Type();
         PredicateOperator markJoin = builder.create<relalg::MarkJoinOp>(loc, relType, markAttrDef, treeVal, relOperator.asRelation());
         markJoin.initPredicate();
         apply(markJoin);
         tuples::ColumnRefAttr markAttrRef = attributeManager.createRef(scopeName, attributeName);
         builder.setInsertionPoint(op);
         auto replacement = builder.create<tuples::GetColumnOp>(loc, builder.getI1Type(), markAttrRef, surroundingOperator.getLambdaRegion().getArgument(0));
         std::vector<mlir::Value> logicalUsers;
         std::function<void(mlir::Operation*)> checkUser = [&](mlir::Operation* op) {
            for (auto user : op->getUsers()) {
               auto orOp = mlir::dyn_cast_or_null<db::OrOp>(*user);
               if (orOp) {
                  logicalUsers.push_back(orOp);
                  checkUser(orOp);
               }
               auto andOp = mlir::dyn_cast_or_null<db::AndOp>(*user);
               if (andOp) {
                  logicalUsers.push_back(andOp);
                  checkUser(andOp);
               }
               auto notOp = mlir::dyn_cast_or_null<db::NotOp>(*user);
               if (notOp) {
                  logicalUsers.push_back(notOp);
                  checkUser(notOp);
               }
            }
         };
         checkUser(op);
         op->replaceAllUsesWith(replacement);
         for (auto user : logicalUsers) {
            bool notNull = true;
            auto* op = user.getDefiningOp();
            for (auto val : op->getOperands()) {
               if (val.getType() != builder.getI1Type()) {
                  notNull = false;
               }
            }
            if (notNull) {
               op->getResult(0).setType(builder.getI1Type());
            }
         }
         op->erase();
         surroundingOperator->setOperand(0, markJoin->getResult(0));
      }
   }
   void runOnOperation() override {
      auto& attributeManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      using namespace mlir;
      getOperation().walk([&](mlir::Operation* op) {
         TupleLamdaOperator surroundingOperator = op->getParentOfType<TupleLamdaOperator>();
         if (!surroundingOperator) {
            return;
         }
         Value treeVal = surroundingOperator->getOperand(0);
         if (auto getscalarop = mlir::dyn_cast_or_null<relalg::GetScalarOp>(op)) {
            OpBuilder builder(surroundingOperator);
            std::string scopeName = attributeManager.getUniqueScope("singlejoin");
            std::string attributeName = "sjattr";
            auto before = getscalarop.getAttr();
            auto fromExisting = ArrayAttr::get(&getContext(), {before});

            auto newAttrType = getscalarop.getType();
            auto newDef = attributeManager.createDef(scopeName, attributeName, fromExisting);
            if (!mlir::isa<db::NullableType>(newAttrType)) {
               newAttrType = db::NullableType::get(builder.getContext(), newAttrType);
            }
            newDef.getColumn().type = newAttrType;

            auto mapping = ArrayAttr::get(&getContext(), {newDef});
            auto singleJoin = builder.create<relalg::SingleJoinOp>(getscalarop->getLoc(), tuples::TupleStreamType::get(builder.getContext()), treeVal, getscalarop.getRel(), mapping);
            singleJoin.initPredicate();
            builder.setInsertionPoint(getscalarop);
            mlir::Value replacement = builder.create<tuples::GetColumnOp>(getscalarop->getLoc(), newAttrType, attributeManager.createRef(scopeName, attributeName), surroundingOperator.getLambdaRegion().getArgument(0));
            getscalarop.replaceAllUsesWith(replacement);
            getscalarop->erase();
            treeVal = singleJoin;
            surroundingOperator->setOperand(0, treeVal);
         } else if (auto getlistop = mlir::dyn_cast_or_null<relalg::GetListOp>(op)) {
            OpBuilder builder(surroundingOperator);
            std::string scopeName = attributeManager.getUniqueScope("collectionjoin");
            std::string attributeName = "collattr";
            auto fromAttrs = getlistop.getCols();

            auto newDef = attributeManager.createDef(scopeName, attributeName);
            newDef.getColumn().type = getlistop.getType();
            auto collectionJoin = builder.create<relalg::CollectionJoinOp>(getlistop->getLoc(), tuples::TupleStreamType::get(builder.getContext()), fromAttrs, newDef, treeVal, getlistop.getRel());
            collectionJoin.initPredicate();
            builder.setInsertionPoint(getlistop);
            Operation* replacement = builder.create<tuples::GetColumnOp>(getlistop->getLoc(), getlistop.getType(), attributeManager.createRef(scopeName, attributeName), surroundingOperator.getLambdaRegion().getArgument(0));
            getlistop.replaceAllUsesWith(replacement);
            getlistop->erase();
            treeVal = collectionJoin;
            surroundingOperator->setOperand(0, treeVal);
         } else if (auto existsop = mlir::dyn_cast_or_null<relalg::ExistsOp>(op)) {
            handleScalarBoolOp(existsop->getLoc(), surroundingOperator, op, mlir::cast<Operator>(existsop.getRel().getDefiningOp()), [](auto) {});
         }
         else if (auto inop = mlir::dyn_cast_or_null<db::OneOfOp>(op)) {
            OpBuilder builder(surroundingOperator);
            auto vals = inop.getVals();
            std::vector<mlir::Attribute> rows;
            for (auto v : vals) {
               auto c = mlir::dyn_cast_or_null<db::ConstantOp>(v.getDefiningOp());
               if (!c) {
                  return;
               }
               rows.push_back(builder.getArrayAttr({c.getValue()}));
            }

            static size_t constRelId = 0;
            std::string symName = "implConstrel" + std::to_string(constRelId++);;
            std::string columnName = "const0";
            auto attrDef = attributeManager.createDef(symName, columnName);
            attrDef.getColumn().type = (*vals.begin()).getType();
            auto constRel = builder.create<relalg::ConstRelationOp>(builder.getUnknownLoc(), builder.getArrayAttr(std::vector<mlir::Attribute>{attrDef}), builder.getArrayAttr(rows));

            const auto* attr = *constRel.getAvailableColumns().begin();
            auto searchInAttr = attributeManager.createRef(attr);
            handleScalarBoolOp(inop->getLoc(), surroundingOperator, op, constRel, [&](PredicateOperator predicateOperator) {
               predicateOperator.addPredicate([&](Value tuple, OpBuilder& builder) {
                  mlir::IRMapping mapping;
                  mapping.map(surroundingOperator.getLambdaArgument(), predicateOperator.getPredicateArgument());
                  relalg::detail::inlineOpIntoBlock(inop.getVal().getDefiningOp(), surroundingOperator.getOperation(), &predicateOperator.getPredicateBlock(), mapping);
                  auto val = mapping.lookup(inop.getVal());
                  auto otherVal = builder.create<tuples::GetColumnOp>(inop->getLoc(), searchInAttr.getColumn().type, searchInAttr, tuple);
                  Value predicate = builder.create<db::CmpOp>(inop->getLoc(), db::DBCmpPredicate::eq, val, otherVal);
                  return predicate;
               });
            });
         }
         else if (auto inop = mlir::dyn_cast_or_null<relalg::InOp>(op)) {
            Operator relOperator = mlir::cast<Operator>(inop.getRel().getDefiningOp());
            //get attribute of relation to search in
            const auto* attr = *relOperator.getAvailableColumns().begin();
            auto searchInAttr = attributeManager.createRef(attr);
            handleScalarBoolOp(inop->getLoc(), surroundingOperator, op, relOperator, [&](PredicateOperator predicateOperator) {
               predicateOperator.addPredicate([&](Value tuple, OpBuilder& builder) {
                  mlir::IRMapping mapping;
                  mapping.map(surroundingOperator.getLambdaArgument(), predicateOperator.getPredicateArgument());
                  relalg::detail::inlineOpIntoBlock(inop.getVal().getDefiningOp(), surroundingOperator.getOperation(), &predicateOperator.getPredicateBlock(), mapping);
                  auto val = mapping.lookup(inop.getVal());
                  auto otherVal = builder.create<tuples::GetColumnOp>(inop->getLoc(), searchInAttr.getColumn().type, searchInAttr, tuple);
                  Value predicate = builder.create<db::CmpOp>(inop->getLoc(), db::DBCmpPredicate::eq, val, otherVal);
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

std::unique_ptr<mlir::Pass> relalg::createImplicitToExplicitJoinsPass() { return std::make_unique<ImplicitToExplicitJoins>(); }
