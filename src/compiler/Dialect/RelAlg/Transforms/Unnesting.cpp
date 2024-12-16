
#include "lingodb/compiler/Dialect/DB/IR/DBOps.h"
#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgOps.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOps.h"

#include "lingodb/compiler/Dialect/RelAlg/Passes.h"
#include "mlir/IR/IRMapping.h"

#include "lingodb/compiler/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <list>
#include <unordered_map>

namespace {
using namespace lingodb::compiler::dialect;

class Unnesting : public mlir::PassWrapper<Unnesting, mlir::OperationPass<mlir::func::FuncOp>> {
   virtual llvm::StringRef getArgument() const override { return "relalg-unnesting"; }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Unnesting)
   private:
   Operator getFirstOfTree(Operator tree) {
      Operator currFirst = tree;
      for (auto child : tree.getChildren()) {
         mlir::Operation* otherFirst = getFirstOfTree(child);
         if (otherFirst->isBeforeInBlock(currFirst)) {
            currFirst = mlir::cast<Operator>(otherFirst);
         }
      }
      return currFirst;
   }

   void handleChildren(mlir::Location loc, Operator d, Operator others) {
      llvm::SmallVector<Operator, 4> newChildren;
      for (auto childOp : others.getChildren()) {
         newChildren.push_back(pushDependJoinDown(loc, d, childOp));
      }
      others.setChildren(newChildren);
   }
   size_t countUses(Operator o) {
      size_t uses = 0;
      for (auto& u : o->getUses()) uses++; // NOLINT(clang-diagnostic-unused-variable)
      return uses;
   }

   Operator pushDependJoinDown(mlir::Location loc, Operator d, Operator op) {
      using namespace relalg;
      auto availableD = d.getAvailableColumns();
      auto relType = tuples::TupleStreamType::get(&getContext());
      if (!op.getFreeColumns().intersects(availableD)) {
         mlir::OpBuilder builder(&getContext());
         builder.setInsertionPointAfter(op.getOperation());
         return mlir::cast<Operator>(builder.create<CrossProductOp>(loc, relType, op.asRelation(), d.asRelation()).getOperation());
      }
      assert(countUses(op) <= 1);

      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(op.getOperation());
      return ::llvm::TypeSwitch<mlir::Operation*, Operator>(op.getOperation())
         .Case<relalg::BaseTableOp, relalg::ConstRelationOp>([&](Operator baserelation) {
            return builder.create<CrossProductOp>(loc, relType, baserelation.asRelation(), d.asRelation()).getOperation();
         })
         .Case<CrossProductOp>([&](Operator cp) {
            llvm::SmallVector<Operator, 4> newChildren;
            bool pushedDownAny = false;
            for (auto childOp : cp.getChildren()) {
               if (!childOp.getFreeColumns().intersects(availableD)) {
                  newChildren.push_back(childOp);
               } else {
                  pushedDownAny = true;
                  newChildren.push_back(pushDependJoinDown(loc, d, childOp));
               }
            }
            if (!pushedDownAny) {
               newChildren[0] = pushDependJoinDown(loc, d, newChildren[0]);
            }
            cp.setChildren(newChildren);
            return cp;
         })
         .Case<AggregationOp>([&](AggregationOp projection) {
            handleChildren(loc, d, projection);
            projection.setGroupByColsAttr(ColumnSet::fromArrayAttr(projection.getGroupByCols()).insert(availableD).asRefArrayAttr(&getContext()));
            return projection;
         })
         .Case<ProjectionOp>([&](ProjectionOp projection) {
            handleChildren(loc, d, projection);
            projection.setColsAttr(ColumnSet::fromArrayAttr(projection.getCols()).insert(availableD).asRefArrayAttr(&getContext()));
            return projection;
         })
         .Case<BinaryOperator>([&](BinaryOperator join) {
            if (relalg::detail::isJoin(join.getOperation())) {
               auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
               auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
               auto freeRight = right.getFreeColumns();
               auto pushDownLeft = left.getFreeColumns().intersects(availableD);
               auto pushDownRight = right.getFreeColumns().intersects(availableD);
               if (!pushDownLeft && !pushDownRight && mlir::cast<Operator>(join.getOperation()).getFreeColumns().intersects(availableD)) {
                  pushDownLeft = true;
               }
               bool renameRight = true;
               if (!mlir::isa<InnerJoinOp>(join.getOperation()) && !mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  if (pushDownRight) {
                     pushDownLeft = true;
                  }
               } else if (mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  if (pushDownLeft || pushDownRight) {
                     pushDownLeft = true;
                     pushDownRight = true;
                  }
               }
               if (!pushDownLeft && !pushDownRight) {
                  //handle case when no pushdown would be necessary
                  return mlir::dyn_cast_or_null<Operator>(builder.create<CrossProductOp>(loc, relType, mlir::dyn_cast_or_null<Operator>(join.getOperation()).asRelation(), d.asRelation()).getOperation());

               } else {
                  handleJoin(loc, join, pushDownLeft ? pushDependJoinDown(loc, d, left) : left, pushDownRight ? pushDependJoinDown(loc, d, right) : right, pushDownLeft && pushDownRight, renameRight, availableD);
                  return mlir::dyn_cast_or_null<Operator>(join.getOperation());
               }
            } else {
               handleChildren(loc, d, mlir::dyn_cast_or_null<Operator>(join.getOperation()));
               return mlir::dyn_cast_or_null<Operator>(join.getOperation());
            }
         })
         .Default([&](mlir::Operation* others) -> Operator {
            handleChildren(loc, d, mlir::cast<Operator>(others));
            return mlir::cast<Operator>(others);
         });
   }
   void handleJoin(mlir::Location loc, BinaryOperator join, Operator newLeft, Operator newRight, bool joinDependent, bool renameRight, relalg::ColumnSet& dependentAttributes) {
      using namespace mlir;
      auto relType = tuples::TupleStreamType::get(&getContext());
      auto& attributeManager = getContext().getLoadedDialect<tuples::TupleStreamDialect>()->getColumnManager();
      Operator joinAsOperator = mlir::dyn_cast_or_null<Operator>(join.getOperation());
      mlir::OpBuilder builder(join.getOperation());
      if (joinDependent) {
         Operator toRename = renameRight ? newRight : newLeft;
         std::unordered_map<const tuples::Column*, const tuples::Column*> renamed;
         std::string scope = attributeManager.getUniqueScope("renaming");
         std::vector<Attribute> renamingDefsAsAttr;
         size_t i = 0;
         for (const auto* attr : dependentAttributes) {
            auto def = attributeManager.createDef(scope, "renamed" + std::to_string(i++), builder.getArrayAttr({attributeManager.createRef(attr)}));
            renamingDefsAsAttr.push_back(def);
            def.getColumn().type = attr->type;
            renamed.insert({attr, &def.getColumn()});
         }
         Operator renamingop = builder.create<relalg::RenamingOp>(loc, relType, toRename->getResult(0), builder.getArrayAttr(renamingDefsAsAttr));
         for (const auto* attr : dependentAttributes) {
            mlir::dyn_cast_or_null<PredicateOperator>(join.getOperation()).addPredicate([&](Value tuple, OpBuilder& builder) {
               auto attrefDependent = attributeManager.createRef(renamed[attr]);
               Value valLeft = builder.create<tuples::GetColumnOp>(loc, attr->type, attributeManager.createRef(attr), tuple);
               Value valRight = builder.create<tuples::GetColumnOp>(loc, attr->type, attrefDependent, tuple);
               return builder.create<db::CmpOp>(loc, db::DBCmpPredicate::isa, valLeft, valRight);
            });
         }

         if (renameRight) {
            newRight = renamingop;
         } else {
            newLeft = renamingop;
         }
      }
      joinAsOperator.setChildren({newLeft, newRight});
   }
   bool collectSimpleDependencies(Operator op, relalg::ColumnSet& attributes, std::vector<relalg::SelectionOp>& selectionOps) {
      if (!op.getFreeColumns().intersects(attributes)) {
         return true;
      }
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op.getOperation())
         .Case<relalg::BaseTableOp, relalg::ConstRelationOp>([&](Operator baserelation) {
            return true;
         })
         .Case<relalg::CrossProductOp>([&](Operator cp) {
            auto subOps = cp.getAllSubOperators();
            return collectSimpleDependencies(subOps[0], attributes, selectionOps) && collectSimpleDependencies(subOps[1], attributes, selectionOps);
         })
         .Case<relalg::SelectionOp>([&](relalg::SelectionOp sel) {
            auto x = sel.getUsedColumns();
            x.remove(sel.getAvailableColumns());
            if (x.isSubsetOf(attributes)) {
               selectionOps.push_back(sel);
            }
            return collectSimpleDependencies(sel.getChildren()[0], attributes, selectionOps);
         })
         .Case<relalg::AggregationOp>([&](relalg::AggregationOp aggregationOp) {
            if (aggregationOp.getUsedColumns().intersects(attributes)) {
               return false;
            }
            std::vector<relalg::SelectionOp> extractedSelections;
            if (!collectSimpleDependencies(aggregationOp.getChildren()[0], attributes, extractedSelections)) {
               return false;
            }

            std::vector<db::CmpOp> toAnalyze;
            for (auto sel : extractedSelections) {
               auto returnOp = mlir::cast<tuples::ReturnOp>(sel.getPredicateBlock().getTerminator());
               if (auto cmpOp = mlir::dyn_cast_or_null<db::CmpOp>(returnOp.getResults()[0].getDefiningOp())) {
                  if (cmpOp.getPredicate() == db::DBCmpPredicate::eq) {
                     toAnalyze.push_back(cmpOp);
                  } else {
                     return false;
                  }
               } else {
                  return false;
               }
            }
            relalg::ColumnSet groupByColumns = relalg::ColumnSet::fromArrayAttr(aggregationOp.getGroupByCols());
            for (auto cmpOp : toAnalyze) {
               auto leftColDef = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getLeft().getDefiningOp());
               auto rightColDef = mlir::dyn_cast_or_null<tuples::GetColumnOp>(cmpOp.getRight().getDefiningOp());
               if (!leftColDef || !rightColDef) return false;
               if (attributes.contains(&rightColDef.getAttr().getColumn())) {
                  std::swap(leftColDef, rightColDef);
               }
               groupByColumns.insert(&rightColDef.getAttr().getColumn());
            }
            selectionOps.insert(selectionOps.end(), extractedSelections.begin(), extractedSelections.end());
            aggregationOp.setGroupByColsAttr(groupByColumns.asRefArrayAttr(&getContext()));
            return true;
         })
         .Case<BinaryOperator>([&](BinaryOperator join) {
            return false;
         })
         .Case<relalg::MapOp>([&](relalg::MapOp mapOp) {
            if (mapOp.getUsedColumns().intersects(attributes)) {
               return false;
            }
            return collectSimpleDependencies(mapOp.getChildren()[0], attributes, selectionOps);
         })
         .Default([&](mlir::Operation* others) {
            return false;
         });
   }
   void combine(mlir::Location loc, std::vector<relalg::SelectionOp> selectionOps, PredicateOperator lower) {
      using namespace mlir;
      auto lowerTerminator = mlir::dyn_cast_or_null<tuples::ReturnOp>(lower.getPredicateBlock().getTerminator());

      OpBuilder builder(lower);

      builder.setInsertionPointToEnd(&lower.getPredicateBlock());
      std::vector<mlir::Value> values;
      bool nullable = false;
      if (!lowerTerminator.getResults().empty()) {
         Value lowerPredVal = lowerTerminator.getResults()[0];
         nullable |= mlir::isa<db::NullableType>(lowerPredVal.getType());
         values.push_back(lowerPredVal);
      }
      for (auto selOp : selectionOps) {
         auto higherTerminator = mlir::dyn_cast_or_null<tuples::ReturnOp>(selOp.getPredicateBlock().getTerminator());
         Value higherPredVal = higherTerminator.getResults()[0];
         mlir::IRMapping mapping;
         mapping.map(selOp.getPredicateArgument(), lower.getPredicateArgument());
         relalg::detail::inlineOpIntoBlock(higherPredVal.getDefiningOp(), higherPredVal.getDefiningOp()->getParentOp(), &lower.getPredicateBlock(), mapping);
         nullable |= mlir::isa<db::NullableType>(higherPredVal.getType());
         values.push_back(mapping.lookup(higherPredVal));
      }
      mlir::Type resType = builder.getI1Type();
      if (nullable) {
         resType = db::NullableType::get(builder.getContext(), resType);
      }
      mlir::Value combined = builder.create<db::AndOp>(loc, resType, values);
      builder.create<tuples::ReturnOp>(loc, combined);
      lowerTerminator->erase();
   }
   bool trySimpleUnnesting(BinaryOperator binaryOperator) {
      if (auto predicateOperator = mlir::dyn_cast_or_null<PredicateOperator>(binaryOperator.getOperation())) {
         auto left = mlir::dyn_cast_or_null<Operator>(binaryOperator.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(binaryOperator.rightChild());
         auto dependentLeft = left.getFreeColumns().intersect(right.getAvailableColumns());
         auto dependentRight = right.getFreeColumns().intersect(left.getAvailableColumns());
         relalg::ColumnSet dependentAttributes = dependentLeft;
         dependentAttributes.insert(dependentRight);
         bool leftProvides = dependentLeft.empty();
         Operator providerChild = leftProvides ? left : right;
         Operator dependentChild = leftProvides ? right : left;
         relalg::ColumnSet providedAttrs = providerChild.getAvailableColumns();
         std::vector<relalg::SelectionOp> selectionOps;
         if (!collectSimpleDependencies(dependentChild, providedAttrs, selectionOps)) {
            return false;
         }
         combine(binaryOperator->getLoc(), selectionOps, predicateOperator);
         for (auto selOp : selectionOps) {
            selOp.replaceAllUsesWith(selOp.getRel());
            selOp->erase();
         }
         return true;
      }
      return false;
   }
   void runOnOperation() override {
      using namespace mlir;
      getOperation()->walk([&](BinaryOperator binaryOperator) {
         if (!relalg::detail::isJoin(binaryOperator.getOperation())) return;
         if (!relalg::detail::isDependentJoin(binaryOperator.getOperation())) return;
         auto left = mlir::dyn_cast_or_null<Operator>(binaryOperator.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(binaryOperator.rightChild());
         auto dependentLeft = left.getFreeColumns().intersect(right.getAvailableColumns());
         auto dependentRight = right.getFreeColumns().intersect(left.getAvailableColumns());
         if (!dependentLeft.empty() && !dependentRight.empty()) {
            return;
         }
         if (trySimpleUnnesting(mlir::cast<BinaryOperator>(binaryOperator.getOperation()))) {
            if (!relalg::detail::isDependentJoin(binaryOperator.getOperation())) return;
         }
         relalg::ColumnSet dependentAttributes = dependentLeft;
         dependentAttributes.insert(dependentRight);
         bool leftProvides = dependentLeft.empty();
         Operator providerChild = leftProvides ? left : right;
         Operator dependentChild = leftProvides ? right : left;
         OpBuilder builder(binaryOperator.getOperation());
         providerChild.moveSubTreeBefore(getFirstOfTree(dependentChild));
         builder.setInsertionPointAfter(providerChild);
         auto proj = builder.create<relalg::ProjectionOp>(binaryOperator->getLoc(), tuples::TupleStreamType::get(&getContext()), relalg::SetSemantic::distinct, providerChild.asRelation(), dependentAttributes.asRefArrayAttr(&getContext()));
         Operator d = mlir::dyn_cast_or_null<Operator>(proj.getOperation());
         Operator unnestedChild = pushDependJoinDown(binaryOperator->getLoc(), d, dependentChild);
         Operator newLeft = leftProvides ? providerChild : unnestedChild;
         Operator newRight = leftProvides ? unnestedChild : providerChild;
         handleJoin(binaryOperator->getLoc(), binaryOperator, newLeft, newRight, true, leftProvides, dependentAttributes);
      });
   }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> relalg::createUnnestingPass() { return std::make_unique<Unnesting>(); }
