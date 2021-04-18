
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <list>
#include <unordered_set>

namespace {

class Unnesting : public mlir::PassWrapper<Unnesting, mlir::FunctionPass> {
   Operator getFirstOfTree(Operator tree) {
      Operator currFirst = tree;
      for (auto child : tree.getChildren()) {
         mlir::Operation* otherFirst = getFirstOfTree(child);
         if (otherFirst->isBeforeInBlock(currFirst)) {
            currFirst = otherFirst;
         }
      }
      return currFirst;
   }

   void handleChildren(Operator d, Operator others) {
      llvm::SmallVector<Operator, 4> newChildren;
      for (auto childOp : others.getChildren()) {
         newChildren.push_back(pushDependJoinDown(d, childOp));
      }
      others.setChildren(newChildren);
   }
   Operator pushDependJoinDown(Operator d, Operator op) {
      auto availableD = d.getAvailableAttributes();

      using namespace mlir::relalg;
      auto relType = RelationType::get(&getContext());
      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(op.getOperation());
      return ::llvm::TypeSwitch<mlir::Operation*, Operator>(op.getOperation())
         .Case<mlir::relalg::BaseTableOp, mlir::relalg::ConstRelationOp>([&](Operator baserelation) {
            return builder.create<CrossProductOp>(builder.getUnknownLoc(), relType, baserelation.asRelation(), d.asRelation()).getOperation();
         })
         .Case<CrossProductOp>([&](Operator cp) {
            llvm::SmallVector<Operator, 4> newChildren;
            bool pushedDownAny = false;
            for (auto childOp : cp.getChildren()) {
               if (!childOp.getFreeAttributes().intersects(availableD)) {
                  newChildren.push_back(childOp);
               } else {
                  pushedDownAny = true;
                  newChildren.push_back(pushDependJoinDown(d, childOp));
               }
            }
            if (!pushedDownAny) {
               newChildren[0] = pushDependJoinDown(d, newChildren[0]);
            }
            cp.setChildren(newChildren);
            return cp;
         })
         .Case<AggregationOp>([&](AggregationOp projection) {
            handleChildren(d, projection);
            projection->setAttr("group_by_attrs", Attributes::fromArrayAttr(projection.group_by_attrs()).insert(availableD).asRefArrayAttr(&getContext()));
            return projection;
         })
         .Case<ProjectionOp>([&](ProjectionOp projection) {
            handleChildren(d, projection);
            projection->setAttr("attrs", Attributes::fromArrayAttr(projection.attrs()).insert(availableD).asRefArrayAttr(&getContext()));
            return projection;
         })
         .Case<BinaryOperator>([&](BinaryOperator join) {
            if (mlir::relalg::detail::isJoin(join.getOperation())) {
               auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
               auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
               auto freeRight = right.getFreeAttributes();
               auto pushDownLeft = left.getFreeAttributes().intersects(availableD);
               auto pushDownRight = right.getFreeAttributes().intersects(availableD);
               bool renameRight = true;
               if (!mlir::isa<InnerJoinOp>(join.getOperation()) && !mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  JoinDirection joinDirection = symbolizeJoinDirection(join->getAttr("join_direction").dyn_cast_or_null<mlir::IntegerAttr>().getInt()).getValue();
                  switch (joinDirection) {
                     case JoinDirection::left:
                        if (pushDownRight) {
                           pushDownLeft = true;
                        }
                        break;
                     case JoinDirection::right:
                        if (pushDownLeft) {
                           pushDownRight = true;
                           renameRight = false;
                        }
                        break;
                  }
               } else if (mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  if (pushDownLeft || pushDownRight) {
                     pushDownLeft = true;
                     pushDownRight = true;
                  }
               }
               if (!pushDownLeft && !pushDownRight) {
                  //handle case when no pushdown would be necessary
                  return mlir::dyn_cast_or_null<Operator>(builder.create<CrossProductOp>(builder.getUnknownLoc(), relType, mlir::dyn_cast_or_null<Operator>(join.getOperation()).asRelation(), d.asRelation()).getOperation());

               } else {
                  handleJoin(join, pushDownLeft ? pushDependJoinDown(d, left) : left, pushDownRight ? pushDependJoinDown(d, right) : right, pushDownLeft && pushDownRight, renameRight, availableD);
                  return mlir::dyn_cast_or_null<Operator>(join.getOperation());
               }
            } else {
               handleChildren(d, mlir::dyn_cast_or_null<Operator>(join.getOperation()));
               return mlir::dyn_cast_or_null<Operator>(join.getOperation());
            }
         })
         .Default([&](Operator others) {
            handleChildren(d, others);
            return others;
         });
   }
   void handleJoin(BinaryOperator join, Operator newLeft, Operator newRight, bool joinDependent, bool renameRight, mlir::relalg::Attributes& dependentAttributes) {
      using namespace mlir;
      auto relType = relalg::RelationType::get(&getContext());
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      Operator joinAsOperator = mlir::dyn_cast_or_null<Operator>(join.getOperation());
      mlir::OpBuilder builder(join.getOperation());
      if (joinDependent) {
         Operator toRename = renameRight ? newRight : newLeft;
         std::unordered_map<relalg::RelationalAttribute*, relalg::RelationalAttribute*> renamed;
         std::string scope = attributeManager.getUniqueScope("renaming");
         attributeManager.setCurrentScope(scope);
         std::vector<Attribute> renamingDefsAsAttr;
         size_t i = 0;
         for (auto* attr : dependentAttributes) {
            auto def = attributeManager.createDef("renamed" + std::to_string(i++), builder.getArrayAttr({attributeManager.createRef(attr)}));
            renamingDefsAsAttr.push_back(def);
            def.getRelationalAttribute().type = attr->type;
            renamed.insert({attr, &def.getRelationalAttribute()});
         }
         Operator renamingop = builder.create<relalg::RenamingOp>(builder.getUnknownLoc(), relType, toRename->getResult(0), scope, builder.getArrayAttr(renamingDefsAsAttr));
         for (auto* attr : dependentAttributes) {
            mlir::dyn_cast_or_null<PredicateOperator>(join.getOperation()).addPredicate([&](Value tuple, OpBuilder& builder) {
               auto attrefDependent = attributeManager.createRef(renamed[attr]);
               Value valLeft = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attributeManager.createRef(attr), tuple);
               Value valRight = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attrefDependent, tuple);
               Value cmpEq = builder.create<db::CmpOp>(builder.getUnknownLoc(), db::DBCmpPredicate::eq, valLeft, valRight);
               return cmpEq;
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
   void runOnFunction() override {
      using namespace mlir;
      getFunction()->walk([&](BinaryOperator binaryOperator) {
         if (!mlir::relalg::detail::isJoin(binaryOperator.getOperation())) return;
         if (!mlir::relalg::detail::isDependentJoin(binaryOperator.getOperation())) return;
         auto left = mlir::dyn_cast_or_null<Operator>(binaryOperator.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(binaryOperator.rightChild());
         auto dependentLeft = left.getFreeAttributes().intersect(right.getAvailableAttributes());
         auto dependentRight = right.getFreeAttributes().intersect(left.getAvailableAttributes());
         if (!dependentLeft.empty() && !dependentRight.empty()) {
            return;
         }
         mlir::relalg::Attributes dependentAttributes = dependentLeft;
         dependentAttributes.insert(dependentRight);
         bool leftProvides = dependentLeft.empty();
         Operator providerChild = leftProvides ? left : right;
         Operator dependentChild = leftProvides ? right : left;
         OpBuilder builder(binaryOperator.getOperation());
         providerChild.moveSubTreeBefore(getFirstOfTree(dependentChild));
         builder.setInsertionPointAfter(providerChild);
         auto proj = builder.create<relalg::ProjectionOp>(builder.getUnknownLoc(), relalg::RelationType::get(&getContext()), relalg::SetSemantic::distinct, providerChild.asRelation(), dependentAttributes.asRefArrayAttr(&getContext()));
         Operator d = mlir::dyn_cast_or_null<Operator>(proj.getOperation());
         Operator unnestedChild = pushDependJoinDown(d, dependentChild);
         Operator newLeft = leftProvides ? providerChild : unnestedChild;
         Operator newRight = leftProvides ? unnestedChild : providerChild;
         handleJoin(binaryOperator, newLeft, newRight, true, leftProvides, dependentAttributes);
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createUnnestingPass() { return std::make_unique<Unnesting>(); }
} // end namespace relalg
} // end namespace mlir