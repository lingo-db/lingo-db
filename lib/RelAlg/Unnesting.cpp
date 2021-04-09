
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
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;
   attribute_set intersect(const attribute_set& a, const attribute_set& b) {
      attribute_set result;
      for (auto* x : a) {
         if (b.contains(x)) {
            result.insert(x);
         }
      }
      return result;
   }
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
   void moveTreeBefore(Operator tree, mlir::Operation* before) {
      tree->moveBefore(before);
      for (auto child : tree.getChildren()) {
         moveTreeBefore(child, tree);
      }
   }
   mlir::ArrayAttr addAttributes(mlir::ArrayAttr current, attribute_set toAdd) {
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      llvm::SmallVector<mlir::Attribute, 8> attributes;
      for (auto attr : current) {
         auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>();
         toAdd.insert(&attrRef.getRelationalAttribute());
      }
      for (auto* attr : toAdd) {
         attributes.push_back(attributeManager.createRef(attr));
      }
      return mlir::ArrayAttr::get(&getContext(), attributes);
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
               if (intersect(childOp.getFreeAttributes(), availableD).empty()) {
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
            projection->setAttr("group_by_attrs", addAttributes(projection.group_by_attrs(), availableD));
            return projection;
         })
         .Case<ProjectionOp>([&](ProjectionOp projection) {
            handleChildren(d, projection);
            projection->setAttr("attrs", addAttributes(projection.attrs(), availableD));
            return projection;
         })
         .Case<BinaryOperator>([&](BinaryOperator join) {
            if (mlir::relalg::detail::isJoin(join.getOperation())) {
               auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
               auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
               auto freeLeft = left.getFreeAttributes();
               auto freeRight = right.getFreeAttributes();
               auto dependentLeft = intersect(freeLeft, availableD);
               auto dependentRight = intersect(freeRight, availableD);

               bool pushDownLeft = !dependentLeft.empty();
               bool pushDownRight = !dependentRight.empty();
               bool renameRight = true;
               if (!mlir::isa<InnerJoinOp>(join.getOperation()) || !mlir::isa<FullOuterJoinOp>(join.getOperation())) {
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
               } else if (!mlir::isa<FullOuterJoinOp>(join.getOperation())) {
                  if (pushDownLeft || pushDownRight) {
                     pushDownLeft = true;
                     pushDownRight = true;
                  }
               }
               handleJoin(join, pushDownLeft ? pushDependJoinDown(d, left) : left, pushDownRight ? pushDependJoinDown(d, right) : right, pushDownLeft && pushDownRight, renameRight, availableD);
               return mlir::dyn_cast_or_null<Operator>(join.getOperation());
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
   void addPredicate(TupleLamdaOperator lambdaOperator, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> predicateProducer) {
      auto* terminator = lambdaOperator.getLambdaBlock().getTerminator();
      mlir::OpBuilder builder(terminator);
      auto additionalPred = predicateProducer(lambdaOperator.getLambdaArgument(), builder);
      if (terminator->getNumOperands() > 0) {
         mlir::Value oldValue = terminator->getOperand(0);
         bool nullable = oldValue.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() || additionalPred.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable();
         mlir::Value anded = builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext(), nullable), mlir::ValueRange({oldValue, additionalPred}));
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), anded);
      } else {
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), additionalPred);
      }
      terminator->remove();
      terminator->destroy();
   }
   void handleJoin(BinaryOperator join, Operator newLeft, Operator newRight, bool joinDependent, bool renameRight, attribute_set& dependentAttributes) {
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
         std::vector<relalg::RelationalAttributeDefAttr> renamingDefs;
         std::vector<Attribute> renamingDefsAsAttr;
         size_t i = 0;
         for (auto* attr : dependentAttributes) {
            std::vector<Attribute> fromExistingVec = {attributeManager.createRef(attr)};
            auto def = attributeManager.createDef("renamed" + std::to_string(i++), builder.getArrayAttr(fromExistingVec));
            renamingDefs.push_back(def);
            renamingDefsAsAttr.push_back(def);
            def.getRelationalAttribute().type = attr->type;
            renamed.insert({attr, &def.getRelationalAttribute()});
         }
         Operator renamingop = builder.create<relalg::RenamingOp>(builder.getUnknownLoc(), relType, toRename->getResult(0), scope, builder.getArrayAttr(renamingDefsAsAttr));
         addPredicate(mlir::dyn_cast_or_null<TupleLamdaOperator>(join.getOperation()), [&](Value tuple, OpBuilder& builder) {
            std::vector<Value> toAnd;
            bool anyNullable = false;
            for (auto* attr : dependentAttributes) {
               auto attrefDependent = attributeManager.createRef(renamed[attr]);
               Value valLeft = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attributeManager.createRef(attr), tuple);
               Value valRight = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attrefDependent, tuple);
               Value cmpEq = builder.create<db::CmpOp>(builder.getUnknownLoc(), db::DBCmpPredicate::eq, valLeft, valRight);
               anyNullable |= cmpEq.getType().dyn_cast_or_null<db::DBType>().isNullable();
               toAnd.push_back(cmpEq);
            }
            if (toAnd.size() == 1) {
               return toAnd[0];
            } else {
               Value anded = builder.create<db::AndOp>(builder.getUnknownLoc(), db::BoolType::get(builder.getContext(), anyNullable), toAnd);
               return anded;
            }
         });
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
      auto relType = relalg::RelationType::get(&getContext());

      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();

      getFunction()->walk([&](BinaryOperator binaryOperator) {
         if (!mlir::relalg::detail::isJoin(binaryOperator.getOperation())) return;
         if (!mlir::relalg::detail::isDependentJoin(binaryOperator.getOperation())) return;
         auto left = mlir::dyn_cast_or_null<Operator>(binaryOperator.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(binaryOperator.rightChild());
         auto availableLeft = left.getAvailableAttributes();
         auto availableRight = right.getAvailableAttributes();
         auto freeLeft = left.getFreeAttributes();
         auto freeRight = right.getFreeAttributes();
         auto dependentLeft = intersect(freeLeft, availableRight);
         auto dependentRight = intersect(freeRight, availableLeft);
         if (!dependentLeft.empty() && !dependentRight.empty()) {
            return;
         }
         Operator dependentChild;
         Operator providerChild;
         attribute_set dependentAttributes;
         bool dependentOnRight = true;
         if (!dependentLeft.empty()) {
            dependentChild = left;
            providerChild = right;
            dependentAttributes = dependentLeft;
            dependentOnRight = false;
         } else {
            dependentChild = right;
            providerChild = left;
            dependentAttributes = dependentRight;
         }

         OpBuilder builder(binaryOperator.getOperation());
         std::vector<Attribute> dependentRefsAsAttr;
         for (auto* attr : dependentAttributes) {
            dependentRefsAsAttr.push_back(attributeManager.createRef(attr));
         }

         moveTreeBefore(providerChild, getFirstOfTree(dependentChild));
         builder.setInsertionPointAfter(providerChild);
         auto proj = builder.create<relalg::ProjectionOp>(builder.getUnknownLoc(), relType, relalg::SetSemantic::distinct, providerChild->getResult(0), builder.getArrayAttr(dependentRefsAsAttr));
         Operator d = mlir::dyn_cast_or_null<Operator>(proj.getOperation());
         Operator unnestedChild = pushDependJoinDown(d, dependentChild);

         Operator newLeft, newRight;
         if (dependentOnRight) {
            newLeft = providerChild;
            newRight = unnestedChild;
         } else {
            newLeft = unnestedChild;
            newRight = providerChild;
         }
         handleJoin(binaryOperator, newLeft, newRight, true, dependentOnRight, dependentAttributes);
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createUnnestingPass() { return std::make_unique<Unnesting>(); }
} // end namespace relalg
} // end namespace mlir