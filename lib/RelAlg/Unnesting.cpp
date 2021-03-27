
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/ADT/TypeSwitch.h>
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>

namespace {

class Unnesting : public mlir::PassWrapper<Unnesting, mlir::FunctionPass> {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;
   attribute_set intersect(const attribute_set& a, const attribute_set& b) {
      attribute_set result;
      for (auto x : a) {
         if (b.contains(x)) {
            result.insert(x);
         }
      }
      return result;
   }
   Operator getFirstOfTree(Operator tree) {
      Operator curr_first = tree;
      for (auto child : tree.getChildren()) {
         mlir::Operation* other_first = getFirstOfTree(child);
         if (other_first->isBeforeInBlock(curr_first)) {
            curr_first = other_first;
         }
      }
      return curr_first;
   }
   void moveTreeBefore(Operator tree, mlir::Operation* before) {
      tree->moveBefore(before);
      for (auto child : tree.getChildren()) {
         moveTreeBefore(child, tree);
      }
   }
   mlir::ArrayAttr addAttributes(mlir::ArrayAttr current, attribute_set to_add) {
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      llvm::SmallVector<mlir::Attribute, 8> attributes;
      for (auto attr : current) {
         auto attr_ref = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>();
         to_add.insert(&attr_ref.getRelationalAttribute());
      }
      for (auto attr : to_add) {
         attributes.push_back(attributeManager.createRef(attr));
      }
      return mlir::ArrayAttr::get(&getContext(), attributes);
   }
   void handleChildren(Operator D,Operator others){
      llvm::SmallVector<Operator, 4> new_children;
      for (auto childOp : others.getChildren()) {
         new_children.push_back(pushDependJoinDown(D, childOp));
      }
      others.setChildren(new_children);
   }
   Operator pushDependJoinDown(Operator D, Operator op) {
      auto available_D = D.getAvailableAttributes();

      using namespace mlir::relalg;
      auto rel_type = RelationType::get(&getContext());
      mlir::OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(op.getOperation());
      return ::llvm::TypeSwitch<mlir::Operation*, Operator>(op.getOperation())
         .Case<mlir::relalg::BaseTableOp, mlir::relalg::ConstRelationOp>([&](Operator baserelation) {
            return builder.create<CrossProductOp>(builder.getUnknownLoc(), rel_type, baserelation.asRelation(), D.asRelation()).getOperation();
         })
         .Case<CrossProductOp>([&](Operator cp) {
           llvm::SmallVector<Operator, 4> new_children;
           bool pushed_down_any=false;
           for (auto childOp : cp.getChildren()) {
              if(intersect(childOp.getFreeAttributes(),available_D).empty()){
                 new_children.push_back(childOp);
              }else {
                 pushed_down_any=true;
                 new_children.push_back(pushDependJoinDown(D, childOp));
              }
           }
           if(!pushed_down_any){
              new_children[0]=pushDependJoinDown(D, new_children[0]);
           }
           cp.setChildren(new_children);
           return cp;
         })
         .Case<AggregationOp>([&](AggregationOp projection) {
            handleChildren(D,projection);
            projection->setAttr("group_by_attrs", addAttributes(projection.group_by_attrs(), available_D));
            return projection;
         })
         .Case<ProjectionOp>([&](ProjectionOp projection) {
           handleChildren(D,projection);
           projection->setAttr("attrs", addAttributes(projection.attrs(), available_D));
            return projection;
         })
         .Case<Join>([&](Join join) {
            auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
            auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
            auto free_left = left.getFreeAttributes();
            auto free_right = right.getFreeAttributes();
            auto dependent_left = intersect(free_left, available_D);
            auto dependent_right = intersect(free_right, available_D);

            bool push_down_left = !dependent_left.empty();
            bool push_down_right = !dependent_right.empty();
            bool rename_right = true;
            if (!mlir::isa<InnerJoinOp>(join.getOperation())) {
               JoinDirection joinDirection = symbolizeJoinDirection(join->getAttr("join_direction").dyn_cast_or_null<mlir::IntegerAttr>().getInt()).getValue();
               switch (joinDirection) {
                  case JoinDirection::left:
                     if (push_down_right) {
                        push_down_left = true;
                     }
                     break;
                  case JoinDirection::right:
                     if (push_down_left) {
                        push_down_right = true;
                        rename_right = false;
                     }
                     break;
                  case JoinDirection::full:
                     if (push_down_left || push_down_right) {
                        push_down_left = true;
                        push_down_right = true;
                     }
                     break;
               }
            }
            handleJoin(join, push_down_left ? pushDependJoinDown(D, left) : left, push_down_right ? pushDependJoinDown(D, right) : right, push_down_left && push_down_right, rename_right, available_D);
            return mlir::dyn_cast_or_null<Operator>(join.getOperation());
         })
         .Default([&](Operator others) {
            handleChildren(D,others);
            return others;
         });
   }
   void addPredicate(TupleLamdaOperator lambdaOperator, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> predicate_producer) {
      auto terminator = lambdaOperator.getLambdaBlock().getTerminator();
      mlir::OpBuilder builder(terminator);
      auto additional_pred = predicate_producer(lambdaOperator.getLambdaArgument(), builder);
      if (terminator->getNumOperands() > 0) {
         mlir::Value oldValue = terminator->getOperand(0);
         bool nullable = oldValue.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() || additional_pred.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable();
         mlir::Value anded = builder.create<mlir::db::AndOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext(), nullable), mlir::ValueRange({oldValue, additional_pred}));
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), anded);
      } else {
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), additional_pred);
      }
      terminator->remove();
      terminator->destroy();
   }
   void handleJoin(Join join, Operator newLeft, Operator newRight, bool join_dependent, bool rename_right, attribute_set& dependent_attributes) {
      using namespace mlir;
      auto rel_type = relalg::RelationType::get(&getContext());
      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();
      Operator joinAsOperator = mlir::dyn_cast_or_null<Operator>(join.getOperation());
      mlir::OpBuilder builder(join.getOperation());
      if (join_dependent) {
         Operator toRename = rename_right ? newRight : newLeft;
         std::unordered_map<relalg::RelationalAttribute*, relalg::RelationalAttribute*> renamed;
         std::string scope = attributeManager.getUniqueScope("renaming");
         attributeManager.setCurrentScope(scope);
         std::vector<relalg::RelationalAttributeDefAttr> renaming_defs;
         std::vector<Attribute> renaming_defs_as_attr;
         size_t i = 0;
         for (auto attr : dependent_attributes) {
            std::vector<Attribute> from_existing_vec = {attributeManager.createRef(attr)};
            auto def = attributeManager.createDef("renamed" + std::to_string(i++), builder.getArrayAttr(from_existing_vec));
            renaming_defs.push_back(def);
            renaming_defs_as_attr.push_back(def);
            def.getRelationalAttribute().type = attr->type;
            renamed.insert({attr, &def.getRelationalAttribute()});
         }
         Operator renamingop = builder.create<relalg::RenamingOp>(builder.getUnknownLoc(), rel_type, toRename->getResult(0), scope, builder.getArrayAttr(renaming_defs_as_attr));
         addPredicate(mlir::dyn_cast_or_null<TupleLamdaOperator>(join.getOperation()), [&](Value tuple, OpBuilder& builder) {
            std::vector<Value> to_and;
            bool any_nullable = false;
            for (auto attr : dependent_attributes) {
               auto attref_dependent = attributeManager.createRef(renamed[attr]);
               Value val_left = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attributeManager.createRef(attr), tuple);
               Value val_right = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), attr->type, attref_dependent, tuple);
               Value cmp_eq = builder.create<db::CmpOp>(builder.getUnknownLoc(), db::DBCmpPredicate::eq, val_left, val_right);
               any_nullable |= cmp_eq.getType().dyn_cast_or_null<db::DBType>().isNullable();
               to_and.push_back(cmp_eq);
            }
            if (to_and.size() == 1) {
               return to_and[0];
            } else {
               Value anded = builder.create<db::AndOp>(builder.getUnknownLoc(), db::BoolType::get(builder.getContext(), any_nullable), to_and);
               return anded;
            }
         });
         if (rename_right) {
            newRight = renamingop;
         } else {
            newLeft = renamingop;
         }
      }
      joinAsOperator.setChildren({newLeft, newRight});
   }
   void runOnFunction() override {
      using namespace mlir;
      auto rel_type = relalg::RelationType::get(&getContext());

      auto& attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();

      getFunction()->walk([&](Join join) {
         if (!join.isDependentJoin()) return;
         auto left = mlir::dyn_cast_or_null<Operator>(join.leftChild());
         auto right = mlir::dyn_cast_or_null<Operator>(join.rightChild());
         auto available_left = left.getAvailableAttributes();
         auto available_right = right.getAvailableAttributes();
         auto free_left = left.getFreeAttributes();
         auto free_right = right.getFreeAttributes();
         auto dependent_left = intersect(free_left, available_right);
         auto dependent_right = intersect(free_right, available_left);
         if (!dependent_left.empty() && !dependent_right.empty()) {
            return;
         }
         Operator dependent_child;
         Operator provider_child;
         attribute_set dependent_attributes;
         bool dependent_on_right = true;
         if (!dependent_left.empty()) {
            dependent_child = left;
            provider_child = right;
            dependent_attributes = dependent_left;
            dependent_on_right = false;
         } else {
            dependent_child = right;
            provider_child = left;
            dependent_attributes = dependent_right;
         }

         OpBuilder builder(join.getOperation());
         std::vector<Attribute> dependent_refs_as_attr;
         for (auto attr : dependent_attributes) {
            dependent_refs_as_attr.push_back(attributeManager.createRef(attr));
         }

         moveTreeBefore(provider_child, getFirstOfTree(dependent_child));
         builder.setInsertionPointAfter(provider_child);
         auto proj = builder.create<relalg::ProjectionOp>(builder.getUnknownLoc(), rel_type, relalg::SetSemantic::distinct, provider_child->getResult(0), builder.getArrayAttr(dependent_refs_as_attr));
         Operator D = mlir::dyn_cast_or_null<Operator>(proj.getOperation());
         Operator unnested_child = pushDependJoinDown(D, dependent_child);

         Operator newLeft, newRight;
         if (dependent_on_right) {
            newLeft = provider_child;
            newRight = unnested_child;
         } else {
            newLeft = unnested_child;
            newRight = provider_child;
         }
         handleJoin(join, newLeft, newRight, true, dependent_on_right, dependent_attributes);
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createUnnestingPass() { return std::make_unique<Unnesting>(); }
} // end namespace relalg
} // end namespace mlir