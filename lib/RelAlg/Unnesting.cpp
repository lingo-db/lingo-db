
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"

#include "mlir/Dialect/RelAlg/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <list>
#include <queue>
#include <unordered_set>
#include <mlir/Dialect/RelAlg/IR/RelAlgDialect.h>

namespace {

class Unnesting : public mlir::PassWrapper<Unnesting, mlir::FunctionPass> {
   using attribute_set = llvm::SmallPtrSet<mlir::relalg::RelationalAttribute*, 8>;
   attribute_set intersect(attribute_set& a, attribute_set& b) {
      attribute_set result;
      for (auto x : a) {
         if (b.contains(x)) {
            result.insert(x);
         }
      }
      return result;
   }
   void addPredicate(TupleLamdaOperator lambdaOperator, std::function<mlir::Value(mlir::Value, mlir::OpBuilder& builder)> predicate_producer) {
      auto terminator = lambdaOperator.getLambdaBlock().getTerminator();
      mlir::OpBuilder builder(terminator);
      auto additional_pred = predicate_producer(lambdaOperator.getLambdaArgument(), builder);
      mlir::Operation* newTerminator;
      if (terminator->getNumOperands() > 0) {
         mlir::Value oldValue = terminator->getOperand(0);
         bool nullable = oldValue.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable() || additional_pred.getType().dyn_cast_or_null<mlir::db::DBType>().isNullable();
         mlir::Value anded = builder.create<mlir::AndOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext(), nullable), oldValue, additional_pred);
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), anded);
      } else {
         builder.create<mlir::relalg::ReturnOp>(builder.getUnknownLoc(), additional_pred);
      }
      terminator->remove();
      terminator->destroy();
   }
   void runOnFunction() override {
      using namespace mlir;
      auto attributeManager = getContext().getLoadedDialect<mlir::relalg::RelAlgDialect>()->getRelationalAttributeManager();

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
         size_t joinoperandid;
         if (!dependent_left.empty()) {
            dependent_child = left;
            provider_child = right;
            dependent_attributes = dependent_left;
            joinoperandid = 0;
         } else {
            dependent_child = right;
            provider_child = left;
            dependent_attributes = dependent_right;
            joinoperandid = 1;
         }
         auto rel_type = relalg::RelationType::get(&getContext());

         OpBuilder builder(join.getOperation());
         std::unordered_map<relalg::RelationalAttribute*, relalg::RelationalAttribute*> renamed;
         std::vector<relalg::RelationalAttributeRefAttr> dependent_refs;
         std::vector<Attribute> dependent_refs_as_attr;
         for (auto attr : dependent_attributes) {
            dependent_refs.push_back(attributeManager.createRef(attr));
            dependent_refs_as_attr.push_back(attributeManager.createRef(attr));
         }
         builder.create<relalg::ProjectionOp>(builder.getUnknownLoc(), rel_type, relalg::SetSemantic::distinct, provider_child->getResult(0), builder.getArrayAttr(dependent_refs_as_attr));
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
         Value renamingop = builder.create<relalg::RenamingOp>(builder.getUnknownLoc(), rel_type, scope, builder.getArrayAttr(renaming_defs_as_attr));
         join->setOperand(joinoperandid, renamingop);
         addPredicate(mlir::dyn_cast_or_null<TupleLamdaOperator>(join.getOperation()), [&](Value tuple, OpBuilder& builder) {
            std::vector<Value> to_and;
            bool any_nullable = false;
            for (auto attref_provider : dependent_refs) {
               auto attref_dependent = attributeManager.createRef(renamed[&attref_provider.getRelationalAttribute()]);
               Value val_left = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), db::BoolType::get(builder.getContext()), attref_provider, tuple);
               Value val_right = builder.create<relalg::GetAttrOp>(builder.getUnknownLoc(), db::BoolType::get(builder.getContext()), attref_dependent, tuple);
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
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createUnnestingPass() { return std::make_unique<Unnesting>(); }
} // end namespace relalg
} // end namespace mlir