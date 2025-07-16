#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/IR/Builders.h"
using namespace lingodb::compiler::dialect;
using namespace lingodb::compiler::dialect::subop;
mlir::Type SubOpStateUsageTransformer::getNewRefType(mlir::Operation* op, mlir::Type oldRefType) {
   return getNewRefTypeFn(op, oldRefType);
}
void SubOpStateUsageTransformer::mapMembers(const std::unordered_map<subop::Member, subop::Member>& memberMapping) {
   this->memberMapping.insert(memberMapping.begin(), memberMapping.end());
}
void SubOpStateUsageTransformer::updateValue(mlir::Value oldValue, mlir::Type newType) {
   for (auto* user : oldValue.getUsers()) {
      if (auto stateUsingSubOp = mlir::dyn_cast_or_null<::StateUsingSubOperator>(user)) {
         if (callBeforeFn) { callBeforeFn(stateUsingSubOp.getOperation()); }
         stateUsingSubOp.updateStateType(*this, oldValue, newType);
         if (callAfterFn) { callAfterFn(stateUsingSubOp.getOperation()); }
      } else {
         user->dump();
         assert(false);
      }
   }
}

void SubOpStateUsageTransformer::replaceColumn(tuples::Column* oldColumn, tuples::Column* newColumn) {
   columnMapping[oldColumn] = newColumn;
   for (auto* user : columnUsageAnalysis.findOperationsUsing(oldColumn)) {
      if (auto stateUsingSubOp = mlir::dyn_cast_or_null<::StateUsingSubOperator>(user)) {
         if (callBeforeFn) { callBeforeFn(stateUsingSubOp.getOperation()); }
         stateUsingSubOp.replaceColumns(*this, oldColumn, newColumn);
         if (callAfterFn) { callAfterFn(stateUsingSubOp.getOperation()); }
      } else {
         user->dump();
         assert(false);
      }
   }
}

tuples::ColumnDefAttr SubOpStateUsageTransformer::createReplacementColumn(tuples::ColumnDefAttr oldColumn, mlir::Type newType) {
   auto [scope, name] = getColumnManager().getName(&oldColumn.getColumn());
   auto newColumnDef = getColumnManager().createDef(getColumnManager().getUniqueScope(scope + "_"), name);
   newColumnDef.getColumn().type = newType;
   replaceColumn(&oldColumn.getColumn(), &newColumnDef.getColumn());
   return newColumnDef;
}
mlir::ArrayAttr SubOpStateUsageTransformer::updateMembers(mlir::ArrayAttr currentMembers) {
   bool anyNeedsReplacement = false;
   for (auto m : currentMembers) {
      anyNeedsReplacement |= memberMapping.contains(mlir::cast<subop::MemberAttr>(m).getMember());
   }
   if (anyNeedsReplacement) {
      std::vector<mlir::Attribute> newMembers;
      for (auto m : currentMembers) {
         auto member = mlir::cast<subop::MemberAttr>(m).getMember();
         if (memberMapping.contains(member)) {
            newMembers.push_back(subop::MemberAttr::get(currentMembers.getContext(), memberMapping.at(member)));
         } else {
            newMembers.push_back(m);
         }
      }
      return mlir::ArrayAttr::get(currentMembers.getContext(), newMembers);
   } else {
      return currentMembers;
   }
}

subop::ColumnDefMemberMappingAttr SubOpStateUsageTransformer::updateMapping(subop::ColumnDefMemberMappingAttr currentMapping) {
   bool anyNeedsReplacement = false;
   for (auto m : currentMapping.getMapping()) {
      anyNeedsReplacement |= memberMapping.contains(m.first);
   }
   if (anyNeedsReplacement) {
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnDefAttr>> newMapping;
      for (auto m : currentMapping.getMapping()) {
         if (memberMapping.contains(m.first)) {
            newMapping.push_back({memberMapping.at(m.first), m.second});
         } else {
            newMapping.push_back(m);
         }
      }
      return subop::ColumnDefMemberMappingAttr::get(currentMapping.getContext(), newMapping);
   } else {
      return currentMapping;
   }
}
subop::ColumnRefMemberMappingAttr SubOpStateUsageTransformer::updateMapping(subop::ColumnRefMemberMappingAttr currentMapping) {
   bool anyNeedsReplacement = false;
   for (auto m : currentMapping.getMapping()) {
      anyNeedsReplacement |= memberMapping.contains(m.first);
   }
   if (anyNeedsReplacement) {
      llvm::SmallVector<std::pair<subop::Member, tuples::ColumnRefAttr>> newMapping;
      for (auto m : currentMapping.getMapping()) {
         if (memberMapping.contains(m.first)) {
            newMapping.push_back({memberMapping.at(m.first), m.second});
         } else {
            newMapping.push_back(m);
         }
      }
      return subop::ColumnRefMemberMappingAttr::get(currentMapping.getContext(), newMapping);
   } else {
      return currentMapping;
   }
}