#include "lingodb/compiler/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/IR/Builders.h"
using namespace lingodb::compiler::dialect;
using namespace lingodb::compiler::dialect::subop;
mlir::Type SubOpStateUsageTransformer::getNewRefType(mlir::Operation* op, mlir::Type oldRefType) {
   return getNewRefTypeFn(op, oldRefType);
}
void SubOpStateUsageTransformer::mapMembers(const std::unordered_map<std::string, std::string>& memberMapping) {
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
      anyNeedsReplacement |= memberMapping.contains(mlir::cast<mlir::StringAttr>(m).str());
   }
   if (anyNeedsReplacement) {
      mlir::OpBuilder b(currentMembers.getContext());
      std::vector<mlir::Attribute> newMembers;
      for (auto m : currentMembers) {
         auto memberName = mlir::cast<mlir::StringAttr>(m).str();
         if (memberMapping.contains(memberName)) {
            newMembers.push_back(b.getStringAttr(memberMapping.at(memberName)));
         } else {
            newMembers.push_back(m);
         }
      }
      return b.getArrayAttr(newMembers);
   } else {
      return currentMembers;
   }
}
mlir::DictionaryAttr SubOpStateUsageTransformer::updateMapping(mlir::DictionaryAttr currentMapping) {
   bool anyNeedsReplacement = false;
   for (auto m : currentMapping) {
      anyNeedsReplacement |= memberMapping.contains(m.getName().str());
   }
   if (anyNeedsReplacement) {
      mlir::OpBuilder b(currentMapping.getContext());
      std::vector<mlir::NamedAttribute> newMapping;
      for (auto m : currentMapping) {
         if (memberMapping.contains(m.getName().str())) {
            newMapping.push_back(b.getNamedAttr(memberMapping.at(m.getName().str()), m.getValue()));
         } else {
            newMapping.push_back(m);
         }
      }
      return b.getDictionaryAttr(newMapping);
   } else {
      return currentMapping;
   }
}
