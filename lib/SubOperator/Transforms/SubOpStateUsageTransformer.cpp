#include "mlir/Dialect/SubOperator/SubOperatorInterfaces.h"
#include "mlir/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/IR/Builders.h"
using namespace mlir::subop;
mlir::Type SubOpStateUsageTransformer::getNewRefType(mlir::Operation* op, mlir::Type oldRefType) {
   return getNewRefTypeFn(op, oldRefType);
}

void SubOpStateUsageTransformer::updateValue(mlir::Value oldValue, mlir::Type newType, const std::unordered_map<std::string, std::string>& memberMapping) {
   this->memberMapping.insert(memberMapping.begin(), memberMapping.end());
   for (auto* user : oldValue.getUsers()) {
      if (auto stateUsingSubOp = mlir::dyn_cast_or_null<mlir::subop::StateUsingSubOperator>(user)) {
         stateUsingSubOp.updateStateType(*this, oldValue, newType);
      } else {
         assert(false);
      }
   }
}

void SubOpStateUsageTransformer::replaceColumn(mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn) {
   for (auto *user : columnUsageAnalysis.findOperationsUsing(oldColumn)) {
      if (auto stateUsingSubOp = mlir::dyn_cast_or_null<mlir::subop::StateUsingSubOperator>(user)) {
         stateUsingSubOp.replaceColumns(*this, oldColumn, newColumn);
      } else {
         assert(false);
      }
   }
}

mlir::tuples::ColumnDefAttr SubOpStateUsageTransformer::createReplacementColumn(mlir::tuples::ColumnDefAttr oldColumn, mlir::Type newType) {
   auto [scope,name]=getColumnManager().getName(&oldColumn.getColumn());
   auto newColumnDef=getColumnManager().createDef(getColumnManager().getUniqueScope(scope+"_"),name);
   newColumnDef.getColumn().type=newType;
   replaceColumn(&oldColumn.getColumn(),&newColumnDef.getColumn());
   return newColumnDef;
}
mlir::DictionaryAttr SubOpStateUsageTransformer::updateMapping(mlir::DictionaryAttr currentMapping) {
   bool anyNeedsReplacement = false;
   for (auto m : currentMapping) {
      anyNeedsReplacement |= memberMapping.contains(m.getName().str());
   }
   if (anyNeedsReplacement) {
      mlir::OpBuilder b(currentMapping.getContext());
      std::vector<NamedAttribute> newMapping;
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