#ifndef MLIR_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
#define MLIR_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
#include "mlir/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "mlir/Dialect/SubOperator/Transforms/ColumnFolding.h"
#include "mlir/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "mlir/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <string>
#include <vector>

namespace mlir::subop {
class ColumnMapping {
   std::unordered_map<mlir::tuples::Column*, mlir::tuples::Column*> mapping;

   public:
   mlir::tuples::ColumnRefAttr remap(mlir::tuples::ColumnRefAttr refAttr) {
      if (mapping.contains(&refAttr.getColumn())) {
         return refAttr.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createRef(mapping[&refAttr.getColumn()]);
      } else {
         return refAttr;
      }
   }
   mlir::tuples::ColumnDefAttr remap(mlir::tuples::ColumnDefAttr defAttr) {
      return defAttr.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager().createDef(mapping[&defAttr.getColumn()], defAttr.getFromExisting());
   }
   mlir::Attribute remap(mlir::Attribute attr) {
      if (auto refAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnRefAttr>()) {
         return remap(refAttr);
      } else if (auto defAttr = attr.dyn_cast_or_null<mlir::tuples::ColumnDefAttr>()) {
         return remap(defAttr);
      } else if (auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
         return remap(arrayAttr);
      } else if (auto dictionaryAttr = attr.dyn_cast_or_null<mlir::DictionaryAttr>()) {
         return remap(dictionaryAttr);
      } else {
         assert(false);
      }
   }
   mlir::ArrayAttr remap(mlir::ArrayAttr arrayAttr) {
      std::vector<mlir::Attribute> remapped;
      for (auto x : arrayAttr) {
         remapped.push_back(remap(x));
      }
      return mlir::ArrayAttr::get(arrayAttr.getContext(), remapped);
   }
   mlir::DictionaryAttr remap(mlir::DictionaryAttr dictionaryAttr) {
      std::vector<mlir::NamedAttribute> remapped;
      for (auto x : dictionaryAttr) {
         remapped.push_back(mlir::NamedAttribute(x.getName(), remap(x.getValue())));
      }
      return mlir::DictionaryAttr::get(dictionaryAttr.getContext(), remapped);
   }
   mlir::tuples::ColumnDefAttr clone(mlir::tuples::ColumnDefAttr defAttr) {
      auto& colManager = defAttr.getContext()->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager();
      auto [scope, name] = colManager.getName(&defAttr.getColumn());
      mlir::Attribute fromExisting = defAttr.getFromExisting();
      if (fromExisting) {
         fromExisting = remap(fromExisting);
      }
      auto newDef = colManager.createDef(colManager.getUniqueScope(scope), name, fromExisting);
      newDef.getColumn().type = defAttr.getColumn().type;
      mapping[&defAttr.getColumn()] = &newDef.getColumn();
      return newDef;
   }
   mlir::DictionaryAttr clone(mlir::DictionaryAttr mapping) {
      std::vector<mlir::NamedAttribute> remapped;
      for (auto x : mapping) {
         remapped.push_back(mlir::NamedAttribute(x.getName(), clone(x.getValue().cast<mlir::tuples::ColumnDefAttr>())));
      }
      return mlir::DictionaryAttr::get(mapping.getContext(), remapped);
   }
   mlir::ArrayAttr clone(mlir::ArrayAttr mapping) {
      std::vector<mlir::Attribute> remapped;
      for (auto x : mapping) {
         remapped.push_back(clone(x.cast<mlir::tuples::ColumnDefAttr>()));
      }
      return mlir::ArrayAttr::get(mapping.getContext(), remapped);
   }
};
} // end namespace mlir::subop
#define GET_OP_CLASSES
#include "mlir/Dialect/SubOperator/SubOperatorOpsInterfaces.h.inc"

#endif //MLIR_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
