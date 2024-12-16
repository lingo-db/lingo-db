#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsAttributes.h"
#include "lingodb/compiler/Dialect/SubOperator/Transforms/StateUsageTransformer.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamOpsAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <string>
#include <vector>

namespace lingodb::compiler::dialect::subop {
class ColumnMapping {
   std::unordered_map<dialect::tuples::Column*, dialect::tuples::Column*> mapping;

   public:
   dialect::tuples::ColumnRefAttr remap(dialect::tuples::ColumnRefAttr refAttr) {
      if (mapping.contains(&refAttr.getColumn())) {
         return refAttr.getContext()->getLoadedDialect<dialect::tuples::TupleStreamDialect>()->getColumnManager().createRef(mapping[&refAttr.getColumn()]);
      } else {
         return refAttr;
      }
   }
   dialect::tuples::ColumnDefAttr remap(dialect::tuples::ColumnDefAttr defAttr) {
      return defAttr.getContext()->getLoadedDialect<dialect::tuples::TupleStreamDialect>()->getColumnManager().createDef(mapping[&defAttr.getColumn()], defAttr.getFromExisting());
   }
   mlir::Attribute remap(mlir::Attribute attr) {
      if (auto refAttr = mlir::dyn_cast_or_null<dialect::tuples::ColumnRefAttr>(attr)) {
         return remap(refAttr);
      } else if (auto defAttr = mlir::dyn_cast_or_null<dialect::tuples::ColumnDefAttr>(attr)) {
         return remap(defAttr);
      } else if (auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(attr)) {
         return remap(arrayAttr);
      } else if (auto dictionaryAttr = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr)) {
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
   dialect::tuples::ColumnDefAttr clone(dialect::tuples::ColumnDefAttr defAttr) {
      auto& colManager = defAttr.getContext()->getLoadedDialect<dialect::tuples::TupleStreamDialect>()->getColumnManager();
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
         remapped.push_back(mlir::NamedAttribute(x.getName(), clone(mlir::cast<dialect::tuples::ColumnDefAttr>(x.getValue()))));
      }
      return mlir::DictionaryAttr::get(mapping.getContext(), remapped);
   }
   mlir::ArrayAttr clone(mlir::ArrayAttr mapping) {
      std::vector<mlir::Attribute> remapped;
      for (auto x : mapping) {
         remapped.push_back(clone(mlir::cast<dialect::tuples::ColumnDefAttr>(x)));
      }
      return mlir::ArrayAttr::get(mapping.getContext(), remapped);
   }
   void mapRaw(dialect::tuples::Column* from, dialect::tuples::Column* to) {
      mapping[from] = to;
   }
};
} // end namespace lingodb::compiler::dialect::subop
#define GET_OP_CLASSES
#include "lingodb/compiler/Dialect/SubOperator/SubOperatorOpsInterfaces.h.inc"

#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_SUBOPERATORINTERFACES_H
