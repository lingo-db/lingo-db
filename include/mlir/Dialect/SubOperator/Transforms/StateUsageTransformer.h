#ifndef MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_STATEUSAGETRANSFORMER_H
#define MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_STATEUSAGETRANSFORMER_H
#include "mlir/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "mlir/Dialect/TupleStream/Column.h"
#include "mlir/Dialect/TupleStream/ColumnManager.h"
#include "mlir/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/IR/Operation.h"
#include <string>
#include <unordered_map>
namespace mlir::subop {
class SubOpStateUsageTransformer {
   mlir::tuples::ColumnManager& columnManager;
   const ColumnUsageAnalysis& columnUsageAnalysis;
   std::function<mlir::Type(mlir::Operation* op, mlir::Type oldRefType)> getNewRefTypeFn;
   std::unordered_map<std::string, std::string> memberMapping;

   public:
   SubOpStateUsageTransformer(const ColumnUsageAnalysis& columnUsageAnalysis, mlir::MLIRContext* context, const std::function<mlir::Type(mlir::Operation* op, mlir::Type oldRefType)>& getNewRefTypeFn) : columnManager(context->getLoadedDialect<mlir::tuples::TupleStreamDialect>()->getColumnManager()), columnUsageAnalysis(columnUsageAnalysis), getNewRefTypeFn(getNewRefTypeFn) {}
   void updateValue(mlir::Value oldValue, mlir::Type newType);
   void replaceColumn(mlir::tuples::Column* oldColumn, mlir::tuples::Column* newColumn);
   mlir::Type getNewRefType(mlir::Operation* op, mlir::Type oldRefType);
   mlir::DictionaryAttr updateMapping(mlir::DictionaryAttr currentMapping);
   void mapMembers(const std::unordered_map<std::string, std::string>& memberMapping);
   mlir::tuples::ColumnDefAttr createReplacementColumn(mlir::tuples::ColumnDefAttr oldColumn, mlir::Type newType);
   tuples::ColumnManager& getColumnManager() const {
      return columnManager;
   }
};
} // namespace mlir::subop
#endif //MLIR_DIALECT_SUBOPERATOR_TRANSFORMS_STATEUSAGETRANSFORMER_H
