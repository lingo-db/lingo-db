#ifndef LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEUSAGETRANSFORMER_H
#define LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEUSAGETRANSFORMER_H
#include "lingodb/compiler/Dialect/SubOperator/Transforms/ColumnUsageAnalysis.h"
#include "lingodb/compiler/Dialect/TupleStream/Column.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"
#include "lingodb/compiler/Dialect/TupleStream/TupleStreamDialect.h"
#include "mlir/IR/Operation.h"
#include <string>
#include <unordered_map>
namespace lingodb::compiler::dialect::subop {
class SubOpStateUsageTransformer {
   dialect::tuples::ColumnManager& columnManager;
   const ColumnUsageAnalysis& columnUsageAnalysis;
   std::function<mlir::Type(mlir::Operation* op, mlir::Type oldRefType)> getNewRefTypeFn;
   std::function<void(mlir::Operation* op)> callBeforeFn;
   std::function<void(mlir::Operation* op)> callAfterFn;
   std::unordered_map<subop::Member, subop::Member> memberMapping;
   std::unordered_map<tuples::Column*, tuples::Column*> columnMapping;

   public:
   SubOpStateUsageTransformer(const ColumnUsageAnalysis& columnUsageAnalysis, mlir::MLIRContext* context, const std::function<mlir::Type(mlir::Operation* op, mlir::Type oldRefType)>& getNewRefTypeFn) : columnManager(context->getLoadedDialect<dialect::tuples::TupleStreamDialect>()->getColumnManager()), columnUsageAnalysis(columnUsageAnalysis), getNewRefTypeFn(getNewRefTypeFn) {}
   void updateValue(mlir::Value oldValue, mlir::Type newType);
   void replaceColumn(dialect::tuples::Column* oldColumn, dialect::tuples::Column* newColumn);
   mlir::Type getNewRefType(mlir::Operation* op, mlir::Type oldRefType);
   subop::ColumnRefMemberMappingAttr updateMapping(subop::ColumnRefMemberMappingAttr currentMapping);
   subop::ColumnDefMemberMappingAttr updateMapping(subop::ColumnDefMemberMappingAttr currentMapping);
   void mapMembers(const std::unordered_map<subop::Member, subop::Member>& memberMapping);
   mlir::ArrayAttr updateMembers(mlir::ArrayAttr currentMembers);
   dialect::tuples::ColumnDefAttr createReplacementColumn(dialect::tuples::ColumnDefAttr oldColumn, mlir::Type newType);
   tuples::ColumnManager& getColumnManager() const {
      return columnManager;
   }
   void setCallBeforeFn(const std::function<void(mlir::Operation*)>& callBeforeFn) {
      SubOpStateUsageTransformer::callBeforeFn = callBeforeFn;
   }
   void setCallAfterFn(const std::function<void(mlir::Operation*)>& callAfterFn) {
      SubOpStateUsageTransformer::callAfterFn = callAfterFn;
   }
   tuples::Column* getNewColumn(tuples::Column* oldColumn) {
      if (columnMapping.contains(oldColumn)) {
         return columnMapping[oldColumn];
      }
      return oldColumn;
   }
};
} // namespace lingodb::compiler::dialect::subop
#endif //LINGODB_COMPILER_DIALECT_SUBOPERATOR_TRANSFORMS_STATEUSAGETRANSFORMER_H
