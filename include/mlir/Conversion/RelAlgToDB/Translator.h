#ifndef MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H

#include "TranslatorContext.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <iostream>
#include <memory>

namespace mlir {
namespace relalg {
class Translator {
   public:
   Translator* consumer;
   Operator op;
   std::vector<std::unique_ptr<Translator>> children;
   mlir::relalg::ColumnSet requiredAttributes;

   std::vector<mlir::Value> mergeRelationalBlock(mlir::Block* dest, mlir::Operation* op, mlir::function_ref<mlir::Block*(mlir::Operation*)> getBlockFn, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);

   void propagateInfo();

   Translator(mlir::ValueRange children);
   Translator(Operator op);

   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes);
   virtual mlir::relalg::ColumnSet getAvailableColumns();
   virtual void consume(Translator* child, mlir::OpBuilder& builder, TranslatorContext& context) = 0;
   virtual void produce(TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   virtual void done() {}
   virtual ~Translator() {}
   static std::unique_ptr<mlir::relalg::Translator> createBaseTableTranslator(BaseTableOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createConstRelTranslator(ConstRelationOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createMaterializeTranslator(MaterializeOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createSelectionTranslator(SelectionOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createMapTranslator(MapOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createSortTranslator(SortOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createAggregationTranslator(AggregationOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createRenamingTranslator(RenamingOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createProjectionTranslator(ProjectionOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createLimitTranslator(LimitOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createTmpTranslator(TmpOp operation);

   static std::unique_ptr<mlir::relalg::Translator> createTranslator(mlir::Operation* operation);
   static std::unique_ptr<mlir::relalg::Translator> createJoinTranslator(mlir::Operation* operation);
   static std::unique_ptr<mlir::relalg::Translator> createSetOpTranslator(mlir::Operation* operation);


};
} // end namespace relalg
} // end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H