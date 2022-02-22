#ifndef MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H
#define MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H

#include "TranslatorContext.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <iostream>
#include <memory>

namespace mlir {
namespace relalg {

class Translator {
   protected:
   Translator* consumer;
   Operator op;
   std::vector<std::unique_ptr<Translator>> children;
   std::vector<size_t> requiredBuilders;
   mlir::relalg::Attributes requiredAttributes;
   const mlir::function_ref<void(mlir::OpBuilder&, mlir::Location)> noBuilder = nullptr;

   std::vector<mlir::Value> mergeRelationalBlock(mlir::Block* dest, mlir::Operation* op, mlir::function_ref<mlir::Block*(mlir::Operation*)> getBlockFn, TranslatorContext& context, TranslatorContext::AttributeResolverScope& scope);

   void propagateInfo();
   std::vector<mlir::Value> getRequiredBuilderValues(TranslatorContext& context);
   void setRequiredBuilderValues(TranslatorContext& context, const mlir::ValueRange& values);
   std::vector<mlir::Type> getRequiredBuilderTypes(TranslatorContext& context);
   std::vector<mlir::Location> getRequiredBuilderLocs(TranslatorContext& context);

   public:
   Translator(mlir::ValueRange children);
   Translator(Operator op);
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders);

   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes);
   virtual mlir::relalg::Attributes getAvailableAttributes();
   virtual void consume(Translator* child, mlir::OpBuilder& builder, TranslatorContext& context) = 0;
   virtual void produce(TranslatorContext& context, mlir::OpBuilder& builder) = 0;
   virtual void done() {}
   virtual ~Translator() {}

   static std::unique_ptr<mlir::relalg::Translator> createBaseTableTranslator(BaseTableOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createConstRelTranslator(ConstRelationOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createMaterializeTranslator(MaterializeOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createSelectionTranslator(SelectionOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createMapTranslator(MapOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createCrossProductTranslator(CrossProductOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createSortTranslator(SortOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createAggregationTranslator(AggregationOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createInnerJoinTranslator(InnerJoinOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createSemiJoinTranslator(SemiJoinOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createAntiSemiJoinTranslator(AntiSemiJoinOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createRenamingTranslator(RenamingOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createProjectionTranslator(ProjectionOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createLimitTranslator(LimitOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createOuterJoinTranslator(OuterJoinOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createSingleJoinTranslator(SingleJoinOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createMarkJoinTranslator(MarkJoinOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createTmpTranslator(TmpOp operation);
   static std::unique_ptr<mlir::relalg::Translator> createCollectionJoinTranslator(CollectionJoinOp operation);

   static std::unique_ptr<mlir::relalg::Translator> createTranslator(mlir::Operation* operation);
};
class DummyTranslator : public mlir::relalg::Translator {
   public:
   DummyTranslator() : mlir::relalg::Translator(ValueRange{}) {
   }
   virtual void setInfo(Translator* consumer, mlir::relalg::Attributes requiredAttributes) override{};
   virtual mlir::relalg::Attributes getAvailableAttributes() override { return {}; };
   virtual void consume(Translator* child, mlir::OpBuilder& builder, TranslatorContext& context) override{};
   virtual void produce(TranslatorContext& context, mlir::OpBuilder& builder) override{};
   virtual ~DummyTranslator() {}
};
} // end namespace relalg
} // end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_TRANSLATOR_H