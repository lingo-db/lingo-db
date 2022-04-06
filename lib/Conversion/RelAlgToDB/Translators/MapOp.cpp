#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MapTranslator : public mlir::relalg::Translator {
   mlir::relalg::MapOp mapOp;

   public:
   MapTranslator(mlir::relalg::MapOp mapOp) : mlir::relalg::Translator(mapOp), mapOp(mapOp) {}

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      auto computedCols = mergeRelationalBlock(
         builder.getInsertionBlock(), op, [](auto x) { return &x->getRegion(0).front(); }, context, scope);
      assert(computedCols.size() == mapOp.computed_cols().size());
      for (size_t i = 0; i < computedCols.size(); i++) {
         context.setValueForAttribute(scope, &mapOp.computed_cols()[i].cast<mlir::relalg::ColumnDefAttr>().getColumn(), computedCols[i]);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMapTranslator(mlir::relalg::MapOp mapOp) {
   return std::make_unique<MapTranslator>(mapOp);
}