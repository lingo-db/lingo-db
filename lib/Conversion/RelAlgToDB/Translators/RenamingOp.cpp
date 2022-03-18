#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class RenamingTranslator : public mlir::relalg::Translator {
   mlir::relalg::RenamingOp renamingOp;
   std::vector<std::pair<mlir::relalg::Column*, mlir::Value>> saved;

   public:
   RenamingTranslator(mlir::relalg::RenamingOp renamingOp) : mlir::relalg::Translator(renamingOp), renamingOp(renamingOp) {}

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      for(mlir::Attribute attr:renamingOp.columns()){
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
         context.setValueForAttribute(scope,&relationDefAttr.getColumn(),context.getValueForAttribute(&relationRefAttr.getColumn()));
      }
      for(auto s:saved){
         context.setValueForAttribute(scope,s.first,s.second);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      for(mlir::Attribute attr:renamingOp.columns()){
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<mlir::relalg::ColumnRefAttr>();
         auto *attrptr=&relationRefAttr.getColumn();
         auto val=context.getUnsafeValueForAttribute(attrptr);
         saved.push_back({attrptr,val});
      }
      children[0]->produce(context, builder);
   }

   virtual ~RenamingTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createRenamingTranslator(mlir::relalg::RenamingOp renamingOp) {
  return std::make_unique<RenamingTranslator>(renamingOp);
}