#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class RenamingTranslator : public mlir::relalg::Translator {
   mlir::relalg::RenamingOp renamingOp;
   std::vector<std::pair<mlir::relalg::RelationalAttribute*, mlir::Value>> saved;

   public:
   RenamingTranslator(mlir::relalg::RenamingOp renamingOp) : mlir::relalg::Translator(renamingOp), renamingOp(renamingOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      for(mlir::Attribute attr:renamingOp.attributes()){
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>();
         context.setValueForAttribute(scope,&relationDefAttr.getRelationalAttribute(),context.getValueForAttribute(&relationRefAttr.getRelationalAttribute()));
      }
      for(auto s:saved){
         context.setValueForAttribute(scope,s.first,s.second);
      }
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      for(mlir::Attribute attr:renamingOp.attributes()){
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>();
         auto *attrptr=&relationRefAttr.getRelationalAttribute();
         auto val=context.getUnsafeValueForAttribute(attrptr);
         saved.push_back({attrptr,val});
      }
      children[0]->produce(context, builder);
   }

   virtual ~RenamingTranslator() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredRenamingOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::RenamingOp renamingOp) {
  return std::make_unique<RenamingTranslator>(renamingOp);
});