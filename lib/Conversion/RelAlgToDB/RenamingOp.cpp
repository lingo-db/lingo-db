#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class RenamingLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::RenamingOp renamingOp;
   std::vector<std::pair<mlir::relalg::RelationalAttribute*, mlir::Value>> saved;

   public:
   RenamingLowering(mlir::relalg::RenamingOp renamingOp) : mlir::relalg::ProducerConsumerNode(renamingOp.rel()), renamingOp(renamingOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(renamingOp.getUsedAttributes());
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
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
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      for(mlir::Attribute attr:renamingOp.attributes()){
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         mlir::Attribute from=relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>()[0];
         auto relationRefAttr = from.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>();
         auto attrptr=&relationRefAttr.getRelationalAttribute();
         auto val=context.getUnsafeValueForAttribute(attrptr);
         saved.push_back({attrptr,val});
      }
      children[0]->produce(context, builder);
   }

   virtual ~RenamingLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredRenamingOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::RenamingOp renamingOp) {
  return std::make_unique<RenamingLowering>(renamingOp);
});