#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class SingleJoinImpl : public mlir::relalg::JoinImpl {
   mlir::Value matchFoundFlag;

   public:
   SingleJoinImpl(mlir::relalg::SingleJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMapping(builder, context, scope);
         auto trueVal = builder.create<mlir::db::ConstantOp>(loc, mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, trueVal);
      });
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(loc, mlir::db::BoolType::get(builder.getContext()), matchFound);
      translator->handlePotentialMatch(builder, context, noMatchFound, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMappingNull(builder, context, scope);
      });
   }
   virtual ~SingleJoinImpl() {}
};
class ConstantSingleJoinTranslator : public mlir::relalg::Translator {
   mlir::relalg::SingleJoinOp joinOp;
   std::vector<const mlir::relalg::RelationalAttribute*> attrs;
   std::vector<const mlir::relalg::RelationalAttribute*> origAttrs;
   std::vector<mlir::Type> types;
   size_t builderId;

   public:
   ConstantSingleJoinTranslator(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::Translator(singleJoinOp), joinOp(singleJoinOp) {
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedAttributes());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         auto* defAttr = &relationDefAttr.getRelationalAttribute();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            const auto* refAttr = *mlir::relalg::Attributes::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
            origAttrs.push_back(refAttr);
            attrs.push_back(defAttr);
            types.push_back(defAttr->type);
         }
      }
      propagateInfo();
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         auto unpacked = builder.create<mlir::util::UnPackOp>(joinOp->getLoc(), context.builders[builderId]);
         for (size_t i = 0; i < attrs.size(); i++) {
            context.setValueForAttribute(scope, attrs[i], unpacked.getResult(i));
         }
         consumer->consume(this, builder, context);
      } else if (child == this->children[1].get()) {
         std::vector<mlir::Value> values;
         for (size_t i = 0; i < origAttrs.size(); i++) {
            mlir::Value value = context.getValueForAttribute(origAttrs[i]);
            if (origAttrs[i]->type != attrs[i]->type) {
               mlir::Value tmp = builder.create<mlir::db::CastOp>(op->getLoc(), attrs[i]->type, value);
               value = tmp;
            }
            values.push_back(value);
         }
         context.builders[builderId] = builder.create<mlir::util::PackOp>(joinOp->getLoc(), values);
      }
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      std::vector<mlir::Value> values;
      for (auto type : types) {
         values.push_back(builder.create<mlir::db::NullOp>(joinOp.getLoc(), type));
      }
      builderId = context.getBuilderId();
      context.builders[builderId] = builder.create<mlir::util::PackOp>(joinOp->getLoc(), values);
      children[1]->addRequiredBuilders({builderId});
      children[1]->produce(context, builder);
      children[0]->produce(context, builder);
   }

   virtual ~ConstantSingleJoinTranslator() {}
};


std::shared_ptr<mlir::relalg::JoinImpl> mlir::relalg::Translator::createSingleJoinImpl(mlir::relalg::SingleJoinOp joinOp) {
   return std::make_shared<SingleJoinImpl>(joinOp);
}
std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createConstSingleJoinTranslator(mlir::relalg::SingleJoinOp joinOp){
   return std::make_unique<ConstantSingleJoinTranslator>(joinOp);
}