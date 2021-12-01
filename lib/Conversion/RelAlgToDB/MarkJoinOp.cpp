#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinUtils.h>
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLMarkJoinTranslator : public mlir::relalg::NLJoinTranslator {
   mlir::Value matchFoundFlag;

   public:
   NLMarkJoinTranslator(mlir::relalg::MarkJoinOp innerJoinOp) : mlir::relalg::NLJoinTranslator(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      context.setValueForAttribute(scope, &cast<mlir::relalg::MarkJoinOp>(joinOp).markattr().getRelationalAttribute(), matchFound);
      consumer->consume(this, builder, context);
   }
   virtual ~NLMarkJoinTranslator() {}
};
class HashMarkJoinTranslator : public mlir::relalg::HJNode {
   mlir::Value matchFoundFlag;

   public:
   HashMarkJoinTranslator(mlir::relalg::MarkJoinOp innerJoinOp) : mlir::relalg::HJNode(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      context.setValueForAttribute(scope, &cast<mlir::relalg::MarkJoinOp>(joinOp).markattr().getRelationalAttribute(), matchFound);
      consumer->consume(this, builder, context);
   }
   virtual ~HashMarkJoinTranslator() {}
};
bool mlir::relalg::ProducerConsumerNodeRegistry::registeredMarkJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::MarkJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::Translator>) std::make_unique<HashMarkJoinTranslator>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::Translator>) std::make_unique<NLMarkJoinTranslator>(joinOp);
});