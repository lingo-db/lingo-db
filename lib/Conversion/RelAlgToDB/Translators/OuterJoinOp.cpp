#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLOuterJoinTranslator : public mlir::relalg::NLJoinTranslator {
   public:
   NLOuterJoinTranslator(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::NLJoinTranslator(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      this->stopOnFlag = false;
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         handleMapping(builder, context, scope);
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
      handlePotentialMatch(builder, context, noMatchFound, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         handleMappingNull(builder, context, scope);
      });
   }

   virtual ~NLOuterJoinTranslator() {}
};
class HashOuterJoinTranslator : public mlir::relalg::HashJoinTranslator {
   public:
   HashOuterJoinTranslator(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::HashJoinTranslator(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      this->stopOnFlag = false;
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         handleMapping(builder, context, scope);
         auto trueVal = builder.create<mlir::db::ConstantOp>(loc, mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, trueVal);
      });
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(loc, mlir::db::BoolType::get(builder.getContext()), matchFound);
      handlePotentialMatch(builder, context, noMatchFound, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         handleMappingNull(builder, context, scope);
      });
   }

   virtual ~HashOuterJoinTranslator() {}
};

class MHashOuterJoinTranslator : public mlir::relalg::HashJoinTranslator {
   public:
   MHashOuterJoinTranslator(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::HashJoinTranslator(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         handleMapping(builder, context, scope);
      });
   }
   virtual void after(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto zero = builder.create<mlir::arith::ConstantOp>(op->getLoc(), marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(op->getLoc(), mlir::arith::CmpIPredicate::eq, marker, zero);
      auto isZeroDB = builder.create<mlir::db::TypeCastOp>(op->getLoc(), mlir::db::BoolType::get(builder.getContext()), isZero);
      handlePotentialMatch(builder, context, isZeroDB, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         handleMappingNull(builder, context, scope);
      });
   }

   virtual ~MHashOuterJoinTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createOuterJoinTranslator(mlir::relalg::OuterJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::Translator>) std::make_unique<HashOuterJoinTranslator>(joinOp);
         }
         if (impl.getValue() == "markhash") {
            return (std::unique_ptr<mlir::relalg::Translator>) std::make_unique<MHashOuterJoinTranslator>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::Translator>) std::make_unique<NLOuterJoinTranslator>(joinOp);
}