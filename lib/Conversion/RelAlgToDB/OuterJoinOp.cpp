#include "mlir/Conversion/RelAlgToDB/HashJoinUtils.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLOuterJoinLowering : public mlir::relalg::NLJoinTranslator {
   mlir::Value matchFoundFlag;
   public:
   NLOuterJoinLowering(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::NLJoinTranslator(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder) {
         auto trueVal = builder.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, trueVal);
      });
   }

   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound, [&](mlir::OpBuilder& builder1, mlir::Location) {
            for (mlir::Attribute attr : cast<mlir::relalg::OuterJoinOp>(joinOp).mapping()) {
               auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
               auto* defAttr = &relationDefAttr.getRelationalAttribute();
               if (this->requiredAttributes.contains(defAttr)) {
                  auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
                  context.setValueForAttribute(scope, defAttr, nullValue);
               }
            }
            consumer->consume(this, builder1, context);
            builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }

   virtual ~NLOuterJoinLowering() {}
};
class HashOuterJoinLowering : public mlir::relalg::HJNode {
   mlir::Value matchFoundFlag;
   public:
   HashOuterJoinLowering(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::HJNode(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder) {
         auto trueVal = builder.create<mlir::db::ConstantOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, trueVal);
      });
   }

   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFound);
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), noMatchFound, [&](mlir::OpBuilder& builder1, mlir::Location) {
            for (mlir::Attribute attr : cast<mlir::relalg::OuterJoinOp>(joinOp).mapping()) {
               auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
               auto* defAttr = &relationDefAttr.getRelationalAttribute();
               if (this->requiredAttributes.contains(defAttr)) {
                  auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
                  context.setValueForAttribute(scope, defAttr, nullValue);
               }
            }
            consumer->consume(this, builder1, context);
            builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }

   virtual ~HashOuterJoinLowering() {}
};

class MHashOuterJoinLowering : public mlir::relalg::HJNode {
   public:

   MHashOuterJoinLowering(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::HJNode(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(),true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      handlePotentialMatch(builder, context, matched);
   }
   virtual void after(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto zero = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(builder.getUnknownLoc(), mlir::arith::CmpIPredicate::eq, marker, zero);
      auto isZeroDB = builder.create<mlir::db::TypeCastOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), isZero);
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), isZeroDB, [&](mlir::OpBuilder& builder1, mlir::Location) {
            for (mlir::Attribute attr : cast<mlir::relalg::OuterJoinOp>(joinOp).mapping()) {
               auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
               auto* defAttr = &relationDefAttr.getRelationalAttribute();
               if (this->requiredAttributes.contains(defAttr)) {
                  auto nullValue = builder1.create<mlir::db::NullOp>(joinOp.getLoc(), defAttr->type);
                  context.setValueForAttribute(scope, defAttr, nullValue);
               }
            }
            consumer->consume(this, builder1, context);
            builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });

      setRequiredBuilderValues(context, ifOp.getResults());
   }

   virtual ~MHashOuterJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredOuterJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::OuterJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashOuterJoinLowering>(joinOp);
         }
         if (impl.getValue() == "markhash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<MHashOuterJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLOuterJoinLowering>(joinOp);
});