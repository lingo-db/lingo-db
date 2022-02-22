#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/HashJoinTranslator.h>
#include <mlir/IR/BlockAndValueMapping.h>

class AntiSemiJoinImpl : public mlir::relalg::JoinImpl {
   public:
   AntiSemiJoinImpl(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {}

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, matched);
   }
   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(loc, mlir::db::BoolType::get(builder.getContext()), matchFound);
      translator->handlePotentialMatch(builder, context, noMatchFound);
   }
   virtual ~AntiSemiJoinImpl() {}
};

class ReversedAntiSemiJoinImpl : public mlir::relalg::JoinImpl {
   public:
   ReversedAntiSemiJoinImpl(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {}

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto builderValuesBefore = translator->getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         loc, translator->getRequiredBuilderTypes(context), matched, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
         auto const1 = builder1.create<mlir::arith::ConstantOp>(loc, builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
         builder1.create<mlir::memref::AtomicRMWOp>(loc, builder1.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
         builder1.create<mlir::db::YieldOp>(loc, translator->getRequiredBuilderValues(context)); }, translator->requiredBuilders.empty() ? translator->noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(loc, builderValuesBefore); });
      translator->setRequiredBuilderValues(context, ifOp.getResults());
   }
   virtual void after(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto zero = builder.create<mlir::arith::ConstantOp>(loc, marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, marker, zero);
      auto isZeroDB = builder.create<mlir::db::TypeCastOp>(loc, mlir::db::BoolType::get(builder.getContext()), isZero);
      translator->handlePotentialMatch(builder, context, isZeroDB);
   }

   virtual ~ReversedAntiSemiJoinImpl() {}
};
std::shared_ptr<mlir::relalg::JoinImpl> mlir::relalg::Translator::createAntiSemiJoinImpl(mlir::relalg::AntiSemiJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<ReversedAntiSemiJoinImpl>(joinOp) : (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<AntiSemiJoinImpl>(joinOp);
};