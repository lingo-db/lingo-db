#include "mlir/Conversion/RelAlgToDB/NLJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

#include <mlir/Conversion/RelAlgToDB/HashJoinUtils.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLSemiJoinLowering : public mlir::relalg::NLJoinTranslator {
   mlir::Value matchFoundFlag;

   public:
   NLSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::NLJoinTranslator(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      handlePotentialMatch(builder, context, matchFound);
   }
   virtual ~NLSemiJoinLowering() {}
};

class HashSemiJoinLowering : public mlir::relalg::HJNode {
   mlir::Value matchFoundFlag;

   public:
   HashSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::HJNode(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(loc, matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(loc, mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(loc, mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      handlePotentialMatch(builder, context, matchFound);
   }
   virtual ~HashSemiJoinLowering() {}
};
class MHashSemiJoinLowering : public mlir::relalg::HJNode {
   public:
   MHashSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::HJNode(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(),true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto beforeBuilderValues = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         loc, getRequiredBuilderTypes(context), matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
            auto const1 = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
            auto markerBefore = builder1.create<mlir::AtomicRMWOp>(builder1.getUnknownLoc(), builder1.getIntegerType(64), mlir::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
            {
               auto zero = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), markerBefore.getType(), builder1.getIntegerAttr(markerBefore.getType(), 0));
               auto isZero = builder1.create<mlir::arith::CmpIOp>(builder1.getUnknownLoc(), mlir::arith::CmpIPredicate::eq, markerBefore, zero);
               auto isZeroDB = builder1.create<mlir::db::TypeCastOp>(builder1.getUnknownLoc(), mlir::db::BoolType::get(builder1.getContext()), isZero);
               handlePotentialMatch(builder,context,isZeroDB);
            }
            builder1.create<mlir::db::YieldOp>(loc, getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(loc, beforeBuilderValues); });
      setRequiredBuilderValues(context, ifOp.getResults());
   }

   virtual ~MHashSemiJoinLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredSemiJoinOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::SemiJoinOp joinOp) {
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<HashSemiJoinLowering>(joinOp);
         }
         if (impl.getValue() == "markhash") {
            return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<MHashSemiJoinLowering>(joinOp);
         }
      }
   }
   return (std::unique_ptr<mlir::relalg::ProducerConsumerNode>) std::make_unique<NLSemiJoinLowering>(joinOp);
});