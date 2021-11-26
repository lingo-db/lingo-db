#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

#include <mlir/Conversion/RelAlgToDB/HashJoinUtils.h>
#include <mlir/IR/BlockAndValueMapping.h>

class NLSemiJoinLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::SemiJoinOp joinOp;
   mlir::Value matchFoundFlag;

   public:
   NLSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::ProducerConsumerNode(innerJoinOp), joinOp(innerJoinOp) {
   }

   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
         children[1]->setFlag(matchFoundFlag);
         children[1]->produce(context, builder);
         mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
         auto builderValuesBefore = getRequiredBuilderValues(context);
         auto ifOp = builder.create<mlir::db::IfOp>(
            joinOp->getLoc(), getRequiredBuilderTypes(context), matchFound, [&](mlir::OpBuilder& builder1, mlir::Location) {
               consumer->consume(this, builder1, context);
               builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
            requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });
         setRequiredBuilderValues(context, ifOp.getResults());
      } else if (child == this->children[1].get()) {
         mlir::Value matched= mergeRelationalBlock(
            builder.getInsertionBlock(), joinOp, [](auto x) { return &x->getRegion(0).front(); }, context, scope)[0];
         builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
      }
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~NLSemiJoinLowering() {}
};

class HashSemiJoinLowering : public mlir::relalg::HJNode<mlir::relalg::SemiJoinOp> {
   mlir::Value matchFoundFlag;

   public:
   HashSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::HJNode<mlir::relalg::SemiJoinOp>(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::db::SetFlag>(joinOp->getLoc(), matchFoundFlag, matched);
   }
   mlir::Value getFlag() override {
      return matchFoundFlag;
   }
   void beforeLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::db::CreateFlag>(joinOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::db::GetFlag>(joinOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), matchFoundFlag);
      auto builderValuesBefore = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), matchFound, [&](mlir::OpBuilder& builder1, mlir::Location) {
            consumer->consume(this, builder1, context);
            builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), builderValuesBefore); });

      setRequiredBuilderValues(context, ifOp.getResults());
   }
   virtual ~HashSemiJoinLowering() {}
};
class MHashSemiJoinLowering : public mlir::relalg::MarkableHJNode<mlir::relalg::SemiJoinOp> {
   public:
   MHashSemiJoinLowering(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::MarkableHJNode<mlir::relalg::SemiJoinOp>(innerJoinOp, innerJoinOp.left(), innerJoinOp.right()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto beforeBuilderValues = getRequiredBuilderValues(context);
      auto ifOp = builder.create<mlir::db::IfOp>(
         joinOp->getLoc(), getRequiredBuilderTypes(context), matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
            auto const1 = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
            auto markerBefore = builder1.create<mlir::AtomicRMWOp>(builder1.getUnknownLoc(), builder1.getIntegerType(64), mlir::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
            {
               auto zero = builder1.create<mlir::arith::ConstantOp>(builder1.getUnknownLoc(), markerBefore.getType(), builder1.getIntegerAttr(markerBefore.getType(), 0));
               auto isZero = builder1.create<mlir::arith::CmpIOp>(builder1.getUnknownLoc(), mlir::arith::CmpIPredicate::eq, markerBefore, zero);
               auto isZeroDB = builder1.create<mlir::db::TypeCastOp>(builder1.getUnknownLoc(), mlir::db::BoolType::get(builder1.getContext()), isZero);
               auto beforeBuilderValues = getRequiredBuilderValues(context);
               auto ifOp = builder1.create<mlir::db::IfOp>(joinOp->getLoc(), getRequiredBuilderTypes(context), isZeroDB, [&](mlir::OpBuilder& builder10, mlir::Location) {
                  consumer->consume(this, builder10, context);
                  builder10.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context));
                  },requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) {
                  builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), beforeBuilderValues);
               });
               setRequiredBuilderValues(context,ifOp.getResults());
            }
            builder1.create<mlir::db::YieldOp>(joinOp->getLoc(), getRequiredBuilderValues(context)); },
         requiredBuilders.empty() ? mlir::relalg::noBuilder : [&](mlir::OpBuilder& builder2, mlir::Location) { builder2.create<mlir::db::YieldOp>(joinOp->getLoc(), beforeBuilderValues); });
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