#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

class ProjectionTranslator : public mlir::relalg::Translator {
   mlir::relalg::ProjectionOp projectionOp;

   public:
   ProjectionTranslator(mlir::relalg::ProjectionOp projectionOp) : mlir::relalg::Translator(projectionOp), projectionOp(projectionOp) {}

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      consumer->consume(this, builder, context);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~ProjectionTranslator() {}
};

class DistinctProjectionTranslator : public mlir::relalg::Translator {
   mlir::relalg::ProjectionOp projectionOp;
   mlir::Value aggrHt;

   mlir::relalg::OrderedAttributes key;

   mlir::TupleType valTupleType;
   mlir::TupleType entryType;

   public:
   DistinctProjectionTranslator(mlir::relalg::ProjectionOp projectionOp) : mlir::relalg::Translator(projectionOp), projectionOp(projectionOp) {
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value emptyVals = builder.create<mlir::util::UndefOp>(projectionOp->getLoc(), valTupleType);
      mlir::Value packedKey = key.pack(context, builder, projectionOp->getLoc());

      auto reduceOp = builder.create<mlir::dsa::HashtableInsert>(projectionOp->getLoc(), aggrHt, packedKey, emptyVals);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      reduceOp.equal().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({packedKey.getType(), packedKey.getType()}, {projectionOp->getLoc(), projectionOp->getLoc()});
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         auto yieldOp = builder.create<mlir::dsa::YieldOp>(projectionOp->getLoc());
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value matches = compareKeys(builder, aggrBuilderBlock->getArgument(0), aggrBuilderBlock->getArgument(1),projectionOp->getLoc());
         builder.create<mlir::dsa::YieldOp>(projectionOp->getLoc(), matches);
         yieldOp.erase();
      }
      {
         mlir::Block* aggrBuilderBlock = new mlir::Block;
         reduceOp.hash().push_back(aggrBuilderBlock);
         aggrBuilderBlock->addArguments({packedKey.getType()}, {projectionOp->getLoc()});
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value hashed = builder.create<mlir::db::Hash>(projectionOp->getLoc(), builder.getIndexType(), aggrBuilderBlock->getArgument(0));
         builder.create<mlir::dsa::YieldOp>(projectionOp->getLoc(), hashed);
      }
   }
   mlir::Value compareKeys(mlir::OpBuilder& rewriter, mlir::Value left, mlir::Value right,mlir::Location loc) {
      mlir::Value equal = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getI1Type(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
      auto leftUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, left);
      auto rightUnpacked = rewriter.create<mlir::util::UnPackOp>(loc, right);
      for (size_t i = 0; i < leftUnpacked.getNumResults(); i++) {
         mlir::Value compared;
         auto currLeftType = leftUnpacked->getResult(i).getType();
         auto currRightType = rightUnpacked.getResult(i).getType();
         auto currLeftNullableType = currLeftType.dyn_cast_or_null<mlir::db::NullableType>();
         auto currRightNullableType = currRightType.dyn_cast_or_null<mlir::db::NullableType>();
         if (currLeftNullableType || currRightNullableType) {
            mlir::Value isNull1 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), leftUnpacked->getResult(i));
            mlir::Value isNull2 = rewriter.create<mlir::db::IsNullOp>(loc, rewriter.getI1Type(), rightUnpacked->getResult(i));
            mlir::Value anyNull = rewriter.create<mlir::arith::OrIOp>(loc, isNull1, isNull2);
            mlir::Value bothNull = rewriter.create<mlir::arith::AndIOp>(loc, isNull1, isNull2);
            compared = rewriter.create<mlir::scf::IfOp>(
                                  loc, rewriter.getI1Type(), anyNull, [&](mlir::OpBuilder& b, mlir::Location loc) { b.create<mlir::scf::YieldOp>(loc, bothNull); },
                                  [&](mlir::OpBuilder& b, mlir::Location loc) {
                                     mlir::Value left = rewriter.create<mlir::db::NullableGetVal>(loc, leftUnpacked->getResult(i));
                                     mlir::Value right = rewriter.create<mlir::db::NullableGetVal>(loc, rightUnpacked->getResult(i));
                                     mlir::Value cmpRes = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, left, right);
                                     b.create<mlir::scf::YieldOp>(loc, cmpRes);
                                  })
                          .getResult(0);
         } else {
            compared = rewriter.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, leftUnpacked->getResult(i), rightUnpacked.getResult(i));
         }
         mlir::Value localEqual = rewriter.create<mlir::arith::AndIOp>(loc, rewriter.getI1Type(), mlir::ValueRange({equal, compared}));
         equal = localEqual;
      }
      return equal;
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      key = mlir::relalg::OrderedAttributes::fromRefArr(projectionOp.cols());
      valTupleType = mlir::TupleType::get(builder.getContext(), {});
      auto keyTupleType = key.getTupleType(builder.getContext());
      mlir::Value emptyTuple = builder.create<mlir::util::UndefOp>(projectionOp.getLoc(), mlir::TupleType::get(builder.getContext()));
      aggrHt = builder.create<mlir::dsa::CreateDS>(projectionOp.getLoc(), mlir::dsa::AggregationHashtableType::get(builder.getContext(), keyTupleType, valTupleType), emptyTuple);

      entryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, valTupleType});
      children[0]->produce(context, builder);

      auto forOp2 = builder.create<mlir::dsa::ForOp>(projectionOp->getLoc(), mlir::TypeRange{}, aggrHt, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(entryType, projectionOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<mlir::util::UnPackOp>(projectionOp->getLoc(), forOp2.getInductionVar()).getResults();
      auto unpackedKey = builder2.create<mlir::util::UnPackOp>(projectionOp->getLoc(), unpacked[0]).getResults();
      key.setValuesForColumns(context, scope, unpackedKey);
      consumer->consume(this, builder2, context);
      builder2.create<mlir::dsa::YieldOp>(projectionOp->getLoc(), mlir::ValueRange{});

      builder.create<mlir::dsa::FreeOp>(projectionOp->getLoc(), aggrHt);
   }
   virtual void done() override {
   }
   virtual ~DistinctProjectionTranslator() {}
};
std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createProjectionTranslator(mlir::relalg::ProjectionOp projectionOp) {
   if (projectionOp.set_semantic() == mlir::relalg::SetSemantic::distinct) {
      return (std::unique_ptr<Translator>) std::make_unique<DistinctProjectionTranslator>(projectionOp);
   } else {
      return (std::unique_ptr<Translator>) std::make_unique<ProjectionTranslator>(projectionOp);
   }
}
