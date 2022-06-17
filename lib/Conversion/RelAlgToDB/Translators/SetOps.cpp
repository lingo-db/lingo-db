#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/SCF/SCF.h"
class SetOpTranslator : public mlir::relalg::Translator {
   Operator unionOp;
   std::unordered_map<const mlir::relalg::Column*, const mlir::relalg::Column*> leftMapping;
   std::unordered_map<const mlir::relalg::Column*, const mlir::relalg::Column*> rightMapping;

   protected:
   mlir::relalg::OrderedAttributes orderedAttributes;
   mlir::TupleType tupleType;

   public:
   SetOpTranslator(Operator unionOp) : mlir::relalg::Translator(unionOp), unionOp(unionOp) {
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      orderedAttributes = mlir::relalg::OrderedAttributes::fromColumns(requiredAttributes);
      this->requiredAttributes.insert(op.getUsedColumns());
      propagateInfo();

      for (auto x : unionOp->getAttr("mapping").cast<mlir::ArrayAttr>()) {
         auto columnDef = x.cast<mlir::relalg::ColumnDefAttr>();
         auto leftRef = columnDef.getFromExisting().cast<mlir::ArrayAttr>()[0].cast<mlir::relalg::ColumnRefAttr>();
         auto rightRef = columnDef.getFromExisting().cast<mlir::ArrayAttr>()[1].cast<mlir::relalg::ColumnRefAttr>();
         leftMapping[&columnDef.getColumn()] = &leftRef.getColumn();
         rightMapping[&columnDef.getColumn()] = &rightRef.getColumn();
      }
      tupleType = orderedAttributes.getTupleType(unionOp.getContext());
   }
   mlir::Value pack(std::unordered_map<const mlir::relalg::Column*, const mlir::relalg::Column*>& mapping, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) {
      std::vector<mlir::Value> values;
      for (const auto* x : orderedAttributes.getAttrs()) {
         const auto* colBefore = mapping.at(x);
         mlir::Value current = context.getValueForAttribute(colBefore);
         if (colBefore->type != x->type) {
            current = builder.create<mlir::db::AsNullableOp>(unionOp->getLoc(), x->type, current);
         }
         values.push_back(current);
      }
      return builder.create<mlir::util::PackOp>(unionOp->getLoc(), values);
   }
   mlir::Value pack(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) {
      mlir::Value tuple;
      if (child == children[0].get()) {
         tuple = pack(leftMapping, builder, context);
      } else {
         tuple = pack(rightMapping, builder, context);
      }
      return tuple;
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

   virtual ~SetOpTranslator() {}
};

class UnionAllTranslator : public SetOpTranslator {
   mlir::relalg::UnionOp unionOp;
   mlir::Value vector;

   public:
   UnionAllTranslator(mlir::relalg::UnionOp unionOp) : SetOpTranslator(unionOp), unionOp(unionOp) {
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      builder.create<mlir::dsa::Append>(unionOp->getLoc(), vector, pack(child, builder, context));
   }

   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      vector = builder.create<mlir::dsa::CreateDS>(unionOp.getLoc(), mlir::dsa::VectorType::get(builder.getContext(), tupleType));
      children[0]->produce(context, builder);
      children[1]->produce(context, builder);

      {
         auto forOp2 = builder.create<mlir::dsa::ForOp>(unionOp->getLoc(), mlir::TypeRange{}, vector, mlir::Value(), mlir::ValueRange{});
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType, unionOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<mlir::util::UnPackOp>(unionOp->getLoc(), forOp2.getInductionVar());
         orderedAttributes.setValuesForColumns(context, scope, unpacked.getResults());
         consumer->consume(this, builder2, context);
         builder2.create<mlir::dsa::YieldOp>(unionOp->getLoc(), mlir::ValueRange{});
      }
      builder.create<mlir::dsa::FreeOp>(unionOp->getLoc(), vector);
   }

   virtual ~UnionAllTranslator() {}
};
class UnionDistinctTranslator : public SetOpTranslator {
   mlir::relalg::UnionOp unionOp;
   mlir::Value aggrHt;

   public:
   UnionDistinctTranslator(mlir::relalg::UnionOp unionOp) : SetOpTranslator(unionOp), unionOp(unionOp) {
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value emptyVals = builder.create<mlir::util::UndefOp>(unionOp->getLoc(), mlir::TupleType::get(builder.getContext()));
      mlir::Value packedKey = pack(child, builder, context);
      auto reduceOp = builder.create<mlir::dsa::HashtableInsert>(unionOp->getLoc(), aggrHt, packedKey, emptyVals);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      reduceOp.equal().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({packedKey.getType(), packedKey.getType()}, {unionOp->getLoc(), unionOp->getLoc()});
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         auto yieldOp = builder.create<mlir::dsa::YieldOp>(unionOp->getLoc());
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value matches = compareKeys(builder, aggrBuilderBlock->getArgument(0), aggrBuilderBlock->getArgument(1),unionOp->getLoc());
         builder.create<mlir::dsa::YieldOp>(unionOp->getLoc(), matches);
         yieldOp.erase();
      }
      {
         mlir::Block* aggrBuilderBlock = new mlir::Block;
         reduceOp.hash().push_back(aggrBuilderBlock);
         aggrBuilderBlock->addArguments({packedKey.getType()}, {unionOp->getLoc()});
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value hashed = builder.create<mlir::db::Hash>(unionOp->getLoc(), builder.getIndexType(), aggrBuilderBlock->getArgument(0));
         builder.create<mlir::dsa::YieldOp>(unionOp->getLoc(), hashed);
      }
   }

   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto keyTupleType = orderedAttributes.getTupleType(builder.getContext());
      mlir::Value emptyTuple = builder.create<mlir::util::UndefOp>(unionOp.getLoc(), mlir::TupleType::get(builder.getContext()));
      aggrHt = builder.create<mlir::dsa::CreateDS>(unionOp.getLoc(), mlir::dsa::AggregationHashtableType::get(builder.getContext(), keyTupleType, mlir::TupleType::get(builder.getContext())), emptyTuple);

      mlir::Type entryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, mlir::TupleType::get(builder.getContext())});
      children[0]->produce(context, builder);
      children[1]->produce(context, builder);

      auto forOp2 = builder.create<mlir::dsa::ForOp>(unionOp->getLoc(), mlir::TypeRange{}, aggrHt, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(entryType, unionOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<mlir::util::UnPackOp>(unionOp->getLoc(), forOp2.getInductionVar()).getResults();
      auto unpackedKey = builder2.create<mlir::util::UnPackOp>(unionOp->getLoc(), unpacked[0]).getResults();
      orderedAttributes.setValuesForColumns(context, scope, unpackedKey);
      consumer->consume(this, builder2, context);
      builder2.create<mlir::dsa::YieldOp>(unionOp->getLoc(), mlir::ValueRange{});

      builder.create<mlir::dsa::FreeOp>(unionOp->getLoc(), aggrHt);
   }

   virtual ~UnionDistinctTranslator() {}
};

class CountingSetTranslator : public SetOpTranslator {
   mlir::Location loc;
   mlir::Value aggrHt;
   bool distinct;
   bool except;
   mlir::TupleType aggrTupleType;

   public:
   CountingSetTranslator(mlir::relalg::IntersectOp intersectOp) : SetOpTranslator(intersectOp), loc(intersectOp->getLoc()), distinct(intersectOp.set_semantic() == mlir::relalg::SetSemantic::distinct), except(false) {}
   CountingSetTranslator(mlir::relalg::ExceptOp exceptOp) : SetOpTranslator(exceptOp), loc(exceptOp->getLoc()), distinct(exceptOp.set_semantic() == mlir::relalg::SetSemantic::distinct), except(true) {}

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      size_t updateWhich = child == children[0].get() ? 0 : 1;
      mlir::Value emptyVals = builder.create<mlir::util::UndefOp>(loc, mlir::TupleType::get(builder.getContext()));
      mlir::Value packedKey = pack(child, builder, context);
      auto reduceOp = builder.create<mlir::dsa::HashtableInsert>(loc, aggrHt, packedKey, emptyVals);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      reduceOp.equal().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({packedKey.getType(), packedKey.getType()}, {loc, loc});
      {
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         auto yieldOp = builder.create<mlir::dsa::YieldOp>(loc);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value matches = compareKeys(builder, aggrBuilderBlock->getArgument(0), aggrBuilderBlock->getArgument(1),loc);
         builder.create<mlir::dsa::YieldOp>(loc, matches);
         yieldOp.erase();
      }
      {
         mlir::Block* aggrBuilderBlock = new mlir::Block;
         reduceOp.hash().push_back(aggrBuilderBlock);
         aggrBuilderBlock->addArguments({packedKey.getType()}, {loc});
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::Value hashed = builder.create<mlir::db::Hash>(loc, builder.getIndexType(), aggrBuilderBlock->getArgument(0));
         builder.create<mlir::dsa::YieldOp>(loc, hashed);
      }
      {
         mlir::Block* aggrBuilderBlock = new mlir::Block;
         reduceOp.reduce().push_back(aggrBuilderBlock);
         aggrBuilderBlock->addArguments({aggrTupleType, emptyVals.getType()}, {loc, loc});
         mlir::OpBuilder::InsertionGuard guard(builder);
         builder.setInsertionPointToStart(aggrBuilderBlock);
         mlir::ValueRange unpacked = builder.create<mlir::util::UnPackOp>(loc, aggrBuilderBlock->getArgument(0)).getResults();
         std::vector<mlir::Value> updatedVals = {unpacked[0], unpacked[1]};
         auto oneI64 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64Type(), builder.getI64IntegerAttr(1));

         updatedVals[updateWhich] = builder.create<mlir::arith::AddIOp>(loc, updatedVals[updateWhich], oneI64);
         mlir::Value packed = builder.create<mlir::util::PackOp>(loc, updatedVals);
         builder.create<mlir::dsa::YieldOp>(loc, packed);
      }
   }

   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto keyTupleType = orderedAttributes.getTupleType(builder.getContext());
      aggrTupleType = mlir::TupleType::get(builder.getContext(), {builder.getI64Type(), builder.getI64Type()});
      auto zeroI64 = builder.create<mlir::arith::ConstantOp>(loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
      mlir::Value emptyTuple = builder.create<mlir::util::PackOp>(loc, mlir::ValueRange{zeroI64, zeroI64});
      aggrHt = builder.create<mlir::dsa::CreateDS>(loc, mlir::dsa::AggregationHashtableType::get(builder.getContext(), keyTupleType, aggrTupleType), emptyTuple);

      mlir::Type entryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, aggrTupleType});
      children[0]->produce(context, builder);
      children[1]->produce(context, builder);

      auto forOp2 = builder.create<mlir::dsa::ForOp>(loc, mlir::TypeRange{}, aggrHt, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(entryType, loc);
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<mlir::util::UnPackOp>(loc, forOp2.getInductionVar()).getResults();
      auto unpackedKey = builder2.create<mlir::util::UnPackOp>(loc, unpacked[0]).getResults();
      orderedAttributes.setValuesForColumns(context, scope, unpackedKey);

      auto unpackedVal = builder2.create<mlir::util::UnPackOp>(loc, unpacked[1]).getResults();
      if (except) {
         if (distinct) {
            mlir::Value leftNonZero = builder2.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, unpackedVal[0], zeroI64);
            mlir::Value rightZero = builder2.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, unpackedVal[1], zeroI64);
            mlir::Value outputTuple = builder2.create<mlir::arith::AndIOp>(loc, leftNonZero, rightZero);
            builder2.create<mlir::scf::IfOp>(
               loc, mlir::TypeRange{}, outputTuple, [&](mlir::OpBuilder& b, mlir::Location) {
               consumer->consume(this, b, context);
               b.create<mlir::scf::YieldOp>(loc); });
         } else {
            mlir::Value remaining = builder2.create<mlir::arith::SubIOp>(loc, unpackedVal[0], unpackedVal[1]);
            mlir::Value ltZ = builder2.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, remaining, zeroI64);
            remaining = builder2.create<mlir::arith::SelectOp>(loc, ltZ, zeroI64, remaining);
            mlir::Value remainingAsIdx = builder2.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), remaining);
            mlir::Value zeroIdx = builder2.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value oneIdx = builder2.create<mlir::arith::ConstantIndexOp>(loc, 1);
            builder2.create<mlir::scf::ForOp>(loc, zeroIdx, remainingAsIdx, oneIdx, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange vr) {
               consumer->consume(this, b, context);
               b.create<mlir::scf::YieldOp>(loc);
            });
         }
      } else {
         //intersect
         if (distinct) {
            mlir::Value leftNonZero = builder2.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, unpackedVal[0], zeroI64);
            mlir::Value rightNonZero = builder2.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, unpackedVal[1], zeroI64);
            mlir::Value outputTuple = builder2.create<mlir::arith::AndIOp>(loc, leftNonZero, rightNonZero);
            builder2.create<mlir::scf::IfOp>(
               loc, mlir::TypeRange{}, outputTuple, [&](mlir::OpBuilder& b, mlir::Location) {
                  consumer->consume(this, b, context);
                  b.create<mlir::scf::YieldOp>(loc); });
         } else {

            mlir::Value lGtR = builder2.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, unpackedVal[0], unpackedVal[1]);

            mlir::Value remaining = builder2.create<mlir::arith::SelectOp>(loc, lGtR, unpackedVal[1], unpackedVal[0]);

            mlir::Value remainingAsIdx = builder2.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(), remaining);
            mlir::Value zeroIdx = builder2.create<mlir::arith::ConstantIndexOp>(loc, 0);
            mlir::Value oneIdx = builder2.create<mlir::arith::ConstantIndexOp>(loc, 1);
            builder2.create<mlir::scf::ForOp>(loc, zeroIdx, remainingAsIdx, oneIdx, mlir::ValueRange{}, [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value idx, mlir::ValueRange vr) {
               consumer->consume(this, b, context);
               b.create<mlir::scf::YieldOp>(loc);
            });
         }
      }

      builder2.create<mlir::dsa::YieldOp>(loc, mlir::ValueRange{});

      builder.create<mlir::dsa::FreeOp>(loc, aggrHt);
   }

   virtual ~CountingSetTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createSetOpTranslator(mlir::Operation* setOp) {
   if (auto unionOp = mlir::dyn_cast<mlir::relalg::UnionOp>(setOp)) {
      if (unionOp.set_semantic() == SetSemantic::all) {
         return std::make_unique<UnionAllTranslator>(unionOp);
      } else {
         return std::make_unique<UnionDistinctTranslator>(unionOp);
      }
   } else if (auto intersectOp = mlir::dyn_cast<mlir::relalg::IntersectOp>(setOp)) {
      return std::make_unique<CountingSetTranslator>(intersectOp);
   }  else if (auto exceptOp = mlir::dyn_cast<mlir::relalg::ExceptOp>(setOp)) {
      return std::make_unique<CountingSetTranslator>(exceptOp);
   }else {
      assert(false && "should not happen");
   }
}