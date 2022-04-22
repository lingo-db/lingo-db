#include "mlir/Conversion/RelAlgToDB/HashJoinTranslator.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <mlir/Conversion/RelAlgToDB/NLJoinTranslator.h>
#include <mlir/Dialect/DSA/IR/DSAOps.h>
#include <mlir/IR/BlockAndValueMapping.h>

class SimpleInnerJoinImpl : public mlir::relalg::JoinImpl {
   public:
   SimpleInnerJoinImpl(mlir::relalg::InnerJoinOp crossProductOp) : mlir::relalg::JoinImpl(crossProductOp, crossProductOp.left(), crossProductOp.right()) {}
   SimpleInnerJoinImpl(mlir::relalg::CrossProductOp crossProductOp) : mlir::relalg::JoinImpl(crossProductOp, crossProductOp.left(), crossProductOp.right()) {}

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched);
   }
   virtual ~SimpleInnerJoinImpl() {}
};
std::shared_ptr<mlir::relalg::JoinImpl> createCrossProductImpl(mlir::relalg::CrossProductOp crossProductOp) {
   return std::make_shared<SimpleInnerJoinImpl>(crossProductOp);
}
std::shared_ptr<mlir::relalg::JoinImpl> createInnerJoinImpl(mlir::relalg::InnerJoinOp joinOp) {
   return std::make_shared<SimpleInnerJoinImpl>(joinOp);
}

class CollectionJoinImpl : public mlir::relalg::JoinImpl {
   mlir::relalg::OrderedAttributes cols;
   mlir::Value vector;

   public:
   CollectionJoinImpl(mlir::relalg::CollectionJoinOp collectionJoinOp) : mlir::relalg::JoinImpl(collectionJoinOp, collectionJoinOp.right(), collectionJoinOp.left()) {
      cols = mlir::relalg::OrderedAttributes::fromRefArr(collectionJoinOp.cols());
   }
   virtual void addAdditionalRequiredColumns() override {
      for (const auto* attr : cols.getAttrs()) {
         translator->requiredAttributes.insert(attr);
      }
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::scf::IfOp>(
         loc, mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder, mlir::Location loc) {
            mlir::Value packed = cols.pack(context,builder,loc);
            builder.create<mlir::dsa::Append>(loc, vector, packed);
            builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{}); }, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{}); });
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      vector = builder.create<mlir::dsa::CreateDS>(joinOp.getLoc(), mlir::dsa::VectorType::get(builder.getContext(), cols.getTupleType(builder.getContext())));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      context.setValueForAttribute(scope, &cast<mlir::relalg::CollectionJoinOp>(joinOp).collAttr().getColumn(), vector);
      translator->forwardConsume(builder, context);
      builder.create<mlir::dsa::FreeOp>(loc, vector);
   }
   virtual ~CollectionJoinImpl() {}
};
std::shared_ptr<mlir::relalg::JoinImpl> createCollectionJoinImpl(mlir::relalg::CollectionJoinOp joinOp) {
   return std::make_shared<CollectionJoinImpl>(joinOp);
}

class OuterJoinTranslator : public mlir::relalg::JoinImpl {
   public:
   OuterJoinTranslator(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      this->stopOnFlag = false;
   }
   OuterJoinTranslator(mlir::relalg::SingleJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      this->stopOnFlag = false;
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMapping(builder, context, scope);
         auto trueVal = builder.create<mlir::db::ConstantOp>(loc, builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 1));
         builder.create<mlir::dsa::SetFlag>(loc, matchFoundFlag, trueVal);
      });
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::dsa::CreateFlag>(loc, mlir::dsa::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::dsa::GetFlag>(loc, builder.getI1Type(), matchFoundFlag);
      mlir::Value noMatchFound = builder.create<mlir::db::NotOp>(loc, builder.getI1Type(), matchFound);
      translator->handlePotentialMatch(builder, context, noMatchFound, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMappingNull(builder, context, scope);
      });
   }

   virtual ~OuterJoinTranslator() {}
};

class ReversedOuterJoinImpl : public mlir::relalg::JoinImpl {
   public:
   ReversedOuterJoinImpl(mlir::relalg::OuterJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->handlePotentialMatch(builder, context, matched, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         auto const1 = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerType(64), builder.getI64IntegerAttr(1));
         builder.create<mlir::memref::AtomicRMWOp>(loc, builder.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
         translator->handleMapping(builder, context, scope);
      });
   }
   virtual void after(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto zero = builder.create<mlir::arith::ConstantOp>(loc, marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, marker, zero);
      translator->handlePotentialMatch(builder, context, isZero, [&](mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context, mlir::relalg::TranslatorContext::AttributeResolverScope& scope) {
         translator->handleMappingNull(builder, context, scope);
      });
   }

   virtual ~ReversedOuterJoinImpl() {}
};

std::shared_ptr<mlir::relalg::JoinImpl> createOuterJoinImpl(mlir::relalg::OuterJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<ReversedOuterJoinImpl>(joinOp) : (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<OuterJoinTranslator>(joinOp);
}

class SemiJoinImpl : public mlir::relalg::JoinImpl {
   bool doAnti = false;

   public:
   SemiJoinImpl(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }
   SemiJoinImpl(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
      doAnti = true;
   }
   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::dsa::SetFlag>(loc, matchFoundFlag, matched);
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::dsa::CreateFlag>(loc, mlir::dsa::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      mlir::Value matchFound = builder.create<mlir::dsa::GetFlag>(loc, builder.getI1Type(), matchFoundFlag);
      mlir::Value emit = matchFound;
      if (doAnti) {
         emit = builder.create<mlir::db::NotOp>(loc, builder.getI1Type(), matchFound);
      }
      translator->handlePotentialMatch(builder, context, emit);
   }
   virtual ~SemiJoinImpl() {}
};
class ReversedSemiJoinImpl : public mlir::relalg::JoinImpl {
   public:
   ReversedSemiJoinImpl(mlir::relalg::SemiJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::scf::IfOp>(
         loc, mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder1, mlir::Location) {
            auto const1 = builder1.create<mlir::arith::ConstantOp>(loc, builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
            auto markerBefore = builder1.create<mlir::memref::AtomicRMWOp>(loc, builder1.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
            {
               auto zero = builder1.create<mlir::arith::ConstantOp>(loc, markerBefore.getType(), builder1.getIntegerAttr(markerBefore.getType(), 0));
               auto isZero = builder1.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, markerBefore, zero);
               translator->handlePotentialMatch(builder,context,isZero);
            }
            builder1.create<mlir::scf::YieldOp>(loc); });
   }

   virtual ~ReversedSemiJoinImpl() {}
};

std::shared_ptr<mlir::relalg::JoinImpl> createSemiJoinImpl(mlir::relalg::SemiJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<ReversedSemiJoinImpl>(joinOp) : (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<SemiJoinImpl>(joinOp);
};

class ReversedAntiSemiJoinImpl : public mlir::relalg::JoinImpl {
   public:
   ReversedAntiSemiJoinImpl(mlir::relalg::AntiSemiJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.left(), innerJoinOp.right(), true) {}

   virtual void handleLookup(mlir::Value matched, mlir::Value markerPtr, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::scf::IfOp>(
         loc, mlir::TypeRange{}, matched, [&](mlir::OpBuilder& builder1, mlir::Location loc) {
            auto const1 = builder1.create<mlir::arith::ConstantOp>(loc, builder1.getIntegerType(64), builder1.getI64IntegerAttr(1));
            builder1.create<mlir::memref::AtomicRMWOp>(loc, builder1.getIntegerType(64), mlir::arith::AtomicRMWKind::assign, const1, markerPtr, mlir::ValueRange{});
            builder1.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{}); });
   }
   virtual void after(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      translator->scanHT(context, builder);
   }
   void handleScanned(mlir::Value marker, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto zero = builder.create<mlir::arith::ConstantOp>(loc, marker.getType(), builder.getIntegerAttr(marker.getType(), 0));
      auto isZero = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, marker, zero);
      translator->handlePotentialMatch(builder, context, isZero);
   }

   virtual ~ReversedAntiSemiJoinImpl() {}
};
std::shared_ptr<mlir::relalg::JoinImpl> createAntiSemiJoinImpl(mlir::relalg::AntiSemiJoinOp joinOp, bool reversed) {
   return reversed ? (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<ReversedAntiSemiJoinImpl>(joinOp) : (std::shared_ptr<mlir::relalg::JoinImpl>) std::make_shared<SemiJoinImpl>(joinOp);
};

class ConstantSingleJoinTranslator : public mlir::relalg::Translator {
   mlir::relalg::SingleJoinOp joinOp;
   std::vector<const mlir::relalg::Column*> cols;
   std::vector<const mlir::relalg::Column*> origAttrs;
   std::vector<mlir::Type> types;
   mlir::Value singleValPtr;

   public:
   ConstantSingleJoinTranslator(mlir::relalg::SingleJoinOp singleJoinOp) : mlir::relalg::Translator(singleJoinOp), joinOp(singleJoinOp) {
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(joinOp.getUsedColumns());
      for (mlir::Attribute attr : joinOp.mapping()) {
         auto relationDefAttr = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         auto* defAttr = &relationDefAttr.getColumn();
         if (this->requiredAttributes.contains(defAttr)) {
            auto fromExisting = relationDefAttr.getFromExisting().dyn_cast_or_null<mlir::ArrayAttr>();
            const auto* refAttr = *mlir::relalg::ColumnSet::fromArrayAttr(fromExisting).begin();
            this->requiredAttributes.insert(refAttr);
            origAttrs.push_back(refAttr);
            cols.push_back(defAttr);
            types.push_back(defAttr->type);
         }
      }
      propagateInfo();
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      auto scope = context.createScope();
      if (child == this->children[0].get()) {
         mlir::Value singleVal = builder.create<mlir::util::LoadOp>(joinOp->getLoc(), singleValPtr.getType().cast<mlir::util::RefType>().getElementType(), singleValPtr, mlir::Value());
         auto unpacked = builder.create<mlir::util::UnPackOp>(joinOp->getLoc(), singleVal);
         for (size_t i = 0; i < cols.size(); i++) {
            context.setValueForAttribute(scope, cols[i], unpacked.getResult(i));
         }
         consumer->consume(this, builder, context);
      } else if (child == this->children[1].get()) {
         std::vector<mlir::Value> values;
         for (size_t i = 0; i < origAttrs.size(); i++) {
            mlir::Value value = context.getValueForAttribute(origAttrs[i]);
            if (origAttrs[i]->type != cols[i]->type) {
               mlir::Value tmp = builder.create<mlir::db::AsNullableOp>(op->getLoc(), cols[i]->type, value);
               value = tmp;
            }
            values.push_back(value);
         }
         mlir::Value singleVal = builder.create<mlir::util::PackOp>(joinOp->getLoc(), values);
         builder.create<mlir::util::StoreOp>(joinOp->getLoc(), singleVal, singleValPtr, mlir::Value());
      }
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      std::vector<mlir::Value> values;
      for (auto type : types) {
         values.push_back(builder.create<mlir::db::NullOp>(joinOp.getLoc(), type));
      }
      mlir::Value singleVal = builder.create<mlir::util::PackOp>(joinOp->getLoc(), values);
      singleValPtr = builder.create<mlir::util::AllocaOp>(joinOp->getLoc(), mlir::util::RefType::get(builder.getContext(), singleVal.getType()), mlir::Value());
      builder.create<mlir::util::StoreOp>(joinOp->getLoc(), singleVal, singleValPtr, mlir::Value());
      children[1]->produce(context, builder);
      children[0]->produce(context, builder);
   }

   virtual ~ConstantSingleJoinTranslator() {}
};

std::shared_ptr<mlir::relalg::JoinImpl> createSingleJoinImpl(mlir::relalg::SingleJoinOp joinOp) {
   return std::make_shared<OuterJoinTranslator>(joinOp);
}
std::unique_ptr<mlir::relalg::Translator> createConstSingleJoinTranslator(mlir::relalg::SingleJoinOp joinOp) {
   return std::make_unique<ConstantSingleJoinTranslator>(joinOp);
}

class MarkJoinImpl : public mlir::relalg::JoinImpl {
   public:
   MarkJoinImpl(mlir::relalg::MarkJoinOp innerJoinOp) : mlir::relalg::JoinImpl(innerJoinOp, innerJoinOp.right(), innerJoinOp.left()) {
   }

   virtual void handleLookup(mlir::Value matched, mlir::Value /*marker*/, mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      builder.create<mlir::dsa::SetFlag>(loc, matchFoundFlag, matched);
   }

   void beforeLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      matchFoundFlag = builder.create<mlir::dsa::CreateFlag>(loc, mlir::dsa::FlagType::get(builder.getContext()));
   }
   void afterLookup(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      mlir::Value matchFound = builder.create<mlir::dsa::GetFlag>(loc, builder.getI1Type(), matchFoundFlag);
      context.setValueForAttribute(scope, &cast<mlir::relalg::MarkJoinOp>(joinOp).markattr().getColumn(), matchFound);
      translator->forwardConsume(builder, context);
   }
   virtual ~MarkJoinImpl() {}
};

std::shared_ptr<mlir::relalg::JoinImpl> createMarkJoinImpl(mlir::relalg::MarkJoinOp joinOp) {
   return std::make_shared<MarkJoinImpl>(joinOp);
}
std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createJoinTranslator(mlir::Operation* joinOp) {
   bool reversed = false;
   bool hash = false;
   bool constant = false;
   if (joinOp->hasAttr("impl")) {
      if (auto impl = joinOp->getAttr("impl").dyn_cast_or_null<mlir::StringAttr>()) {
         if (impl.getValue() == "hash") {
            hash = true;
         }
         if (impl.getValue() == "markhash") {
            hash = true;
            reversed = true;
         }
         if (impl.getValue() == "constant") {
            constant = true;
         }
      }
   }
   if (constant) {
      return createConstSingleJoinTranslator(mlir::cast<SingleJoinOp>(joinOp));
   }
   auto joinImpl = ::llvm::TypeSwitch<mlir::Operation*, std::shared_ptr<mlir::relalg::JoinImpl>>(joinOp)
                      .Case<CrossProductOp>([&](auto x) { return createCrossProductImpl(x); })
                      .Case<InnerJoinOp>([&](auto x) { return createInnerJoinImpl(x); })
                      .Case<SemiJoinOp>([&](auto x) { return createSemiJoinImpl(x, reversed); })
                      .Case<AntiSemiJoinOp>([&](auto x) { return createAntiSemiJoinImpl(x, reversed); })
                      .Case<OuterJoinOp>([&](auto x) { return createOuterJoinImpl(x, reversed); })
                      .Case<SingleJoinOp>([&](auto x) { return createSingleJoinImpl(x); })
                      .Case<MarkJoinOp>([&](auto x) { return createMarkJoinImpl(x); })
                      .Case<CollectionJoinOp>([&](auto x) { return createCollectionJoinImpl(x); })
                      .Default([](auto x) { assert(false&&"should not happen"); return std::shared_ptr<JoinImpl>(); });

   if (hash) {
      return std::make_unique<mlir::relalg::HashJoinTranslator>(joinImpl);
   } else {
      return std::make_unique<mlir::relalg::NLJoinTranslator>(joinImpl);
   }
}