#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class ProjectionTranslator : public mlir::relalg::Translator {
   mlir::relalg::ProjectionOp projectionOp;

   public:
   ProjectionTranslator(mlir::relalg::ProjectionOp projectionOp) : mlir::relalg::Translator(projectionOp), projectionOp(projectionOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }

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
   size_t builderId;
   mlir::Value table;
   std::vector<mlir::Type> keyTypes;
   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;
   mlir::TupleType entryType;

   std::vector<const mlir::relalg::RelationalAttribute*> groupAttributes;
   std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> keyMapping;

   public:
   DistinctProjectionTranslator(mlir::relalg::ProjectionOp projectionOp) : mlir::relalg::Translator(projectionOp), projectionOp(projectionOp) {
   }

   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value htBuilder = context.builders[builderId];
      mlir::Value emptyVals = builder.create<mlir::util::UndefTupleOp>(projectionOp->getLoc(), valTupleType);
      mlir::Value packedKey = packValues(context, builder, projectionOp->getLoc(), groupAttributes);
      mlir::Value packed = builder.create<mlir::util::PackOp>(projectionOp->getLoc(), mlir::ValueRange({packedKey, emptyVals}));

      auto mergedBuilder = builder.create<mlir::db::BuilderMerge>(projectionOp->getLoc(), htBuilder.getType(), htBuilder, packed);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      mergedBuilder.fn().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({valTupleType, valTupleType});
      mlir::OpBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      builder2.create<mlir::db::YieldOp>(projectionOp->getLoc(), aggrBuilderBlock->getArgument(0));
      context.builders[builderId] = mergedBuilder.result_builder();
   }

   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();

      for (auto attr : projectionOp.attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            keyMapping[&attrRef.getRelationalAttribute()] = keyTypes.size();
            keyTypes.push_back(attrRef.getRelationalAttribute().type);
            groupAttributes.push_back(&attrRef.getRelationalAttribute());
         }
      }
      keyTupleType = mlir::TupleType::get(builder.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(builder.getContext(), {});
      mlir::Value emptyTuple = builder.create<mlir::util::UndefTupleOp>(projectionOp.getLoc(), mlir::TupleType::get(builder.getContext()));
      auto aggrBuilder = builder.create<mlir::db::CreateAggrHTBuilder>(projectionOp.getLoc(), mlir::db::AggrHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType, valTupleType), emptyTuple);
      builderId = context.getBuilderId();
      context.builders[builderId] = aggrBuilder;
      entryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, valTupleType});
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      mlir::Value hashtable = builder.create<mlir::db::BuilderBuild>(projectionOp.getLoc(), mlir::db::AggregationHashtableType::get(builder.getContext(), keyTupleType, valTupleType), context.builders[builderId]);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(projectionOp->getLoc(), getRequiredBuilderTypes(context), hashtable, context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(entryType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(projectionOp->getLoc(), forOp2.getInductionVar()).getResults();
         auto unpackedKey = builder2.create<mlir::util::UnPackOp>(projectionOp->getLoc(), unpacked[0]).getResults();

         for (const auto* attr : requiredAttributes) {
            if (keyMapping.count(attr)) {
               context.setValueForAttribute(scope, attr, unpackedKey[keyMapping[attr]]);
            }
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(projectionOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      builder.create<mlir::db::FreeOp>(projectionOp->getLoc(), hashtable);
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