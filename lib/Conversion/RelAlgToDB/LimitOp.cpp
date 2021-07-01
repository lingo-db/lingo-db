#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class LimitLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::LimitOp limitOp;
   size_t builderId;
   size_t counterId;
   mlir::Value vector;
   mlir::Value finishedFlag;

   public:
   LimitLowering(mlir::relalg::LimitOp limitOp) : mlir::relalg::ProducerConsumerNode(limitOp.rel()), limitOp(limitOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(limitOp.getUsedAttributes());
      propagateInfo();
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      std::vector<mlir::Type> types;
      std::vector<mlir::Value> values;
      for (const auto* attr : requiredAttributes) {
         types.push_back(attr->type);
         values.push_back(context.getValueForAttribute(attr));
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Value vectorBuilder = context.builders[builderId];
      mlir::Value counter = context.builders[counterId];
      mlir::Value packed = builder.create<mlir::util::PackOp>(limitOp->getLoc(), tupleType, values);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(limitOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      auto one = builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), counter.getType(), builder.getI64IntegerAttr(1));
      mlir::Value addedCounter = builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), counter.getType(), counter, one);
      mlir::Value upper=builder.create<mlir::db::ConstantOp>(limitOp.getLoc(),counter.getType(),builder.getI64IntegerAttr(limitOp.rows()));
      mlir::Value finished=builder.create<mlir::db::CmpOp>(limitOp.getLoc(),mlir::db::DBCmpPredicate::gte,addedCounter,upper);
      builder.create<mlir::db::SetFlag>(limitOp->getLoc(), finishedFlag,finished);
      context.builders[builderId] = mergedBuilder;
      context.builders[counterId] = addedCounter;
   }

   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();
      std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> attributePos;
      std::vector<mlir::Type> types;
      size_t i = 0;
      for (const auto* attr : requiredAttributes) {
         types.push_back(attr->type);
         attributePos[attr] = i++;
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(limitOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
      mlir::Value counter = builder.create<mlir::db::ConstantOp>(limitOp.getLoc(), mlir::db::IntType::get(builder.getContext(), false, 64),builder.getI64IntegerAttr(0));
      finishedFlag = builder.create<mlir::db::CreateFlag>(limitOp->getLoc(), mlir::db::FlagType::get(builder.getContext()));
      children[0]->setFlag(finishedFlag);
      builderId = context.getBuilderId();
      counterId = context.getBuilderId();
      context.builders[builderId] = vectorBuilder;
      context.builders[counterId] = counter;

      children[0]->addRequiredBuilders({builderId,counterId});
      children[0]->produce(context, builder);
      vector = builder.create<mlir::db::BuilderBuild>(limitOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), vectorBuilder);
      
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(limitOp->getLoc(), getRequiredBuilderTypes(context), vector,flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(limitOp->getLoc(), types, forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : requiredAttributes) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(limitOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }

   virtual ~LimitLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredLimitOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::LimitOp limitOp) {
  return std::make_unique<LimitLowering>(limitOp);
});