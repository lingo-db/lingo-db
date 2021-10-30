#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class TmpLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::TmpOp tmpOp;
   size_t builderId;
   bool materialize;
   std::vector<mlir::relalg::RelationalAttribute*> attributes;
   size_t userCount;
   size_t producedCount;

   public:
   TmpLowering(mlir::relalg::TmpOp tmpOp) : mlir::relalg::ProducerConsumerNode(tmpOp.rel()), tmpOp(tmpOp) {
      std::vector<mlir::Operation*> users(tmpOp->getUsers().begin(),tmpOp->getUsers().end());
      userCount=users.size();
      producedCount=0;

   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(mlir::relalg::Attributes::fromArrayAttr(tmpOp.attrs()));
      propagateInfo();
      for (auto* attr : this->requiredAttributes) {
         attributes.push_back(attr);
      }
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      this->children[0]->addRequiredBuilders(requiredBuilders);
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      if (materialize) {
         std::vector<mlir::Type> types;
         std::vector<mlir::Value> values;
         for (const auto* attr : attributes) {
            types.push_back(attr->type);
            values.push_back(context.getValueForAttribute(attr));
         }
         mlir::Value vectorBuilder = context.builders[builderId];
         mlir::Value packed = builder.create<mlir::util::PackOp>(tmpOp->getLoc(), values);
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(tmpOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
         context.builders[builderId] = mergedBuilder;
         consumer->consume(this, builder, context);
      }
   }

   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();

      materialize = !context.materializedTmp.count(tmpOp.getOperation());
      producedCount++;
      if (materialize) {
         std::vector<mlir::Type> types;
         for (const auto* attr : attributes) {
            types.push_back(attr->type);
         }
         auto tupleType = mlir::TupleType::get(builder.getContext(), types);
         std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> attributePos;

         mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(tmpOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
         builderId = context.getBuilderId();
         context.builders[builderId] = vectorBuilder;

         children[0]->addRequiredBuilders({builderId});
         children[0]->produce(context, builder);
         mlir::Value vector = builder.create<mlir::db::BuilderBuild>(tmpOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), context.builders[builderId]);
         context.materializedTmp[tmpOp.getOperation()]={vector,attributes};
      } else {
         auto [vector,attributes]=context.materializedTmp[tmpOp.getOperation()];
         std::vector<mlir::Type> types;
         for (const auto* attr : attributes) {
            types.push_back(attr->type);
         }
         auto tupleType = mlir::TupleType::get(builder.getContext(), types);
         auto forOp2 = builder.create<mlir::db::ForOp>(tmpOp->getLoc(), getRequiredBuilderTypes(context), vector, flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(tmpOp->getLoc(), forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : attributes) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(tmpOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      if(producedCount>=userCount) {
         auto [vector,attributes]=context.materializedTmp[tmpOp.getOperation()];
         builder.create<mlir::db::FreeOp>(tmpOp->getLoc(), vector);
      }
   }

   virtual ~TmpLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredTmpOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::TmpOp tmpOp) {
   return std::make_unique<TmpLowering>(tmpOp);
});