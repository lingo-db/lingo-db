#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class TopKOpLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::TopKOp topKOp;
   size_t builderId;
   mlir::Value topK;

   public:
   TopKOpLowering(mlir::relalg::TopKOp topKOp) : mlir::relalg::ProducerConsumerNode(topKOp.rel()), topKOp(topKOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(topKOp.getUsedAttributes());
      propagateInfo();
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
      mlir::Value packed = builder.create<mlir::util::PackOp>(topKOp->getLoc(), tupleType, values);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(topKOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }
   mlir::Value createSortPredicate(mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, mlir::Value trueVal, mlir::Value falseVal, size_t pos) {
      if (pos < sortCriteria.size()) {
         auto lt = builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::lt, sortCriteria[pos].first, sortCriteria[pos].second);
         auto ifOp = builder.create<mlir::db::IfOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), lt);
         mlir::Block* ifBlock = new mlir::Block;
         ifOp.thenRegion().push_back(ifBlock);
         mlir::relalg::ProducerConsumerBuilder builder1(ifOp.thenRegion());
         builder1.create<mlir::db::YieldOp>(builder.getUnknownLoc(), trueVal);
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         mlir::relalg::ProducerConsumerBuilder builder2(ifOp.elseRegion());
         auto eq = builder2.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::eq, sortCriteria[pos].first, sortCriteria[pos].second);
         auto ifOp2 = builder2.create<mlir::db::IfOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), eq);
         mlir::Block* ifBlock2 = new mlir::Block;
         ifOp2.thenRegion().push_back(ifBlock2);
         mlir::relalg::ProducerConsumerBuilder builder3(ifOp2.thenRegion());
         builder3.create<mlir::db::YieldOp>(builder.getUnknownLoc(), createSortPredicate(builder3, sortCriteria, trueVal, falseVal, pos + 1));
         mlir::Block* elseBlock2 = new mlir::Block;
         ifOp2.elseRegion().push_back(elseBlock2);
         mlir::relalg::ProducerConsumerBuilder builder4(ifOp2.elseRegion());
         builder4.create<mlir::db::YieldOp>(builder.getUnknownLoc(), falseVal);

         builder2.create<mlir::db::YieldOp>(builder.getUnknownLoc(), ifOp2.getResult(0));

         return ifOp.getResult(0);
      } else {
         return falseVal;
      }
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
      mlir::Value maxRows=builder.create<mlir::ConstantOp>(topKOp->getLoc(),builder.getIndexType(),builder.getIndexAttr(topKOp.rows()));
      auto topKBuilder = builder.create<mlir::db::CreateTopKBuilder>(topKOp.getLoc(), mlir::db::TopKBuilderType::get(builder.getContext(), tupleType),maxRows);
      {
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(tupleType);
            block2->addArguments(tupleType);
            topKBuilder.region().push_back(block2);
            mlir::relalg::ProducerConsumerBuilder builder2(topKBuilder.region());
            auto unpackedLeft = builder2.create<mlir::util::UnPackOp>(topKOp->getLoc(), types, block2->getArgument(0));
            auto unpackedRight = builder2.create<mlir::util::UnPackOp>(topKOp->getLoc(), types, block2->getArgument(1));
            std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
            for (auto attr : topKOp.sortspecs()) {
               auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
               mlir::Value left = unpackedLeft.getResult(attributePos[&sortspecAttr.getAttr().getRelationalAttribute()]);
               mlir::Value right = unpackedRight.getResult(attributePos[&sortspecAttr.getAttr().getRelationalAttribute()]);
               if (sortspecAttr.getSortSpec() == mlir::relalg::SortSpec::desc) {
                  std::swap(left, right);
               }
               sortCriteria.push_back({left, right});
            }
            auto trueVal = builder2.create<mlir::db::ConstantOp>(topKOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
            auto falseVal = builder2.create<mlir::db::ConstantOp>(topKOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 0));

            builder2.create<mlir::db::YieldOp>(topKOp->getLoc(), createSortPredicate(builder2, sortCriteria, trueVal, falseVal, 0));
         }
      builderId = context.getBuilderId();
      context.builders[builderId] = topKBuilder;
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      topK = builder.create<mlir::db::BuilderBuild>(topKOp.getLoc(), mlir::db::TopKType::get(builder.getContext(), tupleType), topKBuilder);

      {
         auto forOp2 = builder.create<mlir::db::ForOp>(topKOp->getLoc(), getRequiredBuilderTypes(context), topK,flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(topKOp->getLoc(), types, forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : requiredAttributes) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(topKOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }

   virtual ~TopKOpLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredTopKOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::TopKOp topKOp) {
  return std::make_unique<TopKOpLowering>(topKOp);
});