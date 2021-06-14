#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class AggregationLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::AggregationOp aggregationOp;
   size_t builderId;

   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;
   mlir::TupleType insertEntryType;

   std::vector<const mlir::relalg::RelationalAttribute*> keyAttributes;
   std::vector<const mlir::relalg::RelationalAttribute*> valAttributes;
   std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> keyMapping;

   public:
   AggregationLowering(mlir::relalg::AggregationOp aggregationOp) : mlir::relalg::ProducerConsumerNode(aggregationOp.rel()), aggregationOp(aggregationOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(aggregationOp.getUsedAttributes());
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return aggregationOp.getAvailableAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      std::vector<mlir::Value> keys, values;
      for (const auto* attr : keyAttributes) {
         keys.push_back(context.getValueForAttribute(attr));
      }
      for (const auto* attr : valAttributes) {
         values.push_back(context.getValueForAttribute(attr));
      }
      mlir::Value htBuilder = context.builders[builderId];
      mlir::Value packedKey = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), keyTupleType, keys);
      mlir::Value packedVal = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), valTupleType, values);
      mlir::Value packed = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), insertEntryType, mlir::ValueRange({packedKey, packedVal}));

      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(aggregationOp->getLoc(), htBuilder.getType(), htBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }

   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      std::vector<mlir::Type> keyTypes;
      std::vector<mlir::Type> valTypes;
      auto scope = context.createScope();

      std::vector<std::function<std::pair<const mlir::relalg::RelationalAttribute*, mlir::Value>(mlir::ValueRange)>> finalizeFunctions;
      std::vector<std::function<std::vector<mlir::Value>(mlir::ValueRange, mlir::ValueRange, mlir::OpBuilder & builder)>> aggregationFunctions;
      for (auto attr : aggregationOp.group_by_attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            keyTypes.push_back(attrRef.getRelationalAttribute().type);
            keyAttributes.push_back(&attrRef.getRelationalAttribute());
            keyMapping.insert({&attrRef.getRelationalAttribute(), keyTypes.size() - 1});
         }
      }
      std::vector<mlir::Value> defaultValues;
      std::vector<mlir::Type> aggrTypes;
      aggregationOp.aggr_func().walk([&](mlir::relalg::AddAttrOp addAttrOp) {
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(addAttrOp.val().getDefiningOp())) {
            auto *destAttr = &addAttrOp.attr().getRelationalAttribute();
            auto *attr = &aggrFn.attr().getRelationalAttribute();
            size_t currValIdx = valAttributes.size();
            valAttributes.push_back(attr);
            valTypes.push_back(attr->type);
            mlir::Type resultingType = addAttrOp.attr().getRelationalAttribute().type;
            if (aggrFn.fn() == mlir::relalg::AggrFunc::sum) {
               size_t currDestIdx = aggrTypes.size();
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               auto initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), resultingType.cast<mlir::db::DBType>().getBaseType(), builder.getI64IntegerAttr(0));
               defaultValues.push_back(builder.create<mlir::db::CastOp>(aggregationOp.getLoc(), resultingType, initVal));
               aggregationFunctions.push_back([currDestIdx = currDestIdx, currValIdx = currValIdx, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  res.push_back(builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), resultingType, aggr[currDestIdx], val[currValIdx]));
                  return res;
               });
            }
         }
      });
      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      auto initTuple = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), aggrTupleType, defaultValues);
      keyTupleType = mlir::TupleType::get(builder.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(builder.getContext(), valTypes);
      auto aggrBuilder = builder.create<mlir::db::CreateAggrHTBuilder>(aggregationOp.getLoc(), mlir::db::AggrHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType), initTuple);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      aggrBuilder.region().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({aggrTupleType, valTupleType});
      mlir::relalg::ProducerConsumerBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      auto unpackedCurr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrTypes, aggrBuilderBlock->getArgument(0))->getResults();
      auto unpackedNew = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), valTypes, aggrBuilderBlock->getArgument(1)).getResults();
      std::vector<mlir::Value> values;
      for (auto aggrFn : aggregationFunctions) {
         auto vec = aggrFn(unpackedCurr, unpackedNew, builder2);
         values.insert(values.end(), vec.begin(), vec.end());
      }

      mlir::Value packed = builder2.create<mlir::util::PackOp>(aggregationOp->getLoc(), aggrTupleType, values);

      builder2.create<mlir::db::YieldOp>(builder.getUnknownLoc(), packed);
      builderId = context.getBuilderId();
      context.builders[builderId] = aggrBuilder;
      this->insertEntryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, valTupleType});

      auto iterEntryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, aggrTupleType});
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      mlir::Value hashtable = builder.create<mlir::db::BuilderBuild>(aggregationOp.getLoc(), mlir::db::AggregationHashtableType::get(builder.getContext(), keyTupleType, aggrTupleType), aggrBuilder);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(aggregationOp->getLoc(), getRequiredBuilderTypes(context), hashtable, flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(iterEntryType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), iterEntryType.getTypes(), forOp2.getInductionVar()).getResults();
         auto unpackedKey = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), keyTypes, unpacked[0]).getResults();
         auto unpackedAggr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrTypes, unpacked[1]).getResults();

         for (auto fn : finalizeFunctions) {
            auto [attr, val] = fn(unpackedAggr);
            context.setValueForAttribute(scope, attr, val);
         }
         for (auto *attr : requiredAttributes) {
            if (keyMapping.count(attr)) {
               context.setValueForAttribute(scope, attr, unpackedKey[keyMapping[attr]]);
            }
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(aggregationOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }
   virtual void done() override {
   }
   virtual ~AggregationLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredAggregationOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::AggregationOp sortOp) {
   return std::make_unique<AggregationLowering>(sortOp);
});