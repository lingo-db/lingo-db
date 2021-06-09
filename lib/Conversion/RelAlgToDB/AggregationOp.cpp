#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class AggregationLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::AggregationOp aggregationOp;
   size_t builderId;
   mlir::Value table;
   enum AggWorkFn {
      SUM
   };
   struct AggWork {
      const mlir::relalg::RelationalAttribute* source;
      const mlir::relalg::RelationalAttribute* res;
      AggWorkFn workfn;
   };
   std::vector<mlir::Type> keyTypes;
   std::vector<mlir::Type> valTypes;
   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;
   mlir::TupleType entryType;

   std::vector<AggWork> work;
   std::vector<const mlir::relalg::RelationalAttribute*> groupAttributes;
   std::vector<const mlir::relalg::RelationalAttribute*> valAttributes;
   std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> keyMapping;
   std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> sourceMapping;
   std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> targetMapping;

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
      return aggregationOp.getCreatedAttributes(); //todo: fix
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      std::vector<mlir::Value> keys, values;
      for (const auto* attr : groupAttributes) {
         keys.push_back(context.getValueForAttribute(attr));
      }
      for (const auto* attr : valAttributes) {
         values.push_back(context.getValueForAttribute(attr));
      }
      mlir::Value htBuilder = context.builders[builderId];
      mlir::Value packedKey = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), keyTupleType, keys);
      mlir::Value packedVal = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), valTupleType, values);
      mlir::Value packed = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), entryType, mlir::ValueRange({packedKey, packedVal}));

      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(aggregationOp->getLoc(), htBuilder.getType(), htBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }

   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();

      for (auto attr : aggregationOp.group_by_attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            keyMapping[&attrRef.getRelationalAttribute()] = keyTypes.size();
            keyTypes.push_back(attrRef.getRelationalAttribute().type);
            groupAttributes.push_back(&attrRef.getRelationalAttribute());
         }
      }
      //todo arguments of aggr_builder can be different?
      size_t i = 0;
      aggregationOp.aggr_func().walk([&](mlir::relalg::AddAttrOp addAttrOp) {
        addAttrOp->dump();
        if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(addAttrOp.val().getDefiningOp())) {
           if (aggrFn.fn() == mlir::relalg::AggrFunc::sum) {
              AggWork aggWork;
              aggWork.workfn = AggWorkFn::SUM;
              aggWork.source = &aggrFn.attr().getRelationalAttribute();
              aggWork.res = &addAttrOp.attr().getRelationalAttribute();
              valTypes.push_back(aggWork.source->type);
              valAttributes.push_back(aggWork.source);
              sourceMapping[aggWork.source] = i;
              targetMapping[aggWork.res] = i;
              i++;
              work.push_back(aggWork);
           }
        }
      });
      keyTupleType = mlir::TupleType::get(builder.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(builder.getContext(), valTypes);
      auto aggrBuilder = builder.create<mlir::db::CreateAggrHTBuilder>(aggregationOp.getLoc(), mlir::db::AggrHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType));
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      aggrBuilder.region().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({valTupleType, valTupleType});
      mlir::relalg::ProducerConsumerBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      auto unpackedCurr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), valTypes, aggrBuilderBlock->getArgument(0))->getResults();
      auto unpackedNew = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), valTypes, aggrBuilderBlock->getArgument(1)).getResults();
      std::vector<mlir::Value> values;
      for (auto w : work) {
         if (w.workfn == SUM) {
            values.push_back(builder2.create<mlir::db::AddOp>(aggregationOp->getLoc(), w.source->type, unpackedCurr[sourceMapping[w.source]], unpackedNew[sourceMapping[w.source]]));
         }
      }

      mlir::Value packed = builder2.create<mlir::util::PackOp>(aggregationOp->getLoc(), valTupleType, values);

      builder2.create<mlir::db::YieldOp>(builder.getUnknownLoc(), packed);
      builderId = context.getBuilderId();
      context.builders[builderId] = aggrBuilder;
      entryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, valTupleType});
      children[0]->setRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      mlir::Value hashtable = builder.create<mlir::db::BuilderBuild>(aggregationOp.getLoc(), mlir::db::AggregationHashtableType::get(builder.getContext(), keyTupleType, valTupleType), aggrBuilder);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(aggregationOp->getLoc(), getRequiredBuilderTypes(context), hashtable, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(entryType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), entryType.getTypes(), forOp2.getInductionVar()).getResults();
         auto unpackedKey = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), keyTypes, unpacked[0]).getResults();
         auto unpackedVal = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), valTypes, unpacked[1]).getResults();

         for (const auto* attr : requiredAttributes) {
            if (targetMapping.count(attr)) {
               mlir::Value casted = builder2.create<mlir::db::CastOp>(aggregationOp->getLoc(), unpackedVal[targetMapping[attr]].getType().dyn_cast_or_null<mlir::db::DBType>().asNullable(), unpackedVal[targetMapping[attr]]); //unpackedVal[targetMapping[attr]]
               context.setValueForAttribute(scope, attr, casted);
            } else if (keyMapping.count(attr)) {
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