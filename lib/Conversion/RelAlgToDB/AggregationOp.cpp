#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <llvm/ADT/TypeSwitch.h>

class AggregationLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::AggregationOp aggregationOp;
   size_t builderId;

   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;

   std::vector<const mlir::relalg::RelationalAttribute*> keyAttributes;
   std::vector<const mlir::relalg::RelationalAttribute*> valAttributes;
   std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> keyMapping;

   std::vector<std::function<std::pair<const mlir::relalg::RelationalAttribute*, mlir::Value>(mlir::ValueRange, mlir::OpBuilder & builder)>> finalizeFunctions;
   std::vector<std::function<std::vector<mlir::Value>(mlir::ValueRange, mlir::ValueRange, mlir::OpBuilder & builder)>> aggregationFunctions;
   std::vector<mlir::Type> keyTypes;
   std::vector<mlir::Type> valTypes;
   std::vector<mlir::Value> defaultValues;
   std::vector<mlir::Type> aggrTypes;

   public:
   AggregationLowering(mlir::relalg::AggregationOp aggregationOp) : mlir::relalg::ProducerConsumerNode(aggregationOp), aggregationOp(aggregationOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      //do not forwared requiredBuilders to children
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {

      std::vector<mlir::Value> keys, values;
      for (const auto* attr : keyAttributes) {
         keys.push_back(context.getValueForAttribute(attr));
      }
      for (const auto* attr : valAttributes) {
         values.push_back(context.getValueForAttribute(attr));
      }
      mlir::Value htBuilder = context.builders[builderId];
      mlir::Value packedKey;
      if (keyTupleType.getTypes().size() == 0) {
         packedKey = builder.create<mlir::util::UndefTupleOp>(aggregationOp->getLoc(), keyTupleType);
      } else {
         packedKey = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), keys);
      }
      mlir::Value packedVal;
      if (valTupleType.getTypes().size() == 0) {
         packedVal = builder.create<mlir::util::UndefTupleOp>(aggregationOp->getLoc(), valTupleType);
      } else {
         packedVal = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), values);
      }
      mlir::Value packed = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), mlir::ValueRange({packedKey, packedVal}));

      auto builderMerge = builder.create<mlir::db::BuilderMerge>(aggregationOp->getLoc(), htBuilder.getType(), htBuilder, packed);
      context.builders[builderId] = builderMerge.result_builder();


      auto scope = context.createScope();


      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      keyTupleType = mlir::TupleType::get(builder.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(builder.getContext(), valTypes);
      aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      builderMerge.fn().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({aggrTupleType, valTupleType});
      mlir::OpBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      auto unpackedCurr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(0))->getResults();
      mlir::ValueRange unpackedNew;
      if (valTypes.size() > 0) {
         unpackedNew = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(1)).getResults();
      }
      std::vector<mlir::Value> valuesx;
      for (auto aggrFn : aggregationFunctions) {
         auto vec = aggrFn(unpackedCurr, unpackedNew, builder2);
         valuesx.insert(valuesx.end(), vec.begin(), vec.end());
      }

      mlir::Value packedx = builder2.create<mlir::util::PackOp>(aggregationOp->getLoc(), valuesx);

      builder2.create<mlir::db::YieldOp>(builder.getUnknownLoc(), packedx);




   }

   mlir::Attribute getMaxValueAttr(mlir::db::DBType type) {
      auto *context = aggregationOp->getContext();
      mlir::OpBuilder builder(context);
      mlir::Attribute maxValAttr = ::llvm::TypeSwitch<::mlir::db::DBType, mlir::Attribute>(type)

                                      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                                         if (t.getP() < 19) {
                                            return (mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         }
                                         std::vector<uint64_t> parts = {0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
                                         return (mlir::Attribute) builder.getIntegerAttr(mlir::IntegerType::get(context, 128), mlir::APInt(128, parts));
                                      })
                                      .Case<::mlir::db::IntType>([&](::mlir::db::IntType t) {
                                         return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                      })
                                      .Case<::mlir::db::UIntType>([&](::mlir::db::UIntType t) {
                                         return builder.getI64IntegerAttr(std::numeric_limits<uint64_t>::max());
                                      })
                                      .Case<::mlir::db::FloatType>([&](::mlir::db::FloatType t) {
                                         if (t.getWidth() == 32) {
                                            return (mlir::Attribute) builder.getF32FloatAttr(std::numeric_limits<float>::max());
                                         } else if (t.getWidth() == 64) {
                                            return (mlir::Attribute) builder.getF64FloatAttr(std::numeric_limits<double>::max());
                                         }else{
                                            assert(false&&"should not happen");
                                            return mlir::Attribute();
                                         }
                                      })
                                      .Default([&](::mlir::Type) { return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max()); });
      return maxValAttr;
   }
   void analyze(mlir::OpBuilder& builder){
      for (auto attr : aggregationOp.group_by_attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            keyTypes.push_back(attrRef.getRelationalAttribute().type);
            keyAttributes.push_back(&attrRef.getRelationalAttribute());
            keyMapping.insert({&attrRef.getRelationalAttribute(), keyTypes.size() - 1});
         }
      }

      auto counterType = mlir::db::IntType::get(builder.getContext(), false, 64);

      aggregationOp.aggr_func().walk([&](mlir::relalg::AddAttrOp addAttrOp) {
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(addAttrOp.val().getDefiningOp())) {
            auto* destAttr = &addAttrOp.attr().getRelationalAttribute();
            auto* attr = &aggrFn.attr().getRelationalAttribute();
            auto attrIsNullable=false;
            auto dbType=attr->type.dyn_cast_or_null<mlir::db::DBType>();
            if(dbType){
               attrIsNullable=dbType.isNullable();
            }
            size_t currValIdx = valAttributes.size();
            valAttributes.push_back(attr);
            valTypes.push_back(attr->type);
            mlir::Type resultingType = addAttrOp.attr().getRelationalAttribute().type;
            size_t currDestIdx = aggrTypes.size();

            if (aggrFn.fn() == mlir::relalg::AggrFunc::sum) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.cast<mlir::db::DBType>().isNullable()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), resultingType.cast<mlir::db::DBType>().getBaseType(), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value added = builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), resultingType, currVal, newVal);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), newVal);
                     updatedVal = builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.cast<mlir::db::DBType>().isNullable()) {
                     mlir::Value casted = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), currVal.getType(), newVal);
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull, casted, added));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::min) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.cast<mlir::db::DBType>().isNullable()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), resultingType.cast<mlir::db::DBType>().getBaseType(), getMaxValueAttr(resultingType.cast<mlir::db::DBType>()));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value currLtNew = builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::lt, currVal, newVal);
                  mlir::Value casted = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), currVal.getType(), newVal);
                  mlir::Value added = builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), currLtNew, currVal, casted);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), newVal);
                     updatedVal = builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.cast<mlir::db::DBType>().isNullable()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull, casted, added));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::max) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.cast<mlir::db::DBType>().isNullable()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), resultingType.cast<mlir::db::DBType>().getBaseType(), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value currGtNew = builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(), mlir::db::DBCmpPredicate::gt, currVal, newVal);
                  mlir::Value casted = builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), currVal.getType(), newVal);
                  mlir::Value added = builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), currGtNew, currVal, casted);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), newVal);
                     updatedVal = builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.cast<mlir::db::DBType>().isNullable()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull, casted, added));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::avg) {
               aggrTypes.push_back(resultingType);
               aggrTypes.push_back(counterType);
               auto initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), resultingType.cast<mlir::db::DBType>().getBaseType(), builder.getI64IntegerAttr(0));
               auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));

               defaultValues.push_back(builder.create<mlir::db::CastOp>(aggregationOp.getLoc(), resultingType, initVal));
               defaultValues.push_back(initCounterVal);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr, resultingType = resultingType](mlir::ValueRange range, mlir::OpBuilder builder) {
                  mlir::Value casted=builder.create<mlir::db::CastOp>(builder.getUnknownLoc(), resultingType, range[currDestIdx+1]);
                  mlir::Value average=builder.create<mlir::db::DivOp>(builder.getUnknownLoc(), resultingType, range[currDestIdx], casted);
                  return std::make_pair(destAttr, average); });
               aggregationFunctions.push_back([currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), counterType, builder.getI64IntegerAttr(1));
                  mlir::Value added1 = builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), resultingType, aggr[currDestIdx], val[currValIdx]);
                  mlir::Value added2 = builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), counterType, aggr[currDestIdx + 1], one);
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), val[currValIdx]);
                     res.push_back(builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull1, aggr[currDestIdx], added1));
                     res.push_back(builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull1, aggr[currDestIdx + 1], added2));
                  } else {
                     res.push_back(added1);
                     res.push_back(added2);
                  }

                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::count) {
               size_t currDestIdx = aggrTypes.size();
               aggrTypes.push_back(resultingType);
               auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               defaultValues.push_back(initCounterVal);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               aggregationFunctions.push_back([currDestIdx = currDestIdx, attrIsNullable, currValIdx = currValIdx, counterType = counterType, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), counterType, builder.getI64IntegerAttr(1));
                  mlir::Value value = builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), resultingType, aggr[currDestIdx], one);
                  if (attrIsNullable) {
                     mlir::Value isNull2 = builder.create<mlir::db::IsNullOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), val[currValIdx]);
                     mlir::Value tmp = builder.create<mlir::db::SelectOp>(builder.getUnknownLoc(), isNull2, aggr[currDestIdx], value);
                     value = tmp;
                  }

                  res.push_back(value);
                  return res;
               });
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(addAttrOp.val().getDefiningOp())) {
            auto* destAttr = &addAttrOp.attr().getRelationalAttribute();
            size_t currDestIdx = aggrTypes.size();
            aggrTypes.push_back(counterType);
            auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
            defaultValues.push_back(initCounterVal);
            finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

            aggregationFunctions.push_back([currDestIdx = currDestIdx, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
               std::vector<mlir::Value> res;
               auto one = builder.create<mlir::db::ConstantOp>(builder.getUnknownLoc(), counterType, builder.getI64IntegerAttr(1));
               mlir::Value added2 = builder.create<mlir::db::AddOp>(builder.getUnknownLoc(), counterType, aggr[currDestIdx], one);
               res.push_back(added2);
               return res;
            });
         }
      });
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();

      analyze(builder);

      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      auto initTuple = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), aggrTupleType, defaultValues);
      keyTupleType = mlir::TupleType::get(builder.getContext(), keyTypes);
      valTupleType = mlir::TupleType::get(builder.getContext(), valTypes);
      aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);

      auto aggrBuilder = builder.create<mlir::db::CreateAggrHTBuilder>(aggregationOp.getLoc(), mlir::db::AggrHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType, aggrTupleType), initTuple);

      builderId = context.getBuilderId();
      context.builders[builderId] = aggrBuilder;

      auto iterEntryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, aggrTupleType});
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      mlir::Value hashtable = builder.create<mlir::db::BuilderBuild>(aggregationOp.getLoc(), mlir::db::AggregationHashtableType::get(builder.getContext(), keyTupleType, aggrTupleType), context.builders[builderId]);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(aggregationOp->getLoc(), getRequiredBuilderTypes(context), hashtable, flag, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(iterEntryType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), forOp2.getInductionVar()).getResults();
         mlir::ValueRange unpackedKey;
         if (!keyTypes.empty()) {
            unpackedKey = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[0]).getResults();
         }
         auto unpackedAggr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[1]).getResults();

         for (auto fn : finalizeFunctions) {
            auto [attr, val] = fn(unpackedAggr, builder2);
            context.setValueForAttribute(scope, attr, val);
         }
         for (auto* attr : requiredAttributes) {
            if (keyMapping.count(attr)) {
               context.setValueForAttribute(scope, attr, unpackedKey[keyMapping[attr]]);
            }
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(aggregationOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      builder.create<mlir::db::FreeOp>(aggregationOp->getLoc(),hashtable);
   }
   virtual void done() override {
   }
   virtual ~AggregationLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredAggregationOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::AggregationOp sortOp) {
   return std::make_unique<AggregationLowering>(sortOp);
});