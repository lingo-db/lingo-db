#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
#include <llvm/ADT/TypeSwitch.h>

class AggregationTranslator : public mlir::relalg::Translator {
   mlir::relalg::AggregationOp aggregationOp;
   size_t builderId;

   mlir::TupleType keyTupleType;
   mlir::TupleType valTupleType;

   mlir::relalg::OrderedAttributes key;
   mlir::relalg::OrderedAttributes val;

   std::vector<std::function<std::pair<const mlir::relalg::Column*, mlir::Value>(mlir::ValueRange, mlir::OpBuilder& builder)>> finalizeFunctions;
   std::vector<std::function<std::vector<mlir::Value>(mlir::ValueRange, mlir::ValueRange, mlir::OpBuilder& builder)>> aggregationFunctions;
   std::vector<mlir::Value> defaultValues;
   std::vector<mlir::Type> aggrTypes;

   public:
   AggregationTranslator(mlir::relalg::AggregationOp aggregationOp) : mlir::relalg::Translator(aggregationOp), aggregationOp(aggregationOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      //do not forwared requiredBuilders to children
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value htBuilder = context.builders[builderId];
      mlir::Value packedKey = key.pack(context, builder, aggregationOp->getLoc());
      mlir::Value packedVal = val.pack(context, builder, aggregationOp->getLoc());
      mlir::Value packed = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), mlir::ValueRange({packedKey, packedVal}));

      auto builderMerge = builder.create<mlir::db::BuilderMerge>(aggregationOp->getLoc(), htBuilder.getType(), htBuilder, packed);
      context.builders[builderId] = builderMerge.result_builder();

      auto scope = context.createScope();

      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      keyTupleType = key.getTupleType(builder.getContext());
      valTupleType = val.getTupleType(builder.getContext());
      aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
      mlir::Block* aggrBuilderBlock = new mlir::Block;
      builderMerge.fn().push_back(aggrBuilderBlock);
      aggrBuilderBlock->addArguments({aggrTupleType, valTupleType}, {aggregationOp->getLoc(), aggregationOp->getLoc()});
      mlir::OpBuilder builder2(builder.getContext());
      builder2.setInsertionPointToStart(aggrBuilderBlock);
      auto unpackedCurr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(0))->getResults();
      mlir::ValueRange unpackedNew;
      if (valTupleType.getTypes().size() > 0) {
         unpackedNew = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), aggrBuilderBlock->getArgument(1)).getResults();
      }
      std::vector<mlir::Value> valuesx;
      for (auto aggrFn : aggregationFunctions) {
         auto vec = aggrFn(unpackedCurr, unpackedNew, builder2);
         valuesx.insert(valuesx.end(), vec.begin(), vec.end());
      }

      mlir::Value packedx = builder2.create<mlir::util::PackOp>(aggregationOp->getLoc(), valuesx);

      builder2.create<mlir::db::YieldOp>(aggregationOp->getLoc(), packedx);
   }

   mlir::Attribute getMaxValueAttr(mlir::Type type) {
      auto* context = aggregationOp->getContext();
      mlir::OpBuilder builder(context);
      mlir::Attribute maxValAttr = ::llvm::TypeSwitch<::mlir::Type, mlir::Attribute>(type)

                                      .Case<::mlir::db::DecimalType>([&](::mlir::db::DecimalType t) {
                                         if (t.getP() < 19) {
                                            return (mlir::Attribute) builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                         }
                                         std::vector<uint64_t> parts = {0xFFFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF};
                                         return (mlir::Attribute) builder.getIntegerAttr(mlir::IntegerType::get(context, 128), mlir::APInt(128, parts));
                                      })
                                      .Case<::mlir::IntegerType>([&](::mlir::IntegerType) {
                                         return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max());
                                      })
                                      .Case<::mlir::FloatType>([&](::mlir::FloatType t) {
                                         if (t.getWidth() == 32) {
                                            return (mlir::Attribute) builder.getF32FloatAttr(std::numeric_limits<float>::max());
                                         } else if (t.getWidth() == 64) {
                                            return (mlir::Attribute) builder.getF64FloatAttr(std::numeric_limits<double>::max());
                                         } else {
                                            assert(false && "should not happen");
                                            return mlir::Attribute();
                                         }
                                      })
                                      .Default([&](::mlir::Type) { return builder.getI64IntegerAttr(std::numeric_limits<int64_t>::max()); });
      return maxValAttr;
   }
   void analyze(mlir::OpBuilder& builder) {
      key = mlir::relalg::OrderedAttributes::fromRefArr(aggregationOp.group_by_colsAttr());

      auto counterType = builder.getI64Type();

      aggregationOp.aggr_func().walk([&](mlir::relalg::AddColumnOp addColumnOp) {
         if (auto aggrFn = mlir::dyn_cast_or_null<mlir::relalg::AggrFuncOp>(addColumnOp.val().getDefiningOp())) {
            auto loc = aggrFn->getLoc();
            auto* destAttr = &addColumnOp.attr().getColumn();
            auto* attr = &aggrFn.attr().getColumn();
            auto attrIsNullable = attr->type.isa<mlir::db::NullableType>();
            size_t currValIdx = val.insert(attr);
            mlir::Type resultingType = addColumnOp.attr().getColumn().type;
            size_t currDestIdx = aggrTypes.size();

            if (aggrFn.fn() == mlir::relalg::AggrFunc::sum) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(loc, resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(loc, getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value added = builder.create<mlir::db::AddOp>(loc, resultingType, currVal, newVal);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value casted = builder.create<mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, added));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::min) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), getMaxValueAttr(resultingType));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value currLtNew = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::lt, currVal, newVal);
                  mlir::Value casted = newVal;
                  if (newVal.getType() != currVal.getType()) {
                     casted = builder.create<mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                  }
                  mlir::Value currLtNewT = builder.create<mlir::db::DeriveTruth>(loc, currLtNew);

                  mlir::Value added = builder.create<mlir::arith::SelectOp>(loc, currLtNewT, currVal, casted);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, added));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::max) {
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });
               aggrTypes.push_back(resultingType);
               mlir::Value initVal;
               if (resultingType.isa<mlir::db::NullableType>()) {
                  initVal = builder.create<mlir::db::NullOp>(aggregationOp.getLoc(), resultingType);
               } else {
                  initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), builder.getI64IntegerAttr(0));
               }
               defaultValues.push_back(initVal);
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, resultingType = resultingType, currValIdx = currValIdx](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  mlir::Value currVal = aggr[currDestIdx];
                  mlir::Value newVal = val[currValIdx];
                  mlir::Value currGtNew = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::gt, currVal, newVal);
                  mlir::Value casted = newVal;
                  if (newVal.getType() != currVal.getType()) {
                     casted = builder.create<mlir::db::AsNullableOp>(loc, currVal.getType(), newVal);
                  }
                  mlir::Value currGTNewT = builder.create<mlir::db::DeriveTruth>(loc, currGtNew);
                  mlir::Value added = builder.create<mlir::arith::SelectOp>(loc, currGTNewT, currVal, casted);
                  mlir::Value updatedVal = added;
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), newVal);
                     updatedVal = builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added);
                  }
                  if (resultingType.isa<mlir::db::NullableType>()) {
                     mlir::Value isNull = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), aggr[currDestIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull, casted, added));
                  } else {
                     res.push_back(updatedVal);
                  }
                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::avg) {
               aggrTypes.push_back(resultingType);
               aggrTypes.push_back(counterType);
               mlir::Value initVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), getBaseType(resultingType), builder.getI64IntegerAttr(0));
               mlir::Value initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               mlir::Value defaultVal=resultingType.isa<mlir::db::NullableType>() ? builder.create<mlir::db::AsNullableOp>(aggregationOp.getLoc(), resultingType, initVal) : initVal;
               defaultValues.push_back(defaultVal);
               defaultValues.push_back(initCounterVal);
               finalizeFunctions.push_back([loc, currDestIdx = currDestIdx, destAttr = destAttr, resultingType = resultingType](mlir::ValueRange range, mlir::OpBuilder builder) {
                  mlir::Value casted=builder.create<mlir::db::CastOp>(loc, getBaseType(resultingType), range[currDestIdx+1]);
                  if(resultingType.isa<mlir::db::NullableType>()){
                     casted=builder.create<mlir::db::AsNullableOp>(loc, resultingType, casted);
                  }
                  mlir::Value average=builder.create<mlir::db::DivOp>(loc, resultingType, range[currDestIdx], casted);
                  return std::make_pair(destAttr, average); });
               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, currValIdx = currValIdx, attrIsNullable, resultingType = resultingType, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  mlir::Value added1 = builder.create<mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], val[currValIdx]);
                  mlir::Value added2 = builder.create<mlir::db::AddOp>(loc, counterType, aggr[currDestIdx + 1], one);
                  if (attrIsNullable) {
                     mlir::Value isNull1 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx], added1));
                     res.push_back(builder.create<mlir::arith::SelectOp>(loc, isNull1, aggr[currDestIdx + 1], added2));
                  } else {
                     res.push_back(added1);
                     res.push_back(added2);
                  }

                  return res;
               });
            } else if (aggrFn.fn() == mlir::relalg::AggrFunc::count) {
               size_t currDestIdx = aggrTypes.size();
               auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
               defaultValues.push_back(initCounterVal);
               aggrTypes.push_back(resultingType);
               finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

               aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, attrIsNullable, currValIdx = currValIdx, counterType = counterType, resultingType = resultingType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
                  std::vector<mlir::Value> res;
                  auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
                  mlir::Value value = builder.create<mlir::db::AddOp>(loc, resultingType, aggr[currDestIdx], one);
                  if (attrIsNullable) {
                     mlir::Value isNull2 = builder.create<mlir::db::IsNullOp>(loc, builder.getI1Type(), val[currValIdx]);
                     mlir::Value tmp = builder.create<mlir::arith::SelectOp>(loc, isNull2, aggr[currDestIdx], value);
                     value = tmp;
                  }

                  res.push_back(value);
                  return res;
               });
            }
         }
         if (auto countOp = mlir::dyn_cast_or_null<mlir::relalg::CountRowsOp>(addColumnOp.val().getDefiningOp())) {
            auto loc = countOp->getLoc();

            auto* destAttr = &addColumnOp.attr().getColumn();
            size_t currDestIdx = aggrTypes.size();
            aggrTypes.push_back(counterType);
            auto initCounterVal = builder.create<mlir::db::ConstantOp>(aggregationOp.getLoc(), counterType, builder.getI64IntegerAttr(0));
            defaultValues.push_back(initCounterVal);
            finalizeFunctions.push_back([currDestIdx = currDestIdx, destAttr = destAttr](mlir::ValueRange range, mlir::OpBuilder& builder) { return std::make_pair(destAttr, range[currDestIdx]); });

            aggregationFunctions.push_back([loc, currDestIdx = currDestIdx, counterType = counterType](mlir::ValueRange aggr, mlir::ValueRange val, mlir::OpBuilder& builder) {
               std::vector<mlir::Value> res;
               auto one = builder.create<mlir::db::ConstantOp>(loc, counterType, builder.getI64IntegerAttr(1));
               mlir::Value added2 = builder.create<mlir::db::AddOp>(loc, counterType, aggr[currDestIdx], one);
               res.push_back(added2);
               return res;
            });
         }
      });
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      auto parentPipeline = context.pipelineManager.getCurrentPipeline();
      auto p = std::make_shared<mlir::relalg::Pipeline>(builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
      context.pipelineManager.setCurrentPipeline(p);
      context.pipelineManager.addPipeline(p);
      auto res = p->addInitFn([&](mlir::OpBuilder& builder) {
         analyze(builder);
         auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);
         auto initTuple = builder.create<mlir::util::PackOp>(aggregationOp->getLoc(), aggrTupleType, defaultValues);
         keyTupleType = key.getTupleType(builder.getContext());
         valTupleType = val.getTupleType(builder.getContext());
         auto aggrBuilder = builder.create<mlir::db::CreateAggrHTBuilder>(aggregationOp.getLoc(), mlir::db::AggrHTBuilderType::get(builder.getContext(), keyTupleType, valTupleType, aggrTupleType), initTuple);
         return std::vector<mlir::Value>({aggrBuilder});
      });
      auto aggrTupleType = mlir::TupleType::get(builder.getContext(), aggrTypes);

      builderId = context.getBuilderId();
      context.builders[builderId] = p->addDependency(res[0]);

      auto iterEntryType = mlir::TupleType::get(builder.getContext(), {keyTupleType, aggrTupleType});
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, p->getBuilder());
      p->finishMainFunction({context.builders[builderId]});
      auto hashtableRes = p->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
         mlir::Value hashtable = builder.create<mlir::db::BuilderBuild>(aggregationOp.getLoc(), mlir::db::AggregationHashtableType::get(builder.getContext(), keyTupleType, aggrTupleType), args[0]);
         return std::vector<mlir::Value>{hashtable};
      });

      context.pipelineManager.setCurrentPipeline(parentPipeline);
      auto hashtable = context.pipelineManager.getCurrentPipeline()->addDependency(hashtableRes[0]);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(aggregationOp->getLoc(), getRequiredBuilderTypes(context), hashtable, context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(iterEntryType, aggregationOp->getLoc());
         block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), forOp2.getInductionVar()).getResults();
         mlir::ValueRange unpackedKey;
         if (!keyTupleType.getTypes().empty()) {
            unpackedKey = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[0]).getResults();
         }
         auto unpackedAggr = builder2.create<mlir::util::UnPackOp>(aggregationOp->getLoc(), unpacked[1]).getResults();

         for (auto fn : finalizeFunctions) {
            auto [attr, val] = fn(unpackedAggr, builder2);
            context.setValueForAttribute(scope, attr, val);
         }
         key.setValuesForColumns(context, scope, unpackedKey);
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(aggregationOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      builder.create<mlir::db::FreeOp>(aggregationOp->getLoc(), hashtable);
   }
   virtual void done() override {
   }
   virtual ~AggregationTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createAggregationTranslator(mlir::relalg::AggregationOp sortOp) {
   return std::make_unique<AggregationTranslator>(sortOp);
}