#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class SortTranslator : public mlir::relalg::Translator {
   mlir::relalg::SortOp sortOp;
   size_t builderId;
   mlir::Value vector;

   public:
   SortTranslator(mlir::relalg::SortOp sortOp) : mlir::relalg::Translator(sortOp), sortOp(sortOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value vectorBuilder = context.builders[builderId];
      mlir::Value packed = packValues(context, builder, sortOp->getLoc(), requiredAttributes);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(sortOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }
   mlir::Value createSortPredicate(mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, mlir::Value trueVal, mlir::Value falseVal, size_t pos) {
      if (pos < sortCriteria.size()) {
         auto lt = builder.create<mlir::db::CmpOp>(sortOp->getLoc(), mlir::db::DBCmpPredicate::lt, sortCriteria[pos].first, sortCriteria[pos].second);
         auto ifOp = builder.create<mlir::db::IfOp>(
            sortOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), lt, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::db::YieldOp>(loc, trueVal); }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
               auto eq = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, sortCriteria[pos].first, sortCriteria[pos].second);
               auto ifOp2 = builder.create<mlir::db::IfOp>(loc, mlir::db::BoolType::get(builder.getContext()), eq,[&](mlir::OpBuilder& builder, mlir::Location loc) {
                  builder.create<mlir::db::YieldOp>(loc, createSortPredicate(builder, sortCriteria, trueVal, falseVal, pos + 1));
                  },[&](mlir::OpBuilder& builder, mlir::Location loc) {
                     builder.create<mlir::db::YieldOp>(loc, falseVal);
               });
               builder.create<mlir::db::YieldOp>(loc, ifOp2.getResult(0)); });
         return ifOp.getResult(0);
      } else {
         return falseVal;
      }
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      std::unordered_map<const mlir::relalg::RelationalAttribute*, size_t> attributePos;
      std::vector<mlir::Type> types;
      size_t i = 0;
      for (const auto* attr : requiredAttributes) {
         types.push_back(attr->type);
         attributePos[attr] = i++;
      }
      auto parentPipeline=context.pipelineManager.getCurrentPipeline();
      auto childPipeline =std::make_shared<mlir::relalg::Pipeline>(builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
      context.pipelineManager.setCurrentPipeline(childPipeline);
      context.pipelineManager.addPipeline(childPipeline);
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      auto res = childPipeline->addInitFn([&](mlir::OpBuilder& builder) {
         return std::vector<mlir::Value>({builder.create<mlir::db::CreateVectorBuilder>(sortOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType))});
      });
      builderId = context.getBuilderId();
      context.builders[builderId] = childPipeline->addDependency(res[0]);
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, childPipeline->getBuilder());
      childPipeline->finishMainFunction({context.builders[builderId]});

      auto sortedRes=childPipeline->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
         mlir::Value vector = builder.create<mlir::db::BuilderBuild>(sortOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), args[0]);
         {
            auto dbSortOp = builder.create<mlir::db::SortOp>(sortOp->getLoc(), vector);
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(tupleType,sortOp->getLoc());
            block2->addArguments(tupleType,sortOp->getLoc());
            dbSortOp.region().push_back(block2);
            mlir::OpBuilder builder2(dbSortOp.region());
            auto unpackedLeft = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), block2->getArgument(0));
            auto unpackedRight = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), block2->getArgument(1));
            std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
            for (auto attr : sortOp.sortspecs()) {
               auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
               mlir::Value left = unpackedLeft.getResult(attributePos[&sortspecAttr.getAttr().getRelationalAttribute()]);
               mlir::Value right = unpackedRight.getResult(attributePos[&sortspecAttr.getAttr().getRelationalAttribute()]);
               if (sortspecAttr.getSortSpec() == mlir::relalg::SortSpec::desc) {
                  std::swap(left, right);
               }
               sortCriteria.push_back({left, right});
            }
            auto trueVal = builder2.create<mlir::db::ConstantOp>(sortOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 1));
            auto falseVal = builder2.create<mlir::db::ConstantOp>(sortOp->getLoc(), mlir::db::BoolType::get(builder.getContext()), builder.getIntegerAttr(builder.getI64Type(), 0));

            builder2.create<mlir::db::YieldOp>(sortOp->getLoc(), createSortPredicate(builder2, sortCriteria, trueVal, falseVal, 0));
         }
         return std::vector<mlir::Value>{vector};
      });
      vector=parentPipeline->addDependency(sortedRes[0]);
      context.pipelineManager.setCurrentPipeline(parentPipeline);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(sortOp->getLoc(), getRequiredBuilderTypes(context), vector, context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType,sortOp->getLoc());
         block2->addArguments(getRequiredBuilderTypes(context),getRequiredBuilderLocs(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : requiredAttributes) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(sortOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      builder.create<mlir::db::FreeOp>(sortOp->getLoc(), vector);
   }

   virtual ~SortTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createSortTranslator(mlir::relalg::SortOp sortOp) {
   return std::make_unique<SortTranslator>(sortOp);
}