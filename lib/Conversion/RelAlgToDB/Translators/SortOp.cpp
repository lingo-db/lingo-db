#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/util/UtilOps.h"

class SortTranslator : public mlir::relalg::Translator {
   mlir::relalg::SortOp sortOp;
   mlir::Value vector;
   mlir::relalg::OrderedAttributes orderedAttributes;

   public:
   SortTranslator(mlir::relalg::SortOp sortOp) : mlir::relalg::Translator(sortOp), sortOp(sortOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      mlir::Value packed = orderedAttributes.pack(context, builder, sortOp->getLoc());
      builder.create<mlir::db::Append>(sortOp->getLoc(), vector, packed);
   }
   mlir::Value createSortPredicate(mlir::OpBuilder& builder, std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria, mlir::Value trueVal, mlir::Value falseVal, size_t pos) {
      if (pos < sortCriteria.size()) {
         mlir::Value lt = builder.create<mlir::db::CmpOp>(sortOp->getLoc(), mlir::db::DBCmpPredicate::lt, sortCriteria[pos].first, sortCriteria[pos].second);
         lt = builder.create<mlir::db::DeriveTruth>(sortOp->getLoc(), lt);
         auto ifOp = builder.create<mlir::scf::IfOp>(
            sortOp->getLoc(), builder.getI1Type(), lt, [&](mlir::OpBuilder& builder, mlir::Location loc) { builder.create<mlir::scf::YieldOp>(loc, trueVal); }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
               mlir::Value eq = builder.create<mlir::db::CmpOp>(loc, mlir::db::DBCmpPredicate::eq, sortCriteria[pos].first, sortCriteria[pos].second);
               eq=builder.create<mlir::db::DeriveTruth>(sortOp->getLoc(),eq);
               auto ifOp2 = builder.create<mlir::scf::IfOp>(loc, builder.getI1Type(), eq,[&](mlir::OpBuilder& builder, mlir::Location loc) {
                  builder.create<mlir::scf::YieldOp>(loc, createSortPredicate(builder, sortCriteria, trueVal, falseVal, pos + 1));
                  },[&](mlir::OpBuilder& builder, mlir::Location loc) {
                     builder.create<mlir::scf::YieldOp>(loc, falseVal);
               });
               builder.create<mlir::scf::YieldOp>(loc, ifOp2.getResult(0)); });
         return ifOp.getResult(0);
      } else {
         return falseVal;
      }
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      orderedAttributes = mlir::relalg::OrderedAttributes::fromColumns(requiredAttributes);
      auto parentPipeline = context.pipelineManager.getCurrentPipeline();
      auto childPipeline = std::make_shared<mlir::relalg::Pipeline>(builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
      context.pipelineManager.setCurrentPipeline(childPipeline);
      context.pipelineManager.addPipeline(childPipeline);
      auto tupleType = orderedAttributes.getTupleType(builder.getContext());
      auto res = childPipeline->addInitFn([&](mlir::OpBuilder& builder) {
         return std::vector<mlir::Value>({builder.create<mlir::db::CreateDS>(sortOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType))});
      });
      vector = childPipeline->addDependency(res[0]);
      children[0]->produce(context, childPipeline->getBuilder());
      childPipeline->finishMainFunction({vector});

      auto sortedRes = childPipeline->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
            mlir::Value vector=args[0];
         {
            auto dbSortOp = builder.create<mlir::db::SortOp>(sortOp->getLoc(), vector);
            mlir::Block* block2 = new mlir::Block;
            block2->addArgument(tupleType, sortOp->getLoc());
            block2->addArguments(tupleType, sortOp->getLoc());
            dbSortOp.region().push_back(block2);
            mlir::OpBuilder builder2(dbSortOp.region());
            auto unpackedLeft = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), block2->getArgument(0));
            auto unpackedRight = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), block2->getArgument(1));
            std::vector<std::pair<mlir::Value, mlir::Value>> sortCriteria;
            for (auto attr : sortOp.sortspecs()) {
               auto sortspecAttr = attr.cast<mlir::relalg::SortSpecificationAttr>();
               auto pos = orderedAttributes.getPos(&sortspecAttr.getAttr().getColumn());
               mlir::Value left = unpackedLeft.getResult(pos);
               mlir::Value right = unpackedRight.getResult(pos);
               if (sortspecAttr.getSortSpec() == mlir::relalg::SortSpec::desc) {
                  std::swap(left, right);
               }
               sortCriteria.push_back({left, right});
            }
            auto trueVal = builder2.create<mlir::db::ConstantOp>(sortOp->getLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 1));
            auto falseVal = builder2.create<mlir::db::ConstantOp>(sortOp->getLoc(), builder.getI1Type(), builder.getIntegerAttr(builder.getI64Type(), 0));

            builder2.create<mlir::db::YieldOp>(sortOp->getLoc(), createSortPredicate(builder2, sortCriteria, trueVal, falseVal, 0));
         }
         return std::vector<mlir::Value>{vector};
      });
      vector = parentPipeline->addDependency(sortedRes[0]);
      context.pipelineManager.setCurrentPipeline(parentPipeline);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(sortOp->getLoc(), getRequiredBuilderTypes(context), vector, context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType, sortOp->getLoc());
         block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), forOp2.getInductionVar());
         orderedAttributes.setValuesForColumns(context, scope, unpacked.getResults());
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