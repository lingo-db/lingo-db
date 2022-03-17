#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
class ConstRelTranslator : public mlir::relalg::Translator {
   mlir::relalg::ConstRelationOp constRelationOp;

   public:
   ConstRelTranslator(mlir::relalg::ConstRelationOp constRelationOp) : mlir::relalg::Translator(constRelationOp), constRelationOp(constRelationOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      mlir::relalg::OrderedAttributes attributes = mlir::relalg::OrderedAttributes::fromRefArr(constRelationOp.columns());
      auto tupleType = attributes.getTupleType(builder.getContext());
      mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(constRelationOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
      for (auto rowAttr : constRelationOp.valuesAttr()) {
         auto row = rowAttr.cast<ArrayAttr>();
         std::vector<Value> values;
         size_t i = 0;
         for (auto entryAttr : row.getValue()) {
            auto entryVal = builder.create<mlir::db::ConstantOp>(constRelationOp->getLoc(), tupleType.getType(i), entryAttr);
            values.push_back(entryVal);
            i++;
         }
         mlir::Value packed = builder.create<mlir::util::PackOp>(constRelationOp->getLoc(), values);
         vectorBuilder = builder.create<mlir::db::BuilderMerge>(constRelationOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      }
      Value vector = builder.create<mlir::db::BuilderBuild>(constRelationOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), vectorBuilder);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(constRelationOp->getLoc(), getRequiredBuilderTypes(context), vector, context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType, constRelationOp->getLoc());
         block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(constRelationOp->getLoc(), forOp2.getInductionVar());
         attributes.setValuesForColumns(context, scope, unpacked.getResults());
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(constRelationOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      builder.create<mlir::db::FreeOp>(constRelationOp->getLoc(), vector);
   }
   virtual ~ConstRelTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createConstRelTranslator(mlir::relalg::ConstRelationOp constRelationOp) {
   return std::make_unique<ConstRelTranslator>(constRelationOp);
}