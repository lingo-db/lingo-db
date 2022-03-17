#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class TmpTranslator : public mlir::relalg::Translator {
   mlir::relalg::TmpOp tmpOp;
   size_t builderId;
   bool materialize;
   mlir::relalg::OrderedAttributes attributes;
   size_t userCount;
   size_t producedCount;

   public:
   TmpTranslator(mlir::relalg::TmpOp tmpOp) : mlir::relalg::Translator(tmpOp), tmpOp(tmpOp) {
      std::vector<mlir::Operation*> users(tmpOp->getUsers().begin(), tmpOp->getUsers().end());
      userCount = users.size();
      producedCount = 0;
   }
   void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(mlir::relalg::ColumnSet::fromArrayAttr(tmpOp.cols()));
      propagateInfo();
      for (const auto* attr : this->requiredAttributes) {
         attributes.insert(attr);
      }
   }

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      if (materialize) {
         mlir::Value vectorBuilder = context.builders[builderId];
         mlir::Value packed = attributes.pack(context, builder, tmpOp->getLoc());
         mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(tmpOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
         context.builders[builderId] = mergedBuilder;
      }
   }

   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();

      materialize = !context.materializedTmp.count(tmpOp.getOperation());
      producedCount++;
      if (materialize) {
         auto parentPipeline = context.pipelineManager.getCurrentPipeline();
         auto p = std::make_shared<mlir::relalg::Pipeline>(builder.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
         context.pipelineManager.setCurrentPipeline(p);
         context.pipelineManager.addPipeline(p);
         auto tupleType = attributes.getTupleType(builder.getContext());
         std::unordered_map<const mlir::relalg::Column*, size_t> attributePos;
         auto res = p->addInitFn([&](mlir::OpBuilder& builder) {
            return std::vector<mlir::Value>({builder.create<mlir::db::CreateVectorBuilder>(tmpOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType))});
         });
         builderId = context.getBuilderId();
         context.builders[builderId] = p->addDependency(res[0]);

         children[0]->addRequiredBuilders({builderId});
         children[0]->produce(context, p->getBuilder());
         p->finishMainFunction({context.builders[builderId]});
         auto vectorRes = p->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
            mlir::Value vector = builder.create<mlir::db::BuilderBuild>(tmpOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), args[0]);
            return std::vector<mlir::Value>{vector};
         });
         context.materializedTmp[tmpOp.getOperation()] = {vectorRes[0], attributes.getAttrs()};
         context.pipelineManager.setCurrentPipeline(parentPipeline);
      }
      auto [vector, attributes_] = context.materializedTmp[tmpOp.getOperation()];
      auto attributes=mlir::relalg::OrderedAttributes::fromVec(attributes_);
      auto tupleType = attributes.getTupleType(builder.getContext());
      auto forOp2 = builder.create<mlir::db::ForOp>(tmpOp->getLoc(), getRequiredBuilderTypes(context), context.pipelineManager.getCurrentPipeline()->addDependency(vector), context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType, tmpOp->getLoc());
      block2->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
      auto unpacked = builder2.create<mlir::util::UnPackOp>(tmpOp->getLoc(), forOp2.getInductionVar());
      attributes.setValuesForColumns(context,scope,unpacked.getResults());
      consumer->consume(this, builder2, context);
      builder2.create<mlir::db::YieldOp>(tmpOp->getLoc(), getRequiredBuilderValues(context));
      setRequiredBuilderValues(context, forOp2.results());

      if (producedCount >= userCount) {
         auto [vector, attributes] = context.materializedTmp[tmpOp.getOperation()];
         //builder.create<mlir::db::FreeOp>(tmpOp->getLoc(), vector);
      }
   }

   virtual ~TmpTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createTmpTranslator(mlir::relalg::TmpOp tmpOp) {
   return std::make_unique<TmpTranslator>(tmpOp);
}