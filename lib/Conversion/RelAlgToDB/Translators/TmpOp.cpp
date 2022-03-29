#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class TmpTranslator : public mlir::relalg::Translator {
   mlir::relalg::TmpOp tmpOp;
   bool materialize;
   mlir::relalg::OrderedAttributes attributes;
   size_t userCount;
   size_t producedCount;
   mlir::Value vector;

   public:
   TmpTranslator(mlir::relalg::TmpOp tmpOp) : mlir::relalg::Translator(tmpOp), tmpOp(tmpOp) {
      std::vector<mlir::Operation*> users(tmpOp->getUsers().begin(), tmpOp->getUsers().end());
      userCount = users.size();
      producedCount = 0;
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
         mlir::Value packed = attributes.pack(context, builder, tmpOp->getLoc());
         builder.create<mlir::dsa::Append>(tmpOp->getLoc(), vector, packed);
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
            return std::vector<mlir::Value>({builder.create<mlir::dsa::CreateDS>(tmpOp.getLoc(), mlir::dsa::VectorType::get(builder.getContext(), tupleType))});
         });
         vector = p->addDependency(res[0]);

         children[0]->produce(context, p->getBuilder());
         p->finishMainFunction({vector});
         auto vectorRes = p->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
            return std::vector<mlir::Value>{args[0]};
         });
         context.materializedTmp[tmpOp.getOperation()] = {vectorRes[0], attributes.getAttrs()};
         context.pipelineManager.setCurrentPipeline(parentPipeline);
      }
      auto [vector, attributes_] = context.materializedTmp[tmpOp.getOperation()];
      auto attributes=mlir::relalg::OrderedAttributes::fromVec(attributes_);
      auto tupleType = attributes.getTupleType(builder.getContext());
      auto forOp2 = builder.create<mlir::dsa::ForOp>(tmpOp->getLoc(), mlir::TypeRange{}, context.pipelineManager.getCurrentPipeline()->addDependency(vector), context.pipelineManager.getCurrentPipeline()->getFlag(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType, tmpOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      auto unpacked = builder2.create<mlir::util::UnPackOp>(tmpOp->getLoc(), forOp2.getInductionVar());
      attributes.setValuesForColumns(context,scope,unpacked.getResults());
      consumer->consume(this, builder2, context);
      builder2.create<mlir::dsa::YieldOp>(tmpOp->getLoc(), mlir::ValueRange{});

      if (producedCount >= userCount) {
         auto [vector, attributes] = context.materializedTmp[tmpOp.getOperation()];
         //builder.create<mlir::dsa::FreeOp>(tmpOp->getLoc(), vector);
      }
   }

   virtual ~TmpTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createTmpTranslator(mlir::relalg::TmpOp tmpOp) {
   return std::make_unique<TmpTranslator>(tmpOp);
}