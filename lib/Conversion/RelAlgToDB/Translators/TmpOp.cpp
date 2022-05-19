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
         auto tupleType = attributes.getTupleType(builder.getContext());
         std::unordered_map<const mlir::relalg::Column*, size_t> attributePos;
         vector=builder.create<mlir::dsa::CreateDS>(tmpOp.getLoc(), mlir::dsa::VectorType::get(builder.getContext(), tupleType));

         children[0]->produce(context, builder);
         context.materializedTmp[tmpOp.getOperation()] = {vector, attributes.getAttrs()};
      }
      auto [vector, attributes_] = context.materializedTmp[tmpOp.getOperation()];
      auto attributes=mlir::relalg::OrderedAttributes::fromVec(attributes_);
      auto tupleType = attributes.getTupleType(builder.getContext());
      auto forOp2 = builder.create<mlir::dsa::ForOp>(tmpOp->getLoc(), mlir::TypeRange{}, vector, mlir::Value(), mlir::ValueRange{});
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