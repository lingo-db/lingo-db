#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
class ConstRelTranslator : public mlir::relalg::Translator {
   mlir::relalg::ConstRelationOp constRelationOp;

   public:
   ConstRelTranslator(mlir::relalg::ConstRelationOp constRelationOp) : mlir::relalg::Translator(constRelationOp), constRelationOp(constRelationOp) {}

   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      mlir::relalg::OrderedAttributes attributes = mlir::relalg::OrderedAttributes::fromRefArr(constRelationOp.columns());
      auto tupleType = attributes.getTupleType(builder.getContext());
      mlir::Value vector = builder.create<mlir::dsa::CreateDS>(constRelationOp.getLoc(), mlir::dsa::VectorType::get(builder.getContext(), tupleType));
      for (auto rowAttr : constRelationOp.valuesAttr()) {
         auto row = rowAttr.cast<ArrayAttr>();
         std::vector<Value> values;
         size_t i = 0;
         for (auto entryAttr : row.getValue()) {
            if (tupleType.getType(i).isa<mlir::db::NullableType>() && entryAttr.isa<mlir::UnitAttr>()) {
               auto entryVal = builder.create<mlir::db::NullOp>(constRelationOp->getLoc(), tupleType.getType(i));
               values.push_back(entryVal);
               i++;
            } else {
               mlir::Value entryVal = builder.create<mlir::db::ConstantOp>(constRelationOp->getLoc(), getBaseType(tupleType.getType(i)), entryAttr);
               if (tupleType.getType(i).isa<mlir::db::NullableType>()) {
                  entryVal = builder.create<mlir::db::AsNullableOp>(constRelationOp->getLoc(), tupleType.getType(i), entryVal);
               }
               values.push_back(entryVal);
               i++;
            }
         }
         mlir::Value packed = builder.create<mlir::util::PackOp>(constRelationOp->getLoc(), values);
         builder.create<mlir::dsa::Append>(constRelationOp->getLoc(), vector, packed);
      }
      {
         auto forOp2 = builder.create<mlir::dsa::ForOp>(constRelationOp->getLoc(), mlir::TypeRange{}, vector, mlir::Value(), mlir::ValueRange{});
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType, constRelationOp->getLoc());
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         auto unpacked = builder2.create<mlir::util::UnPackOp>(constRelationOp->getLoc(), forOp2.getInductionVar());
         attributes.setValuesForColumns(context, scope, unpacked.getResults());
         consumer->consume(this, builder2, context);
         builder2.create<mlir::dsa::YieldOp>(constRelationOp->getLoc(), mlir::ValueRange{});
      }
      builder.create<mlir::dsa::FreeOp>(constRelationOp->getLoc(), vector);
   }
   virtual ~ConstRelTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createConstRelTranslator(mlir::relalg::ConstRelationOp constRelationOp) {
   return std::make_unique<ConstRelTranslator>(constRelationOp);
}