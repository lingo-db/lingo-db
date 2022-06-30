#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class BaseTableTranslator : public mlir::relalg::Translator {
   static bool registered;
   mlir::relalg::BaseTableOp baseTableOp;

   public:
   BaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) : mlir::relalg::Translator(baseTableOp), baseTableOp(baseTableOp) {}
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      std::vector<mlir::Type> types;
      std::vector<const mlir::relalg::Column*> cols;
      std::vector<mlir::Attribute> columnNames;
      std::string tableName = baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>().str();
      std::string scanDescription = R"({ "table": ")" + tableName + R"(", "columns": [ )";
      bool first = true;
      for (auto namedAttr : baseTableOp.columnsAttr().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<mlir::relalg::ColumnDefAttr>();
         if (requiredAttributes.contains(&attrDef.getColumn())) {
            if (!first) {
               scanDescription += ",";
            } else {
               first = false;
            }
            scanDescription += "\"" + identifier.str() + "\"";
            columnNames.push_back(builder.getStringAttr(identifier.strref()));
            types.push_back(getBaseType(attrDef.getColumn().type));
            cols.push_back(&attrDef.getColumn());
         }
      }
      scanDescription += "] }";

      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      auto recordBatch = mlir::dsa::RecordBatchType::get(builder.getContext(), tupleType);
      mlir::Type chunkIterable = mlir::dsa::GenericIterableType::get(builder.getContext(), recordBatch, "table_chunk_iterator");

      auto chunkIterator = builder.create<mlir::dsa::ScanSource>(baseTableOp->getLoc(), chunkIterable, builder.getStringAttr(scanDescription));

      auto forOp = builder.create<mlir::dsa::ForOp>(baseTableOp->getLoc(), mlir::TypeRange{}, chunkIterator, mlir::Value(), mlir::ValueRange{});
      mlir::Block* block = new mlir::Block;
      block->addArgument(recordBatch, baseTableOp->getLoc());
      forOp.getBodyRegion().push_back(block);
      mlir::OpBuilder builder1(forOp.getBodyRegion());
      auto forOp2 = builder1.create<mlir::dsa::ForOp>(baseTableOp->getLoc(), mlir::TypeRange{}, forOp.getInductionVar(), mlir::Value(), mlir::ValueRange{});
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(recordBatch.getElementType(), baseTableOp->getLoc());
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      size_t i = 0;
      for (const auto* attr : cols) {
         std::vector<mlir::Type> types;
         types.push_back(getBaseType(attr->type));
         if (attr->type.isa<mlir::db::NullableType>()) {
            types.push_back(builder.getI1Type());
         }
         auto atOp = builder2.create<mlir::dsa::At>(baseTableOp->getLoc(), types, forOp2.getInductionVar(), i++);
         if (attr->type.isa<mlir::db::NullableType>()) {
            mlir::Value isNull = builder2.create<mlir::db::NotOp>(baseTableOp->getLoc(), atOp.valid());
            mlir::Value val = builder2.create<mlir::db::AsNullableOp>(baseTableOp->getLoc(), attr->type, atOp.val(), isNull);
            context.setValueForAttribute(scope, attr, val);
         } else {
            context.setValueForAttribute(scope, attr, atOp.val());
         }
      }
      consumer->consume(this, builder2, context);
      builder2.create<mlir::dsa::YieldOp>(baseTableOp->getLoc());
      builder1.create<mlir::dsa::YieldOp>(baseTableOp->getLoc());
   }
   virtual ~BaseTableTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createBaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) {
   return std::make_unique<BaseTableTranslator>(baseTableOp);
}