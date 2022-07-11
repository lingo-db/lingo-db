#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/DSA/IR/DSAOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MaterializeTranslator : public mlir::relalg::Translator {
   mlir::relalg::MaterializeOp materializeOp;
   mlir::Value tableBuilder;
   mlir::Value table;
   mlir::relalg::OrderedAttributes orderedAttributes;
   std::string arrowDescrFromType(mlir::Type type) {
      if (isIntegerType(type, 1)) {
         return "bool";
      } else if (auto intWidth = getIntegerWidth(type, false)) {
         return "int[" + std::to_string(intWidth) + "]";
      } else if (auto uIntWidth = getIntegerWidth(type, true)) {
         return "uint[" + std::to_string(uIntWidth) + "]";
      } else if (auto decimalType = type.dyn_cast_or_null<mlir::db::DecimalType>()) {
         // TODO: actually handle cases where 128 bits are insufficient.
         auto prec = std::min(decimalType.getP(), 38);
         return "decimal[" + std::to_string(prec) + "," + std::to_string(decimalType.getS()) + "]";
      } else if (auto floatType = type.dyn_cast_or_null<mlir::FloatType>()) {
         return "float[" + std::to_string(floatType.getWidth()) + "]";
      } else if (auto stringType = type.dyn_cast_or_null<mlir::db::StringType>()) {
         return "string";
      } else if (auto dateType = type.dyn_cast_or_null<mlir::db::DateType>()) {
         if (dateType.getUnit() == mlir::db::DateUnitAttr::day) {
            return "date[32]";
         } else {
            return "date[64]";
         }
      } else if (auto charType = type.dyn_cast_or_null<mlir::db::CharType>()) {
         return "fixed_sized[" + std::to_string(charType.getBytes()) + "]";
      } else if (auto intervalType = type.dyn_cast_or_null<mlir::db::IntervalType>()) {
         if (intervalType.getUnit() == mlir::db::IntervalUnitAttr::months) {
            return "interval_months";
         } else {
            return "interval_daytime";
         }
      } else if (auto timestampType = type.dyn_cast_or_null<mlir::db::TimestampType>()) {
         return "timestamp[" + std::to_string(static_cast<uint32_t>(timestampType.getUnit())) + "]";
      }
      return "";
   }

   public:
   MaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) : mlir::relalg::Translator(materializeOp.rel()), materializeOp(materializeOp) {
      orderedAttributes = mlir::relalg::OrderedAttributes::fromRefArr(materializeOp.cols());
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::ColumnSet requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(mlir::relalg::ColumnSet::fromArrayAttr(materializeOp.cols()));
      propagateInfo();
   }
   virtual mlir::relalg::ColumnSet getAvailableColumns() override {
      return {};
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      for (size_t i = 0; i < orderedAttributes.getAttrs().size(); i++) {
         auto val = orderedAttributes.resolve(context, i);
         mlir::Value valid;
         if (val.getType().isa<mlir::db::NullableType>()) {
            valid = builder.create<mlir::db::IsNullOp>(materializeOp->getLoc(), val);
            valid = builder.create<mlir::db::NotOp>(materializeOp->getLoc(), valid);
            val = builder.create<mlir::db::NullableGetVal>(materializeOp->getLoc(), getBaseType(val.getType()), val);
         }
         builder.create<mlir::dsa::Append>(materializeOp->getLoc(), tableBuilder, val, valid);
      }
      builder.create<mlir::dsa::NextRow>(materializeOp->getLoc(), tableBuilder);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& builder) override {
      std::string descr = "";
      auto tupleType = orderedAttributes.getTupleType(builder.getContext());
      for (size_t i = 0; i < materializeOp.columns().size(); i++) {
         if (!descr.empty()) {
            descr += ";";
         }
         descr += materializeOp.columns()[i].cast<mlir::StringAttr>().str() + ":" + arrowDescrFromType(getBaseType(tupleType.getType(i)));
      }
      tableBuilder = builder.create<mlir::dsa::CreateDS>(materializeOp.getLoc(), mlir::dsa::TableBuilderType::get(builder.getContext(), orderedAttributes.getTupleType(builder.getContext())), builder.getStringAttr(descr));
      children[0]->produce(context, builder);
      table = builder.create<mlir::dsa::Finalize>(materializeOp.getLoc(), mlir::dsa::TableType::get(builder.getContext()), tableBuilder).res();
   }
   virtual void done() override {
      materializeOp.replaceAllUsesWith(table);
   }
   virtual ~MaterializeTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) {
   return std::make_unique<MaterializeTranslator>(materializeOp);
}
