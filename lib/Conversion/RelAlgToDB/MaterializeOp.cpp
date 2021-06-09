#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MaterializeLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::MaterializeOp materializeOp;
   size_t builderId;
   mlir::Value table;

   public:
   MaterializeLowering(mlir::relalg::MaterializeOp materializeOp) : mlir::relalg::ProducerConsumerNode(materializeOp.rel()), materializeOp(materializeOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(mlir::relalg::Attributes::fromArrayAttr(materializeOp.attrs()));
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return {};
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      std::vector<mlir::Type> types;
      std::vector<mlir::Value> values;
      for (auto attr : materializeOp.attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            types.push_back(attrRef.getRelationalAttribute().type);
            values.push_back(context.getValueForAttribute(&attrRef.getRelationalAttribute()));
         }
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Value tableBuilder = context.builders[builderId];
      mlir::Value packed = builder.create<mlir::util::PackOp>(materializeOp->getLoc(), tupleType, values);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(materializeOp->getLoc(), tableBuilder.getType(), tableBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      std::vector<mlir::Type> types;
      for (auto attr : materializeOp.attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            types.push_back(attrRef.getRelationalAttribute().type);
         }
      }
      mlir::Value tableBuilder = builder.create<mlir::db::CreateTableBuilder>(materializeOp.getLoc(), mlir::db::TableBuilderType::get(builder.getContext(), mlir::TupleType::get(builder.getContext(), types)), materializeOp.columns());
      builderId = context.getBuilderId();
      context.builders[builderId] = tableBuilder;
      children[0]->setRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      table = builder.create<mlir::db::BuilderBuild>(materializeOp.getLoc(), mlir::db::TableType::get(builder.getContext()), tableBuilder);
   }
   virtual void done() override {
      materializeOp.replaceAllUsesWith(table);
   }
   virtual ~MaterializeLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredMaterializeOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::MaterializeOp materializeOp) {
  return std::make_unique<MaterializeLowering>(materializeOp);
});