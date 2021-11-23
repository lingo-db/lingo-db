#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"
class ConstRelLowering : public mlir::relalg::ProducerConsumerNode {
   mlir::relalg::ConstRelationOp constRelationOp;

   public:
   ConstRelLowering(mlir::relalg::ConstRelationOp constRelationOp) : mlir::relalg::ProducerConsumerNode({}), constRelationOp(constRelationOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override{
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return constRelationOp.getCreatedAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::OpBuilder& builder, mlir::relalg::LoweringContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::OpBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      std::vector<mlir::Type> types;
      std::vector<const mlir::relalg::RelationalAttribute*> attrs;
      for (auto attr : constRelationOp.attributes().getValue()) {
         auto attrDef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         types.push_back(attrDef.getRelationalAttribute().type);
         attrs.push_back(&attrDef.getRelationalAttribute());
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(constRelationOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
      for (auto rowAttr : constRelationOp.valuesAttr()) {
         auto row = rowAttr.cast<ArrayAttr>();
         std::vector<Value> values;
         size_t i = 0;
         for (auto entryAttr : row.getValue()) {
            auto entryVal = builder.create<mlir::db::ConstantOp>(constRelationOp->getLoc(), types[i], entryAttr);
            values.push_back(entryVal);
            i++;
         }
         mlir::Value packed = builder.create<mlir::util::PackOp>(constRelationOp->getLoc(), values);
         vectorBuilder = builder.create<mlir::db::BuilderMerge>(constRelationOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      }
      Value vector = builder.create<mlir::db::BuilderBuild>(constRelationOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), vectorBuilder);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(constRelationOp->getLoc(), getRequiredBuilderTypes(context), vector, flag,getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         mlir::OpBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(constRelationOp->getLoc(), forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : attrs) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(constRelationOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
      builder.create<mlir::db::FreeOp>(constRelationOp->getLoc(),vector);
   }
   virtual ~ConstRelLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredConstRelOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::ConstRelationOp constRelationOp) {
  return std::make_unique<ConstRelLowering>(constRelationOp);
});