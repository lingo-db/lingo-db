#include "mlir/Conversion/RelAlgToDB/ProducerConsumerNode.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class BaseTableLowering : public mlir::relalg::ProducerConsumerNode {
   static bool registered;
   mlir::relalg::BaseTableOp baseTableOp;

   public:
   BaseTableLowering(mlir::relalg::BaseTableOp baseTableOp) : mlir::relalg::ProducerConsumerNode({}), baseTableOp(baseTableOp) {
   }
   virtual void setInfo(mlir::relalg::ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return baseTableOp.getCreatedAttributes();
   }
   virtual void consume(mlir::relalg::ProducerConsumerNode* child, mlir::relalg::ProducerConsumerBuilder& builder, mlir::relalg::LoweringContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(mlir::relalg::LoweringContext& context, mlir::relalg::ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();
      using namespace mlir;
      mlir::Value table = builder.create<mlir::db::GetTable>(baseTableOp->getLoc(), mlir::db::TableType::get(builder.getContext()), baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>(), context.executionContext);
      std::vector<mlir::Attribute> columnNames;
      std::vector<mlir::Type> types;
      std::vector<const mlir::relalg::RelationalAttribute*> attrs;
      for (auto namedAttr : baseTableOp.columnsAttr().getValue()) {
         auto [identifier, attr] = namedAttr;
         columnNames.push_back(builder.getStringAttr(identifier.strref()));
         auto attrDef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         types.push_back(attrDef.getRelationalAttribute().type);
         attrs.push_back(&attrDef.getRelationalAttribute());
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Type rowIterable = mlir::db::GenericIterableType::get(builder.getContext(), tupleType, "table_row_iterator");
      mlir::Type chunkIterable = mlir::db::GenericIterableType::get(builder.getContext(), rowIterable, "table_chunk_iterator");
      auto chunkIterator = builder.create<mlir::db::TableScan>(baseTableOp->getLoc(), chunkIterable, table, builder.getArrayAttr(columnNames));
      auto forOp = builder.create<mlir::db::ForOp>(baseTableOp->getLoc(), getRequiredBuilderTypes(context), chunkIterator,Value()/*todo*/, getRequiredBuilderValues(context));
      mlir::Block* block = new mlir::Block;
      block->addArgument(rowIterable);
      block->addArguments(getRequiredBuilderTypes(context));
      forOp.getBodyRegion().push_back(block);
      mlir::OpBuilder builder1(forOp.getBodyRegion());
      auto forOp2 = builder1.create<mlir::db::ForOp>(baseTableOp->getLoc(), getRequiredBuilderTypes(context), forOp.getInductionVar(),Value()/*todo*/, block->getArguments().drop_front(1));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType);
      block2->addArguments(getRequiredBuilderTypes(context));
      forOp2.getBodyRegion().push_back(block2);
      mlir::relalg::ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
      setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
      auto unpacked = builder2.create<mlir::util::UnPackOp>(baseTableOp->getLoc(), types, forOp2.getInductionVar());
      size_t i = 0;
      for (const auto* attr : attrs) {
         context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
      }
      consumer->consume(this, builder2, context);
      builder2.create<mlir::db::YieldOp>(baseTableOp->getLoc(), getRequiredBuilderValues(context));
      builder1.create<mlir::db::YieldOp>(baseTableOp->getLoc(), forOp2.getResults());
      setRequiredBuilderValues(context, forOp.results());
   }
   virtual ~BaseTableLowering() {}
};

bool mlir::relalg::ProducerConsumerNodeRegistry::registeredBaseTableOp = mlir::relalg::ProducerConsumerNodeRegistry::registerNode([](mlir::relalg::BaseTableOp baseTableOp) {
  return std::make_unique<BaseTableLowering>(baseTableOp);
});