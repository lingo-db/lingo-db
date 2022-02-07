#include "mlir/Conversion/RelAlgToDB/Pipeline.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class BaseTableTranslator : public mlir::relalg::Translator {
   static bool registered;
   mlir::relalg::BaseTableOp baseTableOp;

   public:
   BaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) : mlir::relalg::Translator(baseTableOp), baseTableOp(baseTableOp) {
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
      std::vector<mlir::Type> types;
      std::vector<const mlir::relalg::RelationalAttribute*> attrs;
      std::vector<mlir::Attribute> columnNames;

      for (auto namedAttr : baseTableOp.columnsAttr().getValue()) {
         auto identifier = namedAttr.getName();
         auto attr = namedAttr.getValue();
         auto attrDef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeDefAttr>();
         if (requiredAttributes.contains(&attrDef.getRelationalAttribute())) {
            columnNames.push_back(builder.getStringAttr(identifier.strref()));
            types.push_back(attrDef.getRelationalAttribute().type);
            attrs.push_back(&attrDef.getRelationalAttribute());
         }
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Type rowIterable = mlir::db::GenericIterableType::get(builder.getContext(), tupleType, "table_row_iterator");
      mlir::Type chunkIterable = mlir::db::GenericIterableType::get(builder.getContext(), rowIterable, "table_chunk_iterator");
      auto currPipeline = context.pipelineManager.getCurrentPipeline();
      auto initRes = currPipeline->addInitFn([&](mlir::OpBuilder& builder) {
         mlir::Value table = builder.create<mlir::db::GetTable>(baseTableOp->getLoc(), mlir::db::TableType::get(builder.getContext()), baseTableOp->getAttr("table_identifier").cast<mlir::StringAttr>());
         auto chunkIterator = builder.create<mlir::db::TableScan>(baseTableOp->getLoc(), chunkIterable, table, builder.getArrayAttr(columnNames));

         return std::vector<Value>({chunkIterator});
      });
      auto forOp = builder.create<mlir::db::ForOp>(baseTableOp->getLoc(), getRequiredBuilderTypes(context), currPipeline->addDependency(initRes[0]), context.pipelineManager.getCurrentPipeline()->getFlag(), getRequiredBuilderValues(context));
      mlir::Block* block = new mlir::Block;
      block->addArgument(rowIterable,baseTableOp->getLoc());
      block->addArguments(getRequiredBuilderTypes(context), getRequiredBuilderLocs(context));
      forOp.getBodyRegion().push_back(block);
      mlir::OpBuilder builder1(forOp.getBodyRegion());
      auto forOp2 = builder1.create<mlir::db::ForOp>(baseTableOp->getLoc(), getRequiredBuilderTypes(context), forOp.getInductionVar(), context.pipelineManager.getCurrentPipeline()->getFlag(), block->getArguments().drop_front(1));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType,baseTableOp->getLoc());
      block2->addArguments(getRequiredBuilderTypes(context),getRequiredBuilderLocs(context));
      forOp2.getBodyRegion().push_back(block2);
      mlir::OpBuilder builder2(forOp2.getBodyRegion());
      setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
      auto unpacked = builder2.create<mlir::util::UnPackOp>(baseTableOp->getLoc(), forOp2.getInductionVar());
      size_t i = 0;
      for (const auto* attr : attrs) {
         context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
      }
      consumer->consume(this, builder2, context);
      builder2.create<mlir::db::YieldOp>(baseTableOp->getLoc(), getRequiredBuilderValues(context));
      builder1.create<mlir::db::YieldOp>(baseTableOp->getLoc(), forOp2.getResults());
      setRequiredBuilderValues(context, forOp.results());
   }
   virtual ~BaseTableTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createBaseTableTranslator(mlir::relalg::BaseTableOp baseTableOp) {
   return std::make_unique<BaseTableTranslator>(baseTableOp);
}