#include "mlir/Conversion/RelAlgToDB/OrderedAttributes.h"
#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MaterializeTranslator : public mlir::relalg::Translator {
   mlir::relalg::MaterializeOp materializeOp;
   mlir::Value tableBuilder;
   mlir::Value table;
   mlir::relalg::OrderedAttributes orderedAttributes;

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
      mlir::Value packed = orderedAttributes.pack(context, builder, materializeOp->getLoc());
      builder.create<mlir::db::AddTableRow>(materializeOp->getLoc(), tableBuilder, packed);
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& bX) override {
      auto p = std::make_shared<mlir::relalg::Pipeline>(bX.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
      context.pipelineManager.setCurrentPipeline(p);
      context.pipelineManager.addPipeline(p);
      auto res = p->addInitFn([&](mlir::OpBuilder& builder) {
         return std::vector<mlir::Value>({builder.create<mlir::db::CreateTableBuilder>(materializeOp.getLoc(), mlir::db::TableBuilderType::get(builder.getContext(), orderedAttributes.getTupleType(builder.getContext())), materializeOp.columns())});
      });
      tableBuilder= p->addDependency(res[0]);
      children[0]->produce(context, p->getBuilder());
      p->finishMainFunction({tableBuilder});
      p->addFinalizeFn([&](mlir::OpBuilder& builder, mlir::ValueRange args) {
         auto table = builder.create<mlir::db::FinalizeTable>(materializeOp.getLoc(), mlir::db::TableType::get(builder.getContext()), args[0]);
         return std::vector<mlir::Value>{table};
      });
      context.pipelineManager.execute(bX);
      table = context.pipelineManager.getResultsFromPipeline(p)[0];
   }
   virtual void done() override {
      materializeOp.replaceAllUsesWith(table);
   }
   virtual ~MaterializeTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) {
   return std::make_unique<MaterializeTranslator>(materializeOp);
}