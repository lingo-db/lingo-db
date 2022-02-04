#include "mlir/Conversion/RelAlgToDB/Translator.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilOps.h"

class MaterializeTranslator : public mlir::relalg::Translator {
   mlir::relalg::MaterializeOp materializeOp;
   size_t builderId;
   mlir::Value table;

   public:
   MaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) : mlir::relalg::Translator(materializeOp.rel()), materializeOp(materializeOp) {
   }
   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) override {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      children[0]->addRequiredBuilders(requiredBuilders);
   }
   virtual void setInfo(mlir::relalg::Translator* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      this->requiredAttributes.insert(mlir::relalg::Attributes::fromArrayAttr(materializeOp.attrs()));
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return {};
   }
   virtual void consume(mlir::relalg::Translator* child, mlir::OpBuilder& builder, mlir::relalg::TranslatorContext& context) override {
      std::vector<mlir::Value> values;
      for (auto attr : materializeOp.attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            values.push_back(context.getValueForAttribute(&attrRef.getRelationalAttribute()));
         }
      }
      mlir::Value tableBuilder = context.builders[builderId];
      mlir::Value packed = builder.create<mlir::util::PackOp>(materializeOp->getLoc(), values);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(materializeOp->getLoc(), tableBuilder.getType(), tableBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }
   virtual void produce(mlir::relalg::TranslatorContext& context, mlir::OpBuilder& bX) override {
      auto p=std::make_shared<mlir::relalg::Pipeline>(bX.getBlock()->getParentOp()->getParentOfType<mlir::ModuleOp>());
      context.pipelineManager.setCurrentPipeline(p);
      context.pipelineManager.addPipeline(p);
      std::vector<mlir::Type> types;
      for (auto attr : materializeOp.attrs()) {
         if (auto attrRef = attr.dyn_cast_or_null<mlir::relalg::RelationalAttributeRefAttr>()) {
            types.push_back(attrRef.getRelationalAttribute().type);
         }
      }
      auto res = p->addInitFn([&](mlir::OpBuilder& builder) {
         return std::vector<mlir::Value>({builder.create<mlir::db::CreateTableBuilder>(materializeOp.getLoc(), mlir::db::TableBuilderType::get(builder.getContext(), mlir::TupleType::get(builder.getContext(), types)), materializeOp.columns())});
      });
      builderId = context.getBuilderId();
      context.builders[builderId] = p->addDependency(res[0]);
      children[0]->addRequiredBuilders({builderId});
      children[0]->produce(context, p->getBuilder());
      p->finishMainFunction({context.builders[builderId]});
      p->addFinalizeFn([&](mlir::OpBuilder& builder,mlir::ValueRange args) {
         auto table = builder.create<mlir::db::BuilderBuild>(materializeOp.getLoc(), mlir::db::TableType::get(builder.getContext()), args[0]);
         return std::vector<mlir::Value>{table};
      });
      context.pipelineManager.execute(bX);
      table=context.pipelineManager.getResultsFromPipeline(p)[0];
   }
   virtual void done() override {
      materializeOp.replaceAllUsesWith(table);
   }
   virtual ~MaterializeTranslator() {}
};

std::unique_ptr<mlir::relalg::Translator> mlir::relalg::Translator::createMaterializeTranslator(mlir::relalg::MaterializeOp materializeOp) {
   return std::make_unique<MaterializeTranslator>(materializeOp);
}