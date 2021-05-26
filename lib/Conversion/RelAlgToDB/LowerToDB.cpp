#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/RelAlgToDB/RelAlgToDBPass.h"
#include "mlir/Dialect/DB/IR/DBOps.h"
#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/util/UtilDialect.h"
#include "mlir/Dialect/util/UtilOps.h"

namespace {

class LoweringContext {
   llvm::ScopedHashTable<const mlir::relalg::RelationalAttribute*, mlir::Value> symbolTable;
   using AttributeResolverScope = llvm::ScopedHashTableScope<const mlir::relalg::RelationalAttribute*, mlir::Value>;

   public:
   mlir::Value getValueForAttribute(const mlir::relalg::RelationalAttribute* attribute) const {
      assert(symbolTable.count(attribute));
      return symbolTable.lookup(attribute);
   }
   void setValueForAttribute(AttributeResolverScope& scope, const mlir::relalg::RelationalAttribute* iu, mlir::Value v) {
      symbolTable.insertIntoScope(&scope, iu, v);
   }
   AttributeResolverScope createScope() {
      return AttributeResolverScope(symbolTable);
   }
   mlir::Value executionContext;
   std::unordered_map<size_t, mlir::Value> builders;
   size_t getBuilderId() {
      static size_t id = 0;
      return id++;
   }
};
class ProducerConsumerBuilder : public mlir::OpBuilder {
   public:
   using mlir::OpBuilder::OpBuilder;
};
class ProducerConsumerNode {
   protected:
   ProducerConsumerNode* consumer;
   std::vector<std::unique_ptr<ProducerConsumerNode>> children;
   std::vector<size_t> requiredBuilders;
   mlir::relalg::Attributes requiredAttributes;
   void propagateInfo() {
      for (auto& c : children) {
         auto available = c->getAvailableAttributes();
         mlir::relalg::Attributes toPropagate = requiredAttributes.intersect(available);
         c->setInfo(this, toPropagate);
      }
   }
   std::vector<mlir::Value> getRequiredBuilderValues(LoweringContext& context) {
      std::vector<mlir::Value> res;
      for (auto x : requiredBuilders) {
         res.push_back(context.builders[x]);
      }
      return res;
   }
   void setRequiredBuilderValues(LoweringContext& context, mlir::ValueRange values) {
      size_t i = 0;
      for (auto x : requiredBuilders) {
         context.builders[x] = values[i++];
      }
   }
   std::vector<mlir::Type> getRequiredBuilderTypes(LoweringContext& context) {
      std::vector<mlir::Type> res;
      for (auto x : requiredBuilders) {
         res.push_back(context.builders[x].getType());
      }
      return res;
   }

   public:
   ProducerConsumerNode(mlir::ValueRange children);
   void setRequiredBuilders(std::vector<size_t> requiredBuilders) {
      this->requiredBuilders = requiredBuilders;
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) = 0;
   virtual mlir::relalg::Attributes getAvailableAttributes() = 0;
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) = 0;
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) = 0;
   virtual void done() {}
   virtual ~ProducerConsumerNode() {}
};

class BaseTableLowering : public ProducerConsumerNode {
   mlir::relalg::BaseTableOp baseTableOp;

   public:
   BaseTableLowering(mlir::relalg::BaseTableOp baseTableOp) : ProducerConsumerNode({}), baseTableOp(baseTableOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return baseTableOp.getCreatedAttributes();
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
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
      auto forOp = builder.create<mlir::db::ForOp>(baseTableOp->getLoc(), getRequiredBuilderTypes(context), chunkIterator, getRequiredBuilderValues(context));
      mlir::Block* block = new mlir::Block;
      block->addArgument(rowIterable);
      block->addArguments(getRequiredBuilderTypes(context));
      forOp.getBodyRegion().push_back(block);
      mlir::OpBuilder builder1(forOp.getBodyRegion());
      auto forOp2 = builder1.create<mlir::db::ForOp>(baseTableOp->getLoc(), getRequiredBuilderTypes(context), forOp.getInductionVar(), block->getArguments().drop_front(1));
      mlir::Block* block2 = new mlir::Block;
      block2->addArgument(tupleType);
      block2->addArguments(getRequiredBuilderTypes(context));
      forOp2.getBodyRegion().push_back(block2);
      ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
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
class MaterializeLowering : public ProducerConsumerNode {
   mlir::relalg::MaterializeOp materializeOp;
   size_t builderId;
   mlir::Value table;

   public:
   MaterializeLowering(mlir::relalg::MaterializeOp materializeOp) : ProducerConsumerNode(materializeOp.rel()), materializeOp(materializeOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return {};
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
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
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
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

std::unique_ptr<ProducerConsumerNode> createNodeFor(mlir::Operation* o) {
   std::unique_ptr<ProducerConsumerNode> res;
   llvm::TypeSwitch<mlir::Operation*>(o)
      .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp baseTableOp) {
         res = std::make_unique<BaseTableLowering>(baseTableOp);
      })
      .Case<mlir::relalg::MaterializeOp>([&](mlir::relalg::MaterializeOp materializeOp) {
         res = std::make_unique<MaterializeLowering>(materializeOp);
      })
      .Default([&](mlir::Operation*) {});

   return res;
}
ProducerConsumerNode::ProducerConsumerNode(mlir::ValueRange potentialChildren) {
   for (auto child : potentialChildren) {
      if (child.getType().isa<mlir::relalg::TupleStreamType>()) {
         children.push_back(createNodeFor(child.getDefiningOp()));
      }
   }
}

class LowerToDBPass : public mlir::PassWrapper<LowerToDBPass, mlir::FunctionPass> {
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::util::UtilDialect>();
   }
   bool isTranslationHook(mlir::Operation* op) {
      return ::llvm::TypeSwitch<mlir::Operation*, bool>(op)

         .Case<mlir::relalg::MaterializeOp>([&](mlir::Operation* op) {
            return true;
         })
         .Default([&](auto x) {
            return false;
         });
   }
   void runOnFunction() override {
      LoweringContext loweringContext;
      loweringContext.executionContext = getFunction().getArgument(0);
      getFunction().walk([&](mlir::Operation* op) {
         if (isTranslationHook(op)) {
            auto node = createNodeFor(op);
            node->setInfo(nullptr, {});
            ProducerConsumerBuilder builder(op);
            node->produce(loweringContext, builder);
            node->done();
         }
      });
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createLowerToDBPass() { return std::make_unique<LowerToDBPass>(); }
} // end namespace relalg
} // end namespace mlir