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

   public:
   using AttributeResolverScope = llvm::ScopedHashTableScope<const mlir::relalg::RelationalAttribute*, mlir::Value>;

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

   void mergeRelatinalBlock(mlir::Block* source, LoweringContext& context, LoweringContext::AttributeResolverScope& scope) {
      mlir::Block* dest = getBlock();

      // Splice the operations of the 'source' block into the 'dest' block and erase
      // it.
      llvm::iplist<mlir::Operation> translated;
      std::vector<mlir::Operation*> toErase;
      for (auto getAttrOp : source->getOps<mlir::relalg::GetAttrOp>()) {
         getAttrOp.replaceAllUsesWith(context.getValueForAttribute(&getAttrOp.attr().getRelationalAttribute()));
         toErase.push_back(getAttrOp.getOperation());
      }
      for (auto addAttrOp : source->getOps<mlir::relalg::AddAttrOp>()) {
         context.setValueForAttribute(scope, &addAttrOp.attr().getRelationalAttribute(), addAttrOp.val());
         toErase.push_back(addAttrOp.getOperation());
      }

      dest->getOperations().splice(dest->end(), source->getOperations());
      for (auto* op : toErase) {
         op->erase();
      }
   }
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
   virtual void setRequiredBuilders(std::vector<size_t> requiredBuilders) {
      this->requiredBuilders = requiredBuilders;
      for (auto& child : children) {
         child->setRequiredBuilders(requiredBuilders);
      }
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
class ConstRelLowering : public ProducerConsumerNode {
   mlir::relalg::ConstRelationOp constRelationOp;

   public:
   ConstRelLowering(mlir::relalg::ConstRelationOp constRelationOp) : ProducerConsumerNode({}), constRelationOp(constRelationOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return constRelationOp.getCreatedAttributes();
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
      assert(false && "should not happen");
   }
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
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
      for (auto rowAttr:constRelationOp.valuesAttr()){
         auto row=rowAttr.cast<ArrayAttr>();
         std::vector<Value> values;
         size_t i=0;
         for(auto entryAttr:row.getValue()){
            entryAttr.dump();
            auto entryVal=builder.create<mlir::db::ConstantOp>(constRelationOp->getLoc(),types[i],entryAttr);
            values.push_back(entryVal);
            i++;
         }
         mlir::Value packed = builder.create<mlir::util::PackOp>(constRelationOp->getLoc(), tupleType, values);
         vectorBuilder = builder.create<mlir::db::BuilderMerge>(constRelationOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      }
      Value vector = builder.create<mlir::db::BuilderBuild>(constRelationOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), vectorBuilder);
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(constRelationOp->getLoc(), getRequiredBuilderTypes(context), vector, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(constRelationOp->getLoc(), types, forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : attrs) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(constRelationOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }
   virtual ~ConstRelLowering() {}
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
      this->requiredAttributes.insert(mlir::relalg::Attributes::fromArrayAttr(materializeOp.attrs()));
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

class SelectionLowering : public ProducerConsumerNode {
   mlir::relalg::SelectionOp selectionOp;

   public:
   SelectionLowering(mlir::relalg::SelectionOp selectionOp) : ProducerConsumerNode(selectionOp.rel()), selectionOp(selectionOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
      auto scope = context.createScope();
      mlir::relalg::SelectionOp clonedSelectionOp = mlir::dyn_cast<mlir::relalg::SelectionOp>(selectionOp->clone());
      mlir::Block* block = &clonedSelectionOp.predicate().getBlocks().front();
      auto* terminator = block->getTerminator();

      builder.mergeRelatinalBlock(block, context, scope);

      auto ifOp = builder.create<mlir::db::IfOp>(selectionOp->getLoc(), getRequiredBuilderTypes(context), mlir::cast<mlir::relalg::ReturnOp>(terminator).results()[0]);
      mlir::Block* ifBlock = new mlir::Block;

      ifOp.thenRegion().push_back(ifBlock);

      ProducerConsumerBuilder builder1(ifOp.thenRegion());
      if (!requiredBuilders.empty()) {
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         ProducerConsumerBuilder builder2(ifOp.elseRegion());
         builder2.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context));
      }
      consumer->consume(this, builder1, context);
      builder1.create<mlir::db::YieldOp>(selectionOp->getLoc(), getRequiredBuilderValues(context));

      size_t i = 0;
      for (auto b : requiredBuilders) {
         context.builders[b] = ifOp.getResult(i++);
      }
      terminator->erase();
      clonedSelectionOp->destroy();
   }
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~SelectionLowering() {}
};
class MapLowering : public ProducerConsumerNode {
   mlir::relalg::MapOp mapOp;

   public:
   MapLowering(mlir::relalg::MapOp mapOp) : ProducerConsumerNode(mapOp.rel()), mapOp(mapOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(mapOp.getCreatedAttributes());
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
      auto scope = context.createScope();
      mlir::relalg::MapOp clonedSelectionOp = mlir::dyn_cast<mlir::relalg::MapOp>(mapOp->clone());
      mlir::Block* block = &clonedSelectionOp.predicate().getBlocks().front();
      auto* terminator = block->getTerminator();

      builder.mergeRelatinalBlock(block, context, scope);
      consumer->consume(this, builder, context);
      terminator->erase();
      clonedSelectionOp->destroy();
   }
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~MapLowering() {}
};
class CrossProductLowering : public ProducerConsumerNode {
   mlir::relalg::CrossProductOp crossProductOp;

   public:
   CrossProductLowering(mlir::relalg::CrossProductOp crossProductOp) : ProducerConsumerNode(mlir::ValueRange({crossProductOp.left(), crossProductOp.right()})), crossProductOp(crossProductOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes().insert(this->children[1]->getAvailableAttributes());
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
      if (child == this->children[0].get()) {
         children[1]->produce(context, builder);
      } else if (child == this->children[1].get()) {
         consumer->consume(this, builder, context);
      }
   }
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
      children[0]->produce(context, builder);
   }

   virtual ~CrossProductLowering() {}
};
class SortLowering : public ProducerConsumerNode {
   mlir::relalg::SortOp sortOp;
   size_t builderId;
   mlir::Value vector;

   public:
   SortLowering(mlir::relalg::SortOp sortOp) : ProducerConsumerNode(sortOp.rel()), sortOp(sortOp) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override {
      this->consumer = consumer;
      this->requiredAttributes = requiredAttributes;
      propagateInfo();
   }
   virtual mlir::relalg::Attributes getAvailableAttributes() override {
      return this->children[0]->getAvailableAttributes();
   }
   virtual void consume(ProducerConsumerNode* child, ProducerConsumerBuilder& builder, LoweringContext& context) override {
      std::vector<mlir::Type> types;
      std::vector<mlir::Value> values;
      for (auto *attr : requiredAttributes) {
         types.push_back(attr->type);
         values.push_back(context.getValueForAttribute(attr));
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Value vectorBuilder = context.builders[builderId];
      mlir::Value packed = builder.create<mlir::util::PackOp>(sortOp->getLoc(), tupleType, values);
      mlir::Value mergedBuilder = builder.create<mlir::db::BuilderMerge>(sortOp->getLoc(), vectorBuilder.getType(), vectorBuilder, packed);
      context.builders[builderId] = mergedBuilder;
   }
   mlir::Value createSortPredicate(mlir::OpBuilder& builder,std::vector<std::pair<mlir::Value,mlir::Value>> sortCriteria,mlir::Value trueVal,mlir::Value falseVal,size_t pos){
      if(pos<sortCriteria.size()){
         auto lt  =builder.create<mlir::db::CmpOp>(builder.getUnknownLoc(),mlir::db::DBCmpPredicate::lt,sortCriteria[pos].first,sortCriteria[pos].second);
         auto ifOp = builder.create<mlir::db::IfOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), lt);
         mlir::Block* ifBlock = new mlir::Block;
         ifOp.thenRegion().push_back(ifBlock);
         ProducerConsumerBuilder builder1(ifOp.thenRegion());
         builder1.create<mlir::db::YieldOp>(builder.getUnknownLoc(), trueVal);
         mlir::Block* elseBlock = new mlir::Block;
         ifOp.elseRegion().push_back(elseBlock);
         ProducerConsumerBuilder builder2(ifOp.elseRegion());
         auto eq  =builder2.create<mlir::db::CmpOp>(builder.getUnknownLoc(),mlir::db::DBCmpPredicate::eq,sortCriteria[pos].first,sortCriteria[pos].second);
         auto ifOp2 = builder2.create<mlir::db::IfOp>(builder.getUnknownLoc(), mlir::db::BoolType::get(builder.getContext()), eq);
         mlir::Block* ifBlock2 = new mlir::Block;
         ifOp2.thenRegion().push_back(ifBlock2);
         ProducerConsumerBuilder builder3(ifOp2.thenRegion());
         builder3.create<mlir::db::YieldOp>(builder.getUnknownLoc(),createSortPredicate(builder3,sortCriteria,trueVal,falseVal,pos+1));
         mlir::Block* elseBlock2 = new mlir::Block;
         ifOp2.elseRegion().push_back(elseBlock2);
         ProducerConsumerBuilder builder4(ifOp2.elseRegion());
         builder4.create<mlir::db::YieldOp>(builder.getUnknownLoc(), falseVal);


         builder2.create<mlir::db::YieldOp>(builder.getUnknownLoc(),ifOp2.getResult(0));

         return ifOp.getResult(0);
      }else{
         return falseVal;

      }
   }
   virtual void produce(LoweringContext& context, ProducerConsumerBuilder& builder) override {
      auto scope = context.createScope();
      std::unordered_map<const mlir::relalg::RelationalAttribute*,size_t> attributePos;
      std::vector<mlir::Type> types;
      size_t i=0;
      for (auto *attr : requiredAttributes) {
         types.push_back(attr->type);
         attributePos[attr]=i++;
      }
      auto tupleType = mlir::TupleType::get(builder.getContext(), types);
      mlir::Value vectorBuilder = builder.create<mlir::db::CreateVectorBuilder>(sortOp.getLoc(), mlir::db::VectorBuilderType::get(builder.getContext(), tupleType));
      builderId = context.getBuilderId();
      context.builders[builderId] = vectorBuilder;
      children[0]->setRequiredBuilders({builderId});
      children[0]->produce(context, builder);
      vector = builder.create<mlir::db::BuilderBuild>(sortOp.getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType), vectorBuilder);
      {
         auto dbSortOp = builder.create<mlir::db::SortOp>(sortOp->getLoc(), mlir::db::VectorType::get(builder.getContext(), tupleType),vector);
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(tupleType);
         dbSortOp.region().push_back(block2);
         ProducerConsumerBuilder builder2(dbSortOp.region());
         auto unpackedLeft = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), types, block2->getArgument(0));
         auto unpackedRight = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), types, block2->getArgument(1));
         std::vector<std::pair<mlir::Value,mlir::Value>> sortCriteria;
         for(auto attr:sortOp.sortspecs()){
            auto sortspecAttr=attr.cast<mlir::relalg::SortSpecificationAttr>();
            mlir::Value left=unpackedLeft.getResult(attributePos[&sortspecAttr.getAttr().getRelationalAttribute()]);
            mlir::Value right=unpackedRight.getResult(attributePos[&sortspecAttr.getAttr().getRelationalAttribute()]);
            if(sortspecAttr.getSortSpec()==mlir::relalg::SortSpec::desc){
               std::swap(left,right);
            }
            sortCriteria.push_back({left,right});
         }
         auto trueVal=builder2.create<mlir::db::ConstantOp>(sortOp->getLoc(),mlir::db::BoolType::get(builder.getContext()),builder.getIntegerAttr(builder.getI64Type(),1));
         auto falseVal=builder2.create<mlir::db::ConstantOp>(sortOp->getLoc(),mlir::db::BoolType::get(builder.getContext()),builder.getIntegerAttr(builder.getI64Type(),0));

         builder2.create<mlir::db::YieldOp>(sortOp->getLoc(),createSortPredicate(builder2,sortCriteria,trueVal,falseVal,0));
         vector=dbSortOp.sorted();
      }
      {
         auto forOp2 = builder.create<mlir::db::ForOp>(sortOp->getLoc(), getRequiredBuilderTypes(context), vector, getRequiredBuilderValues(context));
         mlir::Block* block2 = new mlir::Block;
         block2->addArgument(tupleType);
         block2->addArguments(getRequiredBuilderTypes(context));
         forOp2.getBodyRegion().push_back(block2);
         ProducerConsumerBuilder builder2(forOp2.getBodyRegion());
         setRequiredBuilderValues(context, block2->getArguments().drop_front(1));
         auto unpacked = builder2.create<mlir::util::UnPackOp>(sortOp->getLoc(), types, forOp2.getInductionVar());
         size_t i = 0;
         for (const auto* attr : requiredAttributes) {
            context.setValueForAttribute(scope, attr, unpacked.getResult(i++));
         }
         consumer->consume(this, builder2, context);
         builder2.create<mlir::db::YieldOp>(sortOp->getLoc(), getRequiredBuilderValues(context));
         setRequiredBuilderValues(context, forOp2.results());
      }
   }

   virtual ~SortLowering() {}
};
std::unique_ptr<ProducerConsumerNode> createNodeFor(mlir::Operation* o) {
   std::unique_ptr<ProducerConsumerNode> res;
   llvm::TypeSwitch<mlir::Operation*>(o)
      .Case<mlir::relalg::BaseTableOp>([&](mlir::relalg::BaseTableOp baseTableOp) {
         res = std::make_unique<BaseTableLowering>(baseTableOp);
      })
      .Case<mlir::relalg::ConstRelationOp>([&](mlir::relalg::ConstRelationOp constRelationOp) {
        res = std::make_unique<ConstRelLowering>(constRelationOp);
      })
      .Case<mlir::relalg::MaterializeOp>([&](mlir::relalg::MaterializeOp materializeOp) {
         res = std::make_unique<MaterializeLowering>(materializeOp);
      })
      .Case<mlir::relalg::SelectionOp>([&](mlir::relalg::SelectionOp selectionOp) {
         res = std::make_unique<SelectionLowering>(selectionOp);
      })
      .Case<mlir::relalg::MapOp>([&](mlir::relalg::MapOp mapOp) {
         res = std::make_unique<MapLowering>(mapOp);
      })
      .Case<mlir::relalg::CrossProductOp>([&](mlir::relalg::CrossProductOp crossProductOp) {
         res = std::make_unique<CrossProductLowering>(crossProductOp);
      })
      .Case<mlir::relalg::SortOp>([&](mlir::relalg::SortOp sortOp) {
         res = std::make_unique<SortLowering>(sortOp);
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
      getFunction().dump();
   }
};
} // end anonymous namespace

namespace mlir {
namespace relalg {
std::unique_ptr<Pass> createLowerToDBPass() { return std::make_unique<LowerToDBPass>(); }
} // end namespace relalg
} // end namespace mlir