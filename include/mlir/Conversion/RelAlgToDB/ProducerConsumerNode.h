#ifndef MLIR_CONVERSION_RELALGTODB_PRODUCERCONSUMERNODE_H
#define MLIR_CONVERSION_RELALGTODB_PRODUCERCONSUMERNODE_H

#include "llvm/ADT/ScopedHashTable.h"

#include "mlir/Dialect/RelAlg/IR/RelAlgOps.h"
#include "mlir/Dialect/RelAlg/IR/RelationalAttribute.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <iostream>
#include <memory>

namespace mlir {
namespace relalg {
class LoweringContext {
   llvm::ScopedHashTable<const mlir::relalg::RelationalAttribute*, mlir::Value> symbolTable;

   public:
   using AttributeResolverScope = llvm::ScopedHashTableScope<const mlir::relalg::RelationalAttribute*, mlir::Value>;

   mlir::Value getValueForAttribute(const mlir::relalg::RelationalAttribute* attribute) const {
      if (!symbolTable.lookup(attribute)) {
         assert(symbolTable.count(attribute));
      }

      return symbolTable.lookup(attribute);
   }
   mlir::Value getUnsafeValueForAttribute(const mlir::relalg::RelationalAttribute* attribute) const {
      return symbolTable.lookup(attribute);
   }
   void setValueForAttribute(AttributeResolverScope& scope, const mlir::relalg::RelationalAttribute* iu, mlir::Value v) {
      //      assert(!!v);
      symbolTable.insertIntoScope(&scope, iu, v);
   }
   AttributeResolverScope createScope() {
      return AttributeResolverScope(symbolTable);
   }
   std::unordered_map<size_t, mlir::Value> builders;
   size_t getBuilderId() {
      static size_t id = 0;
      return id++;
   }
   std::unordered_map<mlir::Operation*, std::pair<mlir::Value, std::vector<mlir::relalg::RelationalAttribute*>>> materializedTmp;
};
void mergeRelatinalBlock(mlir::Block* dest,mlir::Block* source, LoweringContext& context, LoweringContext::AttributeResolverScope& scope);
static const mlir::function_ref<void(mlir::OpBuilder&, mlir::Location)> noBuilder=nullptr;
class ProducerConsumerNode {
   protected:
   ProducerConsumerNode* consumer;
   std::vector<std::unique_ptr<ProducerConsumerNode>> children;
   std::vector<size_t> requiredBuilders;
   mlir::relalg::Attributes requiredAttributes;
   Value flag;
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

   virtual void addRequiredBuilders(std::vector<size_t> requiredBuilders) {
      this->requiredBuilders.insert(this->requiredBuilders.end(), requiredBuilders.begin(), requiredBuilders.end());
      for (auto& child : children) {
         child->addRequiredBuilders(requiredBuilders);
      }
   }
   void setFlag(mlir::Value flag) {
      this->flag = flag;
      for (auto& child : children) {
         child->setFlag(flag);
      }
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) = 0;
   virtual mlir::relalg::Attributes getAvailableAttributes() = 0;
   virtual void consume(ProducerConsumerNode* child, mlir::OpBuilder& builder, LoweringContext& context) = 0;
   virtual void produce(LoweringContext& context, mlir::OpBuilder& builder) = 0;
   virtual void done() {}
   virtual ~ProducerConsumerNode() {}
};
class NoopNode : public mlir::relalg::ProducerConsumerNode {
   public:
   NoopNode() : mlir::relalg::ProducerConsumerNode({}) {
   }
   virtual void setInfo(ProducerConsumerNode* consumer, mlir::relalg::Attributes requiredAttributes) override{};
   virtual mlir::relalg::Attributes getAvailableAttributes() override { return {}; };
   virtual void consume(ProducerConsumerNode* child, mlir::OpBuilder& builder, LoweringContext& context) override{};
   virtual void produce(LoweringContext& context, mlir::OpBuilder& builder) override{};
   virtual ~NoopNode() {}
};
class ProducerConsumerNodeRegistry {
   static bool registeredBaseTableOp;
   static bool registeredConstRelOp;
   static bool registeredMaterializeOp;
   static bool registeredSelectionOp;
   static bool registeredMapOp;
   static bool registeredCrossProductOp;
   static bool registeredSortOp;
   static bool registeredAggregationOp;
   static bool registeredInnerJoinOp;
   static bool registeredSemiJoinOp;
   static bool registeredAntiSemiJoinOp;
   static bool registeredRenamingOp;
   static bool registeredProjectionOp;
   static bool registeredLimitOp;
   static bool registeredOuterJoinOp;
   static bool registeredSingleJoinOp;
   static bool registeredMarkJoinOp;
   static bool registeredTmpOp;
   static bool registeredCollectionJoinOp;
   std::unordered_map<std::string, std::function<std::unique_ptr<mlir::relalg::ProducerConsumerNode>(mlir::Operation*)>> nodes;
   ProducerConsumerNodeRegistry() {
      bool res = true;
      res &= registeredBaseTableOp;
      res &= registeredConstRelOp;
      res &= registeredMaterializeOp;
      res &= registeredSelectionOp;
      res &= registeredMapOp;
      res &= registeredCrossProductOp;
      res &= registeredSortOp;
      res &= registeredAggregationOp;
      res &= registeredInnerJoinOp;
      res &= registeredSemiJoinOp;
      res &= registeredAntiSemiJoinOp;
      res &= registeredRenamingOp;
      res &= registeredProjectionOp;
      res &= registeredLimitOp;
      res &= registeredOuterJoinOp;
      res &= registeredSingleJoinOp;
      res &= registeredMarkJoinOp;
      res &= registeredTmpOp;
      res &= registeredCollectionJoinOp;
      if(res){
         llvm::dbgs()<<"loading producer nodes failed\n";
      }
   }

   public:
   static ProducerConsumerNodeRegistry& getRegistry() {
      static ProducerConsumerNodeRegistry registry;
      return registry;
   }
   template <typename FnT, typename T = typename llvm::function_traits<std::decay_t<FnT>>::template arg_t<0>>
   static bool registerNode(FnT&& callBack) {
      std::string x = T::getOperationName().str();
      getRegistry().nodes.insert({x, [callBack = callBack](mlir::Operation* op) { return callBack(mlir::cast<T>(op)); }});
      return true;
   }
   static std::unique_ptr<mlir::relalg::ProducerConsumerNode> createNode(mlir::Operation* operation) {
      std::string opName = operation->getName().getStringRef().str();
      if (getRegistry().nodes.count(opName)) {
         return getRegistry().nodes[opName](operation);
      } else {
         assert("could not create node" && false);
         return std::unique_ptr<mlir::relalg::ProducerConsumerNode>();
      }
   }
};
} // end namespace relalg
} // end namespace mlir

#endif // MLIR_CONVERSION_RELALGTODB_PRODUCERCONSUMERNODE_H