#pragma once
#include "lingodb/compiler/frontend/ast/query_node.h"
#include "lingodb/compiler/frontend/sql_scope.h"
namespace lingodb::ast {
class BoundSetOperationNode : public  QueryNode {
   static constexpr QueryNodeType TYPE = QueryNodeType::SET_OPERATION_NODE;
   public:
   BoundSetOperationNode(std::string alias, SetOperationType setType,  bool setOpAll, std::shared_ptr<TableProducer> boundLeft, std::shared_ptr<TableProducer> boundRight, std::shared_ptr<analyzer::SQLScope> leftScope, std::shared_ptr<analyzer::SQLScope> rightScope);
   SetOperationType setType;
   bool setOpAll = false;
   std::shared_ptr<TableProducer> boundLeft;
   std::shared_ptr<TableProducer> boundRight;

   std::shared_ptr<analyzer::SQLScope> leftScope;
   std::shared_ptr<analyzer::SQLScope> rightScope;

   std::shared_ptr<NamedResult> leftMapping;
   std::shared_ptr<NamedResult> rightMapping;

   std::string toString(uint32_t depth) override;
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

};
}