#pragma once
#include "ast_node.h"
#include "group_by_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
namespace lingodb::ast {
class AggregationNode : public AstNode {
   public:
   AggregationNode();

   std::shared_ptr<GroupByNode> groupByNode;
   std::vector<std::shared_ptr<FunctionExpression>> aggregations;
   std::vector<std::shared_ptr<WindowExpression>> windowFunctions;
   //TODO having clause

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast