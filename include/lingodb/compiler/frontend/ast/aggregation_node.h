#ifndef LINGODB_COMPILER_FRONTEND_AST_AGGREGATION_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_AGGREGATION_NODE_H


#include "ast_node.h"
#include "group_by_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
namespace lingodb::ast {
class AggregationNode : public AstNode {
   public:
   AggregationNode();

   std::shared_ptr<GroupByNode> groupByNode;
   std::vector<std::shared_ptr<FunctionExpression>> aggregations;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast
#endif
