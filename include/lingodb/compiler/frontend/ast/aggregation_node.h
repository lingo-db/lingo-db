#ifndef LINGODB_COMPILER_FRONTEND_AST_AGGREGATION_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_AGGREGATION_NODE_H

#include "ast_node.h"
#include "group_by_node.h"
#include "lingodb/compiler/frontend/ast/parsed_expression.h"
namespace lingodb::ast {
class AggregationNode : public AstNode {
   struct AggregationsHash {
      size_t operator()(const std::pair<std::shared_ptr<FunctionExpression>, size_t>& expr) const {
         return expr.first ? expr.first->hash() : 0;
      }
   };
   struct AggregationsEqual {
      bool operator()(const std::pair<std::shared_ptr<FunctionExpression>, size_t>& lhs,
                      const std::pair<std::shared_ptr<FunctionExpression>, size_t>& rhs) const {
         if (lhs.first == rhs.first) return true;
         if (!lhs.first || !rhs.first) return false;
         return *lhs.first == *rhs.first;
      }
   };
   public:
   AggregationNode() : AstNode(NodeType::AGGREGATION) {}

   std::shared_ptr<GroupByNode> groupByNode;
   std::unordered_set<std::pair<std::shared_ptr<FunctionExpression>, size_t>, AggregationsHash, AggregationsEqual > aggregations;
};
} // namespace lingodb::ast
#endif
