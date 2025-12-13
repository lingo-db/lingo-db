#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_AGGREGATION_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_AGGREGATION_H

#include "bound_expression.h"
#include "bound_extend_node.h"
#include "bound_groupby.h"
namespace lingodb::ast {
class BoundAggregationNode : public AstNode {
   public:
   BoundAggregationNode(std::shared_ptr<BoundGroupByNode> groupByNode, std::vector<std::vector<std::shared_ptr<BoundFunctionExpression>>> aggregations, std::vector<std::shared_ptr<BoundExpression>> toMapExpressions, std::string mapName, std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr) : AstNode(NodeType::BOUND_AGGREGATION), groupByNode(groupByNode), aggregations(aggregations), toMapExpressions(toMapExpressions), mapName(mapName), evalBeforeAggr(std::move(evalBeforeAggr)) {
   }

   std::shared_ptr<BoundGroupByNode> groupByNode;
   /**
    * List of a List of Aggregation Function, used for normal aggregations and rollups
    */
   std::vector<std::vector<std::shared_ptr<BoundFunctionExpression>>> aggregations;
   std::vector<std::shared_ptr<BoundExpression>> toMapExpressions;
   std::string mapName;

   std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr;

   std::vector<std::vector<std::shared_ptr<BoundExpression>>> reconstructs;
};
} // namespace lingodb::ast
#endif
