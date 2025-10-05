#pragma once
#include "bound_expression.h"
#include "bound_groupby.h"
namespace lingodb::ast {
class BoundAggregationNode : public AstNode {
   public:
   BoundAggregationNode(std::shared_ptr<BoundGroupByNode> groupByNode, std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations, std::vector<std::shared_ptr<BoundExpression>> toMapExpressions, std::string mapName, std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr) : AstNode(NodeType::BOUND_AGGREGATION), groupByNode(groupByNode), aggregations(aggregations), toMapExpressions(toMapExpressions), mapName(mapName), evalBeforeAggr(std::move(evalBeforeAggr)) {

   }

   std::shared_ptr<BoundGroupByNode> groupByNode;
   std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations;
   std::vector<std::shared_ptr<BoundExpression>> toMapExpressions;
   std::string mapName;

   std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr;
};
}