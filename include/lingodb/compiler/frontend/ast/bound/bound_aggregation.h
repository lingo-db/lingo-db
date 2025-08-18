#pragma once
#include "bound_expression.h"
#include "bound_groupby.h"
namespace lingodb::ast {
class BoundAggregationNode : public AstNode {
   public:
   BoundAggregationNode(std::shared_ptr<BoundGroupByNode> groupByNode, std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations, std::vector<std::shared_ptr<BoundExpression>> toMapExpressions, std::string mapName, std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr);

   std::shared_ptr<BoundGroupByNode> groupByNode;
   std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations;
   //TODO add name of tmp attr
   std::vector<std::shared_ptr<BoundExpression>> toMapExpressions;
   //TODO use context for this kind of information in future. context->getAttribute(id)....
   [[deprecated]]
   std::string mapName;

   std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr;



   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}