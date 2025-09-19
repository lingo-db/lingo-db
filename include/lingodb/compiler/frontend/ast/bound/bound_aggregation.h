#ifndef LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_AGGREGATION_H
#define LINGODB_COMPILER_FRONTEND_AST_BOUND_BOUND_AGGREGATION_H
#include "bound_groupby.h"


namespace lingodb::ast {
class BoundAggregationNode : public AstNode {
   public:
   BoundAggregationNode(std::shared_ptr<BoundGroupByNode> groupByNode, std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations, std::vector<std::shared_ptr<BoundExpression>> toMapExpressions, std::string mapName, std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr);

   std::shared_ptr<BoundGroupByNode> groupByNode;
   std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations;
   std::vector<std::shared_ptr<BoundExpression>> toMapExpressions;
   std::string mapName;

   std::vector<std::shared_ptr<BoundExpression>> evalBeforeAggr;



   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast
#endif
