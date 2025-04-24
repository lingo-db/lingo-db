#include "lingodb/compiler/frontend/ast/bound/bound_aggregation.h"
namespace lingodb::ast {
BoundAggregationNode::BoundAggregationNode(std::shared_ptr<BoundGroupByNode> groupByNode, std::vector<std::shared_ptr<BoundFunctionExpression>> aggregations, std::vector<std::shared_ptr<BoundExpression>> toMapExpressions, std::string mapName) : AstNode(NodeType::BOUND_AGGREGATION), groupByNode(groupByNode), aggregations(aggregations), toMapExpressions(toMapExpressions), mapName(mapName) {
}
std::string BoundAggregationNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast