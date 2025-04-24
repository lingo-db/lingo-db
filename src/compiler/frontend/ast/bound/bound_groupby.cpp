#include "lingodb/compiler/frontend/ast/bound/bound_groupby.h"
namespace lingodb::ast {
BoundGroupByNode::BoundGroupByNode(std::vector<std::shared_ptr<BoundExpression>> groupExpressions) : AstNode(NodeType::BOUND_GROUP_BY), groupExpressions(groupExpressions) {
}
std::string BoundGroupByNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast