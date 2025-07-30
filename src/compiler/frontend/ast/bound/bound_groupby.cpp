#include "lingodb/compiler/frontend/ast/bound/bound_groupby.h"
namespace lingodb::ast {
BoundGroupByNode::BoundGroupByNode(std::vector<std::shared_ptr<NamedResult>> groupNamedResults, std::vector<std::set<size_t>> groupingSet) : AstNode(NodeType::BOUND_GROUP_BY), groupNamedResults(groupNamedResults), groupingSet(groupingSet) {
}
std::string BoundGroupByNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast