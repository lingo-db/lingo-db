#include "lingodb/compiler/frontend/ast/bound/bound_extend_node.h"
namespace lingodb::ast {
BoundExtendNode::BoundExtendNode(std::string mapName, std::vector<std::shared_ptr<BoundExpression>> extensions) : AstNode(TYPE), mapName(mapName), extensions(std::move(extensions)) {
}
std::string BoundExtendNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
}