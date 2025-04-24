#include "lingodb/compiler/frontend/ast/create_node.h"
namespace lingodb::ast {

CreateNode::CreateNode(std::shared_ptr<CreateInfo> createInfo) : AstNode(NodeType::CREATE_NODE), createInfo(createInfo) {
}

std::string CreateNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}

}