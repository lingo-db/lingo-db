#include "lingodb/compiler/frontend/ast/copy_node.h"
namespace lingodb::ast {
CopyNode::CopyNode() : AstNode(TYPE), copyInfo(std::make_shared<CopyInfo>()) {
}
std::string CopyNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
}