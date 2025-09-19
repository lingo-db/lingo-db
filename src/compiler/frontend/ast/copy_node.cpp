#include "lingodb/compiler/frontend/ast/copy_node.h"
namespace lingodb::ast {
CopyNode::CopyNode() : AstNode(kType), copyInfo(std::make_shared<CopyInfo>()) {
}
std::string CopyNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast