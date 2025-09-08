#include "lingodb/compiler/frontend/ast/set_node.h"
namespace lingodb::ast {
 SetNode::SetNode(SetType setType, std::string name) : AstNode(TYPE), setType(setType), name(name) {

}
 std::string SetNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
 }

 /**
 * SetVariableStatement
 */
SetVariableStatement::SetVariableStatement(std::string name, std::vector<std::shared_ptr<ParsedExpression>> values) : SetNode(SetType::SET, name), values(values) {
}
std::string SetVariableStatement::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
    return "";
}
} // namespace lingodb::ast