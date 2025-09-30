#ifndef LINGODB_COMPILER_FRONTEND_AST_SET_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_SET_NODE_H

#include "ast_node.h"
#include "parsed_expression.h"

#include <memory>

namespace lingodb::ast {
enum class SetType : uint8_t {
   SET = 0,
   RESET = 1,
};
/**
 * Node for the different Set statements: VariableSetStmt, VariableShowStmt
 */
class SetNode : public AstNode {
   static constexpr NodeType cType = NodeType::SET_NODE;

   public:
   SetNode(SetType setType, std::string name) : AstNode(cType), setType(setType), name(name) {}

   SetType setType;
   std::string name;
};

class SetVariableStatement : public SetNode {
   public:
   SetVariableStatement(std::string name, std::vector<std::shared_ptr<ParsedExpression>> values) : SetNode(SetType::SET, name), values(values) {}

   std::vector<std::shared_ptr<ParsedExpression>> values;
};
} // namespace lingodb::ast
#endif
