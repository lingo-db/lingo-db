#pragma once
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
   static constexpr NodeType TYPE = NodeType::SET_NODE;
   public:
   //TODO: Add support SetScope
   SetNode(SetType setType, std::string name);
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   SetType setType;
   std::string name;
};

class SetVariableStatement : public SetNode {
   public:
   SetVariableStatement(std::string name, std::vector<std::shared_ptr<ParsedExpression>> values);
   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;

   std::vector<std::shared_ptr<ParsedExpression>> values;
};
} // namespace lingodb::ast