#pragma once
#include "ast_node.h"
#include "parsed_expression.h"
#include "table_producer.h"
namespace lingodb::ast {
class ExtendNode : public AstNode {
   public:
   static constexpr auto TYPE = NodeType::EXTEND_NODE;
   ExtendNode();
   std::vector<std::shared_ptr<ParsedExpression>> extensions;



   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
}