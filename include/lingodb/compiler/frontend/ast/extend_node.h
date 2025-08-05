#pragma once
#include "ast_node.h"
#include "parsed_expression.h"
#include "table_producer.h"
namespace lingodb::ast {
class ExtendNode : public AstNode {
   public:
   static constexpr auto TYPE = NodeType::EXTEND_NODE;
   ExtendNode();
   explicit ExtendNode(bool hidden);
   std::vector<std::shared_ptr<ParsedExpression>> extensions;

   /**
    * When true, columns defined in this ExtendNode can only be accessed through direct column references
    * and will be excluded from wildcard (*) selections. ExtendNodes are used by the SQLCanonicalizer to define intermediate columns
    * that help in query analysis and translation but are not intended to be part of the final output  unless explicitly referenced.
   */
   bool hidden = false;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};
} // namespace lingodb::ast