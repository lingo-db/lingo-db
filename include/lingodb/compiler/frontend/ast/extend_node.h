#ifndef LINGODB_COMPILER_FRONTEND_AST_EXTEND_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_EXTEND_NODE_H

#include "ast_node.h"
#include "parsed_expression.h"
#include "table_producer.h"
namespace lingodb::ast {
class ExtendNode : public AstNode {
   public:
   static constexpr auto cType = NodeType::EXTEND_NODE;
   ExtendNode() : AstNode(NodeType::EXTEND_NODE) {}
   explicit ExtendNode(bool hidden) : AstNode(NodeType::EXTEND_NODE), hidden(hidden) {}
   std::vector<std::shared_ptr<ParsedExpression>> extensions;

   /**
    * When true, columns defined in this ExtendNode can only be accessed through direct column references
    * and will be excluded from wildcard (*) selections. ExtendNodes are used by the SQLCanonicalizer to define intermediate columns
    * that help in query analysis and translation but are not intended to be part of the final output  unless explicitly referenced.
   */
   bool hidden = false;
};
} // namespace lingodb::ast
#endif
