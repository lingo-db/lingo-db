#ifndef LINGODB_COMPILER_FRONTEND_AST_TABLE_PRODUCER_H
#define LINGODB_COMPILER_FRONTEND_AST_TABLE_PRODUCER_H


#include "ast_node.h"
///A node that produces a relation/table
namespace lingodb::ast {
class TableProducer : public AstNode {
   public:
   TableProducer(NodeType type) : AstNode(type) {}
   TableProducer(NodeType type, std::string alias) : AstNode(type), alias(std::move(alias)) {}

   std::string alias;
};
} // namespace lingodb::ast
#endif
