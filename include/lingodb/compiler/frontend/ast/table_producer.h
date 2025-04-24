#pragma once
#include "ast_node.h"
///A node that produces a relation/table
/// Takes a Table as input and produces a relation/table as output
namespace lingodb::ast {
class TableProducer : public AstNode {
   public:
   TableProducer(NodeType type) : AstNode(type) {}
   TableProducer(NodeType type, std::string alias): AstNode(type), alias(std::move(alias)) {}

   std::string alias;
};
} // namespace lingodb::ast