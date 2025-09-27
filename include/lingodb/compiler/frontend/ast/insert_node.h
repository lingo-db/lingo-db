#pragma once
#include "ast_node.h"
#include "constraint.h"
#include "parsed_expression.h"

#include <memory>
#include <vector>
namespace lingodb::ast {

class InsertNode : public AstNode {
   public:
   InsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer) : AstNode(NodeType::INSERT_NODE), schema(schema), tableName(tableName), producer(producer) {}
   InsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer, std::vector<std::string> columns) : AstNode(NodeType::INSERT_NODE), schema(schema), tableName(tableName), producer(producer), columns(columns) {}

   std::string schema;
   std::string tableName;

   std::shared_ptr<TableProducer> producer;
   std::vector<std::string> columns;
};

} // namespace lingodb::ast