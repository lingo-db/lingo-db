#ifndef LINGODB_COMPILER_FRONTEND_AST_INSERT_NODE_H
#define LINGODB_COMPILER_FRONTEND_AST_INSERT_NODE_H


#include "ast_node.h"
#include "constraint.h"
#include "parsed_expression.h"

#include <memory>
#include <vector>
namespace lingodb::ast {

class InsertNode : public AstNode {
   public:
   InsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer);
   InsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer, std::vector<std::string> columns);

   std::string schema;
   std::string tableName;

   //TODO conflict etc
   std::shared_ptr<TableProducer> producer;
   std::vector<std::string> columns;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

} // namespace lingodb::ast
#endif
