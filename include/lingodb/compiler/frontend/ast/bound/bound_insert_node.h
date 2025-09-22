#pragma once

#include "lingodb/catalog/Column.h"
#include "lingodb/compiler/frontend/ast/ast_node.h"
#include "lingodb/compiler/frontend/ast/table_producer.h"
#include "lingodb/compiler/frontend/frontend_type.h"

#include <memory>
#include <vector>
namespace lingodb::ast {

class BoundInsertNode : public AstNode {
   public:
   BoundInsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer, std::vector<std::string> columnsToInsert, std::unordered_map<std::string, NullableType> allColumnsAndTypes) : AstNode(NodeType::BOUND_INSERT_NODE), schema(schema), tableName(tableName), producer(producer), columnsToInsert(columnsToInsert), allColumnsAndTypes(allColumnsAndTypes) {}

   std::string schema;
   std::string tableName;

   std::shared_ptr<TableProducer> producer;
   std::vector<std::string> columnsToInsert;

   std::unordered_map<std::string, NullableType> allColumnsAndTypes;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override {
      return "";
   };
};

} // namespace lingodb::ast