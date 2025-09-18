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
   BoundInsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer, std::vector<std::string> columnsToInsert, std::unordered_map<std::string, NullableType> allColumnsAndTypes);

   std::string schema;
   std::string tableName;

   std::shared_ptr<TableProducer> producer;
   std::vector<std::string> columnsToInsert;

   std::unordered_map<std::string, NullableType> allColumnsAndTypes;

   std::string toDotGraph(uint32_t depth, NodeIdGenerator& idGen) override;
};

} // namespace lingodb::ast