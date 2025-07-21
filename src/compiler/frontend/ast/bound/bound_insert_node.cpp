#include "lingodb/compiler/frontend/ast/bound/bound_insert_node.h"
namespace lingodb::ast {

BoundInsertNode::BoundInsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer, std::vector<std::string> columnsToInsert, std::unordered_map<std::string, catalog::NullableType> allColumnsAndTypes) : AstNode(NodeType::BOUND_INSERT_NODE), schema(schema), tableName(tableName), producer(producer), columnsToInsert(columnsToInsert), allColumnsAndTypes(allColumnsAndTypes) {
}
std::string BoundInsertNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   return "";
}
} // namespace lingodb::ast