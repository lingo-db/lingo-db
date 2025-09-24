#include "lingodb/compiler/frontend/ast/insert_node.h"
namespace lingodb::ast {
InsertNode::InsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer) : AstNode(NodeType::INSERT_NODE), schema(schema), tableName(tableName), producer(producer) {
}
InsertNode::InsertNode(std::string schema, std::string tableName, std::shared_ptr<TableProducer> producer, std::vector<std::string> columns) : AstNode(NodeType::INSERT_NODE), schema(schema), tableName(tableName), producer(producer), columns(columns) {
}
std::string InsertNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the InsertNode
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create label for the InsertNode
   dot += nodeId + " [label=\"Insert\"];\n";

   // Add edge to producer if present
   if (producer) {
      std::string producerId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(producer.get())));
      dot += nodeId + " -> " + producerId + " [label=\"producer\"];\n";
      dot += producer->toDotGraph(depth + 1, idGen);
   }

   return dot;
}
} // namespace lingodb::ast