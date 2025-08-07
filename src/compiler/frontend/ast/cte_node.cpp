#include "lingodb/compiler/frontend/ast/cte_node.h"
namespace lingodb::ast {
std::string CTENode::toString(uint32_t depth) {
   return "";
}
std::string CTENode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the CTE node
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create label with CTE and column aliases
   std::string label = "CTE";
   if (!columnNames.empty()) {
      label += "\\nColumns: ";
      for (size_t i = 0; i < columnNames.size(); ++i) {
         if (i > 0) label += ", ";
         label += columnNames[i];
      }
   }

   dot += nodeId + " [label=\"" + label + "\"];\n";

   // Add query node if present
   if (query) {
      std::string queryId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(query.get())));
      dot += nodeId + " -> " + queryId + " [label=\"query\"];\n";
      dot += query->toDotGraph(depth + 1, idGen);
   }

   // Add child node if present
   if (child) {
      std::string childId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(child.get())));
      dot += nodeId + " -> " + childId + " [label=\"child\"];\n";
      dot += child->toDotGraph(depth + 1, idGen);
   }

   return dot;
}
}