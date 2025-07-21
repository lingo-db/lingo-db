#include "lingodb/compiler/frontend/ast/extend_node.h"
namespace lingodb::ast {
ExtendNode::ExtendNode() : AstNode(NodeType::EXTEND_NODE) {
}

std::string ExtendNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the ExtendNode
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create the ExtendNode label
   dot += nodeId + " [label=\"ExtendNode\"];\n";

   // Add all extension expressions if present
   for (size_t i = 0; i < extensions.size(); ++i) {
      if (extensions[i]) {
         std::string extId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(extensions[i].get())));
         dot += nodeId + " -> " + extId + " [label=\"extension " + std::to_string(i + 1) + "\"];\n";
         dot += extensions[i]->toDotGraph(depth + 1, idGen);
      }
   }

   // Add input node if present

   return dot;
}
}