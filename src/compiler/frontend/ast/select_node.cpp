#include "lingodb/compiler/frontend/ast/select_node.h"

#include <iostream>
namespace lingodb::ast {
SelectNode::SelectNode() : QueryNode(TYPE) {
}
std::string SelectNode::toString(uint32_t depth) {
   return "SelectNode";
};
SelectNode::~SelectNode() = default;

std::string SelectNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the SELECT node
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));
   dot += nodeId + " [label=\"SelectNode\"];\n";

   // Handle select list
   if (select_list) {
      std::string selectListId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(select_list.get())));
      dot += nodeId + " -> " + selectListId + " [label=\"select list\"];\n";
      dot += select_list->toDotGraph(depth + 1, idGen);
   }

   // Handle FROM clause
   if (from_clause) {
      std::string fromId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(from_clause.get())));
      dot += nodeId + " -> " + fromId + " [label=\"FROM\"];\n";
      dot += from_clause->toDotGraph(depth + 1, idGen);
   }

   // Handle WHERE clause
   if (where_clause) {
      std::string whereId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(where_clause.get())));
      dot += nodeId + " -> " + whereId + " [label=\"WHERE\"];\n";
      dot += where_clause->toDotGraph(depth + 1, idGen);
   }

   // Handle GROUP BY
   if (groups) {
      std::string groupsId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(groups.get())));
      dot += nodeId + " -> " + groupsId + " [label=\"GROUP BY\"];\n";
      dot += groups->toDotGraph(depth + 1, idGen);
   }

   // Handle HAVING clause
   if (having) {
      std::string havingId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(having.get())));
      dot += nodeId + " -> " + havingId + " [label=\"HAVING\"];\n";
      dot += having->toDotGraph(depth + 1, idGen);
   }

   for (const auto& modifier : modifiers) {
      if (modifier) {
         std::string modifierId;
         modifierId.append("node");
         modifierId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(modifier.get()))));

         // Add edge from select node to modifier
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(modifierId);

         // Add appropriate label based on modifier type
         dot.append(" [label=\"");
         switch (modifier->modifierType) {
            case ResultModifierType::ORDER_BY:
               dot.append("ORDER BY");
               break;
            case ResultModifierType::LIMIT:
               dot.append("LIMIT");
               break;
            case ResultModifierType::OFFSET:
               dot.append("OFFSET");
               break;
         }
         dot.append("\"];\n");

         // Add the modifier's graph representation
         dot.append(modifier->toDotGraph(depth + 1));
      }
   }

   std::shared_ptr<PipeOperator> currentOp = startPipeOperator;
   size_t i = 0;
   while (currentOp) {
      dot.append(currentOp->node->toDotGraph(depth + 1, idGen));
      std::string startOpId;
      startOpId.append("node");
      startOpId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(currentOp->node.get()))));

      // Create edge from pipe select to start operator
      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(startOpId);

      dot.append(" [label=\"next\"];\n");

      nodeId.clear();
      nodeId.append("node");
      nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(currentOp->node.get()))));

      currentOp = nullptr;
      i++;
   }

   return dot;
}

} // namespace lingodb::ast