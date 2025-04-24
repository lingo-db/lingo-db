#include "lingodb/compiler/frontend/ast/group_by_node.h"
#include "lingodb/compiler/frontend/ast/query_node.h"
namespace lingodb::ast {

std::string GroupByNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the group by node
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the group by node with its label
   dot.append(nodeId);
   dot.append(" [label=\"GROUP BY\"];\n");

   // Add all group expressions
   for (size_t i = 0; i < group_expressions.size(); ++i) {
      if (group_expressions[i]) {
         std::string exprId;
         exprId.append("node");
         exprId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(group_expressions[i].get()))));

         // Create edge from group by to this expression
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(exprId);
         dot.append(" [label=\"expr ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Add the expression's graph representation
         dot.append(group_expressions[i]->toDotGraph(depth + 1, idGen));
      }
   }

   // Add grouping sets information if present
   if (!grouping_sets.empty()) {
      for (size_t i = 0; i < grouping_sets.size(); ++i) {
         std::string setId;
         setId.append("node");
         setId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));
         setId.append("_set");
         setId.append(std::to_string(i));

         // Create the set node
         dot.append(setId);
         dot.append(" [label=\"Set ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Connect group by to set
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(setId);
         dot.append(" [label=\"grouping set\"];\n");
      }
   }

   return dot;
}
} // namespace lingodb::ast