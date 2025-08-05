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
   dot.append(" [label=\"GROUP BY");

   // Add grouping sets information if they exist
   if (!groupingSet.empty()) {
      dot.append("\\nGrouping Sets: {");
      for (size_t i = 0; i < groupingSet.size(); ++i) {
         if (i > 0) {
            dot.append(", ");
         }
         dot.append("{");
         bool first = true;
         for (const auto& idx : groupingSet[i]) {
            if (!first) {
               dot.append(",");
            }
            dot.append(std::to_string(idx + 1));
            first = false;
         }
         dot.append("}");
      }
      dot.append("}");
   }
   dot.append("\"];\n");

   // Add all group expressions
   for (size_t i = 0; i < groupByExpressions.size(); ++i) {
      if (groupByExpressions[i]) {
         std::string exprId;
         exprId.append("node");
         exprId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(groupByExpressions[i].get()))));

         // Create edge from group by to this expression
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(exprId);
         dot.append(" [label=\"expr ");
         dot.append(std::to_string(i + 1));
         dot.append("\"];\n");

         // Add the expression's graph representation
         dot.append(groupByExpressions[i]->toDotGraph(depth + 1, idGen));
      }
   }

   return dot;
}

} // namespace lingodb::ast