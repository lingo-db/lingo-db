#include "lingodb/compiler/frontend/ast/aggregation_node.h"

#include "lingodb/compiler/frontend/ast/query_node.h"
namespace lingodb::ast {
AggregationNode::AggregationNode() : AstNode(NodeType::AGGREGATION) {
}

std::string AggregationNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the aggregation node
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the aggregation node with its label
   dot.append(nodeId);
   dot.append(" [label=\"Aggregation\"];\n");

   // Add aggregation functions
   for (size_t i = 0; i < aggregations.size(); ++i) {
      if (aggregations[i]) {
         std::string aggId;
         aggId.append("node");
         aggId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(aggregations[i].get()))));

         // Create edge from aggregation node to function
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(aggId);
         dot.append(" [label=\"agg_");
         dot.append(std::to_string(i));
         dot.append("\"];\n");

         // Add the function's graph representation
         dot.append(aggregations[i]->toDotGraph(depth + 1, idGen));
      }
   }

   for (size_t i = 0; i < windowFunctions.size(); ++i) {
      if (windowFunctions[i]) {
         std::string aggId;
         aggId.append("node");
         aggId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(windowFunctions[i].get()))));

         // Create edge from aggregation node to function
         dot.append(nodeId);
         dot.append(" -> ");
         dot.append(aggId);
         dot.append(" [label=\"win_");
         dot.append(std::to_string(i));
         dot.append("\"];\n");

         // Add the function's graph representation
         dot.append(windowFunctions[i]->toDotGraph(depth + 1, idGen));
      }
   }

   // Add GROUP BY node if present
   if (groupByNode) {
      std::string groupId;
      groupId.append("node");
      groupId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(groupByNode.get()))));

      // Create edge from aggregation to group by
      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(groupId);
      dot.append(" [label=\"group_by\"];\n");

      // Add the group by's graph representation
      dot.append(groupByNode->toDotGraph(depth + 1, idGen));
   }

   return dot;
}

} // namespace lingodb::ast