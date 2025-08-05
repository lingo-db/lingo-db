#include "lingodb/compiler/frontend/ast/query_node.h"

#include "lingodb/compiler/frontend/ast/tableref.h"
namespace lingodb::ast {
SetOperationNode::SetOperationNode(SetOperationType setType, std::shared_ptr<TableProducer> left, std::shared_ptr<TableProducer> right) : QueryNode(TYPE), setType(setType), left(left), right(right) {
}
std::string SetOperationNode::toString(uint32_t depth) {
   return "";
}
std::string SetOperationNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the SetOperationNode
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Label with set operation type and ALL flag
   std::string label = "SetOperation\\n";
   switch (setType) {
      case SetOperationType::UNION: label += "UNION"; break;
      case SetOperationType::EXCEPT: label += "EXCEPT"; break;
      case SetOperationType::INTERSECT: label += "INTERSECT"; break;
      case SetOperationType::UNION_BY_NAME: label += "UNION_BY_NAME"; break;
      default: label += "NONE"; break;
   }
   if (setOpAll) label += "\\nALL";
   dot += nodeId + " [label=\"" + label + "\"];\n";

   if (input) {
      std::string rightId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(input.get())));
      dot += nodeId + " -> " + rightId + " [label=\"input\"];\n";
      dot += input->toDotGraph(depth + 1, idGen);
   }

   // Add edge to left child if present
   if (left) {
      std::string leftId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(left.get())));
      dot += nodeId + " -> " + leftId + " [label=\"left\"];\n";
      dot += left->toDotGraph(depth + 1, idGen);
   }

   // Add edge to right child if present
   if (right) {
      std::string rightId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(right.get())));
      dot += nodeId + " -> " + rightId + " [label=\"right\"];\n";
      dot += right->toDotGraph(depth + 1, idGen);
   }

   return dot;
}

/// ExpressionListRef
ValuesQueryNode::ValuesQueryNode(std::shared_ptr<ExpressionListRef> expressionListRef) : QueryNode(TYPE), expressionListRef(std::move(expressionListRef)) {
}
std::string ValuesQueryNode::toString(uint32_t depth) {
   return "";
}
std::string ValuesQueryNode::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the ValuesQueryNode
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Label the node as VALUES
   dot += nodeId + " [label=\"VALUES\"];\n";

   // Add edge to expression list if present
   if (expressionListRef) {
      std::string exprListId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(expressionListRef.get())));
      dot += nodeId + " -> " + exprListId + " [label=\"expressions\"];\n";
      dot += expressionListRef->toDotGraph(depth + 1, idGen);
   }

   if (input) {
      std::string inputId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(input.get())));
      dot += nodeId + " -> " + inputId + " [label=\"input\"];\n";
      dot += input->toDotGraph(depth + 1, idGen);
   }

   return dot;
}


}