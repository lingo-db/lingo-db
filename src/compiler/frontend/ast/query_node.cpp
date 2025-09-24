#include "lingodb/compiler/frontend/ast/query_node.h"

#include "lingodb/compiler/frontend/ast/tableref.h"
namespace lingodb::ast {
/**
 * SelectNode
 */
SelectNode::SelectNode() : QueryNode(TYPE) {
}
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
/**
 * SetOperation
 */
SetOperationNode::SetOperationNode(SetOperationType setType, std::shared_ptr<TableProducer> left, std::shared_ptr<TableProducer> right) : QueryNode(TYPE), setType(setType), left(left), right(right) {
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

/**
 * ValuesQueryNode
 */
ValuesQueryNode::ValuesQueryNode(std::shared_ptr<ExpressionListRef> expressionListRef) : QueryNode(TYPE), expressionListRef(std::move(expressionListRef)) {
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

/**
 * CTENode
 */
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


} // namespace lingodb::ast