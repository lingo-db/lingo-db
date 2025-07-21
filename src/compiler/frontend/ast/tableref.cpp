#include "lingodb/compiler/frontend/ast/tableref.h"
namespace lingodb::ast {
///BaseTableRef
BaseTableRef::BaseTableRef(TableDescription tableDescription) : TableRef(TYPE), catalogName(tableDescription.database), schemaName(tableDescription.schema), tableName(tableDescription.table) {
}

std::string BaseTableRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the base table reference
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create label with all table information
   std::string label = "BaseTable\\n table: " + tableName + "\\n" +
      "schema: " + schemaName + "\\n" +
      "catalog: " + catalogName;

   // Add alias if it's not empty
   if (!alias.empty()) {
      label += "\\nalias: " + alias;
   }

   // Create the node with all information
   dot += nodeId + " [label=\"" + label + "\"];\n";

   return dot;
}

/// JoinRef
JoinRef::JoinRef(JoinType type, JoinCondType refType) : TableRef(TYPE), type(type), refType(refType) {
}

std::string JoinRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};
   // Create node identifier for the join
   std::string nodeId;
   nodeId.append("node");
   nodeId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this))));

   // Create the join node with its label
   dot.append(nodeId);
   dot.append(" [label=\"JoinRef\ntype: ");

   // Add join type
   switch (type) {
      case JoinType::LEFT: dot.append("left"); break;
      case JoinType::RIGHT: dot.append("right"); break;
      case JoinType::INNER: dot.append("inner"); break;
      case JoinType::OUTER: dot.append("outer"); break;
      case JoinType::SEMI: dot.append("left semi"); break;
      case JoinType::ANTI: dot.append("left anti"); break;
      case JoinType::MARK: dot.append("mark"); break;
      case JoinType::SINGLE: dot.append("single"); break;
      case JoinType::RIGHT_SEMI: dot.append("right semi"); break;
      case JoinType::RIGHT_ANTI: dot.append("right anti"); break;
      default: dot.append("invalid"); break;
   }

   dot.append(" JOIN\\n");

   // Add join condition type
   switch (refType) {
      case JoinCondType::NATURAL: dot.append("natural"); break;
      case JoinCondType::CROSS: dot.append("cross"); break;
      case JoinCondType::POSITIONAL: dot.append("positional"); break;
      case JoinCondType::ASOF: dot.append("asof"); break;
      case JoinCondType::DEPENDENT: dot.append("dependent"); break;
      default: break;
   }

   dot.append("\"];\n");

   // Add left side
   if (left) {
      std::string leftId;
      leftId.append("node");
      leftId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(left.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(leftId);
      dot.append(" [label=\"left\"];\n");
      dot.append(left->toDotGraph(depth + 1, idGen));
   }
   if (right) {
      std::string rightId;
      rightId.append("node");
      rightId.append(std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(right.get()))));

      dot.append(nodeId);
      dot.append(" -> ");
      dot.append(rightId);
      dot.append(" [label=\"right\"];\n");
      dot.append(right->toDotGraph(depth + 1, idGen));
   }

   return dot;
}

///SuqueryRef
SubqueryRef::SubqueryRef(std::shared_ptr<QueryNode> subSelectNode) : TableRef(TYPE), subSelectNode(std::move(subSelectNode)) {
}

std::string SubqueryRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot{};

   // Create node identifier for the SELECT node
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));
   dot += nodeId + " [label=\"Subquery\"];\n";

   // Handle select list
   if (subSelectNode) {
      std::string selectListId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(subSelectNode.get())));
      dot += nodeId + " -> " + selectListId + " [label=\"\"];\n";
      dot += subSelectNode->toDotGraph(depth + 1, idGen);
   }

   return dot;
}

/// ExpressionListRef
ExpressionListRef::ExpressionListRef(std::vector<std::vector<std::shared_ptr<ParsedExpression>>> values) : TableRef(TYPE), values(std::move(values)) {
}
std::string ExpressionListRef::toDotGraph(uint32_t depth, NodeIdGenerator& idGen) {
   std::string dot;

   // Create node identifier for the expression list
   std::string nodeId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(this)));

   // Create label for the expression list node
   dot += nodeId + " [label=\"ExpressionList\"];\n";

   // Add edges to each expression in the 2D list
   for (size_t row = 0; row < values.size(); ++row) {
      const auto& rowVec = values[row];
      for (size_t col = 0; col < rowVec.size(); ++col) {
         if (rowVec[col]) {
            std::string exprId = "node" + std::to_string(idGen.getId(reinterpret_cast<uintptr_t>(rowVec[col].get())));
            dot += nodeId + " -> " + exprId + " [label=\"expr[" + std::to_string(row) + "][" + std::to_string(col) + "]\"];\n";
            dot += rowVec[col]->toDotGraph(depth + 1, idGen);
         }
      }
   }

   return dot;
}
} // namespace lingodb::ast